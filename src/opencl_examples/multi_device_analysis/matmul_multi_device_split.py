#!/usr/bin/env python3
"""
PyOpenCL multi-device matrix multiplication:
  C = A x B

Work split (rows):
  - NVIDIA GPU: uncoalesced baseline kernel (TP style)
  - Intel iGPU: Kernel 6 style optimized kernel (tiling + register blocking)

The script measures global performance including host<->device transfers.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Avoid PyOpenCL invoker cache issues on some Windows/OneDrive setups.
os.environ.setdefault("PYOPENCL_NO_CACHE", "1")
import pyopencl as cl


# -----------------------------------------------------------------------------
# Kernels
# -----------------------------------------------------------------------------

NVIDIA_UNCOALESCED_KERNEL = r"""
__kernel void mmul_uncoalesced(
    const int N,              // columns of A / rows+cols of B
    const int M,              // rows of A_sub / C_sub
    __global const float* A,  // [M, N]
    __global const float* B,  // [N, N]
    __global float* C)        // [M, N]
{
    // TP uncoalesced-style mapping: i on dim0, j on dim1
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < M && j < N) {
        float tmp = 0.0f;
        for (int k = 0; k < N; k++) {
            tmp += A[i*N + k] * B[k*N + j];
        }
        C[i*N + j] = tmp;
    }
}
"""


INTEL_KERNEL6 = r"""
#ifndef TSM
#define TSM 128
#endif
#ifndef TSN
#define TSN 128
#endif
#ifndef TSK
#define TSK 16
#endif
#ifndef WPTM
#define WPTM 8
#endif
#ifndef WPTN
#define WPTN 8
#endif

#define RTSM (TSM / WPTM)
#define RTSN (TSN / WPTN)
#define LPTA ((TSM * TSK) / (RTSM * RTSN))
#define LPTB ((TSN * TSK) / (RTSM * RTSN))
#define PAD_K 2

__kernel void mmul_kernel6(
    const int N,               // global matrix width/height
    const int M,               // rows in this split
    __global const float* A,   // [M, N]
    __global const float* BT,  // [N, N] where BT = transpose(B)
    __global float* C)         // [M, N]
{
    const int lidm = get_local_id(0);
    const int lidn = get_local_id(1);

    const int group_m = get_group_id(0);
    const int group_n = get_group_id(1);

    const int tid = lidn * RTSM + lidm;

    __local float Asub[TSM][TSK + PAD_K];
    __local float Bsub[TSN][TSK + PAD_K];

    float acc[WPTM][WPTN];
    #pragma unroll
    for (int wm = 0; wm < WPTM; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WPTN; ++wn) {
            acc[wm][wn] = 0.0f;
        }
    }

    for (int k0 = 0; k0 < N; k0 += TSK) {
        // Load A tile into local memory.
        #pragma unroll
        for (int l = 0; l < LPTA; ++l) {
            const int id = l * RTSM * RTSN + tid;
            const int row = id % TSM;
            const int col = id / TSM;

            const int g_row = group_m * TSM + row;
            const int g_col = k0 + col;

            Asub[row][col] = (g_row < M && g_col < N) ? A[g_row * N + g_col] : 0.0f;
        }

        // Load BT tile into local memory.
        #pragma unroll
        for (int l = 0; l < LPTB; ++l) {
            const int id = l * RTSM * RTSN + tid;
            const int row = id % TSN;
            const int col = id / TSN;

            const int g_row = group_n * TSN + row;
            const int g_col = k0 + col;

            Bsub[row][col] = (g_row < N && g_col < N) ? BT[g_row * N + g_col] : 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        #pragma unroll
        for (int k = 0; k < TSK; ++k) {
            float Areg[WPTM];
            #pragma unroll
            for (int wm = 0; wm < WPTM; ++wm) {
                Areg[wm] = Asub[lidm + wm * RTSM][k];
            }

            #pragma unroll
            for (int wn = 0; wn < WPTN; ++wn) {
                const float b = Bsub[lidn + wn * RTSN][k];
                #pragma unroll
                for (int wm = 0; wm < WPTM; ++wm) {
                    acc[wm][wn] = fma(Areg[wm], b, acc[wm][wn]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const int base_row = group_m * TSM + lidm;
    const int base_col = group_n * TSN + lidn;

    #pragma unroll
    for (int wm = 0; wm < WPTM; ++wm) {
        const int row = base_row + wm * RTSM;
        #pragma unroll
        for (int wn = 0; wn < WPTN; ++wn) {
            const int col = base_col + wn * RTSN;
            if (row < M && col < N) {
                C[row * N + col] = acc[wm][wn];
            }
        }
    }
}
"""


# -----------------------------------------------------------------------------
# Data classes and utilities
# -----------------------------------------------------------------------------


@dataclass
class DeviceRunResult:
    name: str
    rows: int
    run_times: list[float]
    c_sub: np.ndarray
    flops: float

    @property
    def mean_time(self) -> float:
        return float(np.mean(self.run_times)) if self.run_times else float("nan")

    @property
    def total_time(self) -> float:
        return float(np.sum(self.run_times)) if self.run_times else float("nan")

    @property
    def gflops_mean(self) -> float:
        return (self.flops / self.mean_time) / 1e9 if self.run_times else float("nan")


def round_up(x: int, tile: int) -> int:
    return ((x + tile - 1) // tile) * tile


def build_program_with_log(ctx: cl.Context, src: str, options: Optional[list[str]] = None) -> cl.Program:
    program = cl.Program(ctx, src)
    try:
        return program.build(options=options or [])
    except Exception as exc:
        logs = []
        for dev in ctx.devices:
            try:
                logs.append(f"[{dev.name}] {program.get_build_info(dev, cl.program_build_info.LOG)}")
            except Exception:
                pass
        full_log = "\n".join(logs) if logs else "<no build log>"
        raise RuntimeError(f"OpenCL build failed: {exc}\n{full_log}") from exc


def find_devices() -> tuple[cl.Device, cl.Device]:
    nvidia = None
    intel = None
    for platform in cl.get_platforms():
        for dev in platform.get_devices():
            dtype = cl.device_type.to_string(dev.type).upper()
            if "GPU" not in dtype:
                continue
            text = f"{dev.vendor} {dev.name}".lower()
            if nvidia is None and "nvidia" in text:
                nvidia = dev
            if intel is None and "intel" in text:
                intel = dev
    if nvidia is None:
        raise RuntimeError("NVIDIA GPU not found.")
    if intel is None:
        raise RuntimeError("Intel integrated GPU not found.")
    return nvidia, intel


# -----------------------------------------------------------------------------
# Per-device execution functions
# -----------------------------------------------------------------------------


def run_nvidia_uncoalesced(
    a_sub: np.ndarray,
    b_full: np.ndarray,
    runs: int,
    warmup: int,
    device: cl.Device,
    local_size: tuple[int, int] = (16, 16),
) -> DeviceRunResult:
    """
    Execute TP-style uncoalesced SGEMM on NVIDIA for one split:
      A_sub [M, N] x B_full [N, N] => C_sub [M, N]
    Includes H2D + kernel + D2H in run timing.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")

    a_sub = np.ascontiguousarray(a_sub, dtype=np.float32)
    b_full = np.ascontiguousarray(b_full, dtype=np.float32)
    m, n = a_sub.shape

    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    program = build_program_with_log(
        ctx,
        NVIDIA_UNCOALESCED_KERNEL,
        options=["-cl-fast-relaxed-math", "-cl-mad-enable"],
    )
    kernel = program.mmul_uncoalesced
    kernel.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None])

    mf = cl.mem_flags
    d_a = cl.Buffer(ctx, mf.READ_ONLY, a_sub.nbytes)
    d_b = cl.Buffer(ctx, mf.READ_ONLY, b_full.nbytes)
    d_c = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * np.dtype(np.float32).itemsize)

    global_size = (round_up(m, local_size[0]), round_up(n, local_size[1]))
    c_last = np.empty((m, n), dtype=np.float32)

    def one_pass(out: np.ndarray) -> None:
        cl.enqueue_copy(queue, d_a, a_sub, is_blocking=False)
        cl.enqueue_copy(queue, d_b, b_full, is_blocking=False)
        kernel(queue, global_size, local_size, np.int32(n), np.int32(m), d_a, d_b, d_c)
        cl.enqueue_copy(queue, out, d_c, is_blocking=False)
        queue.finish()

    warm_buf = np.empty_like(c_last)
    for _ in range(warmup):
        one_pass(warm_buf)

    run_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        one_pass(c_last)
        run_times.append(time.perf_counter() - t0)

    flops = 2.0 * m * n * n
    return DeviceRunResult(name=device.name.strip(), rows=m, run_times=run_times, c_sub=c_last, flops=flops)


def run_intel_kernel6(
    a_sub: np.ndarray,
    b_full: np.ndarray,
    runs: int,
    warmup: int,
    device: cl.Device,
) -> DeviceRunResult:
    """
    Execute Kernel 6 style SGEMM on Intel iGPU for one split:
      A_sub [M, N] x B_full [N, N] => C_sub [M, N]
    B is transposed once on host and sent as BT to improve access patterns.
    Includes H2D + kernel + D2H in run timing.
    """
    if runs < 1:
        raise ValueError("runs must be >= 1")

    # Fixed Kernel 6 parameters requested by TP.
    tsm, tsn, tsk = 128, 128, 16
    wptm, wptn = 8, 8
    rtsm, rtsn = tsm // wptm, tsn // wptn  # 16 x 16 local size

    a_sub = np.ascontiguousarray(a_sub, dtype=np.float32)
    b_full = np.ascontiguousarray(b_full, dtype=np.float32)
    bt_full = np.ascontiguousarray(b_full.T, dtype=np.float32)
    m, n = a_sub.shape

    ctx = cl.Context([device])
    queue = cl.CommandQueue(ctx)

    build_options = [
        f"-DTSM={tsm}",
        f"-DTSN={tsn}",
        f"-DTSK={tsk}",
        f"-DWPTM={wptm}",
        f"-DWPTN={wptn}",
        "-cl-fast-relaxed-math",
        "-cl-mad-enable",
    ]
    program = build_program_with_log(ctx, INTEL_KERNEL6, options=build_options)
    kernel = program.mmul_kernel6
    kernel.set_scalar_arg_dtypes([np.int32, np.int32, None, None, None])

    mf = cl.mem_flags
    d_a = cl.Buffer(ctx, mf.READ_ONLY, a_sub.nbytes)
    d_bt = cl.Buffer(ctx, mf.READ_ONLY, bt_full.nbytes)
    d_c = cl.Buffer(ctx, mf.WRITE_ONLY, m * n * np.dtype(np.float32).itemsize)

    groups_m = (m + tsm - 1) // tsm
    groups_n = (n + tsn - 1) // tsn
    local_size = (rtsm, rtsn)            # fixed 16x16
    global_size = (groups_m * rtsm, groups_n * rtsn)

    c_last = np.empty((m, n), dtype=np.float32)

    def one_pass(out: np.ndarray) -> None:
        cl.enqueue_copy(queue, d_a, a_sub, is_blocking=False)
        cl.enqueue_copy(queue, d_bt, bt_full, is_blocking=False)
        kernel(queue, global_size, local_size, np.int32(n), np.int32(m), d_a, d_bt, d_c)
        cl.enqueue_copy(queue, out, d_c, is_blocking=False)
        queue.finish()

    warm_buf = np.empty_like(c_last)
    for _ in range(warmup):
        one_pass(warm_buf)

    run_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        one_pass(c_last)
        run_times.append(time.perf_counter() - t0)

    flops = 2.0 * m * n * n
    return DeviceRunResult(name=device.name.strip(), rows=m, run_times=run_times, c_sub=c_last, flops=flops)


# -----------------------------------------------------------------------------
# Split orchestration
# -----------------------------------------------------------------------------


def split_and_run(
    a: np.ndarray,
    b: np.ndarray,
    alpha: float,
    runs: int,
    warmup: int,
    verify: bool,
) -> None:
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1].")
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != a.shape[1] or b.shape[0] != b.shape[1] or a.shape != b.shape:
        raise ValueError("A and B must be square matrices with identical shape.")

    n = a.shape[0]
    split_rows = int(round(alpha * n))
    split_rows = max(0, min(n, split_rows))

    a0 = np.ascontiguousarray(a[:split_rows, :], dtype=np.float32)   # NVIDIA
    a1 = np.ascontiguousarray(a[split_rows:, :], dtype=np.float32)   # Intel
    b = np.ascontiguousarray(b, dtype=np.float32)

    nvidia_dev, intel_dev = find_devices()

    print("\nDevices:")
    print(f"  NVIDIA: {nvidia_dev.name} ({nvidia_dev.vendor})")
    print(f"  Intel : {intel_dev.name} ({intel_dev.vendor})")

    print("\nSplit configuration:")
    print(f"  N = {n}")
    print(f"  alpha = {alpha:.3f}")
    print(f"  NVIDIA rows = {a0.shape[0]} ({100.0 * a0.shape[0] / n:.2f}%)")
    print(f"  Intel  rows = {a1.shape[0]} ({100.0 * a1.shape[0] / n:.2f}%)")

    # Multi-device run (parallel host dispatch).
    t_global0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_nv = pool.submit(run_nvidia_uncoalesced, a0, b, runs, warmup, nvidia_dev) if a0.shape[0] > 0 else None
        fut_in = pool.submit(run_intel_kernel6, a1, b, runs, warmup, intel_dev) if a1.shape[0] > 0 else None
        res_nv = fut_nv.result() if fut_nv is not None else None
        res_in = fut_in.result() if fut_in is not None else None
    orchestration_wall = time.perf_counter() - t_global0

    parts = []
    if res_nv is not None:
        parts.append(res_nv.c_sub)
    if res_in is not None:
        parts.append(res_in.c_sub)
    c = np.vstack(parts)

    # Global FLOPS across all timed runs for full NxN multiplication.
    total_flops = 2.0 * n * n * n * runs
    # Parallel timed section estimate (transfers + kernels), excluding build/setup overhead.
    active_results = [r for r in (res_nv, res_in) if r is not None]
    total_time = max(r.total_time for r in active_results)
    mean_time = total_time / runs
    gflops_total = (total_flops / total_time) / 1e9

    print("\nPer-device timing:")
    if res_nv is not None:
        print(f"  NVIDIA mean time: {res_nv.mean_time:.6f} s | GFLOPS(device): {res_nv.gflops_mean:.2f}")
    if res_in is not None:
        print(f"  Intel  mean time: {res_in.mean_time:.6f} s | GFLOPS(device): {res_in.gflops_mean:.2f}")

    print("\nGlobal timing (transfers + kernels):")
    print(f"  Total time (parallel estimate, {runs} runs): {total_time:.6f} s")
    print(f"  Mean time per run (parallel estimate): {mean_time:.6f} s")
    print(f"  Global GFLOPS: {gflops_total:.2f}")
    print(f"  Host orchestration wall time (with setup/build): {orchestration_wall:.6f} s")

    # Baseline: NVIDIA only on full matrix, same runs/warmup.
    print("\nRunning NVIDIA-only baseline (full matrix, uncoalesced)...")
    base_nv = run_nvidia_uncoalesced(a, b, runs=runs, warmup=warmup, device=nvidia_dev)
    baseline_time = base_nv.total_time
    baseline_gflops = (total_flops / baseline_time) / 1e9
    speedup = baseline_time / total_time

    print(f"  Baseline NVIDIA-only time: {baseline_time:.6f} s")
    print(f"  Baseline NVIDIA-only GFLOPS: {baseline_gflops:.2f}")
    print(f"  Speedup vs NVIDIA-only: x{speedup:.3f}")

    if verify:
        print("\nVerification (sampled entries using numpy.dot):")
        checks = [(0, 0), (n // 2, n // 2), (n - 1, n - 1)]
        max_abs = 0.0
        max_rel = 0.0
        for i, j in checks:
            ref = float(np.dot(a[i, :].astype(np.float64), b[:, j].astype(np.float64)))
            got = float(c[i, j])
            abs_err = abs(got - ref)
            rel_err = abs_err / (abs(ref) + 1e-12)
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
            print(
                f"  C[{i},{j}] -> gpu={got:.6e}, ref={ref:.6e}, "
                f"abs_err={abs_err:.3e}, rel_err={rel_err:.3e}"
            )
        print(f"  max_abs_err={max_abs:.3e}, max_rel_err={max_rel:.3e}")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyOpenCL multi-device SGEMM split (NVIDIA + Intel iGPU).")
    parser.add_argument("--n", type=int, default=8192, help="Matrix size N.")
    parser.add_argument("--alpha", type=float, default=0.8, help="Row split ratio to NVIDIA.")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs.")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up runs.")
    parser.add_argument("--verify", action="store_true", help="Enable sampled correctness checks.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    n = int(args.n)
    if n <= 0:
        raise ValueError("--n must be > 0")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    print("Generating matrices...")
    rng = np.random.default_rng(args.seed)
    a = rng.standard_normal((n, n), dtype=np.float32)
    b = rng.standard_normal((n, n), dtype=np.float32)

    split_and_run(
        a=a,
        b=b,
        alpha=float(args.alpha),
        runs=int(args.runs),
        warmup=int(args.warmup),
        verify=bool(args.verify),
    )


if __name__ == "__main__":
    main()
