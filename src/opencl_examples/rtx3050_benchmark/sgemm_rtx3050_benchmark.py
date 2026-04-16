#!/usr/bin/env python3
"""
OpenCL SGEMM benchmark tuned for large square matrices (target N=8192) on NVIDIA RTX 3050 Laptop GPU.

Main implementation:
- Kernel 6 style (2D register blocking + local-memory tiling)

Optional variant:
- Kernel 7 style with float4 global loads

Constraints:
- OpenCL only
- PyOpenCL host
- no CUDA intrinsics
"""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

# Some Windows/OneDrive setups expose a read-only cache path for PyOpenCL.
# Must be set before importing pyopencl.
os.environ.setdefault("PYOPENCL_NO_CACHE", "1")

import numpy as np
import pandas as pd
import pyopencl as cl


BASE_DIR = Path(__file__).resolve().parent
KERNEL6_PATH = BASE_DIR / "sgemm_kernel6.cl"
KERNEL7_PATH = BASE_DIR / "sgemm_kernel7_vec4.cl"


def gflops_from_ms(n: int, ms: float) -> float:
    # SGEMM does 2*N^3 floating-point operations.
    return (2.0 * (n**3)) / (ms * 1.0e6)


def parse_csv_ints(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def select_nvidia_device(device_substr: str) -> tuple[cl.Platform, cl.Device]:
    candidates: list[tuple[cl.Platform, cl.Device]] = []
    for platform in cl.get_platforms():
        for dev in platform.get_devices():
            dtype = cl.device_type.to_string(dev.type)
            if "GPU" not in dtype.upper():
                continue
            name_vendor = f"{dev.name} {dev.vendor}".lower()
            if "nvidia" in name_vendor:
                candidates.append((platform, dev))

    if not candidates:
        raise RuntimeError("No NVIDIA OpenCL GPU found.")

    wanted = device_substr.lower()
    for platform, dev in candidates:
        name_vendor = f"{dev.name} {dev.vendor}".lower()
        if wanted in name_vendor:
            return platform, dev

    # Fallback: first NVIDIA GPU
    return candidates[0]


def build_program(
    ctx: cl.Context,
    kernel_path: Path,
    tsm: int,
    tsn: int,
    tsk: int,
    wptm: int,
    wptn: int,
    width: int,
    fast_math: bool = True,
) -> cl.Program:
    src = kernel_path.read_text(encoding="utf-8")
    options = [
        f"-DTSM={tsm}",
        f"-DTSN={tsn}",
        f"-DTSK={tsk}",
        f"-DWPTM={wptm}",
        f"-DWPTN={wptn}",
        f"-DWIDTH={width}",
    ]
    if fast_math:
        options.extend(["-cl-fast-relaxed-math", "-cl-mad-enable"])
    return cl.Program(ctx, src).build(options=options)


def run_one_variant(
    n: int,
    kernel_kind: str,  # "k6" or "k7"
    tsm: int,
    tsn: int,
    tsk: int,
    wptm: int,
    wptn: int,
    width: int,
    runs: int,
    warmup: int,
    seed: int,
    check_count: int,
    device_substr: str,
    fast_math: bool = True,
) -> dict:
    if n % tsm != 0 or n % tsn != 0 or n % tsk != 0:
        raise ValueError("N must be divisible by TSM, TSN, and TSK for this fixed-tile kernel.")
    if tsm % wptm != 0 or tsn % wptn != 0:
        raise ValueError("TSM%WPTM and TSN%WPTN must be 0.")
    if kernel_kind == "k7" and width != 4:
        raise ValueError("Kernel k7 expects WIDTH=4.")
    if kernel_kind == "k6" and width != 1:
        raise ValueError("Kernel k6 expects WIDTH=1 in this script.")

    rtsm = tsm // wptm
    rtsn = tsn // wptn
    local_size = (rtsm, rtsn)
    wg_size = rtsm * rtsn
    global_size = ((n // tsm) * rtsm, (n // tsn) * rtsn)

    platform, device = select_nvidia_device(device_substr)
    if wg_size > int(device.max_work_group_size):
        raise ValueError(
            f"Local work-group size {wg_size} exceeds device max_work_group_size={device.max_work_group_size}"
        )

    print("=" * 80)
    print("Device")
    print(f"  Platform: {platform.name}")
    print(f"  Device:   {device.name}")
    print(f"  Vendor:   {device.vendor}")
    print(f"  Max WG:   {device.max_work_group_size}")
    print(f"  Max WI:   {device.max_work_item_sizes}")
    print("Kernel config")
    print(
        f"  kind={kernel_kind}, N={n}, TSM={tsm}, TSN={tsn}, TSK={tsk}, "
        f"WPTM={wptm}, WPTN={wptn}, WIDTH={width}"
    )
    print(f"  local_size={local_size}, global_size={global_size}")

    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    rng = np.random.default_rng(seed)
    a = (rng.random((n, n), dtype=np.float32) - 0.5).astype(np.float32, copy=False)
    b = (rng.random((n, n), dtype=np.float32) - 0.5).astype(np.float32, copy=False)
    bt = np.ascontiguousarray(b.T)
    c = np.empty((n, n), dtype=np.float32)

    mf = cl.mem_flags
    d_a = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    d_bt = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bt)
    d_c = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

    kernel_path = KERNEL6_PATH if kernel_kind == "k6" else KERNEL7_PATH
    program = build_program(
        ctx=ctx,
        kernel_path=kernel_path,
        tsm=tsm,
        tsn=tsn,
        tsk=tsk,
        wptm=wptm,
        wptn=wptn,
        width=width,
        fast_math=fast_math,
    )
    kernel_name = "sgemm_kernel6" if kernel_kind == "k6" else "sgemm_kernel7_vec4"
    kernel = cl.Kernel(program, kernel_name)
    kernel.set_arg(0, np.int32(n))
    kernel.set_arg(1, d_a)
    kernel.set_arg(2, d_bt)
    kernel.set_arg(3, d_c)

    # Warm-up runs (not measured in summary).
    for _ in range(warmup):
        evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        evt.wait()

    ms_list: list[float] = []
    for run_idx in range(1, runs + 1):
        evt = cl.enqueue_nd_range_kernel(queue, kernel, global_size, local_size)
        evt.wait()
        ms = (evt.profile.end - evt.profile.start) * 1.0e-6
        ms_list.append(ms)
        print(f"run {run_idx:02d}: {ms:9.3f} ms  |  {gflops_from_ms(n, ms):9.2f} GFLOPS")

    best_ms = float(np.min(ms_list))
    mean_ms = float(np.mean(ms_list))
    std_ms = float(np.std(ms_list))
    best_gflops = gflops_from_ms(n, best_ms)
    mean_gflops = gflops_from_ms(n, mean_ms)

    cl.enqueue_copy(queue, c, d_c).wait()

    # Correctness checks on selected C[i,j], comparing with CPU dot products.
    checks = [
        (0, 0),
        (n // 2, n // 2),
        (n - 1, n - 1),
        (123 % n, 456 % n),
        (2047 % n, 777 % n),
    ][: max(1, check_count)]

    print("-" * 80)
    print("Correctness checks (sampled entries):")
    max_abs_err = 0.0
    max_rel_err = 0.0
    t0 = time.time()
    for i, j in checks:
        ref = float(np.dot(a[i, :].astype(np.float64), b[:, j].astype(np.float64)))
        got = float(c[i, j])
        abs_err = abs(got - ref)
        rel_err = abs_err / (abs(ref) + 1e-12)
        max_abs_err = max(max_abs_err, abs_err)
        max_rel_err = max(max_rel_err, rel_err)
        print(
            f"C[{i:4d},{j:4d}]  gpu={got: .6e}  ref={ref: .6e}  "
            f"abs_err={abs_err: .3e}  rel_err={rel_err: .3e}"
        )
    print(f"check_time={time.time() - t0:.3f} s")

    print("-" * 80)
    print(f"best: {best_ms:.3f} ms  |  {best_gflops:.2f} GFLOPS")
    print(f"mean: {mean_ms:.3f} ms  |  {mean_gflops:.2f} GFLOPS  (std={std_ms:.3f} ms)")

    return {
        "kernel_kind": kernel_kind,
        "n": n,
        "tsm": tsm,
        "tsn": tsn,
        "tsk": tsk,
        "wptm": wptm,
        "wptn": wptn,
        "width": width,
        "local_x": local_size[0],
        "local_y": local_size[1],
        "best_ms": best_ms,
        "mean_ms": mean_ms,
        "std_ms": std_ms,
        "best_gflops": best_gflops,
        "mean_gflops": mean_gflops,
        "max_abs_err": max_abs_err,
        "max_rel_err": max_rel_err,
        "device_name": device.name.strip(),
        "platform_name": platform.name.strip(),
    }


def run_sweep(args: argparse.Namespace) -> None:
    tsk_values = parse_csv_ints(args.tsk_values)
    wpt_values = parse_csv_ints(args.wpt_values)
    width_values = parse_csv_ints(args.width_values)

    rows = []
    for tsk, wptm, wptn, width in itertools.product(tsk_values, wpt_values, wpt_values, width_values):
        kernel_kind = "k6" if width == 1 else "k7"
        try:
            row = run_one_variant(
                n=args.n,
                kernel_kind=kernel_kind,
                tsm=args.tsm,
                tsn=args.tsn,
                tsk=tsk,
                wptm=wptm,
                wptn=wptn,
                width=width,
                runs=args.runs,
                warmup=args.warmup,
                seed=args.seed,
                check_count=args.check_count,
                device_substr=args.device_substr,
                fast_math=(not args.no_fast_math),
            )
            rows.append(row)
        except Exception as exc:
            print(
                f"SKIP: TSK={tsk}, WPTM={wptm}, WPTN={wptn}, WIDTH={width} -> {exc}"
            )

    if not rows:
        print("No valid sweep variants executed.")
        return

    df = pd.DataFrame(rows).sort_values("best_gflops", ascending=False)
    print("=" * 80)
    print("Top variants by best GFLOPS:")
    print(df[["kernel_kind", "tsk", "wptm", "wptn", "width", "best_ms", "best_gflops"]].head(10).to_string(index=False))

    if args.out_csv:
        out = Path(args.out_csv)
        df.to_csv(out, index=False)
        print(f"Saved sweep results to: {out}")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="OpenCL SGEMM benchmark for RTX 3050 (PyOpenCL).")
    p.add_argument("--n", type=int, default=8192, help="Matrix size N (square NxN).")

    # Default kernel: main Kernel 6 implementation
    p.add_argument("--kernel", choices=["k6", "k7"], default="k6", help="Kernel variant to run.")

    p.add_argument("--tsm", type=int, default=128)
    p.add_argument("--tsn", type=int, default=128)
    p.add_argument("--tsk", type=int, default=16, help="Try nearby values: 8, 16, 32")
    p.add_argument("--wptm", type=int, default=8, help="Try nearby values: 4 or 8")
    p.add_argument("--wptn", type=int, default=8, help="Try nearby values: 4 or 8")
    p.add_argument("--width", type=int, default=1, help="WIDTH=1 for k6, WIDTH=4 for k7")

    p.add_argument("--runs", type=int, default=5, help="Measured benchmark runs.")
    p.add_argument("--warmup", type=int, default=1, help="Warm-up runs.")
    p.add_argument("--check-count", type=int, default=5, help="Number of sampled correctness checks.")
    p.add_argument("--seed", type=int, default=7)

    p.add_argument("--device-substr", type=str, default="RTX 3050", help="Device-name substring filter.")
    p.add_argument("--no-fast-math", action="store_true", help="Disable -cl-fast-relaxed-math and -cl-mad-enable.")

    # Optional nearby-variant sweep.
    p.add_argument("--sweep", action="store_true", help="Run nearby variants sweep.")
    p.add_argument("--tsk-values", type=str, default="8,16,32")
    p.add_argument("--wpt-values", type=str, default="4,8")
    p.add_argument("--width-values", type=str, default="1,4")
    p.add_argument("--out-csv", type=str, default="", help="Optional CSV path for sweep results.")

    return p


def main() -> None:
    args = build_argparser().parse_args()

    if args.sweep:
        run_sweep(args)
        return

    # Enforce consistent defaults for the chosen kernel.
    if args.kernel == "k6" and args.width != 1:
        raise ValueError("For kernel k6, use --width 1.")
    if args.kernel == "k7" and args.width != 4:
        raise ValueError("For kernel k7, use --width 4.")

    run_one_variant(
        n=args.n,
        kernel_kind=args.kernel,
        tsm=args.tsm,
        tsn=args.tsn,
        tsk=args.tsk,
        wptm=args.wptm,
        wptn=args.wptn,
        width=args.width,
        runs=args.runs,
        warmup=args.warmup,
        seed=args.seed,
        check_count=args.check_count,
        device_substr=args.device_substr,
        fast_math=(not args.no_fast_math),
    )


if __name__ == "__main__":
    main()
