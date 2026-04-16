import argparse
import os
import warnings

import numpy as np


# Avoid filesystem cache writes in restricted/read-only environments.
os.environ.setdefault("PYOPENCL_NO_CACHE", "1")

import pyopencl as cl

warnings.filterwarnings("ignore", category=cl.CompilerWarning)


TSM = 128
TSN = 128
TSK = 16
WPTM = 8
WPTN = 8
WIDTH = 4


KERNEL_SOURCE = r"""
#define TSM 128
#define TSN 128
#define TSK 16
#define WPTM 8
#define WPTN 8
#define WIDTH 4

#define RTSM (TSM / WPTM)
#define RTSN (TSN / WPTN)
#define TSKV (TSK / WIDTH)
#define LPTA ((TSM * TSKV) / (RTSM * RTSN))
#define LPTB ((TSN * TSKV) / (RTSM * RTSN))

#if WIDTH == 4
typedef float4 floatX;
#endif

__kernel void sgemm_kernel6_advanced(
    const int N,
    __global const float* A,
    __global const float* BT,
    __global float* C)
{
    const int localRow = get_local_id(0);
    const int localCol = get_local_id(1);
    const int groupRow = get_group_id(0);
    const int groupCol = get_group_id(1);
    const int tid = localCol * RTSM + localRow;

    const int Nvec = N / WIDTH;
    const int numTiles = N / TSK;

    __global const floatX* A4 = (__global const floatX*)A;
    __global const floatX* B4 = (__global const floatX*)BT;

    __local floatX Asub[2][TSM][TSKV];
    __local floatX Bsub[2][TSN][TSKV];

    float acc[WPTM][WPTN];
    #pragma unroll
    for (int wm = 0; wm < WPTM; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WPTN; ++wn) {
            acc[wm][wn] = 0.0f;
        }
    }

    int readBuf = 0;
    int writeBuf = 1;

    // Preload tile 0
    #pragma unroll
    for (int la = 0; la < LPTA; ++la) {
        const int idx = tid + la * RTSM * RTSN;
        const int row = idx / TSKV;
        const int kvec = idx - row * TSKV;
        const int globalRow = groupRow * TSM + row;
        Asub[readBuf][row][kvec] = A4[globalRow * Nvec + kvec];
    }
    #pragma unroll
    for (int lb = 0; lb < LPTB; ++lb) {
        const int idx = tid + lb * RTSM * RTSN;
        const int col = idx / TSKV;
        const int kvec = idx - col * TSKV;
        const int globalCol = groupCol * TSN + col;
        Bsub[readBuf][col][kvec] = B4[globalCol * Nvec + kvec];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int t = 0; t < numTiles; ++t) {
        // Prefetch the next tile into the alternate local-memory buffer.
        if (t + 1 < numTiles) {
            const int nextBase = ((t + 1) * TSK) / WIDTH;

            #pragma unroll
            for (int la = 0; la < LPTA; ++la) {
                const int idx = tid + la * RTSM * RTSN;
                const int row = idx / TSKV;
                const int kvec = idx - row * TSKV;
                const int globalRow = groupRow * TSM + row;
                Asub[writeBuf][row][kvec] = A4[globalRow * Nvec + nextBase + kvec];
            }
            #pragma unroll
            for (int lb = 0; lb < LPTB; ++lb) {
                const int idx = tid + lb * RTSM * RTSN;
                const int col = idx / TSKV;
                const int kvec = idx - col * TSKV;
                const int globalCol = groupCol * TSN + col;
                Bsub[writeBuf][col][kvec] = B4[globalCol * Nvec + nextBase + kvec];
            }
        }

        // Compute with the current tile.
        #pragma unroll
        for (int kvec = 0; kvec < TSKV; ++kvec) {
            floatX regA[WPTM];
            floatX regB[WPTN];

            #pragma unroll
            for (int wm = 0; wm < WPTM; ++wm) {
                const int row = localRow + wm * RTSM;
                regA[wm] = Asub[readBuf][row][kvec];
            }

            #pragma unroll
            for (int wn = 0; wn < WPTN; ++wn) {
                const int col = localCol + wn * RTSN;
                regB[wn] = Bsub[readBuf][col][kvec];
            }

            #pragma unroll
            for (int wm = 0; wm < WPTM; ++wm) {
                const float a0 = regA[wm].x;
                const float a1 = regA[wm].y;
                const float a2 = regA[wm].z;
                const float a3 = regA[wm].w;

                #pragma unroll
                for (int wn = 0; wn < WPTN; ++wn) {
                    const floatX b = regB[wn];
                    acc[wm][wn] = fma(a0, b.x, acc[wm][wn]);
                    acc[wm][wn] = fma(a1, b.y, acc[wm][wn]);
                    acc[wm][wn] = fma(a2, b.z, acc[wm][wn]);
                    acc[wm][wn] = fma(a3, b.w, acc[wm][wn]);
                }
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const int tmp = readBuf;
        readBuf = writeBuf;
        writeBuf = tmp;
    }

    #pragma unroll
    for (int wm = 0; wm < WPTM; ++wm) {
        const int globalRow = groupRow * TSM + localRow + wm * RTSM;
        #pragma unroll
        for (int wn = 0; wn < WPTN; ++wn) {
            const int globalCol = groupCol * TSN + localCol + wn * RTSN;
            C[globalRow * N + globalCol] = acc[wm][wn];
        }
    }
}
"""


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def select_gpu_device() -> cl.Device:
    gpu_devices = []
    for platform in cl.get_platforms():
        for device in platform.get_devices(device_type=cl.device_type.GPU):
            gpu_devices.append(device)

    if not gpu_devices:
        raise RuntimeError("No GPU device found.")

    for device in gpu_devices:
        vendor = f"{device.vendor} {device.name}".upper()
        if "NVIDIA" in vendor:
            return device

    return gpu_devices[0]


def benchmark_sgemm(n: int, warmup: int, runs: int) -> tuple[float, float, float]:
    n = round_up(n, TSM)

    device = select_gpu_device()
    ctx = cl.Context(devices=[device])
    queue = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    np.random.seed(0)
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    bt = b.T.copy()
    c = np.empty((n, n), dtype=np.float32)

    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    bt_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=bt)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, c.nbytes)

    program = cl.Program(ctx, KERNEL_SOURCE).build(
        options=["-cl-fast-relaxed-math", "-cl-mad-enable"]
    )
    kernel = program.sgemm_kernel6_advanced

    local_size = (TSM // WPTM, TSN // WPTN)
    global_size = (n // WPTM, n // WPTN)
    n_arg = np.int32(n)

    for _ in range(warmup):
        event = kernel(queue, global_size, local_size, n_arg, a_buf, bt_buf, c_buf)
        event.wait()

    times = []
    for _ in range(runs):
        event = kernel(queue, global_size, local_size, n_arg, a_buf, bt_buf, c_buf)
        event.wait()
        elapsed_s = (event.profile.end - event.profile.start) * 1.0e-9
        times.append(elapsed_s)

    cl.enqueue_copy(queue, c, c_buf).wait()

    # --- VALIDATION ---
    ref = a @ b
    max_err = np.max(np.abs(c - ref))
    print("Max error:", max_err)
    # ------------------
    best_time = float(np.min(times))
    mean_time = float(np.mean(times))
    gflops = (2.0 * (n**3)) / best_time / 1.0e9
    return best_time, mean_time, gflops


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_time, mean_time, gflops = benchmark_sgemm(args.n, args.warmup, args.runs)

    print(f"Best time (ms): {best_time * 1.0e3:.3f}")
    print(f"Mean time (ms): {mean_time * 1.0e3:.3f}")
    print(f"GFLOPS: {gflops:.2f}")


if __name__ == "__main__":
    main()
