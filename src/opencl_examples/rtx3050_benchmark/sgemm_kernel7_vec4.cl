// Kernel 7 style SGEMM variant with wider float4 loads.
// Same mapping as kernel6 (2D register blocking + local tiling), but
// global-memory tile loads move 4 contiguous k-elements at once.
//
// Computes: C = A * B, with BT = B^T passed to the kernel.

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

#ifndef WIDTH
#define WIDTH 4
#endif

#define RTSM (TSM / WPTM)
#define RTSN (TSN / WPTN)

#define LPTA_VEC ((TSM * (TSK / WIDTH)) / (RTSM * RTSN))
#define LPTB_VEC ((TSN * (TSK / WIDTH)) / (RTSM * RTSN))

#define PAD_K 2

#if (WIDTH != 4)
#error "This kernel expects WIDTH=4"
#endif

#if (TSK % WIDTH) != 0
#error "TSK must be divisible by WIDTH=4"
#endif

#if (TSM % WPTM) != 0
#error "TSM must be divisible by WPTM"
#endif

#if (TSN % WPTN) != 0
#error "TSN must be divisible by WPTN"
#endif

#if ((TSM * (TSK / WIDTH)) % (RTSM * RTSN)) != 0
#error "(TSM*(TSK/WIDTH)) must be divisible by (RTSM*RTSN)"
#endif

#if ((TSN * (TSK / WIDTH)) % (RTSM * RTSN)) != 0
#error "(TSN*(TSK/WIDTH)) must be divisible by (RTSM*RTSN)"
#endif

__kernel void sgemm_kernel7_vec4(
    const int N,
    __global const float* A,
    __global const float* BT,
    __global float* C)
{
    // Local thread coordinates.
    const int lidm = get_local_id(0);
    const int lidn = get_local_id(1);

    // Tile coordinates in C.
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
        // Vectorized load for A tile.
        #pragma unroll
        for (int l = 0; l < LPTA_VEC; ++l) {
            const int id = l * RTSM * RTSN + tid;
            const int row = id % TSM;
            const int col_vec = id / TSM;       // 0 .. (TSK/WIDTH)-1
            const int col = col_vec * WIDTH;    // actual k-offset

            const int g_row = group_m * TSM + row;
            const int g_col = k0 + col;

            float4 v = vload4(0, A + g_row * N + g_col);
            vstore4(v, 0, Asub[row] + col);
        }

        // Vectorized load for B^T tile.
        #pragma unroll
        for (int l = 0; l < LPTB_VEC; ++l) {
            const int id = l * RTSM * RTSN + tid;
            const int row = id % TSN;
            const int col_vec = id / TSN;
            const int col = col_vec * WIDTH;

            const int g_row = group_n * TSN + row;
            const int g_col = k0 + col;

            float4 v = vload4(0, BT + g_row * N + g_col);
            vstore4(v, 0, Bsub[row] + col);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute tile products.
        #pragma unroll
        for (int k = 0; k < TSK; ++k) {
            float Areg[WPTM];

            #pragma unroll
            for (int wm = 0; wm < WPTM; ++wm) {
                const int local_row = lidm + wm * RTSM;
                Areg[wm] = Asub[local_row][k];
            }

            #pragma unroll
            for (int wn = 0; wn < WPTN; ++wn) {
                const int local_col = lidn + wn * RTSN;
                const float b = Bsub[local_col][k];

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
            C[row * N + col] = acc[wm][wn];
        }
    }
}

