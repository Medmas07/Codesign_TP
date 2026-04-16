// Kernel 6 style SGEMM for large square matrices.
// Strategy:
// - 2D register blocking (each thread computes WPTM x WPTN outputs)
// - local memory tiling for A and B^T
// - padded local-memory K dimension to reduce bank conflicts
//
// Computes: C = A * B, but B must be provided transposed as BT = B^T.
// Then: C[i,j] = dot(A[i,:], BT[j,:]).

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

// Padding the K dimension in local memory helps reduce bank conflicts.
#define PAD_K 2

#if (TSM % WPTM) != 0
#error "TSM must be divisible by WPTM"
#endif

#if (TSN % WPTN) != 0
#error "TSN must be divisible by WPTN"
#endif

#if ((TSM * TSK) % (RTSM * RTSN)) != 0
#error "(TSM*TSK) must be divisible by (RTSM*RTSN)"
#endif

#if ((TSN * TSK) % (RTSM * RTSN)) != 0
#error "(TSN*TSK) must be divisible by (RTSM*RTSN)"
#endif

__kernel void sgemm_kernel6(
    const int N,
    __global const float* A,
    __global const float* BT,
    __global float* C)
{
    // Local thread coordinates inside the work-group.
    const int lidm = get_local_id(0);  // row-thread index inside tile
    const int lidn = get_local_id(1);  // col-thread index inside tile

    // Work-group coordinates: one group computes one C tile (TSM x TSN).
    const int group_m = get_group_id(0);
    const int group_n = get_group_id(1);

    // Flattened thread id used for cooperative tile loading.
    const int tid = lidn * RTSM + lidm;

    // Local tiles:
    // Asub stores A tile rows [0..TSM), k [0..TSK)
    // Bsub stores BT tile rows [0..TSN), k [0..TSK)
    __local float Asub[TSM][TSK + PAD_K];
    __local float Bsub[TSN][TSK + PAD_K];

    // Registers: each thread accumulates WPTM x WPTN outputs.
    float acc[WPTM][WPTN];

    #pragma unroll
    for (int wm = 0; wm < WPTM; ++wm) {
        #pragma unroll
        for (int wn = 0; wn < WPTN; ++wn) {
            acc[wm][wn] = 0.0f;
        }
    }

    // Sweep K dimension in chunks of TSK.
    for (int k0 = 0; k0 < N; k0 += TSK) {
        // Cooperative load for Asub.
        #pragma unroll
        for (int l = 0; l < LPTA; ++l) {
            const int id = l * RTSM * RTSN + tid;
            const int row = id % TSM;
            const int col = id / TSM;

            const int g_row = group_m * TSM + row;
            const int g_col = k0 + col;

            Asub[row][col] = A[g_row * N + g_col];
        }

        // Cooperative load for Bsub from transposed B.
        #pragma unroll
        for (int l = 0; l < LPTB; ++l) {
            const int id = l * RTSM * RTSN + tid;
            const int row = id % TSN;
            const int col = id / TSN;

            const int g_row = group_n * TSN + row;
            const int g_col = k0 + col;

            Bsub[row][col] = BT[g_row * N + g_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        // Compute outer products for this K tile.
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

    // Store the WPTM x WPTN thread results.
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

