// FP16 GEMM using simdgroup_matrix instructions (Apple's tensor-core-equivalent).
//
// Computes  C = A @ B^T   where  A:[M, K], B:[N, K] row-major, C:[M, N] row-major.
// This is the LLM projection layout: Y = X · Wᵀ with W stored as [outDim, inDim].
//
// v3 — tiled, with boundary handling on M, N, and K:
//   * 32×32 output tile per threadgroup, served by 4 simdgroups in a 2×2 layout.
//   * Each simdgroup owns a 16×16 quadrant = 4 sub-tiles of 8×8.
//   * A and B slabs (32×8 each) are cooperatively staged in threadgroup memory
//     per K-step. Out-of-bounds rows/cols load as 0 so partial K, M or N tiles
//     accumulate the correct values.
//   * FP32 accumulation; per-element bounds-checked write to C (so the last
//     row/column tile may be partial without out-of-bounds stores).
//
// Constraints (now soft):
//   * M ≥ 1, N ≥ 1, K ≥ 1.
//   * Best efficiency when M, N are multiples of 32 and K is a multiple of 8.

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

kernel void gemm_f16_smm_ntb(
    device const half* A [[ buffer(0) ]],   // [M, K]
    device const half* B [[ buffer(1) ]],   // [N, K]   (read transposed)
    device       half* C [[ buffer(2) ]],   // [M, N]
    constant int& M     [[ buffer(3) ]],
    constant int& N     [[ buffer(4) ]],
    constant int& K     [[ buffer(5) ]],
    uint2 tg_pos        [[ threadgroup_position_in_grid ]],
    uint  sgid          [[ simdgroup_index_in_threadgroup ]],
    uint  lane          [[ thread_index_in_simdgroup ]],
    uint  tid           [[ thread_index_in_threadgroup ]])
{
    const uint row0 = tg_pos.y * 32;
    const uint col0 = tg_pos.x * 32;

    const uint sg_row = sgid >> 1;   // 0..1
    const uint sg_col = sgid & 1u;   // 0..1

    simdgroup_matrix<float, 8, 8> acc00 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc01 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc10 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> acc11 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup half a_shared[32 * 8];
    threadgroup half b_shared[32 * 8];

    // K-loop. The last iteration may run with k0 + 8 > K; in that case the
    // cooperative load zero-fills the out-of-range columns and the matmul
    // accumulates 0 for those K positions — correct partial sum.
    for (int k0 = 0; k0 < K; k0 += 8) {
        {
            uint p0 = tid;
            uint p1 = tid + 128u;

            uint r0 = p0 >> 3, c0 = p0 & 7u;
            uint r1 = p1 >> 3, c1 = p1 & 7u;

            uint a_row0_idx = row0 + r0;
            uint a_row1_idx = row0 + r1;
            uint k0_idx     = (uint)k0 + c0;
            uint k1_idx     = (uint)k0 + c1;

            a_shared[p0] = (a_row0_idx < (uint)M && k0_idx < (uint)K)
                ? A[(size_t)a_row0_idx * K + k0_idx]
                : half(0);
            a_shared[p1] = (a_row1_idx < (uint)M && k1_idx < (uint)K)
                ? A[(size_t)a_row1_idx * K + k1_idx]
                : half(0);

            // B is [N, K]. Row index here = output column = col0 + r.
            uint b_row0_idx = col0 + r0;
            uint b_row1_idx = col0 + r1;

            b_shared[p0] = (b_row0_idx < (uint)N && k0_idx < (uint)K)
                ? B[(size_t)b_row0_idx * K + k0_idx]
                : half(0);
            b_shared[p1] = (b_row1_idx < (uint)N && k1_idx < (uint)K)
                ? B[(size_t)b_row1_idx * K + k1_idx]
                : half(0);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_matrix<half, 8, 8> a0, a1, b0, b1;

        simdgroup_load(a0, a_shared + (size_t)(sg_row * 16)     * 8, 8);
        simdgroup_load(a1, a_shared + (size_t)(sg_row * 16 + 8) * 8, 8);

        simdgroup_load(b0, b_shared + (size_t)(sg_col * 16)     * 8, 8, ulong2(0, 0), true);
        simdgroup_load(b1, b_shared + (size_t)(sg_col * 16 + 8) * 8, 8, ulong2(0, 0), true);

        simdgroup_multiply_accumulate(acc00, a0, b0, acc00);
        simdgroup_multiply_accumulate(acc01, a0, b1, acc01);
        simdgroup_multiply_accumulate(acc10, a1, b0, acc10);
        simdgroup_multiply_accumulate(acc11, a1, b1, acc11);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── Bounds-checked write-back ────────────────────────────────────────────
    // Each simdgroup converts its 4 accumulators FP32 → FP16 via threadgroup
    // memory, then each lane writes its 2 elements to C with a per-position
    // bounds check (no simdgroup_store at the device boundary, since we cannot
    // partially store a simdgroup_matrix tile).
    threadgroup float wb_f[4 * 64];
    threadgroup half  wb_h[4 * 64];
    threadgroup float* sg_f = wb_f + (size_t)sgid * 64;
    threadgroup half*  sg_h = wb_h + (size_t)sgid * 64;

    const uint dst_row0 = row0 + sg_row * 16;
    const uint dst_col0 = col0 + sg_col * 16;

    // (DROW_OFS, DCOL_OFS) = offset of this 8×8 sub-tile within the 16×16 quadrant.
    #define STORE_TILE(ACC, DROW_OFS, DCOL_OFS)                                                 \
        do {                                                                                     \
            simdgroup_store(ACC, sg_f, 8);                                                       \
            sg_h[lane * 2]     = half(sg_f[lane * 2]);                                           \
            sg_h[lane * 2 + 1] = half(sg_f[lane * 2 + 1]);                                       \
            uint p0 = lane * 2u, p1 = lane * 2u + 1u;                                            \
            uint r0 = p0 >> 3, c0 = p0 & 7u;                                                     \
            uint r1 = p1 >> 3, c1 = p1 & 7u;                                                     \
            uint gr0 = dst_row0 + (DROW_OFS) + r0;                                               \
            uint gc0 = dst_col0 + (DCOL_OFS) + c0;                                               \
            uint gr1 = dst_row0 + (DROW_OFS) + r1;                                               \
            uint gc1 = dst_col0 + (DCOL_OFS) + c1;                                               \
            if (gr0 < (uint)M && gc0 < (uint)N) C[(size_t)gr0 * N + gc0] = sg_h[p0];             \
            if (gr1 < (uint)M && gc1 < (uint)N) C[(size_t)gr1 * N + gc1] = sg_h[p1];             \
        } while (0)

    STORE_TILE(acc00, 0, 0);
    STORE_TILE(acc01, 0, 8);
    STORE_TILE(acc10, 8, 0);
    STORE_TILE(acc11, 8, 8);

    #undef STORE_TILE
}
