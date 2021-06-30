#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode gemm_ref_fp32(
    const float *A,
    const float *B,
    const float *V,
    const float *H,
    const int32_t trans_A,
    const int32_t trans_B,
    const int32_t M,
    const int32_t N,
    const int32_t K,
    const float alpha,
    const float beta,
    float *Y)
{
#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#endif
    for (int64_t m = 0; m < M; ++m) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
        PRAGMA_OMP_PARALLEL_FOR()
#endif
        for (int64_t n = 0; n < N; ++n) {
            float y = 0.0f;
            if (!trans_A && !trans_B) { // MK, KN; NN
                for (int64_t k = 0; k < K; ++k) {
                    y += A[m * K + k] * B[k * N + n];
                }
            }
            if (trans_A && !trans_B) { // KM, KN; TN
                for (int64_t k = 0; k < K; ++k) {
                    y += A[k * M + m] * B[k * N + n];
                }
            }
            if (trans_A && trans_B) { // KM, NK; TT
                for (int64_t k = 0; k < K; ++k) {
                    y += A[k * M + m] * B[n * K + k];
                }
            }
            if (!trans_A && trans_B) { // MK, NK; NT
                for (int64_t k = 0; k < K; ++k) {
                    y += A[m * K + k] * B[n * K + k];
                }
            }
            y *= alpha;
            if (V) y += beta * V[n];
            if (H) y += beta * H[m * N + n];
            Y[m * N + n] = y;
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
