#ifndef __ST_PPL_KERNEL_X86_FP32_GEMM_V2_GEMM_KERNEL_FP32_SSE_H_
#define __ST_PPL_KERNEL_X86_FP32_GEMM_V2_GEMM_KERNEL_FP32_SSE_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

typedef void (*gemm_kernel_fp32_sse_func_type_t)(const float*, const float*, const int32_t, const int32_t, const int32_t, const int32_t, float*);

// 4x12 kernels
extern const gemm_kernel_fp32_sse_func_type_t gemm_kernel_max4x12_fp32_sse_func_tab[5][4];

void gemm_kernel_4x12_fp32_sse(
    const float* A,
    const float* B,
    const int32_t k_len,
    const int32_t lda,
    const int32_t ldb,
    const int32_t ldc,
    float* C);

}}} // namespace ppl::kernel::x86

#endif // !__ST_PPL_KERNEL_X86_FP32_GEMM_V2_GEMM_KERNEL_FP32_SSE_H_