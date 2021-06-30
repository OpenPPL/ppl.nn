#ifndef __ST_PPL_KERNEL_X86_FP32_GEMM_H_
#define __ST_PPL_KERNEL_X86_FP32_GEMM_H_

#include "ppl/kernel/x86/common/general_include.h"

namespace ppl { namespace kernel { namespace x86 {

ppl::common::RetCode gemm_ref_fp32(
    const float *A,
    const float *B,
    const float *V, // vector C
    const float *H, // matrix C
    const int32_t trans_A,
    const int32_t trans_B,
    const int32_t M,
    const int32_t N,
    const int32_t K,
    const float alpha,
    const float beta,
    float *Y);

}}}; // namespace ppl::kernel::x86

#endif //! __ST_PPL_KERNEL_X86_FP32_GEMM_H_
