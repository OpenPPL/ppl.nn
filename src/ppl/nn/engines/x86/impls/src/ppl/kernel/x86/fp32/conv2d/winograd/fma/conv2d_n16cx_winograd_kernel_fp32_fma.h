#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_FMA_CONV2D_N16CX_WINOGRAD_KERNEL_FP32_FMA_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_FMA_CONV2D_N16CX_WINOGRAD_KERNEL_FP32_FMA_H_

#include "ppl/kernel/x86/common/internal_include.h"

#define CH_DT_BLK()   16
#define CH_RF_BLK()   8

#define TILE_RF_CNT() 6
#define OC_RF_CNT()   2

namespace ppl { namespace kernel { namespace x86 {

typedef void (*conv2d_n16cx_winograd_kernel_fp32_fma_func_t)(
    const float *,
    const float *,
    const int64_t,
    const int64_t,
    const int64_t,
    const int64_t,
    float *);

extern conv2d_n16cx_winograd_kernel_fp32_fma_func_t
    conv2d_n16cx_winograd_kernel_fp32_fma_table[TILE_RF_CNT()];

}}}; // namespace ppl::kernel::x86

#endif
