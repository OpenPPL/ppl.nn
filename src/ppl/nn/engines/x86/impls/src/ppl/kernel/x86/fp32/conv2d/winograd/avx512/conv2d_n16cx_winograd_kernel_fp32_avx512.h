#ifndef __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_AVX512_CONV2D_N16CX_WINOGRAD_KERNEL_FP32_AVX512_H_
#define __ST_PPL_KERNEL_X86_FP32_CONV2D_WINOGRAD_AVX512_CONV2D_N16CX_WINOGRAD_KERNEL_FP32_AVX512_H_

#include "ppl/kernel/x86/common/internal_include.h"

#define PICK_PARAM(T, PARAM, IDX) *(T*)(PARAM + IDX)

#define KERNEL_PARAM_LEN()   10
#define SRC_IDX()            0
#define DST_IDX()            1
#define FLT_IDX()            2
#define TILES_IDX()          3
#define CHANNELS_IDX()       4
#define SRC_TKB_STRIDE_IDX() 5
#define DST_OCB_STRIDE_IDX() 6
#define FLT_OCB_STRIDE_IDX() 7
#define LOAD_DST_IDX()       8

#define CH_DT_BLK() 16

#define MAX_OC_RF() 2
#define MAX_TILES_RF() 14

#define T6_OC_RF() 4
#define T6_TILES_RF() 6

#define T9_OC_RF() 3
#define T9_TILES_RF() 9

#define T14_OC_RF() 2
#define T14_TILES_RF() 14

#define T31_OC_RF() 1
#define T31_TILES_RF() 31

namespace ppl { namespace kernel { namespace x86 {

typedef void (*conv2d_n16cx_winograd_kernel_fp32_avx512_func_t)(const int64_t *);

extern conv2d_n16cx_winograd_kernel_fp32_avx512_func_t
    conv2d_n16cx_winograd_kernel_fp32_avx512_o16_table[T14_TILES_RF()];

extern conv2d_n16cx_winograd_kernel_fp32_avx512_func_t
    conv2d_n16cx_winograd_kernel_fp32_avx512_o32_table[T14_TILES_RF()];

extern conv2d_n16cx_winograd_kernel_fp32_avx512_func_t
    conv2d_n16cx_winograd_kernel_fp32_avx512_o48_table[T9_TILES_RF()];

extern conv2d_n16cx_winograd_kernel_fp32_avx512_func_t
    conv2d_n16cx_winograd_kernel_fp32_avx512_o64_table[T6_TILES_RF()];

}}}; // namespace ppl::kernel::x86

#endif
