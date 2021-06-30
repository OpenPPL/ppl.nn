#include "ppl/kernel/x86/fp32/conv2d/depthwise/fma/conv2d_n16cx_depthwise_blk1x1_kernel_fp32_fma.h"
#include "ppl/kernel/x86/fp32/conv2d/depthwise/fma/conv2d_n16cx_depthwise_blk1x7_kernel_fp32_fma.h"

namespace ppl { namespace kernel { namespace x86 {

conv2d_n16cx_depthwise_kernel_fp32_fma_func_t
conv2d_n16cx_depthwise_kernel_fp32_fma_pad_table[NT_STORE_OPT()] =
{
    conv2d_n16cx_depthwise_fp32_fma_blk1x1_kernel<false>,
    conv2d_n16cx_depthwise_fp32_fma_blk1x1_kernel<true>,
};

#define DEPTHWISE_BLK_KERNEL_TABLE_BLK(NT_STORE, STRIDE_W) \
    {\
        conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 1>,\
        conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 2>,\
        conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 3>,\
        conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 4>,\
        conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 5>,\
        conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 6>,\
        conv2d_n16cx_depthwise_fp32_fma_blk1x7_kernel<NT_STORE, STRIDE_W, 7>,\
    }

conv2d_n16cx_depthwise_kernel_fp32_fma_func_t
conv2d_n16cx_depthwise_kernel_fp32_fma_blk_table[NT_STORE_OPT()][STRIDE_W_OPT()][MAX_OW_RF()] =
{
    {
        DEPTHWISE_BLK_KERNEL_TABLE_BLK(false, 0),
        DEPTHWISE_BLK_KERNEL_TABLE_BLK(false, 1),
        DEPTHWISE_BLK_KERNEL_TABLE_BLK(false, 2),
    },
    {
        DEPTHWISE_BLK_KERNEL_TABLE_BLK(true, 0),
        DEPTHWISE_BLK_KERNEL_TABLE_BLK(true, 1),
        DEPTHWISE_BLK_KERNEL_TABLE_BLK(true, 2),
    },
};

}}};
