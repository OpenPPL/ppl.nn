// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/avx512/conv2d_n16cx_direct_ndarray_blk1x1_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/direct_ndarray/avx512/conv2d_n16cx_direct_ndarray_blk1x14_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_func_t
conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_pad_table[NT_STORE_OPT()][MAX_OC_RF()] =
{
    {
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x1_kernel<false, 1 * OC_DT_BLK()>,
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x1_kernel<false, 2 * OC_DT_BLK()>,
    },
    {
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x1_kernel<true, 1 * OC_DT_BLK()>,
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x1_kernel<true, 2 * OC_DT_BLK()>,
    },
};

#define DIRECT_NDARRAY_O16_KERNEL_TABLE_BLK(NT_STORE) \
    {\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 1 * OC_DT_BLK(), 1>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 1 * OC_DT_BLK(), 2>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 1 * OC_DT_BLK(), 3>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 1 * OC_DT_BLK(), 4>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 1 * OC_DT_BLK(), 5>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 1 * OC_DT_BLK(), 6>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 1 * OC_DT_BLK(), 7>,\
    }

#define DIRECT_NDARRAY_O32_KERNEL_TABLE_BLK(NT_STORE) \
    {\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 2 * OC_DT_BLK(), 1>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 2 * OC_DT_BLK(), 2>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 2 * OC_DT_BLK(), 3>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 2 * OC_DT_BLK(), 4>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 2 * OC_DT_BLK(), 5>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 2 * OC_DT_BLK(), 6>,\
        conv2d_n16cx_direct_ndarray_fp32_avx512_blk1x14_kernel<NT_STORE, 2 * OC_DT_BLK(), 7>,\
    }

conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_func_t
conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_o16_table[NT_STORE_OPT()][MAX_OW_RF()] =
{
    DIRECT_NDARRAY_O16_KERNEL_TABLE_BLK(false),
    DIRECT_NDARRAY_O16_KERNEL_TABLE_BLK(true),
};

conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_func_t
conv2d_n16cx_direct_ndarray_kernel_fp32_avx512_o32_table[NT_STORE_OPT()][MAX_OW_RF()] =
{
    DIRECT_NDARRAY_O32_KERNEL_TABLE_BLK(false),
    DIRECT_NDARRAY_O32_KERNEL_TABLE_BLK(true),
};

}}};
