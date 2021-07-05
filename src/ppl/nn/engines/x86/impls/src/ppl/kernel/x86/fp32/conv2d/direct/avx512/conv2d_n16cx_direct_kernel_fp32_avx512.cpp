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

#include "ppl/kernel/x86/fp32/conv2d/direct/avx512/conv2d_n16cx_direct_blk1x1_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/direct/avx512/conv2d_n16cx_direct_blk1x6_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/direct/avx512/conv2d_n16cx_direct_blk1x9_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/direct/avx512/conv2d_n16cx_direct_blk1x14_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

conv2d_n16cx_direct_kernel_fp32_avx512_func_t
conv2d_n16cx_direct_kernel_fp32_avx512_pad_table[NT_STORE_OPT()][MAX_OC_RF()] =
{
    {
        conv2d_n16cx_direct_fp32_avx512_blk1x1_kernel<false, 1 * CH_DT_BLK()>,
        conv2d_n16cx_direct_fp32_avx512_blk1x1_kernel<false, 2 * CH_DT_BLK()>,
        conv2d_n16cx_direct_fp32_avx512_blk1x1_kernel<false, 3 * CH_DT_BLK()>,
        conv2d_n16cx_direct_fp32_avx512_blk1x1_kernel<false, 4 * CH_DT_BLK()>,
    },
    {
        conv2d_n16cx_direct_fp32_avx512_blk1x1_kernel<true, 1 * CH_DT_BLK()>,
        conv2d_n16cx_direct_fp32_avx512_blk1x1_kernel<true, 2 * CH_DT_BLK()>,
        conv2d_n16cx_direct_fp32_avx512_blk1x1_kernel<true, 3 * CH_DT_BLK()>,
        conv2d_n16cx_direct_fp32_avx512_blk1x1_kernel<true, 4 * CH_DT_BLK()>,
    },
};

#define DIRECT_O16_KERNEL_TABLE_BLK(NT_STORE, STRIDE_W) \
    {\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 1>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 2>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 3>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 4>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 5>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 6>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 7>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 8>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 9>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 10>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 11>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 12>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 13>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 1 * CH_DT_BLK(), 14>,\
    }

#define DIRECT_O32_KERNEL_TABLE_BLK(NT_STORE, STRIDE_W) \
    {\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 1>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 2>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 3>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 4>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 5>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 6>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 7>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 8>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 9>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 10>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 11>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 12>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 13>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x14_kernel<NT_STORE, STRIDE_W, 2 * CH_DT_BLK(), 14>,\
    }

#define DIRECT_O48_KERNEL_TABLE_BLK(NT_STORE, STRIDE_W) \
    {\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 3 * CH_DT_BLK(), 1>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 3 * CH_DT_BLK(), 2>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 3 * CH_DT_BLK(), 3>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 3 * CH_DT_BLK(), 4>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 3 * CH_DT_BLK(), 5>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 3 * CH_DT_BLK(), 6>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel<NT_STORE, STRIDE_W, 3 * CH_DT_BLK(), 7>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel<NT_STORE, STRIDE_W, 3 * CH_DT_BLK(), 8>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x9_kernel<NT_STORE, STRIDE_W, 3 * CH_DT_BLK(), 9>,\
        nullptr,\
        nullptr,\
        nullptr,\
        nullptr,\
        nullptr,\
    }

#define DIRECT_O64_KERNEL_TABLE_BLK(NT_STORE, STRIDE_W) \
    {\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 4 * CH_DT_BLK(), 1>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 4 * CH_DT_BLK(), 2>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 4 * CH_DT_BLK(), 3>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 4 * CH_DT_BLK(), 4>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 4 * CH_DT_BLK(), 5>,\
        conv2d_n16cx_direct_fp32_avx512_blk1x6_kernel<NT_STORE, STRIDE_W, 4 * CH_DT_BLK(), 6>,\
        nullptr,\
        nullptr,\
        nullptr,\
        nullptr,\
        nullptr,\
        nullptr,\
        nullptr,\
        nullptr,\
    }

conv2d_n16cx_direct_kernel_fp32_avx512_func_t
conv2d_n16cx_direct_kernel_fp32_avx512_o16_table[NT_STORE_OPT()][STRIDE_W_OPT()][BLK1X14_OW_RF()] =
{
    {
        DIRECT_O16_KERNEL_TABLE_BLK(false, 0),
        DIRECT_O16_KERNEL_TABLE_BLK(false, 1),
        DIRECT_O16_KERNEL_TABLE_BLK(false, 2),
    },
    {
        DIRECT_O16_KERNEL_TABLE_BLK(true, 0),
        DIRECT_O16_KERNEL_TABLE_BLK(true, 1),
        DIRECT_O16_KERNEL_TABLE_BLK(true, 2),
    },
};

conv2d_n16cx_direct_kernel_fp32_avx512_func_t
conv2d_n16cx_direct_kernel_fp32_avx512_o32_table[NT_STORE_OPT()][STRIDE_W_OPT()][BLK1X14_OW_RF()] =
{
    {
        DIRECT_O32_KERNEL_TABLE_BLK(false, 0),
        DIRECT_O32_KERNEL_TABLE_BLK(false, 1),
        DIRECT_O32_KERNEL_TABLE_BLK(false, 2),
    },
    {
        DIRECT_O32_KERNEL_TABLE_BLK(true, 0),
        DIRECT_O32_KERNEL_TABLE_BLK(true, 1),
        DIRECT_O32_KERNEL_TABLE_BLK(true, 2),
    },
};

conv2d_n16cx_direct_kernel_fp32_avx512_func_t
conv2d_n16cx_direct_kernel_fp32_avx512_o48_table[NT_STORE_OPT()][STRIDE_W_OPT()][BLK1X14_OW_RF()] =
{
    {
        DIRECT_O48_KERNEL_TABLE_BLK(false, 0),
        DIRECT_O48_KERNEL_TABLE_BLK(false, 1),
        DIRECT_O48_KERNEL_TABLE_BLK(false, 2),
    },
    {
        DIRECT_O48_KERNEL_TABLE_BLK(true, 0),
        DIRECT_O48_KERNEL_TABLE_BLK(true, 1),
        DIRECT_O48_KERNEL_TABLE_BLK(true, 2),
    },
};

conv2d_n16cx_direct_kernel_fp32_avx512_func_t
conv2d_n16cx_direct_kernel_fp32_avx512_o64_table[NT_STORE_OPT()][STRIDE_W_OPT()][BLK1X14_OW_RF()] =
{
    {
        DIRECT_O64_KERNEL_TABLE_BLK(false, 0),
        DIRECT_O64_KERNEL_TABLE_BLK(false, 1),
        DIRECT_O64_KERNEL_TABLE_BLK(false, 2),
    },
    {
        DIRECT_O64_KERNEL_TABLE_BLK(true, 0),
        DIRECT_O64_KERNEL_TABLE_BLK(true, 1),
        DIRECT_O64_KERNEL_TABLE_BLK(true, 2),
    },
};

}}};
