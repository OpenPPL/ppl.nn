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

#include "ppl/kernel/x86/fp32/conv2d/winograd/avx512/conv2d_n16cx_winograd_t6_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/winograd/avx512/conv2d_n16cx_winograd_t9_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/winograd/avx512/conv2d_n16cx_winograd_t14_kernel_fp32_avx512.h"
#include "ppl/kernel/x86/fp32/conv2d/winograd/avx512/conv2d_n16cx_winograd_t31_kernel_fp32_avx512.h"

namespace ppl { namespace kernel { namespace x86 {

conv2d_n16cx_winograd_kernel_fp32_avx512_func_t
conv2d_n16cx_winograd_kernel_fp32_avx512_o16_table[T14_TILES_RF()] =
{
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 1>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 2>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 3>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 4>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 5>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 6>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 7>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 8>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 9>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 10>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 11>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 12>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 13>,
    conv2d_n16cx_winograd_t31_kernel_fp32_avx512<1 * CH_DT_BLK(), 14>,
};

conv2d_n16cx_winograd_kernel_fp32_avx512_func_t
conv2d_n16cx_winograd_kernel_fp32_avx512_o32_table[T14_TILES_RF()] =
{
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 1>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 2>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 3>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 4>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 5>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 6>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 7>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 8>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 9>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 10>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 11>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 12>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 13>,
    conv2d_n16cx_winograd_t14_kernel_fp32_avx512<2 * CH_DT_BLK(), 14>,
};

// conv2d_n16cx_winograd_kernel_fp32_avx512_func_t
// conv2d_n16cx_winograd_kernel_fp32_avx512_o48_table[T9_TILES_RF()] =
// {
//     conv2d_n16cx_winograd_t9_kernel_fp32_avx512<3 * CH_DT_BLK(), 1>,
//     conv2d_n16cx_winograd_t9_kernel_fp32_avx512<3 * CH_DT_BLK(), 2>,
//     conv2d_n16cx_winograd_t9_kernel_fp32_avx512<3 * CH_DT_BLK(), 3>,
//     conv2d_n16cx_winograd_t9_kernel_fp32_avx512<3 * CH_DT_BLK(), 4>,
//     conv2d_n16cx_winograd_t9_kernel_fp32_avx512<3 * CH_DT_BLK(), 5>,
//     conv2d_n16cx_winograd_t9_kernel_fp32_avx512<3 * CH_DT_BLK(), 6>,
//     conv2d_n16cx_winograd_t9_kernel_fp32_avx512<3 * CH_DT_BLK(), 7>,
//     conv2d_n16cx_winograd_t9_kernel_fp32_avx512<3 * CH_DT_BLK(), 8>,
//     conv2d_n16cx_winograd_t9_kernel_fp32_avx512<3 * CH_DT_BLK(), 9>,
// };

// conv2d_n16cx_winograd_kernel_fp32_avx512_func_t
// conv2d_n16cx_winograd_kernel_fp32_avx512_o64_table[T6_TILES_RF()] =
// {
//     conv2d_n16cx_winograd_t6_kernel_fp32_avx512<4 * CH_DT_BLK(), 1>,
//     conv2d_n16cx_winograd_t6_kernel_fp32_avx512<4 * CH_DT_BLK(), 2>,
//     conv2d_n16cx_winograd_t6_kernel_fp32_avx512<4 * CH_DT_BLK(), 3>,
//     conv2d_n16cx_winograd_t6_kernel_fp32_avx512<4 * CH_DT_BLK(), 4>,
//     conv2d_n16cx_winograd_t6_kernel_fp32_avx512<4 * CH_DT_BLK(), 5>,
//     conv2d_n16cx_winograd_t6_kernel_fp32_avx512<4 * CH_DT_BLK(), 6>,
// };

}}};
