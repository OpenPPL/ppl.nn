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

#include "ppl/kernel/riscv/common/conv_transpose/conv_transpose_common.h"
#include "ppl/kernel/riscv/fp16/conv2d/common/gemm_common_kernel.h"

namespace ppl { namespace kernel { namespace riscv {

int64_t conv_transpose_n8cx_get_buffer_bytes_fp16_vec128(
    const int32_t batch,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t num_output,
    const int32_t channels,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w)
{
    constexpr int32_t c_blk = 8;
    return conv_transpose_nxcx_get_buffer_bytes_common<__fp16, c_blk, c_blk>(
        batch,
        src_h,
        src_w,
        num_output,
        channels,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w);
}

static void conv_transpose_n8cx_gemm_func_fp16_vec128(
    const __fp16 *A,
    const __fp16 *B,
    __fp16 *C,
    const int32_t M,
    const int32_t N,
    const int32_t K)
{
    constexpr bool gemm_first_flag = true;
    auto gemm_func                 = conv_gemm_select_kernel_fp16<gemm_first_flag>(N);
    gemm_func(A, B, C, M, N, K);
}

ppl::common::RetCode conv_transpose_n8cx_fp16_vec128(
    const __fp16 *input,
    const __fp16 *filter,
    const __fp16 *bias,
    const int32_t src_h,
    const int32_t src_w,
    const int32_t dst_h,
    const int32_t dst_w,
    const int32_t batch,
    const int32_t channels,
    const int32_t num_output,
    const int32_t kernel_h,
    const int32_t kernel_w,
    const int32_t stride_h,
    const int32_t stride_w,
    const int32_t pad_h,
    const int32_t pad_w,
    const int32_t hole_h,
    const int32_t hole_w,
    __fp16 *tmp_buffer,
    __fp16 *output)
{
    constexpr int32_t c_blk = 8;

    return conv_transpose_nxcx_common<__fp16, c_blk, conv_transpose_n8cx_gemm_func_fp16_vec128>(
        input,
        filter,
        bias,
        src_h,
        src_w,
        dst_h,
        dst_w,
        batch,
        channels,
        num_output,
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        hole_h,
        hole_w,
        tmp_buffer,
        output);
}

}}}; // namespace ppl::kernel::riscv