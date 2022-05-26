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

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/conv_transpose.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t conv_transpose_ndarray_fp32_get_buffer_bytes(
    const ppl::common::isa_t isa,
    const int64_t group,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t num_output,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return conv_transpose_ndarray_fp32_avx512_get_buffer_bytes(
            group, src_h, src_w, num_output,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return conv_transpose_ndarray_fp32_fma_get_buffer_bytes(
            group, src_h, src_w, num_output,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w);
    }
    return conv_transpose_ndarray_fp32_sse_get_buffer_bytes(
            group, src_h, src_w, num_output,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w);
}

ppl::common::RetCode conv_transpose_ndarray_fp32(
    const ppl::common::isa_t isa,
    const float *input,
    const float *filter,
    const float *bias,
    const int64_t src_h,
    const int64_t src_w,
    const int64_t dst_h,
    const int64_t dst_w,
    const int64_t batch,
    const int64_t group,
    const int64_t channels,
    const int64_t num_output,
    const int64_t kernel_h,
    const int64_t kernel_w,
    const int64_t stride_h,
    const int64_t stride_w,
    const int64_t pad_h,
    const int64_t pad_w,
    const int64_t hole_h,
    const int64_t hole_w,
    void *tmp_buffer,
    float *output)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return conv_transpose_ndarray_fp32_avx512(
            input, filter, bias,
            src_h, src_w, dst_h, dst_w,
            batch, group, channels, num_output,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, hole_h, hole_w,
            tmp_buffer, output);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return conv_transpose_ndarray_fp32_fma(
            input, filter, bias,
            src_h, src_w, dst_h, dst_w,
            batch, group, channels, num_output,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, hole_h, hole_w,
            tmp_buffer, output);
    }
    return conv_transpose_ndarray_fp32_sse(
            input, filter, bias,
            src_h, src_w, dst_h, dst_w,
            batch, group, channels, num_output,
            kernel_h, kernel_w, stride_h, stride_w,
            pad_h, pad_w, hole_h, hole_w,
            tmp_buffer, output);
}

}}}; // namespace ppl::kernel::x86
