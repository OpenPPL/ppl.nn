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

#include "ppl/kernel/x86/fp32/arithmetic_multi_array/avx/arithmetic_multi_array_fp32_avx.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t min_fp32_avx_get_temp_buffer_bytes(
    const uint32_t input_num)
{
    return arithmetic_multi_array_fp32_get_temp_buffer_bytes(input_num);
}

ppl::common::RetCode min_eltwise_fp32_avx(
    const ppl::common::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    float *dst)
{
    if (num_src == 2) {
        return arithmetic_multi_array_eltwise_fp32_avx<ARRAY_MIN, true>(dst_shape, src_list, num_src, dst);
    } else {
        return arithmetic_multi_array_eltwise_fp32_avx<ARRAY_MIN, false>(dst_shape, src_list, num_src, dst);
    }
}

ppl::common::RetCode min_ndarray_fp32_avx(
    const ppl::common::TensorShape **src_shape_list,
    const ppl::common::TensorShape *dst_shape,
    const float **src_list,
    const uint32_t num_src,
    void *temp_buffer,
    float *dst)
{
    if (num_src == 2) {
        return arithmetic_multi_array_ndarray_fp32_avx<ARRAY_MIN, true>(src_shape_list, dst_shape, src_list, num_src, temp_buffer, dst);
    } else {
        return arithmetic_multi_array_ndarray_fp32_avx<ARRAY_MIN, false>(src_shape_list, dst_shape, src_list, num_src, temp_buffer, dst);
    }
}

}}}; // namespace ppl::kernel::x86
