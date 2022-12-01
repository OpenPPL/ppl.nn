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

#include <string.h>

#include "ppl/kernel/riscv/common/internal_include.h"
#include "ppl/kernel/riscv/fp32/transpose.h"

namespace ppl { namespace kernel { namespace riscv {

ppl::common::RetCode channel_shuffle_ndarray_fp32(
    const ppl::common::TensorShape* src_shape,
    const float* src,
    const int32_t group,
    float* dst)
{
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);
    const int64_t mid_c1   = group;
    const int64_t mid_c2   = channels / group;
    int64_t mid_dims[5]    = {batch, mid_c1, mid_c2, src_h, src_w};

    ppl::common::TensorShape& mid_shape = *(new ppl::common::TensorShape());
    mid_shape.Reshape(mid_dims, 5);

    return transpose_ndarray_continous2d_fp32(src, dst, &mid_shape, 1, 2);
}

ppl::common::RetCode channel_shuffle_ndarray_concat_split_fp32(
    const ppl::common::TensorShape* src0_shape,
    const ppl::common::TensorShape* src1_shape,
    const float* src0,
    const float* src1,
    const int32_t group,
    float* dst0,
    float* dst1_optional)
{
    const int64_t in_c1    = src0_shape->GetDim(1);
    const int64_t in_c2    = src1_shape->GetDim(1);
    const int64_t channels = in_c1 + in_c2;
    if (dst1_optional && channels % 2) {
        return ppl::common::RC_INVALID_VALUE;
    }
    float* dst1          = dst1_optional;
    const int64_t out_c1 = dst1 ? channels / 2 : channels;
    const int64_t out_c2 = dst1 ? channels / 2 : 0;

    const int64_t batch      = src0_shape->GetDim(0);
    const int64_t src_h      = src0_shape->GetDim(2);
    const int64_t src_w      = src0_shape->GetDim(3);
    const int64_t inner_dims = src_h * src_w;
    const int64_t mid_c1     = group;
    const int64_t mid_c2     = channels / group;

    // (batch, in_c1 + in_c2, src_h, src_w) -> (batch, mid_c1, mid_c2, src_h, src_w)
    for (int64_t b = 0; b < batch; b++) {
        for (int64_t j = 0; j < mid_c2; j++) {
            for (int64_t i = 0; i < mid_c1; i++) {
                const int64_t cur_in_c  = i * mid_c2 + j;
                const int64_t cur_out_c = j * mid_c1 + i;
                const float* src_;
                float* dst_;
                if (cur_in_c < in_c1) {
                    src_ = src0 + b * in_c1 * inner_dims + cur_in_c * inner_dims;
                } else {
                    src_ = src1 + b * in_c2 * inner_dims + (cur_in_c - in_c1) * inner_dims;
                }
                if (cur_out_c < out_c1) {
                    dst_ = dst0 + b * out_c1 * inner_dims + cur_out_c * inner_dims;
                } else {
                    dst_ = dst1 + b * out_c2 * inner_dims + (cur_out_c - out_c1) * inner_dims;
                }
                memcpy(dst_, src_, inner_dims * sizeof(float));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} //  namespace ppl::kernel::riscv
