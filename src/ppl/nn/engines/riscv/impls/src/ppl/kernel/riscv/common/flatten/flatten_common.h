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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_FLATTEN_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_FLATTEN_COMMON_H_

#include <vector>
#include <cstring>

#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

template <typename T, int32_t c_blk>
ppl::common::RetCode flatten_nbcx(
    const T* src,
    T* dst,

    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape)
{
    if (src_shape->GetDimCount() != 4) {
        return ppl::common::RC_UNSUPPORTED;
    }
    const int64_t batch    = src_shape->GetDim(0);
    const int64_t channels = src_shape->GetDim(1);
    const int64_t src_h    = src_shape->GetDim(2);
    const int64_t src_w    = src_shape->GetDim(3);

    const int64_t pad_c   = round_up(channels, c_blk);
    const int64_t size_2D = src_h * src_w;
    // const int64_t size_3D  = channels * src_h * src_w;
    const int64_t size_3D = pad_c * size_2D;
    // c8-w-h-(pad_c/8)-n -> w-h-c8-(pad_c/8)-n
    // for (int64_t b = 0; b < batch; ++b) {
    //     const T *src_ = src + b * size_3D;
    //     T *dst_       = dst + b * size_3D;
    //     for (int64_t c = 0; c < pad_c; ++c) {
    //         for (int64_t hw = 0; hw < size_2D; ++hw) {
    //             dst_[c * size_2D + hw] = src_[(c / c_blk) * size_2D * c_blk + hw * c_blk + c % c_blk];
    //         }
    //     }
    // }

    // c8-w-h-(pad_c/8)-n -> w-h-c-pad_chw-n
    const int64_t pad_3D = round_up(channels * size_2D, c_blk);
    for (int64_t b = 0; b < batch; ++b) {
        const T* src_ = src + b * size_3D;
        T* dst_       = dst + b * pad_3D;
        for (int64_t i = 0; i < pad_3D; ++i) {
            if (i < size_3D) {
                int64_t c_idx   = i / size_2D;
                int64_t hw_idx  = i % size_2D;
                int64_t src_idx = (c_idx / c_blk) * size_2D * c_blk + hw_idx * c_blk + c_idx % c_blk;
                dst_[i]         = src_[src_idx];
            } else {
                dst_[i] = 0;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_COMMON_FLATTEN_COMMON_H_
