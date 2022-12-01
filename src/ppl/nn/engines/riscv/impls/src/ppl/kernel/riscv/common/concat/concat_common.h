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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_CONCAT_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_CONCAT_COMMON_H_

#include <vector>
#include <cstring>

#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

template <typename T, int32_t c_blk>
ppl::common::RetCode concat_nbcx(
    const T** src_list,
    T* dst,

    const ppl::common::TensorShape** src_shape_list,
    const int32_t num_src,
    const int32_t c_axis)
{
    const int32_t ndims     = int32_t(src_shape_list[0]->GetDimCount());
    const int32_t axis      = c_axis < 0 ? ndims + c_axis : c_axis;
    const int32_t c_dim_idx = 1;

    int64_t outer_dim = 1;
    int64_t inner_dim = c_blk;
    for (int32_t i = 0; i < axis; i++) {
        if (i == c_dim_idx) {
            outer_dim *= div_up(src_shape_list[0]->GetDim(i), c_blk);
        } else {
            outer_dim *= src_shape_list[0]->GetDim(i);
        }
    }
    for (int32_t i = axis + 1; i < ndims; i++) {
        if (i == c_dim_idx) {
            inner_dim *= div_up(src_shape_list[0]->GetDim(i), c_blk);
        } else {
            inner_dim *= src_shape_list[0]->GetDim(i);
        }
    }

    std::vector<int64_t> dst_offset(num_src);
    dst_offset[0]          = 0;
    int64_t dst_concat_dim = 0;
    if (axis == c_dim_idx) {
        for (int32_t i = 1; i < num_src; i++) {
            dst_offset[i] = dst_offset[i - 1] + div_up(src_shape_list[i - 1]->GetDim(axis), c_blk);
        }
        dst_concat_dim = dst_offset[num_src - 1] + div_up(src_shape_list[num_src - 1]->GetDim(axis), c_blk);
    } else {
        for (int32_t i = 1; i < num_src; i++) {
            dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(axis);
        }
        dst_concat_dim = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(axis);
    }

    if (axis == c_dim_idx) {
        for (int64_t i = 0; i < outer_dim; i++) {
            for (int64_t n = 0; n < num_src; n++) {
                memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                       src_list[n] + i * div_up(src_shape_list[n]->GetDim(axis), c_blk) * inner_dim,
                       div_up(src_shape_list[n]->GetDim(axis), c_blk) * inner_dim * sizeof(T));
            }
        }
    } else {
        for (int64_t i = 0; i < outer_dim; i++) {
            for (int64_t n = 0; n < num_src; n++) {
                memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                       src_list[n] + i * src_shape_list[n]->GetDim(axis) * inner_dim,
                       src_shape_list[n]->GetDim(axis) * inner_dim * sizeof(T));
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename T>
ppl::common::RetCode concat_ndarray(
    const T** src_list,
    T* dst,

    const ppl::common::TensorShape** src_shape_list,
    const int32_t num_src,
    const int32_t c_axis)
{
    const int32_t ndims = int32_t(src_shape_list[0]->GetDimCount());
    const int32_t axis  = c_axis < 0 ? ndims + c_axis : c_axis;

    int64_t outer_dim = 1;
    int64_t inner_dim = 1;
    for (int32_t i = 0; i < axis; i++) {
        outer_dim *= src_shape_list[0]->GetDim(i);
    }
    for (int32_t i = axis + 1; i < ndims; i++) {
        inner_dim *= src_shape_list[0]->GetDim(i);
    }

    std::vector<int64_t> dst_offset(num_src);
    dst_offset[0] = 0;
    for (int32_t i = 1; i < num_src; i++) {
        dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(axis);
    }
    const int64_t dst_concat_dim = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(axis);

    for (int64_t i = 0; i < outer_dim; i++) {
        for (int64_t n = 0; n < num_src; n++) {
            memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                   src_list[n] + i * src_shape_list[n]->GetDim(axis) * inner_dim,
                   src_shape_list[n]->GetDim(axis) * inner_dim * sizeof(T));
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename T, int32_t c_blk>
ppl::common::RetCode concat_nbcx_interleave_channels(
    const T** src_list,
    T* dst,

    const ppl::common::TensorShape** src_shape_list,
    const int32_t num_src,
    const int32_t axis,
    const int32_t c_dim_idx)
{
    const int32_t ndims = int32_t(src_shape_list[0]->GetDimCount());
    int64_t outer_dim   = 1;
    int64_t inner_dim   = 1;
    for (int32_t i = 0; i < c_dim_idx; ++i) {
        outer_dim *= src_shape_list[0]->GetDim(i);
    }
    for (int32_t i = c_dim_idx + 1; i < ndims; ++i) {
        inner_dim *= src_shape_list[0]->GetDim(i);
    }

    std::vector<int64_t> dst_offset(num_src);
    dst_offset[0] = 0;
    for (int32_t i = 1; i < num_src; ++i) {
        dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(c_dim_idx);
    }

    const int64_t dst_channels = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(c_dim_idx);
    const int64_t padded_oc    = round_up(dst_channels, c_blk);

    for (int64_t i = 0; i < outer_dim; i++) {
        for (int64_t n = 0; n < num_src; n++) {
            const int32_t src_channels = src_shape_list[n]->GetDim(c_dim_idx);
            const int32_t padded_ic    = round_up(src_channels, c_blk);
            for (int32_t ic = 0; ic < padded_ic; ic += c_blk) {
                const int32_t oc = dst_offset[n] + ic;
                const T* src_    = src_list[n] + i * padded_ic * inner_dim + ic * inner_dim;
                T* dst_          = dst + i * padded_oc * inner_dim + round(oc, c_blk) * inner_dim;
                if (oc % c_blk == 0) { //  no interleave on this xc
                    memcpy(dst_, src_, inner_dim * c_blk * sizeof(T));
                } else { //  has interleave on this xc
                    const int32_t c_offset = c_blk - (oc % c_blk);
                    const int32_t c_end    = min(src_channels - ic, (int32_t)c_blk);
                    T* dst_next_xc         = dst_ + c_blk * inner_dim;
                    for (int64_t id = 0; id < inner_dim; id++) {
                        // interleave copy
                        for (int32_t c = 0; c < c_offset; c++) {
                            dst_[id * c_blk + c_blk - c_offset + c] = src_[id * c_blk + c];
                        }
                        for (int32_t c = c_offset; c < c_end; c++) {
                            dst_next_xc[id * c_blk + c - c_offset] = src_[id * c_blk + c];
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_COMMON_CONCAT_COMMON_H_
