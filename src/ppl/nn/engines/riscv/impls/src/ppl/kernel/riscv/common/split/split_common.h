
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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_SPLIT_SPLIT_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_SPLIT_SPLIT_COMMON_H_

#include <vector>
#include <cstring>

#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

template <typename eT>
ppl::common::RetCode split_ndarray(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape** dst_shape_list,
    const eT* src,
    const int32_t slice_axis,
    const int32_t num_dst,
    eT** dst_list)
{
    const int32_t ndims         = src_shape->GetDimCount();
    const int32_t fixed_axis    = slice_axis < 0 ? slice_axis + ndims : slice_axis;
    const int64_t src_split_dim = src_shape->GetDim(fixed_axis);

    int64_t outer_dims = 1;
    int64_t inner_dims = 1;
    for (int32_t i = 0; i < fixed_axis; i++) {
        outer_dims *= src_shape->GetDim(i);
    }
    for (int32_t i = fixed_axis + 1; i < ndims; i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    std::vector<int64_t> src_offset;
    src_offset.resize(num_dst);
    src_offset[0] = 0;
    for (int32_t i = 1; i < num_dst; i++) {
        src_offset[i] = src_offset[i - 1] + dst_shape_list[i - 1]->GetDim(fixed_axis);
    }

    for (int32_t i = 0; i < outer_dims; i++) {
        for (int32_t n = 0; n < num_dst; n++) {
            const eT* p_src = src + i * src_split_dim * inner_dims + src_offset[n] * inner_dims;
            eT* p_dst       = dst_list[n] + i * dst_shape_list[n]->GetDim(fixed_axis) * inner_dims;

            const size_t size = dst_shape_list[n]->GetDim(fixed_axis) * inner_dims * sizeof(eT);
            memcpy(p_dst, p_src, size);
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int64_t c_blk>
ppl::common::RetCode split_nxcx_interleave_channels(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape** dst_shape_list,
    const eT* src,
    const int32_t slice_axis,
    const int32_t num_dst,
    const int32_t c_dim_idx,
    eT** dst_list)
{
    const int32_t ndims = src_shape->GetDimCount();

    int64_t outer_dims = 1;
    int64_t inner_dims = 1;
    for (int32_t i = 0; i < c_dim_idx; i++) {
        outer_dims *= src_shape->GetDim(i);
    }
    for (int32_t i = c_dim_idx + 1; i < ndims; i++) {
        inner_dims *= src_shape->GetDim(i);
    }

    std::vector<int64_t> src_offset;
    src_offset.resize(num_dst);
    src_offset[0] = 0;
    for (int32_t i = 1; i < num_dst; i++) {
        src_offset[i] = src_offset[i - 1] + dst_shape_list[i - 1]->GetDim(c_dim_idx);
    }

    const int64_t src_channels = src_shape->GetDim(c_dim_idx);
    const int64_t padded_ic    = round_up(src_channels, c_blk);

    const int64_t start_inner_dims = 0;
    const int64_t end_inner_dims   = inner_dims;

    for (int64_t i = 0; i < outer_dims; i++) {
        for (int32_t n = 0; n < num_dst; n++) {
            const int32_t dst_channels = dst_shape_list[n]->GetDim(c_dim_idx);
            const int32_t padded_oc    = round_up(dst_channels, c_blk);
            for (int32_t oc = 0; oc < padded_oc; oc += c_blk) {
                const int32_t ic = src_offset[n] + oc;
                const eT* p_src  = src + i * padded_ic * inner_dims + round(ic, c_blk) * inner_dims;
                eT* p_dst        = dst_list[n] + i * padded_oc * inner_dims + oc * inner_dims;
                if (ic % c_blk == 0) { // no interleave on this xc
                    memcpy(p_dst + start_inner_dims * c_blk, p_src + start_inner_dims * c_blk, (end_inner_dims - start_inner_dims) * c_blk * sizeof(eT));
                } else { // has interleave on this xc
                    const int32_t c_offset  = c_blk - (ic % c_blk);
                    const int32_t c_end     = min(dst_channels - oc, (int32_t)c_blk);
                    const eT* p_src_next_xc = p_src + c_blk * inner_dims;

                    if (oc + c_blk == padded_oc && dst_channels < padded_oc) { // last xc need to pad 0
                        for (int64_t id = start_inner_dims; id < end_inner_dims; id++) {
                            // interleave copy
                            for (int32_t c = 0; c < c_offset; c++) {
                                p_dst[id * c_blk + c] = p_src[id * c_blk + c_blk - c_offset + c];
                            }
                            for (int32_t c = c_offset; c < c_end; c++) {
                                p_dst[id * c_blk + c] = p_src_next_xc[id * c_blk + c - c_offset];
                            }
                        }
                    } else {
                        for (int64_t id = start_inner_dims; id < end_inner_dims; id++) {
                            // interleave copy
                            for (int32_t c = 0; c < c_offset; c++) {
                                p_dst[id * c_blk + c] = p_src[id * c_blk + c_blk - c_offset + c];
                            }
                            for (int32_t c = c_offset; c < c_end; c++) {
                                p_dst[id * c_blk + c] = p_src_next_xc[id * c_blk + c - c_offset];
                            }
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int64_t c_blk>
ppl::common::RetCode split_nxcx(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape** dst_shape_list,
    const eT* src,
    const int32_t slice_axis,
    const int32_t num_dst,
    eT** dst_list)
{
    const int32_t ndims      = src_shape->GetDimCount();
    const int32_t fixed_axis = slice_axis < 0 ? slice_axis + ndims : slice_axis;
    const int64_t c_dim_idx  = 1;

    if (fixed_axis == 1) {
        for (int32_t i = 0; i < num_dst - 1; i++) {
            if (dst_shape_list[i]->GetDim(c_dim_idx) % c_blk != 0) {
                return split_nxcx_interleave_channels<eT, c_blk>(
                    src_shape,
                    dst_shape_list,
                    src,
                    slice_axis,
                    num_dst,
                    c_dim_idx,
                    dst_list);
            }
        }
    }

    int64_t outer_dims = 1;
    int64_t inner_dims = c_blk;
    for (int32_t i = 0; i < fixed_axis; i++) {
        if (i == c_dim_idx) {
            outer_dims *= div_up(src_shape->GetDim(i), c_blk);
        } else {
            outer_dims *= src_shape->GetDim(i);
        }
    }
    for (int32_t i = fixed_axis + 1; i < ndims; i++) {
        if (i == c_dim_idx) {
            inner_dims *= div_up(src_shape->GetDim(i), c_blk);
        } else {
            inner_dims *= src_shape->GetDim(i);
        }
    }

    std::vector<int64_t> src_offset;
    src_offset.resize(num_dst);
    src_offset[0]         = 0;
    int64_t src_split_dim = 0;
    if (fixed_axis == c_dim_idx) {
        for (int32_t i = 1; i < num_dst; i++) {
            src_offset[i] = src_offset[i - 1] + div_up(dst_shape_list[i - 1]->GetDim(fixed_axis), c_blk);
        }
        src_split_dim = div_up(src_shape->GetDim(fixed_axis), c_blk);
    } else {
        for (int32_t i = 1; i < num_dst; i++) {
            src_offset[i] = src_offset[i - 1] + dst_shape_list[i - 1]->GetDim(fixed_axis);
        }
        src_split_dim = src_shape->GetDim(fixed_axis);
    }

    if (fixed_axis == c_dim_idx) {
        for (int32_t i = 0; i < outer_dims; i++) {
            for (int32_t n = 0; n < num_dst; n++) {
                const eT* p_src = src + i * src_split_dim * inner_dims + src_offset[n] * inner_dims;
                eT* p_dst       = dst_list[n] + i * div_up(dst_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dims;

                const size_t size = div_up(dst_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dims * sizeof(eT);
                memcpy(p_dst, p_src, size);
            }
        }
    } else {
        for (int32_t i = 0; i < outer_dims; i++) {
            for (int32_t n = 0; n < num_dst; n++) {
                const eT* p_src = src + i * src_split_dim * inner_dims + src_offset[n] * inner_dims;
                eT* p_dst       = dst_list[n] + i * dst_shape_list[n]->GetDim(fixed_axis) * inner_dims;

                const size_t size = dst_shape_list[n]->GetDim(fixed_axis) * inner_dims * sizeof(eT);
                memcpy(p_dst, p_src, size);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::kernel::riscv

#endif // __ST_PPL_KERNEL_RISCV_COMMON_SPLIT_SPLIT_COMMON_H_
