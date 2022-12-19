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

#include <vector>
#include <string.h>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/simd_tools.h"
#include "ppl/kernel/arm_server/common/pad_channel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

struct concat_parallel_info {
    concat_parallel_info() {}
    concat_parallel_info(
        const int64_t blk_idx,
        const int64_t start,
        const int64_t end)
    {
        this->blk_idx = blk_idx;
        this->start   = start;
        this->end     = end;
    }

    int64_t blk_idx;
    int64_t start;
    int64_t end;
};

template <typename eT>
static ppl::common::RetCode concat_ndarray(
    const ppl::common::TensorShape **src_shape_list,
    const eT **src_list,
    const int64_t num_src,
    const int64_t axis,
    eT *dst)
{
    const int64_t ndims      = int64_t(src_shape_list[0]->GetDimCount());
    const int64_t fixed_axis = axis < 0 ? ndims + axis : axis;

    int64_t outer_dim = 1;
    int64_t inner_dim = 1;
    for (int64_t i = 0; i < fixed_axis; ++i) {
        outer_dim *= src_shape_list[0]->GetDim(i);
    }
    for (int64_t i = fixed_axis + 1; i < ndims; ++i) {
        inner_dim *= src_shape_list[0]->GetDim(i);
    }

    std::vector<int64_t> dst_offset(num_src);
    dst_offset[0] = 0;
    for (int64_t i = 1; i < num_src; ++i) {
        dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(fixed_axis);
    }
    const int64_t dst_concat_dim = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(fixed_axis);

    int64_t num_threads = PPL_OMP_MAX_THREADS();
    if (dst_concat_dim * inner_dim * sizeof(eT) < 16 && num_src >= num_threads) { // when has small inner dims(usually occurred when fixed_axis == ndims - 1), use memcpy_neon to replace memcpy & change index calculating method
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t n = 0; n < num_src; n++) {
            eT *p_dst                   = dst + dst_offset[n] * inner_dim;
            const eT *p_src             = src_list[n];
            const int64_t copy_elements = src_shape_list[n]->GetDim(fixed_axis) * inner_dim;
            for (int64_t i = 0; i < outer_dim; i++) {
                memcpy_neon(p_dst, p_src, copy_elements * sizeof(eT));
                p_dst += dst_concat_dim * inner_dim;
                p_src += copy_elements;
            }
        }
    } else if (num_src * outer_dim >= num_threads) {
        PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
        for (int64_t i = 0; i < outer_dim; i++) {
            for (int64_t n = 0; n < num_src; n++) {
                memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                       src_list[n] + i * src_shape_list[n]->GetDim(fixed_axis) * inner_dim,
                       src_shape_list[n]->GetDim(fixed_axis) * inner_dim * sizeof(eT));
            }
        }
    } else { // parallel divide along concat dim, may across different src
        num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), dst_concat_dim);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t concat_dim_per_thread = div_up(dst_concat_dim, num_threads);
            const int64_t start_concat_dim      = concat_dim_per_thread * thread_id;
            const int64_t end_concat_dim        = min(concat_dim_per_thread * (thread_id + 1), dst_concat_dim);

            if (start_concat_dim < end_concat_dim) {
                int64_t start_blk_idx = num_src - 1;
                int64_t end_blk_idx   = num_src - 1;
                for (int64_t i = 0; i < num_src - 1; i++) {
                    if (start_concat_dim >= dst_offset[i] && start_concat_dim < dst_offset[i + 1]) {
                        start_blk_idx = i;
                    }
                    if (end_concat_dim >= dst_offset[i] && end_concat_dim < dst_offset[i + 1]) {
                        end_blk_idx = i;
                    }
                }
                int64_t start_axis_idx = start_concat_dim - dst_offset[start_blk_idx];
                int64_t end_axis_idx   = end_concat_dim - dst_offset[end_blk_idx];

                std::vector<concat_parallel_info> infos;
                if (start_blk_idx == end_blk_idx) { // copy from single src
                    infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                } else { // start_blk_idx < end_blk_idx, copy from multiple src
                    infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, src_shape_list[start_blk_idx]->GetDim(fixed_axis))); // start blk, from start to dim(fixed_axis)
                    for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                        infos.emplace_back(concat_parallel_info(i, 0, src_shape_list[i]->GetDim(fixed_axis))); // mid blk, from 0 to dim(fixed_axis)
                    }
                    infos.emplace_back(concat_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                }

                for (int64_t i = 0; i < outer_dim; i++) {
                    for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                        const concat_parallel_info &info = infos[j];

                        const eT *p_src = src_list[info.blk_idx] + (i * src_shape_list[info.blk_idx]->GetDim(fixed_axis) + info.start) * inner_dim;
                        eT *p_dst       = dst + (i * dst_concat_dim + dst_offset[info.blk_idx] + info.start) * inner_dim;

                        const size_t size = (info.end - info.start) * inner_dim * sizeof(eT);
                        memcpy(p_dst, p_src, size);
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int32_t c_blk>
static ppl::common::RetCode concat_nbcx_interleave_channels(
    const ppl::common::TensorShape **src_shape_list,
    const eT **src_list,
    const int64_t num_src,
    const int64_t axis,
    const int64_t c_dim_idx,
    eT *dst)
{
    const int64_t ndims = int64_t(src_shape_list[0]->GetDimCount());
    int64_t outer_dim   = 1;
    int64_t inner_dim   = 1;
    for (int64_t i = 0; i < c_dim_idx; ++i) {
        outer_dim *= src_shape_list[0]->GetDim(i);
    }
    for (int64_t i = c_dim_idx + 1; i < ndims; ++i) {
        inner_dim *= src_shape_list[0]->GetDim(i);
    }

    std::vector<int64_t> dst_offset(num_src);
    dst_offset[0] = 0;
    for (int64_t i = 1; i < num_src; ++i) {
        dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(c_dim_idx);
    }

    const int64_t dst_channels = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(c_dim_idx);
    const int64_t padded_oc    = round_up(dst_channels, c_blk);

    const int64_t num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), inner_dim);
    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
        const int64_t inner_dim_per_thread = div_up(inner_dim, num_threads);
        const int64_t start_inner_dim      = inner_dim_per_thread * thread_id;
        const int64_t end_inner_dim        = min(inner_dim_per_thread * (thread_id + 1), inner_dim);

        if (start_inner_dim < end_inner_dim) {
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int64_t n = 0; n < num_src; n++) {
                    const int64_t src_channels = src_shape_list[n]->GetDim(c_dim_idx);
                    const int64_t padded_ic    = round_up(src_channels, c_blk);
                    for (int64_t ic = 0; ic < padded_ic; ic += c_blk) {
                        const int64_t oc = dst_offset[n] + ic;
                        const eT *p_src  = src_list[n] + i * padded_ic * inner_dim + ic * inner_dim;
                        eT *p_dst        = dst + i * padded_oc * inner_dim + round(oc, c_blk) * inner_dim;
                        if (oc % c_blk == 0) { // no interleave on this c_blk
                            memcpy(p_dst + start_inner_dim * c_blk, p_src + start_inner_dim * c_blk, (end_inner_dim - start_inner_dim) * c_blk * sizeof(eT));
                        } else { // has interleave on this c_blk
                            const int64_t c_offset = c_blk - (oc % c_blk);
                            const int64_t c_end    = min(src_channels - ic, (int64_t)c_blk);
                            if (c_end > c_offset) {    // has two dst c_blk
                                eT *p_dst_next_c_blk   = p_dst + c_blk * inner_dim;
                                if (n == num_src - 1 && ic + c_blk == padded_ic && dst_channels < padded_oc) {  // last c_blk need to pad 0
                                    for (int64_t id = start_inner_dim; id < end_inner_dim; id++) {
                                        // interleave copy
                                        for (int64_t c = 0; c < c_offset; c++) {
                                            p_dst[id * c_blk + c_blk - c_offset + c] = p_src[id * c_blk + c];
                                        }
                                        for (int64_t c = c_offset; c < c_end; c++) {
                                            p_dst_next_c_blk[id * c_blk + c - c_offset] = p_src[id * c_blk + c];
                                        }
                                        for (int64_t c = c_end; c < c_blk; c++) {   // pad 0
                                            p_dst_next_c_blk[id * c_blk + c - c_offset] = 0;
                                        }
                                    }
                                } else {
                                    for (int64_t id = start_inner_dim; id < end_inner_dim; id++) {
                                        // interleave copy
                                        for (int64_t c = 0; c < c_offset; c++) {
                                            p_dst[id * c_blk + c_blk - c_offset + c] = p_src[id * c_blk + c];
                                        }
                                        for (int64_t c = c_offset; c < c_end; c++) {
                                            p_dst_next_c_blk[id * c_blk + c - c_offset] = p_src[id * c_blk + c];
                                        }
                                    }
                                }
                            } else {    // only has one dst c_blk
                                for (int64_t id = start_inner_dim; id < end_inner_dim; id++) {
                                    // interleave copy
                                    for (int64_t c = 0; c < c_offset; c++) {
                                        p_dst[id * c_blk + c_blk - c_offset + c] = p_src[id * c_blk + c];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int32_t c_blk>
static ppl::common::RetCode concat_nbcx(
    const ppl::common::TensorShape **src_shape_list,
    const eT **src_list,
    const int64_t num_src,
    const int64_t axis,
    eT *dst)
{
    const int64_t ndims      = int64_t(src_shape_list[0]->GetDimCount());
    const int64_t fixed_axis = axis < 0 ? ndims + axis : axis;
    const int64_t c_dim_idx  = 1;

    if (fixed_axis == c_dim_idx) { // concat on C dim
        for (int64_t i = 0; i < num_src - 1; i++) {
            if (src_shape_list[i]->GetDim(c_dim_idx) % c_blk != 0) {
                return concat_nbcx_interleave_channels<eT, c_blk>(
                    src_shape_list,
                    src_list,
                    num_src,
                    axis,
                    c_dim_idx,
                    dst);
            }
        }
    }

    int64_t outer_dim = 1;
    int64_t inner_dim = c_blk;
    for (int64_t i = 0; i < fixed_axis; ++i) {
        if (i == c_dim_idx) {
            outer_dim *= div_up(src_shape_list[0]->GetDim(i), c_blk);
        } else {
            outer_dim *= src_shape_list[0]->GetDim(i);
        }
    }
    for (int64_t i = fixed_axis + 1; i < ndims; ++i) {
        if (i == c_dim_idx) {
            inner_dim *= div_up(src_shape_list[0]->GetDim(i), c_blk);
        } else {
            inner_dim *= src_shape_list[0]->GetDim(i);
        }
    }

    std::vector<int64_t> dst_offset(num_src);
    int64_t dst_concat_dim = 0;
    dst_offset[0]          = 0;
    if (fixed_axis == c_dim_idx) {
        for (int64_t i = 1; i < num_src; ++i) {
            dst_offset[i] = dst_offset[i - 1] + div_up(src_shape_list[i - 1]->GetDim(fixed_axis), c_blk);
        }
        dst_concat_dim = dst_offset[num_src - 1] + div_up(src_shape_list[num_src - 1]->GetDim(fixed_axis), c_blk);
    } else {
        for (int64_t i = 1; i < num_src; ++i) {
            dst_offset[i] = dst_offset[i - 1] + src_shape_list[i - 1]->GetDim(fixed_axis);
        }
        dst_concat_dim = dst_offset[num_src - 1] + src_shape_list[num_src - 1]->GetDim(fixed_axis);
    }

    int64_t num_threads = PPL_OMP_MAX_THREADS();
    if (num_src * outer_dim >= num_threads) {
        if (fixed_axis == c_dim_idx) {
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int64_t n = 0; n < num_src; n++) {
                    memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                           src_list[n] + i * div_up(src_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dim,
                           div_up(src_shape_list[n]->GetDim(fixed_axis), c_blk) * inner_dim * sizeof(eT));
                }
            }
        } else {
            PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
            for (int64_t i = 0; i < outer_dim; i++) {
                for (int64_t n = 0; n < num_src; n++) {
                    memcpy(dst + (i * dst_concat_dim + dst_offset[n]) * inner_dim,
                           src_list[n] + i * src_shape_list[n]->GetDim(fixed_axis) * inner_dim,
                           src_shape_list[n]->GetDim(fixed_axis) * inner_dim * sizeof(eT));
                }
            }
        }
    } else { // parallel divide along concat dim, may across different src
        num_threads = min((int64_t)PPL_OMP_MAX_THREADS(), dst_concat_dim);
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t thread_id = 0; thread_id < num_threads; thread_id++) {
            const int64_t concat_dim_per_thread = div_up(dst_concat_dim, num_threads);
            const int64_t start_concat_dim      = concat_dim_per_thread * thread_id;
            const int64_t end_concat_dim        = min(concat_dim_per_thread * (thread_id + 1), dst_concat_dim);

            if (start_concat_dim < end_concat_dim) {
                int64_t start_blk_idx = num_src - 1;
                int64_t end_blk_idx   = num_src - 1;
                for (int64_t i = 0; i < num_src - 1; i++) {
                    if (start_concat_dim >= dst_offset[i] && start_concat_dim < dst_offset[i + 1]) {
                        start_blk_idx = i;
                    }
                    if (end_concat_dim >= dst_offset[i] && end_concat_dim < dst_offset[i + 1]) {
                        end_blk_idx = i;
                    }
                }
                int64_t start_axis_idx = start_concat_dim - dst_offset[start_blk_idx];
                int64_t end_axis_idx   = end_concat_dim - dst_offset[end_blk_idx];

                if (fixed_axis == c_dim_idx) {
                    std::vector<concat_parallel_info> infos;
                    if (start_blk_idx == end_blk_idx) { // copy from single src
                        infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                    } else { // start_blk_idx < end_blk_idx, copy from multiple src
                        infos.emplace_back(concat_parallel_info(
                            start_blk_idx, start_axis_idx, div_up(src_shape_list[start_blk_idx]->GetDim(fixed_axis), c_blk))); // start blk, from start to dim(fixed_axis)
                        for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                            infos.emplace_back(concat_parallel_info(i, 0, div_up(src_shape_list[i]->GetDim(fixed_axis), c_blk))); // mid blk, from 0 to dim(fixed_axis)
                        }
                        infos.emplace_back(concat_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                    }

                    for (int64_t i = 0; i < outer_dim; i++) {
                        for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                            const concat_parallel_info &info = infos[j];

                            const eT *p_src = src_list[info.blk_idx] + (i * div_up(src_shape_list[info.blk_idx]->GetDim(fixed_axis), c_blk) + info.start) * inner_dim;
                            eT *p_dst       = dst + (i * dst_concat_dim + dst_offset[info.blk_idx] + info.start) * inner_dim;

                            const size_t size = (info.end - info.start) * inner_dim * sizeof(eT);
                            memcpy(p_dst, p_src, size);
                        }
                    }
                } else {
                    std::vector<concat_parallel_info> infos;
                    if (start_blk_idx == end_blk_idx) { // copy from single src
                        infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, end_axis_idx)); // from start to end
                    } else { // start_blk_idx < end_blk_idx, copy from multiple src
                        infos.emplace_back(concat_parallel_info(start_blk_idx, start_axis_idx, src_shape_list[start_blk_idx]->GetDim(fixed_axis))); // start blk, from start to dim(fixed_axis)
                        for (int64_t i = start_blk_idx + 1; i < end_blk_idx; i++) {
                            infos.emplace_back(concat_parallel_info(i, 0, src_shape_list[i]->GetDim(fixed_axis))); // mid blk, from 0 to dim(fixed_axis)
                        }
                        infos.emplace_back(concat_parallel_info(end_blk_idx, 0, end_axis_idx)); // end blk, from 0 to end
                    }

                    for (int64_t i = 0; i < outer_dim; i++) {
                        for (int64_t j = 0; j < (int64_t)infos.size(); j++) {
                            const concat_parallel_info &info = infos[j];

                            const eT *p_src = src_list[info.blk_idx] + (i * src_shape_list[info.blk_idx]->GetDim(fixed_axis) + info.start) * inner_dim;
                            eT *p_dst       = dst + (i * dst_concat_dim + dst_offset[info.blk_idx] + info.start) * inner_dim;

                            const size_t size = (info.end - info.start) * inner_dim * sizeof(eT);
                            memcpy(p_dst, p_src, size);
                        }
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT>
static ppl::common::RetCode concat_wrapper(
    const ppl::common::TensorShape **src_shape_list,
    const void **src_list,
    const int64_t num_src,
    const int64_t axis,
    void *dst)
{
    const auto data_format = src_shape_list[0]->GetDataFormat();
    if (data_format == ppl::common::DATAFORMAT_NDARRAY) {
        return concat_ndarray<eT>(src_shape_list, (const eT **)src_list, num_src, axis, (eT *)dst);
    }

    // NBCX
    if (sizeof(eT) == 4) {
        if (data_format == ppl::common::DATAFORMAT_N4CX) { // 32bit n4cx
            return concat_nbcx<uint32_t, 4>(src_shape_list, (const uint32_t **)src_list, num_src, axis, (uint32_t *)dst);
        }
    }
    if (sizeof(eT) == 2) {
        if (data_format == ppl::common::DATAFORMAT_N8CX) { // 16bit n8cx
            return concat_nbcx<uint16_t, 8>(src_shape_list, (const uint16_t **)src_list, num_src, axis, (uint16_t *)dst);
        }
    }

    return ppl::common::RC_UNSUPPORTED;
}

ppl::common::RetCode concat(
    const ppl::common::TensorShape **src_shape_list,
    const void **src_list,
    const int64_t num_src,
    const int64_t axis,
    void *dst)
{
    const auto data_type = src_shape_list[0]->GetDataType();
    switch (ppl::common::GetSizeOfDataType(data_type)) {
        case 1: return concat_wrapper<uint8_t>(src_shape_list, src_list, num_src, axis, dst);
        case 2: return concat_wrapper<uint16_t>(src_shape_list, src_list, num_src, axis, dst);
        case 4: return concat_wrapper<uint32_t>(src_shape_list, src_list, num_src, axis, dst);
        case 8: return concat_wrapper<uint64_t>(src_shape_list, src_list, num_src, axis, dst);
        default: break;
    }
    return ppl::common::RC_UNSUPPORTED;
}

}}}} // namespace ppl::kernel::arm_server::neon
