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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_REDUCE_NEON_REDUCE_NBCX_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_REDUCE_NEON_REDUCE_NBCX_COMMON_H_

#include <vector>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/common/threading_tools.h"
#include "ppl/kernel/arm_server/reduce/neon/reduce_common.h"
#include "ppl/kernel/arm_server/common/pad_channel.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, int32_t c_blk, reduce_op_type_t op_type>
static inline void reduce_nbcx_lastdim_no_reduce_c_no_reduce_common(
    const eT* src,
    const int64_t length,
    const int64_t remain_c,
    eT* dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    for (int64_t i = 0; i < length; i++) {
        vecType v_src = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i * c_blk));
        vecType v_dst = vld<eT, eN>(dst + i * c_blk);
        vst<eT, eN>(dst + i * c_blk, reduce_vector_kernel<vecType, op_type>(v_src, v_dst));
    }
}

template <typename eT, int32_t c_blk, reduce_op_type_t op_type>
static inline void reduce_nbcx_lastdim_reduce_c_no_reduce_common(
    const eT* src,
    const int64_t length,
    const int64_t remain_c,
    eT* dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    vecType v_reduced = vdup_n<eT, eN>(reduce_init_val<eT, op_type>());

    for (int64_t i = 0; i < length; i++) {
        vecType v_src = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i * c_blk));
        v_reduced     = reduce_vector_kernel<vecType, op_type>(v_src, v_reduced);
    }

    v_reduced = reduce_vector_kernel<vecType, op_type>(v_reduced, vld<eT, eN>(dst));
    vst<eT, eN>(dst, v_reduced);
}

template <typename eT, int32_t c_blk, reduce_op_type_t op_type>
static inline void reduce_nbcx_lastdim_no_reduce_c_reduce_common(
    const eT* src,
    const int64_t length,
    const int64_t remain_c,
    eT* dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    if (remain_c >= c_blk) {
        for (int64_t i = 0; i < length; i++) {
            vecType v_src  = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i * c_blk));
            dst[i * c_blk] = reduce_vector_to_scalar_kernel<eT, vecType, op_type>(v_src, dst[i * c_blk]);
        }
    } else {
        for (int64_t i = 0; i < length; i++) {
            eT s_reduced = reduce_init_val<eT, op_type>();
            for (int64_t c = 0; c < remain_c; c++) {
                s_reduced = reduce_scalar_kernel<eT, op_type>(reduce_first_process_kernel<eT, op_type>(src[i * c_blk + c]), s_reduced);
            }
            dst[i * c_blk] = reduce_scalar_kernel<eT, op_type>(s_reduced, dst[i * c_blk]);
        }
    }
}

template <typename eT, int32_t c_blk, reduce_op_type_t op_type>
static inline void reduce_nbcx_lastdim_reduce_c_reduce_common(
    const eT* src,
    const int64_t length,
    const int64_t remain_c,
    eT* dst)
{
    constexpr int32_t eN = c_blk;
    typedef typename DT<eT, eN>::vecDT vecType;

    if (remain_c >= c_blk) {
        eT s_reduced      = reduce_init_val<eT, op_type>();
        vecType v_reduced = vdup_n<eT, eN>(s_reduced);
        for (int64_t i = 0; i < length; i++) {
            vecType v_src = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i * c_blk));
            v_reduced     = reduce_vector_kernel<vecType, op_type>(v_src, v_reduced);
        }
        s_reduced = reduce_vector_to_scalar_kernel<eT, vecType, op_type>(v_reduced, s_reduced);
        dst[0]    = reduce_scalar_kernel<eT, op_type>(s_reduced, dst[0]);
    } else {
        eT s_reduced = reduce_init_val<eT, op_type>();
        for (int64_t i = 0; i < length; i++) {
            for (int64_t c = 0; c < remain_c; c++) {
                s_reduced = reduce_scalar_kernel<eT, op_type>(reduce_first_process_kernel<eT, op_type>(src[i * c_blk + c]), s_reduced);
            }
        }
        dst[0] = reduce_scalar_kernel<eT, op_type>(s_reduced, dst[0]);
    }
}

template <typename eT, int32_t c_blk, reduce_op_type_t op_type>
static ppl::common::RetCode reduce_nbcx_recursive_common(
    const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& dst_dims,
    const eT* src,
    const int64_t dim_idx,
    const std::vector<int64_t>& inc_src,
    const std::vector<int64_t>& inc_dst,
    const single_parallel_loop_config_t* pc,
    const int64_t c_dim_idx,
    int64_t remain_c,
    eT* dst)
{
    const int64_t length = dim_idx == c_dim_idx ? div_up(src_dims[dim_idx], c_blk) : src_dims[dim_idx];
    if (dim_idx == (int64_t)src_dims.size() - 1) { // last dim
        const bool lastdim_reduce = dst_dims[dim_idx] != src_dims[dim_idx];
        const bool c_reduce       = dst_dims[c_dim_idx] != src_dims[c_dim_idx];
        if (!lastdim_reduce && !c_reduce) {
            reduce_nbcx_lastdim_no_reduce_c_no_reduce_common<eT, c_blk, op_type>(src, length, remain_c, dst);
        } else if (lastdim_reduce && !c_reduce) {
            reduce_nbcx_lastdim_reduce_c_no_reduce_common<eT, c_blk, op_type>(src, length, remain_c, dst);
        } else if (!lastdim_reduce && c_reduce) {
            reduce_nbcx_lastdim_no_reduce_c_reduce_common<eT, c_blk, op_type>(src, length, remain_c, dst);
        } else {
            reduce_nbcx_lastdim_reduce_c_reduce_common<eT, c_blk, op_type>(src, length, remain_c, dst);
        }
    } else {
        if (pc->depth_of_loop == dim_idx && pc->num_threads > 1) { // parallel on this dim
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc->num_threads; t++) {
                const int64_t length_per_thread = div_up(length, pc->num_threads);
                const int64_t start_idx         = t * length_per_thread;
                const int64_t end_idx           = min(start_idx + length_per_thread, length);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    if (dim_idx == c_dim_idx) {
                        remain_c = src_dims[c_dim_idx] - i * c_blk;
                    }
                    reduce_nbcx_recursive_common<eT, c_blk, op_type>(
                        src_dims,
                        dst_dims,
                        src + i * inc_src[dim_idx],
                        dim_idx + 1,
                        inc_src,
                        inc_dst,
                        pc,
                        c_dim_idx,
                        remain_c,
                        dst + i * inc_dst[dim_idx]);
                }
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                if (dim_idx == c_dim_idx) {
                    remain_c = src_dims[c_dim_idx] - i * c_blk;
                }
                reduce_nbcx_recursive_common<eT, c_blk, op_type>(
                    src_dims,
                    dst_dims,
                    src + i * inc_src[dim_idx],
                    dim_idx + 1,
                    inc_src,
                    inc_dst,
                    pc,
                    c_dim_idx,
                    remain_c,
                    dst + i * inc_dst[dim_idx]);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, int32_t c_blk, reduce_op_type_t op_type>
static ppl::common::RetCode reduce_nbcx_common(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const eT* src,
    const int32_t* axes,
    const int32_t num_axes,
    eT* dst)
{
    const int64_t c_dim_idx = 1;

    // pad 1 to dst shape to keepdims
    ppl::common::TensorShape padded_dst_shape = *src_shape;
    for (int64_t i = 0; i < num_axes; i++) {
        padded_dst_shape.SetDim(axes[i], 1);
    }
    const int64_t dim_count = src_shape->GetDimCount();
    std::vector<int64_t> src_dims(src_shape->GetDims(), src_shape->GetDims() + dim_count);
    std::vector<int64_t> padded_dst_dims(padded_dst_shape.GetDims(), padded_dst_shape.GetDims() + dim_count);

    // preprocess output data
    reduce_preprocess_common<eT, op_type>(dst, dst_shape->CalcElementsIncludingPadding());

    // compress dims
    std::vector<int64_t> compressed_src_dims;
    std::vector<int64_t> compressed_dst_dims;
    reduce_compress_dims(src_dims, padded_dst_dims, axes, num_axes, compressed_src_dims, compressed_dst_dims, c_dim_idx);
    const int64_t compressed_dim_count = compressed_src_dims.size();

    // prepare incs
    std::vector<int64_t> inc_src(compressed_dim_count);
    std::vector<int64_t> inc_dst(compressed_dim_count);
    reduce_prepare_incs<c_blk>(compressed_src_dims, compressed_dst_dims, inc_src, inc_dst, c_dim_idx);

    // calc parallel config
    const float omp_div_task_time_ratio = 10.0f; // assume omp create thread may be 10x slower than one element reduce process
    std::vector<uint8_t> forbid_mask(compressed_dim_count, 0);
    for (int64_t i = 0; i < num_axes; i++) { // reduce dims cannot parallel
        forbid_mask[axes[i]] = 1;
    }
    forbid_mask[compressed_dim_count - 1]     = 1; // last dim will not parallel
    single_parallel_loop_config_t loop_config = select_single_parallel_loop(compressed_src_dims, omp_div_task_time_ratio, forbid_mask);

    reduce_nbcx_recursive_common<eT, c_blk, op_type>(
        compressed_src_dims,
        compressed_dst_dims,
        src,
        0,
        inc_src,
        inc_dst,
        &loop_config,
        c_dim_idx,
        src_dims[c_dim_idx],
        dst);

    const int64_t reduce_factor = src_shape->CalcElementsExcludingPadding() / dst_shape->CalcElementsExcludingPadding();
    reduce_postprocess_common<eT, op_type>(dst, dst_shape->CalcElementsIncludingPadding(), reduce_factor);

    return pad_channel_zero(&padded_dst_shape, dst);
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_REDUCE_NEON_REDUCE_NBCX_COMMON_H_