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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_REDUCE_NEON_REDUCE_NDARRAY_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_REDUCE_NEON_REDUCE_NDARRAY_COMMON_H_

#include <vector>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/common/threading_tools.h"
#include "ppl/kernel/arm_server/reduce/neon/reduce_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, reduce_op_type_t op_type>
inline void reduce_ndarray_lastdim_no_reduce_common(
    const eT* src,
    const int64_t length,
    eT* dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        const vecType v_src_0 = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i + simd_w * 0));
        const vecType v_src_1 = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i + simd_w * 1));
        const vecType v_src_2 = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i + simd_w * 2));
        const vecType v_src_3 = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i + simd_w * 3));

        const vecType v_dst_0 = vld<eT, eN>(dst + i + simd_w * 0);
        const vecType v_dst_1 = vld<eT, eN>(dst + i + simd_w * 1);
        const vecType v_dst_2 = vld<eT, eN>(dst + i + simd_w * 2);
        const vecType v_dst_3 = vld<eT, eN>(dst + i + simd_w * 3);

        vst<eT, eN>(dst + i + simd_w * 0, reduce_vector_kernel<vecType, op_type>(v_src_0, v_dst_0));
        vst<eT, eN>(dst + i + simd_w * 1, reduce_vector_kernel<vecType, op_type>(v_src_1, v_dst_1));
        vst<eT, eN>(dst + i + simd_w * 2, reduce_vector_kernel<vecType, op_type>(v_src_2, v_dst_2));
        vst<eT, eN>(dst + i + simd_w * 3, reduce_vector_kernel<vecType, op_type>(v_src_3, v_dst_3));
    }
    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = reduce_scalar_kernel<eT, op_type>(reduce_first_process_kernel<eT, op_type>(src[i]), dst[i]);
    }
}

template <typename eT, reduce_op_type_t op_type>
inline void reduce_ndarray_lastdim_reduce_common(
    const eT* src,
    const int64_t length,
    eT* dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    eT s_reduced        = reduce_init_val<eT, op_type>();
    vecType v_reduced_0 = vdup_n<eT, eN>(s_reduced);
    vecType v_reduced_1 = v_reduced_0;
    vecType v_reduced_2 = v_reduced_0;
    vecType v_reduced_3 = v_reduced_0;

    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        const vecType v_src_0 = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i + simd_w * 0));
        const vecType v_src_1 = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i + simd_w * 1));
        const vecType v_src_2 = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i + simd_w * 2));
        const vecType v_src_3 = reduce_first_process_kernel<vecType, op_type>(vld<eT, eN>(src + i + simd_w * 3));

        v_reduced_0 = reduce_vector_kernel<vecType, op_type>(v_src_0, v_reduced_0);
        v_reduced_1 = reduce_vector_kernel<vecType, op_type>(v_src_1, v_reduced_1);
        v_reduced_2 = reduce_vector_kernel<vecType, op_type>(v_src_2, v_reduced_2);
        v_reduced_3 = reduce_vector_kernel<vecType, op_type>(v_src_3, v_reduced_3);
    }
    for (int64_t i = unroll_body; i < length; i++) {
        s_reduced = reduce_scalar_kernel<eT, op_type>(reduce_first_process_kernel<eT, op_type>(src[i]), s_reduced);
    }

    if (length >= unroll_len) {
        v_reduced_0 = reduce_vector_kernel<vecType, op_type>(v_reduced_2, v_reduced_0);
        v_reduced_1 = reduce_vector_kernel<vecType, op_type>(v_reduced_3, v_reduced_1);
        v_reduced_0 = reduce_vector_kernel<vecType, op_type>(v_reduced_1, v_reduced_0);
        s_reduced   = reduce_vector_to_scalar_kernel<eT, vecType, op_type>(v_reduced_0, s_reduced);
    }
    dst[0] = reduce_scalar_kernel<eT, op_type>(s_reduced, dst[0]);
}

template <typename eT, reduce_op_type_t op_type>
static ppl::common::RetCode reduce_ndarray_recursive_common(
    const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& dst_dims,
    const eT* src,
    const int64_t dim_idx,
    const std::vector<int64_t>& inc_src,
    const std::vector<int64_t>& inc_dst,
    const single_parallel_loop_config_t* pc,
    eT* dst)
{
    const int64_t length = src_dims[dim_idx];
    if (dim_idx == (int64_t)src_dims.size() - 1) { // last dim
        if (src_dims[dim_idx] == dst_dims[dim_idx]) {
            reduce_ndarray_lastdim_no_reduce_common<eT, op_type>(src, length, dst);
        } else {
            reduce_ndarray_lastdim_reduce_common<eT, op_type>(src, length, dst);
        }
    } else {
        if (pc->depth_of_loop == dim_idx && pc->num_threads > 1) { // parallel on this dim
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t t = 0; t < pc->num_threads; t++) {
                const int64_t length_per_thread = div_up(length, pc->num_threads);
                const int64_t start_idx         = t * length_per_thread;
                const int64_t end_idx           = min(start_idx + length_per_thread, length);
                for (int64_t i = start_idx; i < end_idx; i++) {
                    reduce_ndarray_recursive_common<eT, op_type>(
                        src_dims,
                        dst_dims,
                        src + i * inc_src[dim_idx],
                        dim_idx + 1,
                        inc_src,
                        inc_dst,
                        pc,
                        dst + i * inc_dst[dim_idx]);
                }
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                reduce_ndarray_recursive_common<eT, op_type>(
                    src_dims,
                    dst_dims,
                    src + i * inc_src[dim_idx],
                    dim_idx + 1,
                    inc_src,
                    inc_dst,
                    pc,
                    dst + i * inc_dst[dim_idx]);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <typename eT, reduce_op_type_t op_type>
static ppl::common::RetCode reduce_ndarray_common(
    const ppl::common::TensorShape* src_shape,
    const ppl::common::TensorShape* dst_shape,
    const eT* src,
    const int32_t* axes,
    const int32_t num_axes,
    eT* dst)
{
    // pad 1 to dst shape to keepdims
    const int64_t dim_count = src_shape->GetDimCount();
    std::vector<int64_t> src_dims(src_shape->GetDims(), src_shape->GetDims() + dim_count);
    std::vector<int64_t> padded_dst_dims = src_dims;
    for (int64_t i = 0; i < num_axes; i++) {
        padded_dst_dims[axes[i]] = 1;
    }

    // preprocess output data
    reduce_preprocess_common<eT, op_type>(dst, dst_shape->CalcElementsIncludingPadding());

    // compress dims
    std::vector<int64_t> compressed_src_dims;
    std::vector<int64_t> compressed_dst_dims;
    reduce_compress_dims(src_dims, padded_dst_dims, axes, num_axes, compressed_src_dims, compressed_dst_dims);
    const int64_t compressed_dim_count = compressed_src_dims.size();

    // prepare incs
    std::vector<int64_t> inc_src(compressed_dim_count);
    std::vector<int64_t> inc_dst(compressed_dim_count);
    reduce_prepare_incs<1>(compressed_src_dims, compressed_dst_dims, inc_src, inc_dst);

    // calc parallel config
    const float omp_div_task_time_ratio = 10.0f; // assume omp create thread may be 10x slower than one element reduce process
    std::vector<uint8_t> forbid_mask(compressed_dim_count, 0);
    for (int64_t i = 0; i < num_axes; i++) { // reduce dims cannot parallel
        forbid_mask[axes[i]] = 1;
    }
    forbid_mask[compressed_dim_count - 1]     = 1; // last dim will not parallel
    single_parallel_loop_config_t loop_config = select_single_parallel_loop(compressed_src_dims, omp_div_task_time_ratio, forbid_mask);

    reduce_ndarray_recursive_common<eT, op_type>(
        compressed_src_dims,
        compressed_dst_dims,
        src,
        0,
        inc_src,
        inc_dst,
        &loop_config,
        dst);

    const int64_t reduce_factor = src_shape->CalcElementsExcludingPadding() / dst_shape->CalcElementsExcludingPadding();
    reduce_postprocess_common<eT, op_type>(dst, dst_shape->CalcElementsIncludingPadding(), reduce_factor);

    return ppl::common::RC_SUCCESS;
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_REDUCE_NEON_REDUCE_NDARRAY_COMMON_H_