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
#ifndef __ST_PPL_KERNEL_ARM_SERVER_RELATION_NEON_RELATION_BROADCAST_NDARRAY_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_RELATION_NEON_RELATION_BROADCAST_NDARRAY_COMMON_H_

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/common/threading_tools.h"
#include "ppl/kernel/arm_server/relation/neon/relation_common.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

template <typename eT, relation_op_type_t op_type>
inline void relation_broadcast_ndarray_lastdim_no_broadcast_common(
    const eT *src0,
    const eT *src1,
    const int64_t length,
    const bool parallel,
    uint8_t *dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    if (parallel) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < unroll_body; i += unroll_len) {
            vecType v_src0_0 = vld<eT, eN>(src0 + i + simd_w * 0);
            vecType v_src0_1 = vld<eT, eN>(src0 + i + simd_w * 1);
            vecType v_src0_2 = vld<eT, eN>(src0 + i + simd_w * 2);
            vecType v_src0_3 = vld<eT, eN>(src0 + i + simd_w * 3);

            vecType v_src1_0 = vld<eT, eN>(src1 + i + simd_w * 0);
            vecType v_src1_1 = vld<eT, eN>(src1 + i + simd_w * 1);
            vecType v_src1_2 = vld<eT, eN>(src1 + i + simd_w * 2);
            vecType v_src1_3 = vld<eT, eN>(src1 + i + simd_w * 3);

            vecType v_dst_0 = relation_vector_kernel<vecType, op_type>(v_src0_0, v_src1_0);
            vecType v_dst_1 = relation_vector_kernel<vecType, op_type>(v_src0_1, v_src1_1);
            vecType v_dst_2 = relation_vector_kernel<vecType, op_type>(v_src0_2, v_src1_2);
            vecType v_dst_3 = relation_vector_kernel<vecType, op_type>(v_src0_3, v_src1_3);

            pack_four(v_dst_0, v_dst_1, v_dst_2, v_dst_3, dst + i);
        }
        for (int64_t i = unroll_body; i < length; i++) {
            dst[i] = relation_scalar_kernel<eT, op_type>(src0[i], src1[i]);
        }
    } else {
        for (int64_t i = 0; i < unroll_body; i += unroll_len) {
            vecType v_src0_0 = vld<eT, eN>(src0 + i + simd_w * 0);
            vecType v_src0_1 = vld<eT, eN>(src0 + i + simd_w * 1);
            vecType v_src0_2 = vld<eT, eN>(src0 + i + simd_w * 2);
            vecType v_src0_3 = vld<eT, eN>(src0 + i + simd_w * 3);

            vecType v_src1_0 = vld<eT, eN>(src1 + i + simd_w * 0);
            vecType v_src1_1 = vld<eT, eN>(src1 + i + simd_w * 1);
            vecType v_src1_2 = vld<eT, eN>(src1 + i + simd_w * 2);
            vecType v_src1_3 = vld<eT, eN>(src1 + i + simd_w * 3);

            vecType v_dst_0 = relation_vector_kernel<vecType, op_type>(v_src0_0, v_src1_0);
            vecType v_dst_1 = relation_vector_kernel<vecType, op_type>(v_src0_1, v_src1_1);
            vecType v_dst_2 = relation_vector_kernel<vecType, op_type>(v_src0_2, v_src1_2);
            vecType v_dst_3 = relation_vector_kernel<vecType, op_type>(v_src0_3, v_src1_3);

            pack_four(v_dst_0, v_dst_1, v_dst_2, v_dst_3, dst + i);
        }
        for (int64_t i = unroll_body; i < length; i++) {
            dst[i] = relation_scalar_kernel<eT, op_type>(src0[i], src1[i]);
        }
    }
}

template <typename eT, relation_op_type_t op_type>
inline void relation_broadcast_ndarray_lastdim_broadcast0_common(
    const eT *src0,
    const eT *src1,
    const int64_t length,
    const bool parallel,
    uint8_t *dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    const eT s_broadcast_val      = src0[0];
    const vecType v_broadcast_val = vdup_n<eT, eN>(s_broadcast_val);

    if (parallel) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < unroll_body; i += unroll_len) {
            vecType v_src1_0 = vld<eT, eN>(src1 + i + simd_w * 0);
            vecType v_src1_1 = vld<eT, eN>(src1 + i + simd_w * 1);
            vecType v_src1_2 = vld<eT, eN>(src1 + i + simd_w * 2);
            vecType v_src1_3 = vld<eT, eN>(src1 + i + simd_w * 3);

            vecType v_dst_0 = relation_vector_kernel<vecType, op_type>(v_broadcast_val, v_src1_0);
            vecType v_dst_1 = relation_vector_kernel<vecType, op_type>(v_broadcast_val, v_src1_1);
            vecType v_dst_2 = relation_vector_kernel<vecType, op_type>(v_broadcast_val, v_src1_2);
            vecType v_dst_3 = relation_vector_kernel<vecType, op_type>(v_broadcast_val, v_src1_3);

            pack_four(v_dst_0, v_dst_1, v_dst_2, v_dst_3, dst + i);
        }
        for (int64_t i = unroll_body; i < length; i++) {
            dst[i] = relation_scalar_kernel<eT, op_type>(s_broadcast_val, src1[i]);
        }
    } else {
        for (int64_t i = 0; i < unroll_body; i += unroll_len) {
            vecType v_src1_0 = vld<eT, eN>(src1 + i + simd_w * 0);
            vecType v_src1_1 = vld<eT, eN>(src1 + i + simd_w * 1);
            vecType v_src1_2 = vld<eT, eN>(src1 + i + simd_w * 2);
            vecType v_src1_3 = vld<eT, eN>(src1 + i + simd_w * 3);

            vecType v_dst_0 = relation_vector_kernel<vecType, op_type>(v_broadcast_val, v_src1_0);
            vecType v_dst_1 = relation_vector_kernel<vecType, op_type>(v_broadcast_val, v_src1_1);
            vecType v_dst_2 = relation_vector_kernel<vecType, op_type>(v_broadcast_val, v_src1_2);
            vecType v_dst_3 = relation_vector_kernel<vecType, op_type>(v_broadcast_val, v_src1_3);

            pack_four(v_dst_0, v_dst_1, v_dst_2, v_dst_3, dst + i);
        }
        for (int64_t i = unroll_body; i < length; i++) {
            dst[i] = relation_scalar_kernel<eT, op_type>(s_broadcast_val, src1[i]);
        }
    }
}

template <typename eT, relation_op_type_t op_type>
inline void relation_broadcast_ndarray_lastdim_broadcast1_common(
    const eT *src0,
    const eT *src1,
    const int64_t length,
    const bool parallel,
    uint8_t *dst)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    const eT s_broadcast_val      = src1[0];
    const vecType v_broadcast_val = vdup_n<eT, eN>(s_broadcast_val);

    if (parallel) {
        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < unroll_body; i += unroll_len) {
            vecType v_src0_0 = vld<eT, eN>(src0 + i + simd_w * 0);
            vecType v_src0_1 = vld<eT, eN>(src0 + i + simd_w * 1);
            vecType v_src0_2 = vld<eT, eN>(src0 + i + simd_w * 2);
            vecType v_src0_3 = vld<eT, eN>(src0 + i + simd_w * 3);

            vecType v_dst_0 = relation_vector_kernel<vecType, op_type>(v_src0_0, v_broadcast_val);
            vecType v_dst_1 = relation_vector_kernel<vecType, op_type>(v_src0_1, v_broadcast_val);
            vecType v_dst_2 = relation_vector_kernel<vecType, op_type>(v_src0_2, v_broadcast_val);
            vecType v_dst_3 = relation_vector_kernel<vecType, op_type>(v_src0_3, v_broadcast_val);

            pack_four(v_dst_0, v_dst_1, v_dst_2, v_dst_3, dst + i);
        }
        for (int64_t i = unroll_body; i < length; i++) {
            dst[i] = relation_scalar_kernel<eT, op_type>(src0[i], s_broadcast_val);
        }
    } else {
        for (int64_t i = 0; i < unroll_body; i += unroll_len) {
            vecType v_src0_0 = vld<eT, eN>(src0 + i + simd_w * 0);
            vecType v_src0_1 = vld<eT, eN>(src0 + i + simd_w * 1);
            vecType v_src0_2 = vld<eT, eN>(src0 + i + simd_w * 2);
            vecType v_src0_3 = vld<eT, eN>(src0 + i + simd_w * 3);

            vecType v_dst_0 = relation_vector_kernel<vecType, op_type>(v_src0_0, v_broadcast_val);
            vecType v_dst_1 = relation_vector_kernel<vecType, op_type>(v_src0_1, v_broadcast_val);
            vecType v_dst_2 = relation_vector_kernel<vecType, op_type>(v_src0_2, v_broadcast_val);
            vecType v_dst_3 = relation_vector_kernel<vecType, op_type>(v_src0_3, v_broadcast_val);

            pack_four(v_dst_0, v_dst_1, v_dst_2, v_dst_3, dst + i);
        }
        for (int64_t i = unroll_body; i < length; i++) {
            dst[i] = relation_scalar_kernel<eT, op_type>(src0[i], s_broadcast_val);
        }
    }
}

template <typename eT, relation_op_type_t op_type>
static ppl::common::RetCode relation_broadcast_ndarray_recursive_common(
    const int64_t *src0_shape,
    const int64_t *src1_shape,
    const int64_t *dst_shape,
    const eT *src0,
    const eT *src1,
    const int64_t *inc0,
    const int64_t *inc1,
    const int64_t *inc_out,
    const int64_t dim_count,
    const int64_t dim_idx,
    const single_parallel_loop_config_t *loop_config,
    uint8_t *dst)
{
    const int64_t length = dst_shape[dim_idx];

    if (dim_idx == dim_count - 1) { // last dim
        const bool lastdim_parallel = dim_idx == loop_config->depth_of_loop && loop_config->num_threads > 1;
        if (src0_shape[dim_idx] == src1_shape[dim_idx]) {
            relation_broadcast_ndarray_lastdim_no_broadcast_common<eT, op_type>(src0, src1, length, lastdim_parallel, dst);
        } else if (src0_shape[dim_idx] == 1) {
            relation_broadcast_ndarray_lastdim_broadcast0_common<eT, op_type>(src0, src1, length, lastdim_parallel, dst);
        } else {
            relation_broadcast_ndarray_lastdim_broadcast1_common<eT, op_type>(src0, src1, length, lastdim_parallel, dst);
        }
    } else {
        if (dim_idx == loop_config->depth_of_loop && loop_config->num_threads > 1) { // parallel on this dim
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t i = 0; i < length; i++) {
                relation_broadcast_ndarray_recursive_common<eT, op_type>(
                    src0_shape,
                    src1_shape,
                    dst_shape,
                    src0 + i * inc0[dim_idx],
                    src1 + i * inc1[dim_idx],
                    inc0,
                    inc1,
                    inc_out,
                    dim_count,
                    dim_idx + 1,
                    loop_config,
                    dst + i * inc_out[dim_idx]);
            }
        } else {
            for (int64_t i = 0; i < length; i++) {
                relation_broadcast_ndarray_recursive_common<eT, op_type>(
                    src0_shape,
                    src1_shape,
                    dst_shape,
                    src0 + i * inc0[dim_idx],
                    src1 + i * inc1[dim_idx],
                    inc0,
                    inc1,
                    inc_out,
                    dim_count,
                    dim_idx + 1,
                    loop_config,
                    dst + i * inc_out[dim_idx]);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

inline void relation_ndarray_prepare_incs(
    const int64_t *src0_shape,
    const int64_t *src1_shape,
    const int64_t *dst_shape,
    const int64_t dim_count,
    int64_t *inc0,
    int64_t *inc1,
    int64_t *inc_out)
{
    int64_t stride0    = 1;
    int64_t stride1    = 1;
    int64_t stride_out = 1;

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        inc0[i]    = src0_shape[i] == 1 ? 0 : stride0;
        inc1[i]    = src1_shape[i] == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= src0_shape[i];
        stride1 *= src1_shape[i];
        stride_out *= dst_shape[i];
    }
}

template <typename eT, relation_op_type_t op_type>
static ppl::common::RetCode relation_broadcast_ndarray_common(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const eT *src0,
    const eT *src1,
    uint8_t *dst)
{
    const int64_t max_dim_count              = dst_shape->GetDimCount();
    std::vector<int64_t> padded_src0_shape(max_dim_count);
    std::vector<int64_t> padded_src1_shape(max_dim_count);
    relation_pad_shape(src0_shape, max_dim_count, padded_src0_shape.data());
    relation_pad_shape(src1_shape, max_dim_count, padded_src1_shape.data());

    int64_t compressed_dim_count                 = 0;
    std::vector<int64_t> compressed_src0_shape(max_dim_count);
    std::vector<int64_t> compressed_src1_shape(max_dim_count);
    std::vector<int64_t> compressed_dst_shape(max_dim_count);
    relation_compress_shape(padded_src0_shape.data(), padded_src1_shape.data(), max_dim_count, &compressed_dim_count, compressed_src0_shape.data(), compressed_src1_shape.data(), compressed_dst_shape.data());

    std::vector<int64_t> inc0(compressed_dim_count);
    std::vector<int64_t> inc1(compressed_dim_count);
    std::vector<int64_t> inc_out(compressed_dim_count);
    relation_ndarray_prepare_incs(compressed_src0_shape.data(), compressed_src1_shape.data(), compressed_dst_shape.data(), compressed_dim_count, inc0.data(), inc1.data(), inc_out.data());

    const float omp_div_task_time_ratio       = 20.0f; // assume omp create thread may be 20x slower than one element arthimetic process
    single_parallel_loop_config_t loop_config = select_single_parallel_loop(compressed_dst_shape.data(), compressed_dim_count, omp_div_task_time_ratio);

    return relation_broadcast_ndarray_recursive_common<eT, op_type>(
        compressed_src0_shape.data(),
        compressed_src1_shape.data(),
        compressed_dst_shape.data(),
        src0,
        src1,
        inc0.data(),
        inc1.data(),
        inc_out.data(),
        compressed_dim_count,
        0,
        &loop_config,
        dst);
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_RELATION_NEON_RELATION_BROADCAST_NDARRAY_COMMON_H_
