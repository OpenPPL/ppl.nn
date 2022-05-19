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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_REDUCE_NEON_REDUCE_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_REDUCE_NEON_REDUCE_COMMON_H_

#include <vector>

#include "ppl/kernel/arm_server/common/internal_include.h"
#include "ppl/kernel/arm_server/common/math.h"
#include "ppl/kernel/arm_server/common/type_traits.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

enum reduce_op_type_t {
    REDUCE_MAX        = 0,
    REDUCE_MIN        = 1,
    REDUCE_SUM        = 2,
    REDUCE_MEAN       = 3,
    REDUCE_PROD       = 4,
    REDUCE_SUM_SQUARE = 5,
    REDUCE_ABS_SUM    = 6,
};

template <typename eT, reduce_op_type_t op_type>
inline eT reduce_scalar_kernel(const eT val, const eT reduced);

template <typename vT, reduce_op_type_t op_type>
inline vT reduce_vector_kernel(const vT val, const vT reduced);

template <typename eT, typename vT, reduce_op_type_t op_type>
inline eT reduce_vector_to_scalar_kernel(const vT val, const eT reduced);

template <typename eT, reduce_op_type_t op_type>
inline eT reduce_first_process_kernel(const eT val)
{
    return val;
}

// TODO: partial specialization this
template <typename eT, reduce_op_type_t op_type>
inline eT reduce_init_val(void)
{
    if (op_type == REDUCE_MAX) {
        return numeric_min<eT>();
    } else if (op_type == REDUCE_MIN) {
        return numeric_max<eT>();
    } else if (op_type == REDUCE_PROD) {
        return (eT)1;
    }
    return 0;
}

template <typename eT, reduce_op_type_t op_type>
static void reduce_preprocess_common(
    eT* dst,
    const int64_t len)
{
    constexpr int32_t eN = 128 / 8 / sizeof(eT);
    typedef typename DT<eT, eN>::vecDT vecType;

    const eT init_val        = reduce_init_val<eT, op_type>();
    const vecType v_init_val = vdup_n<eT, eN>(init_val);

    const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(len, unroll_len);

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        vst<eT, eN>(dst + i + simd_w * 0, v_init_val);
        vst<eT, eN>(dst + i + simd_w * 1, v_init_val);
        vst<eT, eN>(dst + i + simd_w * 2, v_init_val);
        vst<eT, eN>(dst + i + simd_w * 3, v_init_val);
    }
    for (int64_t i = unroll_body; i < len; i++) {
        dst[i] = init_val;
    }
}

template <typename eT, reduce_op_type_t op_type>
static void reduce_postprocess_common(
    eT* dst,
    const int64_t len,
    const int64_t reduce_factor)
{
    if (op_type == REDUCE_MEAN) {
        constexpr int32_t eN = 128 / 8 / sizeof(eT);
        typedef typename DT<eT, eN>::vecDT vecType;

        const float rdiv     = 1.0f / reduce_factor;
        const vecType v_rdiv = vdup_n<eT, eN>(rdiv);

        const int64_t simd_w      = sizeof(vecType) / sizeof(eT);
        const int64_t unroll_len  = simd_w * 4;
        const int64_t unroll_body = round(len, unroll_len);

        PRAGMA_OMP_PARALLEL_FOR()
        for (int64_t i = 0; i < unroll_body; i += unroll_len) {
            vecType v_dst_0 = vld<eT, eN>(dst + i + simd_w * 0);
            vecType v_dst_1 = vld<eT, eN>(dst + i + simd_w * 1);
            vecType v_dst_2 = vld<eT, eN>(dst + i + simd_w * 2);
            vecType v_dst_3 = vld<eT, eN>(dst + i + simd_w * 3);

            v_dst_0 = vmul<vecType>(v_dst_0, v_rdiv);
            v_dst_1 = vmul<vecType>(v_dst_1, v_rdiv);
            v_dst_2 = vmul<vecType>(v_dst_2, v_rdiv);
            v_dst_3 = vmul<vecType>(v_dst_3, v_rdiv);

            vst<eT, eN>(dst + i + simd_w * 0, v_dst_0);
            vst<eT, eN>(dst + i + simd_w * 1, v_dst_1);
            vst<eT, eN>(dst + i + simd_w * 2, v_dst_2);
            vst<eT, eN>(dst + i + simd_w * 3, v_dst_3);
        }
        for (int64_t i = unroll_body; i < len; i++) {
            dst[i] *= rdiv;
        }
    }
}

inline void reduce_compress_dims(
    const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& dst_dims,
    const int32_t* axes,
    const int32_t num_axes,
    std::vector<int64_t>& compressed_src_dims,
    std::vector<int64_t>& compressed_dst_dims,
    const int64_t c_dim_idx = -1)
{ // for nbcx dataformat, c_dim_idx should be set to disable compress on channel dim

    const int64_t dim_count = src_dims.size();
    std::vector<bool> dim_reduced(dim_count, false);
    for (int64_t i = 0; i < num_axes; i++) {
        dim_reduced[axes[i]] = true;
    }

    compressed_src_dims.resize(dim_count);
    compressed_dst_dims.resize(dim_count);

    int64_t compressed_dim_idx = 0;
    compressed_src_dims[0]     = src_dims[0];
    compressed_dst_dims[0]     = dst_dims[0];

    for (int64_t i = 1; i < dim_count; i++) {
        if (i == c_dim_idx) { // for nbcx dataformat, channel dim cannot be compressed
            compressed_dim_idx++; // flush before
            compressed_src_dims[compressed_dim_idx] = src_dims[i];
            compressed_dst_dims[compressed_dim_idx] = dst_dims[i];

            compressed_dim_idx++; // move to next
            i++;
            compressed_src_dims[compressed_dim_idx] = src_dims[i];
            compressed_dst_dims[compressed_dim_idx] = dst_dims[i];

            continue;
        }

        if (dim_reduced[i] == dim_reduced[i - 1]) {
            compressed_src_dims[compressed_dim_idx] *= src_dims[i];
            compressed_dst_dims[compressed_dim_idx] *= dst_dims[i];
        } else {
            compressed_dim_idx++;
            compressed_src_dims[compressed_dim_idx] = src_dims[i];
            compressed_dst_dims[compressed_dim_idx] = dst_dims[i];
        }
    }

    const int64_t compressed_dim_count = compressed_dim_idx + 1;
    compressed_src_dims.resize(compressed_dim_count);
    compressed_dst_dims.resize(compressed_dim_count);
}

template <int32_t c_blk>
inline void reduce_prepare_incs(
    const std::vector<int64_t>& src_dims,
    const std::vector<int64_t>& dst_dims,
    std::vector<int64_t>& inc_src,
    std::vector<int64_t>& inc_dst,
    const int64_t c_dim_idx = -1)
{
    const int64_t dim_count = src_dims.size();
    int64_t src_stride      = c_blk;
    int64_t dst_stride      = c_blk;

    for (int64_t i = dim_count - 1; i >= 0; i--) {
        inc_src[i] = src_dims[i] == 1 ? 0 : src_stride;
        inc_dst[i] = dst_dims[i] == 1 ? 0 : dst_stride;

        src_stride *= i == c_dim_idx ? div_up(src_dims[i], c_blk) : src_dims[i];
        dst_stride *= i == c_dim_idx ? div_up(dst_dims[i], c_blk) : dst_dims[i];
    }
}

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_REDUCE_NEON_REDUCE_COMMON_H_