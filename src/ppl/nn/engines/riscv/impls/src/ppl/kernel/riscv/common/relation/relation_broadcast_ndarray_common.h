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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_BROADCAST_NDARRAY_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_BROADCAST_NDARRAY_COMMON_H_

#include <vector>
#include "ppl/kernel/riscv/common/relation/relation_common.h"

namespace ppl { namespace kernel { namespace riscv {

template <relation_op_type_t op, typename T, int32_t vlen>
inline void relation_broadcast_ndarray_lastdim_no_broadcast_scalar_common(
    const T* src0,
    const T* src1,
    const int64_t length,
    uint8_t* dst)
{
    for (int64_t i = 0; i < length; i++) {
        dst[i] = relation_scalar_kernel<op, T>(src0[i], src1[i]);
    }
}

template <relation_op_type_t op, typename T, int32_t vlen>
inline void relation_broadcast_ndarray_lastdim_broadcast0_scalar_common(
    const T* src0,
    const T* src1,
    const int64_t length,
    uint8_t* dst)
{
    const T broadcast_val = src0[0];
    for (int64_t i = 0; i < length; i++) {
        dst[i] = relation_scalar_kernel<op, T>(broadcast_val, src1[i]);
    }
}

template <relation_op_type_t op, typename T, int32_t vlen>
inline void relation_broadcast_ndarray_lastdim_broadcast1_scalar_common(
    const T* src0,
    const T* src1,
    const int64_t length,
    uint8_t* dst)
{
    const T broadcast_val = src1[0];
    for (int64_t i = 0; i < length; i++) {
        dst[i] = relation_scalar_kernel<op, T>(src0[i], broadcast_val);
    }
}

template <relation_op_type_t op, typename T, int32_t vlen>
static ppl::common::RetCode relation_broadcast_ndarray_recursive_scalar_common(
    const int64_t* src0_shape,
    const int64_t* src1_shape,
    const int64_t* dst_shape,
    const T* src0,
    const T* src1,
    const int64_t* inc0,
    const int64_t* inc1,
    const int64_t* inc_out,
    const int64_t dim_count,
    const int64_t dim_idx,
    uint8_t* dst)
{
    const int64_t length = dst_shape[dim_idx];
    if (dim_idx == dim_count - 1) {
        if (src0_shape[dim_idx] == src1_shape[dim_idx]) {
            relation_broadcast_ndarray_lastdim_no_broadcast_scalar_common<op, T, vlen>(src0, src1, length, dst);
        } else if (src0_shape[dim_idx] == 1) {
            relation_broadcast_ndarray_lastdim_broadcast0_scalar_common<op, T, vlen>(src0, src1, length, dst);
        } else if (src1_shape[dim_idx] == 1) {
            relation_broadcast_ndarray_lastdim_broadcast1_scalar_common<op, T, vlen>(src0, src1, length, dst);
        }
    } else {
        for (int64_t i = 0; i < length; i++) {
            relation_broadcast_ndarray_recursive_scalar_common<op, T, vlen>(
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
                dst + i * inc_out[dim_idx]);
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <relation_op_type_t op, typename T, int32_t vlen>
static ppl::common::RetCode relation_broadcast_ndarray_scalar_common(
    const ppl::common::TensorShape* src0_shape,
    const ppl::common::TensorShape* src1_shape,
    const ppl::common::TensorShape* dst_shape,
    const T* src0,
    const T* src1,
    uint8_t* dst)
{
    const int64_t max_dim_count              = dst_shape->GetDimCount();
    int64_t padded_src0_shape[max_dim_count] = {0};
    int64_t padded_src1_shape[max_dim_count] = {0};
    pad_shape(src0_shape, max_dim_count, padded_src0_shape);
    pad_shape(src1_shape, max_dim_count, padded_src1_shape);

    int64_t compressed_dim_count                 = 0;
    int64_t compressed_src0_shape[max_dim_count] = {0};
    int64_t compressed_src1_shape[max_dim_count] = {0};
    int64_t compressed_dst_shape[max_dim_count]  = {0};
    compress_shape(
        padded_src0_shape,
        padded_src1_shape,
        max_dim_count,
        &compressed_dim_count,
        compressed_src0_shape,
        compressed_src1_shape,
        compressed_dst_shape);

    int64_t inc0[compressed_dim_count]    = {0};
    int64_t inc1[compressed_dim_count]    = {0};
    int64_t inc_out[compressed_dim_count] = {0};
    int64_t stride0                       = 1;
    int64_t stride1                       = 1;
    int64_t stride_out                    = 1;
    for (int64_t i = compressed_dim_count - 1; i >= 0; i--) {
        inc0[i]    = compressed_src0_shape[i] == 1 ? 0 : stride0;
        inc1[i]    = compressed_src1_shape[i] == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= compressed_src0_shape[i];
        stride1 *= compressed_src1_shape[i];
        stride_out *= compressed_dst_shape[i];
    }

    return relation_broadcast_ndarray_recursive_scalar_common<op, T, vlen>(
        compressed_src0_shape,
        compressed_src1_shape,
        compressed_dst_shape,
        src0,
        src1,
        inc0,
        inc1,
        inc_out,
        compressed_dim_count,
        0,
        dst);
}

template <relation_op_type_t op, typename T, int32_t vlen>
inline void relation_broadcast_ndarray_lastdim_no_broadcast_common(
    const T* src0,
    const T* src1,
    const int64_t length,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    uint64_t vl             = vsetvli<T, vlen>(c_blk);

    const int64_t simd_w      = c_blk;
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    std::vector<uint_type<T>> lst(c_blk, 1);
    register_v<uint_type<T>, vlen> v_mask = vlev<uint_type<T>, vlen>(lst.data(), vl);

    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        register_v<T, vlen> v_src0_0 = vlev<T, vlen>(src0 + i + simd_w * 0, vl);
        register_v<T, vlen> v_src0_1 = vlev<T, vlen>(src0 + i + simd_w * 1, vl);
        register_v<T, vlen> v_src0_2 = vlev<T, vlen>(src0 + i + simd_w * 2, vl);
        register_v<T, vlen> v_src0_3 = vlev<T, vlen>(src0 + i + simd_w * 3, vl);

        register_v<T, vlen> v_src1_0 = vlev<T, vlen>(src1 + i + simd_w * 0, vl);
        register_v<T, vlen> v_src1_1 = vlev<T, vlen>(src1 + i + simd_w * 1, vl);
        register_v<T, vlen> v_src1_2 = vlev<T, vlen>(src1 + i + simd_w * 2, vl);
        register_v<T, vlen> v_src1_3 = vlev<T, vlen>(src1 + i + simd_w * 3, vl);

        register_ve<T, vlen> v_dst_0 = vrelation_vv<op, T, vlen>(v_src0_0, v_src1_0, vl);
        register_ve<T, vlen> v_dst_1 = vrelation_vv<op, T, vlen>(v_src0_1, v_src1_1, vl);
        register_ve<T, vlen> v_dst_2 = vrelation_vv<op, T, vlen>(v_src0_2, v_src1_2, vl);
        register_ve<T, vlen> v_dst_3 = vrelation_vv<op, T, vlen>(v_src0_3, v_src1_3, vl);

        pack_four<T, vlen>(v_dst_0, v_dst_1, v_dst_2, v_dst_3, v_mask, dst + i);
    }

    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = relation_scalar_kernel<op, T>(src0[i], src1[i]);
    }
}

template <relation_op_type_t op, typename T, int32_t vlen>
inline void relation_broadcast_ndarray_lastdim_broadcast0_common(
    const T* src0,
    const T* src1,
    const int64_t length,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    uint64_t vl             = vsetvli<T, vlen>(c_blk);

    const int64_t simd_w      = c_blk;
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    const T broadcast_val               = src0[0];
    register_v<T, vlen> v_broadcast_val = vmvvx<T, vlen>(broadcast_val, vl);
    std::vector<uint_type<T>> lst(c_blk, 1);
    register_v<uint_type<T>, vlen> v_mask = vlev<uint_type<T>, vlen>(lst.data(), vl);

    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        register_v<T, vlen> v_src1_0 = vlev<T, vlen>(src1 + i + simd_w * 0, vl);
        register_v<T, vlen> v_src1_1 = vlev<T, vlen>(src1 + i + simd_w * 1, vl);
        register_v<T, vlen> v_src1_2 = vlev<T, vlen>(src1 + i + simd_w * 2, vl);
        register_v<T, vlen> v_src1_3 = vlev<T, vlen>(src1 + i + simd_w * 3, vl);

        register_ve<T, vlen> v_dst_0 = vrelation_vv<op, T, vlen>(v_broadcast_val, v_src1_0, vl);
        register_ve<T, vlen> v_dst_1 = vrelation_vv<op, T, vlen>(v_broadcast_val, v_src1_1, vl);
        register_ve<T, vlen> v_dst_2 = vrelation_vv<op, T, vlen>(v_broadcast_val, v_src1_2, vl);
        register_ve<T, vlen> v_dst_3 = vrelation_vv<op, T, vlen>(v_broadcast_val, v_src1_3, vl);

        pack_four<T, vlen>(v_dst_0, v_dst_1, v_dst_2, v_dst_3, v_mask, dst + i);
    }
    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = relation_scalar_kernel<op, T>(broadcast_val, src1[i]);
    }
}

template <relation_op_type_t op, typename T, int32_t vlen>
inline void relation_broadcast_ndarray_lastdim_broadcast1_common(
    const T* src0,
    const T* src1,
    const int64_t length,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    uint64_t vl             = vsetvli<T, vlen>(c_blk);

    const int64_t simd_w      = c_blk;
    const int64_t unroll_len  = simd_w * 4;
    const int64_t unroll_body = round(length, unroll_len);

    const T broadcast_val               = src1[0];
    register_v<T, vlen> v_broadcast_val = vmvvx<T, vlen>(broadcast_val, vl);
    std::vector<uint_type<T>> lst(c_blk, 1);
    register_v<uint_type<T>, vlen> v_mask = vlev<uint_type<T>, vlen>(lst.data(), vl);

    for (int64_t i = 0; i < unroll_body; i += unroll_len) {
        register_v<T, vlen> v_src0_0 = vlev<T, vlen>(src0 + i + simd_w * 0, vl);
        register_v<T, vlen> v_src0_1 = vlev<T, vlen>(src0 + i + simd_w * 1, vl);
        register_v<T, vlen> v_src0_2 = vlev<T, vlen>(src0 + i + simd_w * 2, vl);
        register_v<T, vlen> v_src0_3 = vlev<T, vlen>(src0 + i + simd_w * 3, vl);

        register_ve<T, vlen> v_dst_0 = vrelation_vv<op, T, vlen>(v_src0_0, v_broadcast_val, vl);
        register_ve<T, vlen> v_dst_1 = vrelation_vv<op, T, vlen>(v_src0_1, v_broadcast_val, vl);
        register_ve<T, vlen> v_dst_2 = vrelation_vv<op, T, vlen>(v_src0_2, v_broadcast_val, vl);
        register_ve<T, vlen> v_dst_3 = vrelation_vv<op, T, vlen>(v_src0_3, v_broadcast_val, vl);

        pack_four<T, vlen>(v_dst_0, v_dst_1, v_dst_2, v_dst_3, v_mask, dst + i);
    }
    for (int64_t i = unroll_body; i < length; i++) {
        dst[i] = relation_scalar_kernel<op, T>(src0[i], broadcast_val);
    }
}

template <relation_op_type_t op, typename T, int32_t vlen>
static ppl::common::RetCode relation_broadcast_ndarray_recursive_common(
    const int64_t* src0_shape,
    const int64_t* src1_shape,
    const int64_t* dst_shape,
    const T* src0,
    const T* src1,
    const int64_t* inc0,
    const int64_t* inc1,
    const int64_t* inc_out,
    const int64_t dim_count,
    const int64_t dim_idx,
    uint8_t* dst)
{
    const int64_t length = dst_shape[dim_idx];
    if (dim_idx == dim_count - 1) {
        if (src0_shape[dim_idx] == src1_shape[dim_idx]) {
            relation_broadcast_ndarray_lastdim_no_broadcast_common<op, T, vlen>(src0, src1, length, dst);
        } else if (src0_shape[dim_idx] == 1) {
            relation_broadcast_ndarray_lastdim_broadcast0_common<op, T, vlen>(src0, src1, length, dst);
        } else if (src1_shape[dim_idx] == 1) {
            relation_broadcast_ndarray_lastdim_broadcast1_common<op, T, vlen>(src0, src1, length, dst);
        }
    } else {
        for (int64_t i = 0; i < length; i++) {
            relation_broadcast_ndarray_recursive_common<op, T, vlen>(
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
                dst + i * inc_out[dim_idx]);
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <relation_op_type_t op, typename T, int32_t vlen>
static ppl::common::RetCode relation_broadcast_ndarray_common(
    const ppl::common::TensorShape* src0_shape,
    const ppl::common::TensorShape* src1_shape,
    const ppl::common::TensorShape* dst_shape,
    const T* src0,
    const T* src1,
    uint8_t* dst)
{
    const int64_t max_dim_count              = dst_shape->GetDimCount();
    int64_t padded_src0_shape[max_dim_count] = {0};
    int64_t padded_src1_shape[max_dim_count] = {0};
    pad_shape(src0_shape, max_dim_count, padded_src0_shape);
    pad_shape(src1_shape, max_dim_count, padded_src1_shape);

    int64_t compressed_dim_count                 = 0;
    int64_t compressed_src0_shape[max_dim_count] = {0};
    int64_t compressed_src1_shape[max_dim_count] = {0};
    int64_t compressed_dst_shape[max_dim_count]  = {0};
    compress_shape(
        padded_src0_shape,
        padded_src1_shape,
        max_dim_count,
        &compressed_dim_count,
        compressed_src0_shape,
        compressed_src1_shape,
        compressed_dst_shape);

    int64_t inc0[compressed_dim_count]    = {0};
    int64_t inc1[compressed_dim_count]    = {0};
    int64_t inc_out[compressed_dim_count] = {0};
    int64_t stride0                       = 1;
    int64_t stride1                       = 1;
    int64_t stride_out                    = 1;
    for (int64_t i = compressed_dim_count - 1; i >= 0; i--) {
        inc0[i]    = compressed_src0_shape[i] == 1 ? 0 : stride0;
        inc1[i]    = compressed_src1_shape[i] == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= compressed_src0_shape[i];
        stride1 *= compressed_src1_shape[i];
        stride_out *= compressed_dst_shape[i];
    }

    return relation_broadcast_ndarray_recursive_common<op, T, vlen>(
        compressed_src0_shape,
        compressed_src1_shape,
        compressed_dst_shape,
        src0,
        src1,
        inc0,
        inc1,
        inc_out,
        compressed_dim_count,
        0,
        dst);
}

}}}; // namespace ppl::kernel::riscv

#endif
