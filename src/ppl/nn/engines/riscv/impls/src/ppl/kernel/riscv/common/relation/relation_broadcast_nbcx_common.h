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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_BROADCAST_NBCX_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_BROADCAST_NBCX_COMMON_H_

#include "ppl/kernel/riscv/common/relation/relation_common.h"

namespace ppl { namespace kernel { namespace riscv {

template <relation_op_type_t op, typename T, int32_t vlen>
inline void relation_broadcast_nbcx_lastdim_no_broadcast_common(
    const T* src0,
    const T* src1,
    const int64_t length,
    const bool c0_broadcast,
    const bool c1_broadcast,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    uint64_t vl             = vsetvli<T, vlen>(c_blk);

    std::vector<uint_type<T>> lst(c_blk, 1);
    register_v<uint_type<T>, vlen> v_mask = vlev<uint_type<T>, vlen>(lst.data(), vl);

    if (!c0_broadcast && !c1_broadcast) {
        for (int64_t i = 0; i < length; i++) {
            register_v<T, vlen> v_src0 = vlev<T, vlen>(src0 + i * c_blk, vl);
            register_v<T, vlen> v_src1 = vlev<T, vlen>(src1 + i * c_blk, vl);
            register_ve<T, vlen> v_dst = vrelation_vv<op, T, vlen>(v_src0, v_src1, vl);
            pack_one<T, vlen>(v_dst, v_mask, dst + i * c_blk);
        }
    } else if (c0_broadcast) {
        for (int64_t i = 0; i < length; i++) {
            register_v<T, vlen> v_src0 = vmvvx<T, vlen>(src0[i * c_blk], vl);
            register_v<T, vlen> v_src1 = vlev<T, vlen>(src1 + i * c_blk, vl);
            register_ve<T, vlen> v_dst = vrelation_vv<op, T, vlen>(v_src0, v_src1, vl);
            pack_one<T, vlen>(v_dst, v_mask, dst + i * c_blk);
        }
    } else {
        for (int64_t i = 0; i < length; i++) {
            register_v<T, vlen> v_src0 = vlev<T, vlen>(src0 + i * c_blk, vl);
            register_v<T, vlen> v_src1 = vmvvx<T, vlen>(src1[i * c_blk], vl);
            register_ve<T, vlen> v_dst = vrelation_vv<op, T, vlen>(v_src0, v_src1, vl);
            pack_one<T, vlen>(v_dst, v_mask, dst + i * c_blk);
        }
    }
}

template <relation_op_type_t op, typename T, int32_t vlen>
inline void relation_broadcast_nbcx_lastdim_broadcast0_common(
    const T* src0,
    const T* src1,
    const int64_t length,
    const bool c0_broadcast,
    const bool c1_broadcast,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    uint64_t vl             = vsetvli<T, vlen>(c_blk);

    std::vector<uint_type<T>> lst(c_blk, 1);
    register_v<uint_type<T>, vlen> v_mask = vlev<uint_type<T>, vlen>(lst.data(), vl);

    if (!c0_broadcast && !c1_broadcast) {
        register_v<T, vlen> v_src0 = vlev<T, vlen>(src0, vl);
        for (int64_t i = 0; i < length; i++) {
            register_v<T, vlen> v_src1 = vlev<T, vlen>(src1 + i * c_blk, vl);
            register_ve<T, vlen> v_dst = vrelation_vv<op, T, vlen>(v_src0, v_src1, vl);
            pack_one<T, vlen>(v_dst, v_mask, dst + i * c_blk);
        }
    } else if (c0_broadcast) {
        register_v<T, vlen> v_src0 = vmvvx<T, vlen>(src0[0], vl);
        for (int64_t i = 0; i < length; i++) {
            register_v<T, vlen> v_src1 = vlev<T, vlen>(src1 + i * c_blk, vl);
            register_ve<T, vlen> v_dst = vrelation_vv<op, T, vlen>(v_src0, v_src1, vl);
            pack_one<T, vlen>(v_dst, v_mask, dst + i * c_blk);
        }
    } else {
        register_v<T, vlen> v_src0 = vlev<T, vlen>(src0, vl);
        for (int64_t i = 0; i < length; i++) {
            register_v<T, vlen> v_src1 = vmvvx<T, vlen>(src1[i * c_blk], vl);
            register_ve<T, vlen> v_dst = vrelation_vv<op, T, vlen>(v_src0, v_src1, vl);
            pack_one<T, vlen>(v_dst, v_mask, dst + i * c_blk);
        }
    }
}

template <relation_op_type_t op, typename T, int32_t vlen>
inline void relation_broadcast_nbcx_lastdim_broadcast1_common(
    const T* src0,
    const T* src1,
    const int64_t length,
    const bool c0_broadcast,
    const bool c1_broadcast,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    uint64_t vl             = vsetvli<T, vlen>(c_blk);

    std::vector<uint_type<T>> lst(c_blk, 1);
    register_v<uint_type<T>, vlen> v_mask = vlev<uint_type<T>, vlen>(lst.data(), vl);

    if (!c0_broadcast && !c1_broadcast) {
        register_v<T, vlen> v_src1 = vlev<T, vlen>(src1, vl);
        for (int64_t i = 0; i < length; i++) {
            register_v<T, vlen> v_src0 = vlev<T, vlen>(src0 + i * c_blk, vl);
            register_ve<T, vlen> v_dst = vrelation_vv<op, T, vlen>(v_src0, v_src1, vl);
            pack_one<T, vlen>(v_dst, v_mask, dst + i * c_blk);
        }
    } else if (c0_broadcast) {
        register_v<T, vlen> v_src1 = vlev<T, vlen>(src1, vl);
        for (int64_t i = 0; i < length; i++) {
            register_v<T, vlen> v_src0 = vmvvx<T, vlen>(src0[i * c_blk], vl);
            register_ve<T, vlen> v_dst = vrelation_vv<op, T, vlen>(v_src0, v_src1, vl);
            pack_one<T, vlen>(v_dst, v_mask, dst + i * c_blk);
        }
    } else {
        register_v<T, vlen> v_src1 = vmvvx<T, vlen>(src1[0], vl);
        for (int64_t i = 0; i < length; i++) {
            register_v<T, vlen> v_src0 = vlev<T, vlen>(src0 + i * c_blk, vl);
            register_ve<T, vlen> v_dst = vrelation_vv<op, T, vlen>(v_src0, v_src1, vl);
            pack_one<T, vlen>(v_dst, v_mask, dst + i * c_blk);
        }
    }
}

template <relation_op_type_t op, typename T, int32_t vlen>
static ppl::common::RetCode relation_broadcast_nbcx_recursive_common(
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
    const int64_t c_dim_idx,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    const int64_t length    = dim_idx == c_dim_idx ? div_up(dst_shape[dim_idx], c_blk) : dst_shape[dim_idx];
    if (dim_idx == dim_count - 1) {
        const bool c0_broadcast = src0_shape[c_dim_idx] != src1_shape[c_dim_idx] && src0_shape[c_dim_idx] == 1;
        const bool c1_broadcast = src0_shape[c_dim_idx] != src1_shape[c_dim_idx] && src1_shape[c_dim_idx] == 1;
        if (src0_shape[dim_idx] == src1_shape[dim_idx]) {
            relation_broadcast_nbcx_lastdim_no_broadcast_common<op, T, vlen>(src0, src1, length, c0_broadcast, c1_broadcast, dst);
        } else if (src0_shape[dim_idx] == 1) {
            relation_broadcast_nbcx_lastdim_broadcast0_common<op, T, vlen>(src0, src1, length, c0_broadcast, c1_broadcast, dst);
        } else {
            relation_broadcast_nbcx_lastdim_broadcast1_common<op, T, vlen>(src0, src1, length, c0_broadcast, c1_broadcast, dst);
        }
    } else {
        for (int64_t i = 0; i < length; i++) {
            relation_broadcast_nbcx_recursive_common<op, T, vlen>(
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
                c_dim_idx,
                dst + i * inc_out[dim_idx]);
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <relation_op_type_t op, typename T, int32_t vlen>
static ppl::common::RetCode relation_broadcast_nbcx_common(
    const ppl::common::TensorShape* src0_shape,
    const ppl::common::TensorShape* src1_shape,
    const ppl::common::TensorShape* dst_shape,
    const T* src0,
    const T* src1,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    const int64_t c_dim_idx = 1;

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
    int64_t stride0                       = c_blk;
    int64_t stride1                       = c_blk;
    int64_t stride_out                    = c_blk;
    for (int64_t i = compressed_dim_count - 1; i >= 0; i--) {
        inc0[i]    = compressed_src0_shape[i] == 1 ? 0 : stride0;
        inc1[i]    = compressed_src1_shape[i] == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= i == c_dim_idx ? div_up(compressed_src0_shape[i], c_blk) : compressed_src0_shape[i];
        stride1 *= i == c_dim_idx ? div_up(compressed_src1_shape[i], c_blk) : compressed_src1_shape[i];
        stride_out *= i == c_dim_idx ? div_up(compressed_dst_shape[i], c_blk) : compressed_dst_shape[i];
    }

    return relation_broadcast_nbcx_recursive_common<op, T, vlen>(
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
        c_dim_idx,
        dst);
}

}}}; // namespace ppl::kernel::riscv

#endif
