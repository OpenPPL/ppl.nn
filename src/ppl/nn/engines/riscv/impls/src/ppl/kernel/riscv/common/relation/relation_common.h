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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_RELATION_RELATION_COMMON_H_

#include "ppl/kernel/riscv/common/internal_include.h"
#include "ppl/kernel/riscv/common/relation/relation_kernel.h"

namespace ppl { namespace kernel { namespace riscv {

// enum relation_op_type_t {
//     RELATION_GREATER          = 0,
//     RELATION_GREATER_OR_EQUAL = 1,
//     RELATION_LESS             = 2,
//     RELATION_LESS_OR_EQUAL    = 3,
//     RELATION_EQUAL            = 4,
//     RELATION_NOT_EQUAL        = 5
// };

template <typename T, int32_t vlen>
static void pack_four(
    register_ve<T, vlen>& v0,
    register_ve<T, vlen>& v1,
    register_ve<T, vlen>& v2,
    register_ve<T, vlen>& v3,
    register_v<uint_type<T>, vlen>& vmask,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    uint64_t vl             = vsetvli<T, vlen>(c_blk);

    uint8_t* tmp_dst = dst;

    uint_type<T> tmp[c_blk * 4];
    vsev_mask<T, vlen>((tmp + c_blk * 0), vmask, v0, vl);
    vsev_mask<T, vlen>((tmp + c_blk * 1), vmask, v1, vl);
    vsev_mask<T, vlen>((tmp + c_blk * 2), vmask, v2, vl);
    vsev_mask<T, vlen>((tmp + c_blk * 3), vmask, v3, vl);

    for (int32_t i = 0; i < c_blk * 4; i++) {
        tmp_dst[i] = tmp[i] & 1;
    }
}

template <typename T, int32_t vlen>
static void pack_one(
    register_ve<T, vlen>& v0,
    register_v<uint_type<T>, vlen>& vmask,
    uint8_t* dst)
{
    constexpr int32_t c_blk = vlen / 8 / sizeof(T);
    uint64_t vl             = vsetvli<T, vlen>(c_blk);

    uint8_t* tmp_dst = dst;

    uint_type<T> tmp[c_blk];
    vsev_mask<T, vlen>(tmp, vmask, v0, vl);
    for (int32_t i = 0; i < c_blk; i++) {
        tmp_dst[i] = tmp[i] & 1;
    }
}

inline void pad_shape(
    const ppl::common::TensorShape* shape,
    const int64_t padded_dim_count,
    int64_t* padded_shape)
{
    const int64_t dim_diff = padded_dim_count - shape->GetRealDimCount();
    for (int64_t i = 0; i < dim_diff; i++) {
        padded_shape[i] = 1;
    }
    for (int64_t i = dim_diff; i < padded_dim_count; i++) {
        padded_shape[i] = shape->GetDim(i - dim_diff);
    }
}

inline void compress_shape(
    const int64_t* src0_shape,
    const int64_t* src1_shape,
    const int64_t dim_count,
    int64_t* compressed_dim_count,
    int64_t* compressed_src0_shape,
    int64_t* compressed_src1_shape,
    int64_t* compressed_dst_shape,
    const int64_t c_dim_idx = -1)
{
    bool src0_broadcast[dim_count];
    bool src1_broadcast[dim_count];
    for (int64_t i = 0; i < dim_count; i++) {
        src0_broadcast[i] = src0_shape[i] != src1_shape[i] && src0_shape[i] == 1;
        src1_broadcast[i] = src0_shape[i] != src1_shape[i] && src1_shape[i] == 1;
    }

    int64_t compressed_dim_idx = 0;
    compressed_src0_shape[0]   = src0_shape[0];
    compressed_src1_shape[0]   = src1_shape[0];

    for (int64_t i = 1; i < dim_count; i++) {
        if (i == c_dim_idx) {
            compressed_dim_idx++;
            compressed_src0_shape[compressed_dim_idx] = src0_shape[i];
            compressed_src1_shape[compressed_dim_idx] = src1_shape[i];

            compressed_dim_idx++;
            i++;
            compressed_src0_shape[compressed_dim_idx] = src0_shape[i];
            compressed_src1_shape[compressed_dim_idx] = src1_shape[i];

            continue;
        }

        if (src0_broadcast[i] == src0_broadcast[compressed_dim_idx] && src1_broadcast[i] == src1_broadcast[compressed_dim_idx]) {
            compressed_src0_shape[compressed_dim_idx] *= src0_shape[i];
            compressed_src1_shape[compressed_dim_idx] *= src0_shape[i];
        } else {
            compressed_dim_idx++;
            compressed_src0_shape[compressed_dim_idx] = src0_shape[i];
            compressed_src1_shape[compressed_dim_idx] = src1_shape[i];
        }
    }

    *compressed_dim_count = compressed_dim_idx + 1;
    for (int64_t i = 0; i < *compressed_dim_count; i++) {
        compressed_dst_shape[i] = max(compressed_src0_shape[i], compressed_src1_shape[i]);
    }
}

}}}; // namespace ppl::kernel::riscv

#endif
