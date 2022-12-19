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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_COMMON_H_

#include "ppl/kernel/arm_server/common/internal_include.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

enum arithmetic_op_type_t {
    ARITHMETIC_ADD = 0,
    ARITHMETIC_SUB = 1,
    ARITHMETIC_MUL = 2,
    ARITHMETIC_DIV = 3,
    ARITHMETIC_POW = 4
};

template <typename eT, arithmetic_op_type_t op_type>
inline eT arithmetic_scalar_kernel(const eT s0, const eT s1);

template <typename vT, arithmetic_op_type_t op_type>
inline vT arithmetic_vector_kernel(const vT v0, const vT v1);

inline void arithmetic_pad_shape(const ppl::common::TensorShape* shape, const int64_t padded_dim_count, int64_t* padded_shape)
{
    const int64_t dim_diff = padded_dim_count - shape->GetRealDimCount();
    for (int64_t i = 0; i < dim_diff; i++) {
        padded_shape[i] = 1;
    }
    for (int64_t i = dim_diff; i < padded_dim_count; i++) {
        padded_shape[i] = shape->GetDim(i - dim_diff);
    }
}

inline void arithmetic_compress_shape(
    const int64_t* src0_shape,
    const int64_t* src1_shape,
    const int64_t dim_count,
    int64_t* compressed_dim_count,
    int64_t* compressed_src0_shape,
    int64_t* compressed_src1_shape,
    int64_t* compressed_dst_shape,
    const int64_t c_dim_idx = -1) // for nbcx dataformat, c_dim_idx should be set to disable compress on channel dim
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
        if (i == c_dim_idx) { // for nbcx dataformat, channel dim cannot be compressed
            compressed_dim_idx++; // flush before
            compressed_src0_shape[compressed_dim_idx] = src0_shape[i];
            compressed_src1_shape[compressed_dim_idx] = src1_shape[i];

            compressed_dim_idx++; // move to next
            i++;
            compressed_src0_shape[compressed_dim_idx] = src0_shape[i];
            compressed_src1_shape[compressed_dim_idx] = src1_shape[i];

            continue;
        }

        if (src0_broadcast[i] == src0_broadcast[i - 1] && src1_broadcast[i] == src1_broadcast[i - 1]) {
            compressed_src0_shape[compressed_dim_idx] *= src0_shape[i];
            compressed_src1_shape[compressed_dim_idx] *= src1_shape[i];
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

}}}}; // namespace ppl::kernel::arm_server::neon

#endif // __ST_PPL_KERNEL_ARM_SERVER_ARTIHMETIC_NEON_ARITHMETIC_COMMON_H_