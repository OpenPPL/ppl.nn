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
#ifndef __ST_PPL_KERNEL_ARM_SERVER_RELATION_NEON_RELATION_COMMON_H_
#define __ST_PPL_KERNEL_ARM_SERVER_RELATION_NEON_RELATION_COMMON_H_

#include "ppl/kernel/arm_server/common/type_traits.h"
#include "ppl/kernel/arm_server/common/internal_include.h"
namespace ppl { namespace kernel { namespace arm_server { namespace neon {

enum relation_op_type_t{
    RELATION_GREATER          = 0,
    RELATION_GREATER_OR_EQUAL = 1,
    RELATION_LESS             = 2,
    RELATION_LESS_OR_EQUAL    = 3,
    RELATION_EQUAL            = 4,
    RELATION_NOT_EQUAL        = 5
};

template <typename eT, relation_op_type_t _op>
inline uint8_t relation_scalar_kernel(eT a, eT b);


template <typename vT, relation_op_type_t _op>
inline vT relation_vector_kernel(vT va, vT vb);

template <typename vecType>
static void pack_four(
    const vecType& v0,
    const vecType& v1, 
    const vecType& v2, 
    const vecType& v3, 
    uint8_t* dst)
{
    uint32_t len = 0;
    uint8_t* tmp_dst = dst;

    if(std::is_same<vecType, float32x4_t>::value){
        len = 16;
        uint32_t* tmp = (uint32_t *)malloc(sizeof(uint32_t) * len);
        vst1q_f32((float *)(tmp + 0), (const float32x4_t)v0);
        vst1q_f32((float *)(tmp + 4), (const float32x4_t)v1);
        vst1q_f32((float *)(tmp + 8), (const float32x4_t)v2);
        vst1q_f32((float *)(tmp + 12), (const float32x4_t)v3);

        for (uint32_t i = 0; i < len; i++)
        {
            tmp_dst[i] = tmp[i] & 1;
        }

        free(tmp);
    } else if (std::is_same<vecType, int64x2_t>::value) {
        len = 8;
        uint64_t* tmp = (uint64_t *)malloc(sizeof(uint64_t) * len);
        vst1q_s64((int64_t *)(tmp + 0), (const int64x2_t)v0);
        vst1q_s64((int64_t *)(tmp + 2), (const int64x2_t)v1);
        vst1q_s64((int64_t *)(tmp + 4), (const int64x2_t)v2);
        vst1q_s64((int64_t *)(tmp + 6), (const int64x2_t)v3); 

        for (uint32_t i = 0; i < len; i++)
        {
            tmp_dst[i] = tmp[i] & 1;
        }

        free(tmp); 
    } 
#ifdef PPLNN_USE_ARMV8_2_FP16
    else if (std::is_same<vecType, float16x8_t>::value) {
        len = 32;
        uint16_t* tmp = (uint16_t *)malloc(sizeof(uint16_t) * len);
        vst1q_f16((__fp16 *)(tmp + 0), (const float16x8_t)v0);
        vst1q_f16((__fp16 *)(tmp + 8), (const float16x8_t)v1);
        vst1q_f16((__fp16 *)(tmp + 16), (const float16x8_t)v2);
        vst1q_f16((__fp16 *)(tmp + 24), (const float16x8_t)v3); 

        for (uint32_t i = 0; i < len; i++)
        {
            tmp_dst[i] = tmp[i] & 1;
        }

        free(tmp);
    }
#endif 
    else {
        len = 0;
    }
}


template <typename vecType>
static void pack_one(
    const vecType& v0, 
    uint8_t* dst)
{
    uint32_t len = 0;
    uint8_t* tmp_dst = dst;
    if(std::is_same<vecType, float32x4_t>::value){
        len = 4;
        uint32_t* tmp = (uint32_t *)malloc(sizeof(uint32_t) * len);
        vst1q_f32((float *)(tmp + 0), (const float32x4_t)v0);
        for (uint32_t i = 0; i < len; i++)
        {
            tmp_dst[i] = tmp[i] & 1;
        }

        free(tmp);
    } else if (std::is_same<vecType, int64x2_t>::value){
        len = 2;
        uint64_t* tmp = (uint64_t *)malloc(sizeof(uint64_t) * len);
        vst1q_s64((int64_t *)(tmp + 0), (const int64x2_t)v0);
        for (uint32_t i = 0; i < len; i++)
        {
            tmp_dst[i] = tmp[i] & 1;
        }

        free(tmp);
    } 
#ifdef PPLNN_USE_ARMV8_2_FP16
    else if (std::is_same<vecType, float16x8_t>::value){
        len = 2;
        uint16_t* tmp = (uint16_t *)malloc(sizeof(uint16_t) * len);
        vst1q_f16((__fp16 *)(tmp + 0), (const float16x8_t)v0);
        for (uint32_t i = 0; i < len; i++)
        {
            tmp_dst[i] = tmp[i] & 1;
        }
        free(tmp);
    }
#endif
}


inline void relation_pad_shape(
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

inline void relation_compress_shape(
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

        if (src0_broadcast[i] == src0_broadcast[compressed_dim_idx] && src1_broadcast[i] == src1_broadcast[compressed_dim_idx]) {
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

#endif // __ST_PPL_KERNEL_ARM_SERVER_RELATION_NEON_RELATION_COMMON_H_