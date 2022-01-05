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

#ifndef __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_ARITHMETIC_BROADCAST_N16CX_INT64_H_
#define __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_ARITHMETIC_BROADCAST_N16CX_INT64_H_

#include "arithmetic_kernel_int64.h"

namespace ppl { namespace kernel { namespace x86 {

template <arithmetic_op_type_t _op>
static inline void arithmetic_broadcast_lastdim_no_broadcast_n16cx_int64(
    const int64_t *src0,
    const int64_t *src1,
    const int64_t start,
    const int64_t end,
    const bool c0_broadcast,
    const bool c1_broadcast,
    int64_t *dst)
{
    int64_t i = start;
    if (!c0_broadcast && !c1_broadcast) {
        for (; i <= end; i++) {
            dst[i * 16 + 0]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 0], src1[i * 16 + 0]);
            dst[i * 16 + 1]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 1], src1[i * 16 + 1]);
            dst[i * 16 + 2]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 2], src1[i * 16 + 2]);
            dst[i * 16 + 3]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 3], src1[i * 16 + 3]);
            dst[i * 16 + 4]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 4], src1[i * 16 + 4]);
            dst[i * 16 + 5]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 5], src1[i * 16 + 5]);
            dst[i * 16 + 6]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 6], src1[i * 16 + 6]);
            dst[i * 16 + 7]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 7], src1[i * 16 + 7]);
            dst[i * 16 + 8]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 8], src1[i * 16 + 8]);
            dst[i * 16 + 9]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 9], src1[i * 16 + 9]);
            dst[i * 16 + 10] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 10], src1[i * 16 + 10]);
            dst[i * 16 + 11] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 11], src1[i * 16 + 11]);
            dst[i * 16 + 12] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 12], src1[i * 16 + 12]);
            dst[i * 16 + 13] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 13], src1[i * 16 + 13]);
            dst[i * 16 + 14] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 14], src1[i * 16 + 14]);
            dst[i * 16 + 15] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 15], src1[i * 16 + 15]);
        }
    } else if (c0_broadcast) {
        for (; i <= end; i++) {
            int64_t broadcast_val = src0[i * 16];
            dst[i * 16 + 0]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 0]);
            dst[i * 16 + 1]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 1]);
            dst[i * 16 + 2]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 2]);
            dst[i * 16 + 3]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 3]);
            dst[i * 16 + 4]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 4]);
            dst[i * 16 + 5]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 5]);
            dst[i * 16 + 6]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 6]);
            dst[i * 16 + 7]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 7]);
            dst[i * 16 + 8]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 8]);
            dst[i * 16 + 9]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 9]);
            dst[i * 16 + 10]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 10]);
            dst[i * 16 + 11]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 11]);
            dst[i * 16 + 12]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 12]);
            dst[i * 16 + 13]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 13]);
            dst[i * 16 + 14]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 14]);
            dst[i * 16 + 15]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 15]);
        }
    } else if (c1_broadcast) {
        for (; i <= end; i++) {
            int64_t broadcast_val = src1[i * 16];
            dst[i * 16 + 0]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 0], broadcast_val);
            dst[i * 16 + 1]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 1], broadcast_val);
            dst[i * 16 + 2]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 2], broadcast_val);
            dst[i * 16 + 3]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 3], broadcast_val);
            dst[i * 16 + 4]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 4], broadcast_val);
            dst[i * 16 + 5]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 5], broadcast_val);
            dst[i * 16 + 6]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 6], broadcast_val);
            dst[i * 16 + 7]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 7], broadcast_val);
            dst[i * 16 + 8]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 8], broadcast_val);
            dst[i * 16 + 9]       = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 9], broadcast_val);
            dst[i * 16 + 10]      = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 10], broadcast_val);
            dst[i * 16 + 11]      = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 11], broadcast_val);
            dst[i * 16 + 12]      = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 12], broadcast_val);
            dst[i * 16 + 13]      = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 13], broadcast_val);
            dst[i * 16 + 14]      = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 14], broadcast_val);
            dst[i * 16 + 15]      = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 15], broadcast_val);
        }
    }
}

template <arithmetic_op_type_t _op>
static inline void arithmetic_broadcast_lastdim_src0_broadcast_n16cx_int64(
    const int64_t *src0,
    const int64_t *src1,
    const int64_t start,
    const int64_t end,
    const bool c0_broadcast,
    const bool c1_broadcast,
    int64_t *dst)
{
    int64_t i = start;
    if (!c0_broadcast && !c1_broadcast) {
        for (; i <= end; i++) {
            dst[i * 16 + 0]  = arithmetic_scalar_kernel_int64<_op>(src0[0], src1[i * 16 + 0]);
            dst[i * 16 + 1]  = arithmetic_scalar_kernel_int64<_op>(src0[1], src1[i * 16 + 1]);
            dst[i * 16 + 2]  = arithmetic_scalar_kernel_int64<_op>(src0[2], src1[i * 16 + 2]);
            dst[i * 16 + 3]  = arithmetic_scalar_kernel_int64<_op>(src0[3], src1[i * 16 + 3]);
            dst[i * 16 + 4]  = arithmetic_scalar_kernel_int64<_op>(src0[4], src1[i * 16 + 4]);
            dst[i * 16 + 5]  = arithmetic_scalar_kernel_int64<_op>(src0[5], src1[i * 16 + 5]);
            dst[i * 16 + 6]  = arithmetic_scalar_kernel_int64<_op>(src0[6], src1[i * 16 + 6]);
            dst[i * 16 + 7]  = arithmetic_scalar_kernel_int64<_op>(src0[7], src1[i * 16 + 7]);
            dst[i * 16 + 8]  = arithmetic_scalar_kernel_int64<_op>(src0[8], src1[i * 16 + 8]);
            dst[i * 16 + 9]  = arithmetic_scalar_kernel_int64<_op>(src0[9], src1[i * 16 + 9]);
            dst[i * 16 + 10] = arithmetic_scalar_kernel_int64<_op>(src0[10], src1[i * 16 + 10]);
            dst[i * 16 + 11] = arithmetic_scalar_kernel_int64<_op>(src0[11], src1[i * 16 + 11]);
            dst[i * 16 + 12] = arithmetic_scalar_kernel_int64<_op>(src0[12], src1[i * 16 + 12]);
            dst[i * 16 + 13] = arithmetic_scalar_kernel_int64<_op>(src0[13], src1[i * 16 + 13]);
            dst[i * 16 + 14] = arithmetic_scalar_kernel_int64<_op>(src0[14], src1[i * 16 + 14]);
            dst[i * 16 + 15] = arithmetic_scalar_kernel_int64<_op>(src0[15], src1[i * 16 + 15]);
        }
    } else if (c0_broadcast) {
        int64_t broadcast_val = src0[0];
        for (; i <= end; i++) {
            dst[i * 16 + 0]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 0]);
            dst[i * 16 + 1]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 1]);
            dst[i * 16 + 2]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 2]);
            dst[i * 16 + 3]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 3]);
            dst[i * 16 + 4]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 4]);
            dst[i * 16 + 5]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 5]);
            dst[i * 16 + 6]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 6]);
            dst[i * 16 + 7]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 7]);
            dst[i * 16 + 8]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 8]);
            dst[i * 16 + 9]  = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 9]);
            dst[i * 16 + 10] = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 10]);
            dst[i * 16 + 11] = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 11]);
            dst[i * 16 + 12] = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 12]);
            dst[i * 16 + 13] = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 13]);
            dst[i * 16 + 14] = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 14]);
            dst[i * 16 + 15] = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[i * 16 + 15]);
        }
    } else if (c1_broadcast) {
        for (; i <= end; i++) {
            int64_t broadcast_val = src1[i * 16];
            dst[i * 16 + 0]       = arithmetic_scalar_kernel_int64<_op>(src0[0], broadcast_val);
            dst[i * 16 + 1]       = arithmetic_scalar_kernel_int64<_op>(src0[1], broadcast_val);
            dst[i * 16 + 2]       = arithmetic_scalar_kernel_int64<_op>(src0[2], broadcast_val);
            dst[i * 16 + 3]       = arithmetic_scalar_kernel_int64<_op>(src0[3], broadcast_val);
            dst[i * 16 + 4]       = arithmetic_scalar_kernel_int64<_op>(src0[4], broadcast_val);
            dst[i * 16 + 5]       = arithmetic_scalar_kernel_int64<_op>(src0[5], broadcast_val);
            dst[i * 16 + 6]       = arithmetic_scalar_kernel_int64<_op>(src0[6], broadcast_val);
            dst[i * 16 + 7]       = arithmetic_scalar_kernel_int64<_op>(src0[7], broadcast_val);
            dst[i * 16 + 8]       = arithmetic_scalar_kernel_int64<_op>(src0[8], broadcast_val);
            dst[i * 16 + 9]       = arithmetic_scalar_kernel_int64<_op>(src0[9], broadcast_val);
            dst[i * 16 + 10]      = arithmetic_scalar_kernel_int64<_op>(src0[10], broadcast_val);
            dst[i * 16 + 11]      = arithmetic_scalar_kernel_int64<_op>(src0[11], broadcast_val);
            dst[i * 16 + 12]      = arithmetic_scalar_kernel_int64<_op>(src0[12], broadcast_val);
            dst[i * 16 + 13]      = arithmetic_scalar_kernel_int64<_op>(src0[13], broadcast_val);
            dst[i * 16 + 14]      = arithmetic_scalar_kernel_int64<_op>(src0[14], broadcast_val);
            dst[i * 16 + 15]      = arithmetic_scalar_kernel_int64<_op>(src0[15], broadcast_val);
        }
    }
}

template <arithmetic_op_type_t _op>
static inline void arithmetic_broadcast_lastdim_src1_broadcast_n16cx_int64(
    const int64_t *src0,
    const int64_t *src1,
    const int64_t start,
    const int64_t end,
    const bool c0_broadcast,
    const bool c1_broadcast,
    int64_t *dst)
{
    // const int64_t simd_w   = 8;
    // const int64_t simd_num = 2;

    int64_t i = start;
    if (!c0_broadcast && !c1_broadcast) {
        for (; i <= end; i++) {
            dst[i * 16 + 0]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 0], src1[0]);
            dst[i * 16 + 1]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 1], src1[1]);
            dst[i * 16 + 2]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 2], src1[2]);
            dst[i * 16 + 3]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 3], src1[3]);
            dst[i * 16 + 4]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 4], src1[4]);
            dst[i * 16 + 5]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 5], src1[5]);
            dst[i * 16 + 6]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 6], src1[6]);
            dst[i * 16 + 7]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 7], src1[7]);
            dst[i * 16 + 8]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 8], src1[8]);
            dst[i * 16 + 9]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 9], src1[9]);
            dst[i * 16 + 10] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 10], src1[10]);
            dst[i * 16 + 11] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 11], src1[11]);
            dst[i * 16 + 12] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 12], src1[12]);
            dst[i * 16 + 13] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 13], src1[13]);
            dst[i * 16 + 14] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 14], src1[14]);
            dst[i * 16 + 15] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 15], src1[15]);
        }
    } else if (c0_broadcast) {
        for (; i <= end; i++) {
            int64_t broadcast_val = src0[i * 16];
            dst[i * 16 + 0]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[0]);
            dst[i * 16 + 1]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[1]);
            dst[i * 16 + 2]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[2]);
            dst[i * 16 + 3]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[3]);
            dst[i * 16 + 4]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[4]);
            dst[i * 16 + 5]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[5]);
            dst[i * 16 + 6]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[6]);
            dst[i * 16 + 7]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[7]);
            dst[i * 16 + 8]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[8]);
            dst[i * 16 + 9]       = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[9]);
            dst[i * 16 + 10]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[10]);
            dst[i * 16 + 11]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[11]);
            dst[i * 16 + 12]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[12]);
            dst[i * 16 + 13]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[13]);
            dst[i * 16 + 14]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[14]);
            dst[i * 16 + 15]      = arithmetic_scalar_kernel_int64<_op>(broadcast_val, src1[15]);
        }
    } else if (c1_broadcast) {
        int64_t broadcast_val = src1[0];
        for (; i <= end; i++) {
            dst[i * 16 + 0]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 0], broadcast_val);
            dst[i * 16 + 1]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 1], broadcast_val);
            dst[i * 16 + 2]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 2], broadcast_val);
            dst[i * 16 + 3]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 3], broadcast_val);
            dst[i * 16 + 4]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 4], broadcast_val);
            dst[i * 16 + 5]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 5], broadcast_val);
            dst[i * 16 + 6]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 6], broadcast_val);
            dst[i * 16 + 7]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 7], broadcast_val);
            dst[i * 16 + 8]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 8], broadcast_val);
            dst[i * 16 + 9]  = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 9], broadcast_val);
            dst[i * 16 + 10] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 10], broadcast_val);
            dst[i * 16 + 11] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 11], broadcast_val);
            dst[i * 16 + 12] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 12], broadcast_val);
            dst[i * 16 + 13] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 13], broadcast_val);
            dst[i * 16 + 14] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 14], broadcast_val);
            dst[i * 16 + 15] = arithmetic_scalar_kernel_int64<_op>(src0[i * 16 + 15], broadcast_val);
        }
    }
}

template <arithmetic_op_type_t _op>
static ppl::common::RetCode arithmetic_broadcast_recursive_n16cx_int64(
    const int64_t *src0,
    const int64_t *src1,
    const int64_t *src0_shape,
    const int64_t *src1_shape,
    const int64_t *dst_shape,
    const int64_t *inc0,
    const int64_t *inc1,
    const int64_t *inc_out,
    const int64_t dim_count,
    const int64_t dim_idx,
    const bool c0_broadcast,
    const bool c1_broadcast,
    parallel_block *block,
    int64_t *dst)
{
    bool is_first       = is_first_dim(block, dim_idx);
    bool is_last        = is_last_dim(block, dim_idx);
    const int64_t start = is_first ? block->start[dim_idx] : 0;
    const int64_t end   = is_last ? block->end[dim_idx] : dst_shape[dim_idx] - 1;

    if (dim_idx == dim_count - 1) { // last dim
        if (src0_shape[dim_idx] == src1_shape[dim_idx]) {
            arithmetic_broadcast_lastdim_no_broadcast_n16cx_int64<_op>(src0, src1, start, end, c0_broadcast, c1_broadcast, dst);
        } else if (src0_shape[dim_idx] == 1) { // broadcast src0
            arithmetic_broadcast_lastdim_src0_broadcast_n16cx_int64<_op>(src0, src1, start, end, c0_broadcast, c1_broadcast, dst);
        } else if (src1_shape[dim_idx] == 1) { // broadcast src1
            arithmetic_broadcast_lastdim_src1_broadcast_n16cx_int64<_op>(src0, src1, start, end, c0_broadcast, c1_broadcast, dst);
        }
    } else {
        for (block->idx[dim_idx] = start; block->idx[dim_idx] <= end; block->idx[dim_idx]++) {
            int64_t i = block->idx[dim_idx];
            arithmetic_broadcast_recursive_n16cx_int64<_op>(
                src0 + i * inc0[dim_idx],
                src1 + i * inc1[dim_idx],
                src0_shape,
                src1_shape,
                dst_shape,
                inc0,
                inc1,
                inc_out,
                dim_count,
                dim_idx + 1,
                c0_broadcast,
                c1_broadcast,
                block,
                dst + i * inc_out[dim_idx]);
        }
    }

    return ppl::common::RC_SUCCESS;
}

template <arithmetic_op_type_t _op>
static ppl::common::RetCode arithmetic_broadcast_n16cx_int64(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const int64_t *src0,
    const int64_t *src1,
    const int64_t c_dim_idx,
    int64_t *dst)
{
    // pad 1 to input's high dims
    const int64_t dim_count = dst_shape->GetDimCount();
    if (dim_count > PPL_X86_TENSOR_MAX_DIMS()) {
        return ppl::common::RC_UNSUPPORTED;
    }

    int64_t padded_src0_shape[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t padded_src1_shape[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    pad_shape(src0_shape, dim_count, padded_src0_shape);
    pad_shape(src1_shape, dim_count, padded_src1_shape);
    const bool c0_broadcast = padded_src0_shape[c_dim_idx] != padded_src1_shape[c_dim_idx] &&
                              padded_src0_shape[c_dim_idx] == 1;
    const bool c1_broadcast = padded_src0_shape[c_dim_idx] != padded_src1_shape[c_dim_idx] &&
                              padded_src1_shape[c_dim_idx] == 1;

    // compress dims
    int64_t real_dim_count = 0;
    int64_t real_c_dim_idx = c_dim_idx;

    int64_t real_src0_shape[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t real_src1_shape[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t real_dst_shape[PPL_X86_TENSOR_MAX_DIMS()] = {0};

    // remove 1 on high dims to compress dim count
    // stop at C dim
    for (int64_t i = 0; i < dim_count; i++) {
        if (dst_shape->GetDim(i) <= 1 && i < c_dim_idx) {
            real_c_dim_idx--;
            continue;
        }
        real_src0_shape[real_dim_count] = padded_src0_shape[i];
        real_src1_shape[real_dim_count] = padded_src1_shape[i];
        real_dst_shape[real_dim_count] = dst_shape->GetDim(i);
        real_dim_count++;
    }

    // merge low dims
    // stop at C dim
    for (int64_t i = real_dim_count - 1; i >= real_c_dim_idx + 2; i--) {
        bool cur_dim_input0_need_broadcast  = real_src0_shape[i] != real_src1_shape[i] && real_src0_shape[i] == 1;
        bool cur_dim_input1_need_broadcast  = real_src0_shape[i] != real_src1_shape[i] && real_src1_shape[i] == 1;
        bool prev_dim_input0_need_broadcast = real_src0_shape[i - 1] != real_src1_shape[i - 1] && real_src0_shape[i - 1] == 1;
        bool prev_dim_input1_need_broadcast = real_src0_shape[i - 1] != real_src1_shape[i - 1] && real_src1_shape[i - 1] == 1;

        if (cur_dim_input0_need_broadcast == prev_dim_input0_need_broadcast && // can merge
            cur_dim_input1_need_broadcast == prev_dim_input1_need_broadcast) {
            real_src0_shape[i - 1] *= real_src0_shape[i];
            real_src1_shape[i - 1] *= real_src1_shape[i];
            real_dst_shape[i - 1] *= real_dst_shape[i];
            real_dim_count--;
        } else {
            break;
        }
    }

    int64_t inc0[PPL_X86_TENSOR_MAX_DIMS()]    = {0};
    int64_t inc1[PPL_X86_TENSOR_MAX_DIMS()]    = {0};
    int64_t inc_out[PPL_X86_TENSOR_MAX_DIMS()] = {0};

    // div C dim by 16 and set stride_w to 16
    int64_t stride0                   = 16;
    int64_t stride1                   = 16;
    int64_t stride_out                = 16;
    real_src0_shape[real_c_dim_idx] = div_up(real_src0_shape[real_c_dim_idx], 16);
    real_src1_shape[real_c_dim_idx] = div_up(real_src1_shape[real_c_dim_idx], 16);
    real_dst_shape[real_c_dim_idx] = div_up(real_dst_shape[real_c_dim_idx], 16);

    // prepare incs
    for (int64_t i = real_dim_count - 1; i >= 0; i--) {
        inc0[i]    = real_src0_shape[i] == 1 ? 0 : stride0;
        inc1[i]    = real_src1_shape[i] == 1 ? 0 : stride1;
        inc_out[i] = stride_out;

        stride0 *= real_src0_shape[i];
        stride1 *= real_src1_shape[i];
        stride_out *= real_dst_shape[i];
    }

    // split task for each thread
    const int64_t total_len   = dst_shape->GetElementsIncludingPadding() /
                              16; // because C dim has been divided by 16, len should also div 16
    const int64_t len_per_thread = div_up(total_len, PPL_OMP_MAX_THREADS());
    const int64_t num_threads    = div_up(total_len, len_per_thread);

    parallel_block blocks[num_threads];
    for (int64_t i = 0; i < num_threads; i++) {
        int64_t start_idx = i * len_per_thread;
        int64_t end_idx   = (i + 1) * len_per_thread - 1;
        if (end_idx >= total_len) {
            end_idx = total_len - 1;
        }
        idx2dims(start_idx, real_dst_shape, real_dim_count, blocks[i].start);
        idx2dims(end_idx, real_dst_shape, real_dim_count, blocks[i].end);
        blocks[i].id = i;
        for (int64_t j = 0; j < real_dim_count; j++) {
            blocks[i].idx[j] = blocks[i].start[j];
        }
    }

    PRAGMA_OMP_PARALLEL_FOR()
    for (int64_t i = 0; i < num_threads; i++) {
        arithmetic_broadcast_recursive_n16cx_int64<_op>(
            src0,
            src1,
            real_src0_shape,
            real_src1_shape,
            real_dst_shape,
            inc0,
            inc1,
            inc_out,
            real_dim_count,
            0,
            c0_broadcast,
            c1_broadcast,
            blocks + i,
            dst);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_INT64_ARITHMETIC_ARITHMETIC_BROADCAST_N16CX_INT64_H_
