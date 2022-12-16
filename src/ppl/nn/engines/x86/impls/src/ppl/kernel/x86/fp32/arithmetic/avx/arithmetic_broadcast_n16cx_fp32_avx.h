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

#ifndef __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_AVX_ARITHMETIC_BROADCAST_N16CX_FP32_AVX_H_
#define __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_AVX_ARITHMETIC_BROADCAST_N16CX_FP32_AVX_H_

#include "arithmetic_kernel_fp32_avx.h"

namespace ppl { namespace kernel { namespace x86 {

template <arithmetic_op_type_t _op, bool fuse_relu>
static inline void arithmetic_broadcast_lastdim_no_broadcast_n16cx_fp32_avx(
    const float *src0,
    const float *src1,
    const int64_t start,
    const int64_t end,
    const bool c0_broadcast,
    const bool c1_broadcast,
    float *dst)
{
    const int64_t simd_w   = 8;
    const int64_t simd_num = 2;
    int64_t i              = start;

    __m256 zero_vec = _mm256_set1_ps(0.0f);

    if (!c0_broadcast && !c1_broadcast) {
        for (; i <= end; i++) {
            __m256 vsrc0_0 = _mm256_loadu_ps(src0 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vsrc0_1 = _mm256_loadu_ps(src0 + i * simd_num * simd_w + 1 * simd_w);
            __m256 vsrc1_0 = _mm256_loadu_ps(src1 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vsrc1_1 = _mm256_loadu_ps(src1 + i * simd_num * simd_w + 1 * simd_w);
            __m256 vdst_0  = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_0, vsrc1_0);
            __m256 vdst_1  = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_1, vsrc1_1);
            if (fuse_relu) {
                vdst_0     = _mm256_max_ps(vdst_0, zero_vec);
                vdst_1     = _mm256_max_ps(vdst_1, zero_vec);
            }
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 0 * simd_w, vdst_0);
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 1 * simd_w, vdst_1);
        }
    } else if (c0_broadcast) {
        for (; i <= end; i++) {
            __m256 vsrc0   = _mm256_broadcast_ss(src0 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vsrc1_0 = _mm256_loadu_ps(src1 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vsrc1_1 = _mm256_loadu_ps(src1 + i * simd_num * simd_w + 1 * simd_w);
            __m256 vdst_0  = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0, vsrc1_0);
            __m256 vdst_1  = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0, vsrc1_1);
            if (fuse_relu) {
                vdst_0 = _mm256_max_ps(vdst_0, zero_vec);
                vdst_1 = _mm256_max_ps(vdst_1, zero_vec);
            }
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 0 * simd_w, vdst_0);
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 1 * simd_w, vdst_1);
        }
    } else if (c1_broadcast) {
        for (; i <= end; i++) {
            __m256 vsrc0_0 = _mm256_loadu_ps(src0 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vsrc0_1 = _mm256_loadu_ps(src0 + i * simd_num * simd_w + 1 * simd_w);
            __m256 vsrc1   = _mm256_broadcast_ss(src1 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vdst_0  = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_0, vsrc1);
            __m256 vdst_1  = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_1, vsrc1);
            if (fuse_relu) {
                vdst_0 = _mm256_max_ps(vdst_0, zero_vec);
                vdst_1 = _mm256_max_ps(vdst_1, zero_vec);
            }
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 0 * simd_w, vdst_0);
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 1 * simd_w, vdst_1);
        }
    }
}

template <arithmetic_op_type_t _op, bool fuse_relu>
static inline void arithmetic_broadcast_lastdim_src0_broadcast_n16cx_fp32_avx(
    const float *src0,
    const float *src1,
    const int64_t start,
    const int64_t end,
    const bool c0_broadcast,
    const bool c1_broadcast,
    float *dst)
{
    const int64_t simd_w   = 8;
    const int64_t simd_num = 2;

    __m256 zero_vec = _mm256_set1_ps(0.0f);

    __m256 vbroadcast_val_0, vbroadcast_val_1;
    if (!c0_broadcast) {
        vbroadcast_val_0 = _mm256_loadu_ps(src0);
        vbroadcast_val_1 = _mm256_loadu_ps(src0 + simd_w);
    } else {
        vbroadcast_val_0 = _mm256_broadcast_ss(src0);
        vbroadcast_val_1 = vbroadcast_val_0;
    }

    int64_t i = start;
    if (!c1_broadcast) {
        for (; i <= end; i++) {
            __m256 vsrc1_0 = _mm256_loadu_ps(src1 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vsrc1_1 = _mm256_loadu_ps(src1 + i * simd_num * simd_w + 1 * simd_w);
            __m256 vdst_0  = arithmetic_vector_kernel_fp32_avx<_op>(vbroadcast_val_0, vsrc1_0);
            __m256 vdst_1  = arithmetic_vector_kernel_fp32_avx<_op>(vbroadcast_val_1, vsrc1_1);
            if (fuse_relu) {
                vdst_0 = _mm256_max_ps(vdst_0, zero_vec);
                vdst_1 = _mm256_max_ps(vdst_1, zero_vec);
            }
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 0 * simd_w, vdst_0);
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 1 * simd_w, vdst_1);
        }
    } else {
        for (; i <= end; i++) {
            __m256 vsrc1  = _mm256_broadcast_ss(src1 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vdst_0 = arithmetic_vector_kernel_fp32_avx<_op>(vbroadcast_val_0, vsrc1);
            __m256 vdst_1 = arithmetic_vector_kernel_fp32_avx<_op>(vbroadcast_val_1, vsrc1);
            if (fuse_relu) {
                vdst_0 = _mm256_max_ps(vdst_0, zero_vec);
                vdst_1 = _mm256_max_ps(vdst_1, zero_vec);
            }
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 0 * simd_w, vdst_0);
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 1 * simd_w, vdst_1);
        }
    }
}

template <arithmetic_op_type_t _op, bool fuse_relu>
static inline void arithmetic_broadcast_lastdim_src1_broadcast_n16cx_fp32_avx(
    const float *src0,
    const float *src1,
    const int64_t start,
    const int64_t end,
    const bool c0_broadcast,
    const bool c1_broadcast,
    float *dst)
{
    const int64_t simd_w   = 8;
    const int64_t simd_num = 2;

    __m256 zero_vec = _mm256_set1_ps(0.0f);

    __m256 vbroadcast_val_0, vbroadcast_val_1;
    if (!c1_broadcast) {
        vbroadcast_val_0 = _mm256_loadu_ps(src1);
        vbroadcast_val_1 = _mm256_loadu_ps(src1 + simd_w);
    } else {
        vbroadcast_val_0 = _mm256_broadcast_ss(src1);
        vbroadcast_val_1 = vbroadcast_val_0;
    }

    int64_t i = start;
    if (!c0_broadcast) {
        for (; i <= end; i++) {
            __m256 vsrc0_0 = _mm256_loadu_ps(src0 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vsrc0_1 = _mm256_loadu_ps(src0 + i * simd_num * simd_w + 1 * simd_w);
            __m256 vdst_0  = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_0, vbroadcast_val_0);
            __m256 vdst_1  = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0_1, vbroadcast_val_1);
            if (fuse_relu) {
                vdst_0 = _mm256_max_ps(vdst_0, zero_vec);
                vdst_1 = _mm256_max_ps(vdst_1, zero_vec);
            }
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 0 * simd_w, vdst_0);
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 1 * simd_w, vdst_1);
        }
    } else {
        for (; i <= end; i++) {
            __m256 vsrc0  = _mm256_broadcast_ss(src0 + i * simd_num * simd_w + 0 * simd_w);
            __m256 vdst_0 = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0, vbroadcast_val_0);
            __m256 vdst_1 = arithmetic_vector_kernel_fp32_avx<_op>(vsrc0, vbroadcast_val_1);
            if (fuse_relu) {
                vdst_0 = _mm256_max_ps(vdst_0, zero_vec);
                vdst_1 = _mm256_max_ps(vdst_1, zero_vec);
            }
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 0 * simd_w, vdst_0);
            _mm256_storeu_ps(dst + i * simd_num * simd_w + 1 * simd_w, vdst_1);
        }
    }
}

template <arithmetic_op_type_t _op, bool fuse_relu>
static ppl::common::RetCode arithmetic_broadcast_recursive_n16cx_fp32_avx(
    const float *src0,
    const float *src1,
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
    float *dst)
{
    bool is_first       = is_first_dim(block, dim_idx);
    bool is_last        = is_last_dim(block, dim_idx);
    const int64_t start = is_first ? block->start[dim_idx] : 0;
    const int64_t end   = is_last ? block->end[dim_idx] : dst_shape[dim_idx] - 1;

    if (dim_idx == dim_count - 1) { // last dim
        if (src0_shape[dim_idx] == src1_shape[dim_idx]) {
            arithmetic_broadcast_lastdim_no_broadcast_n16cx_fp32_avx<_op, fuse_relu>(
                src0, src1, start, end, c0_broadcast, c1_broadcast, dst);
        } else if (src0_shape[dim_idx] == 1) { // broadcast src0
            arithmetic_broadcast_lastdim_src0_broadcast_n16cx_fp32_avx<_op, fuse_relu>(
                src0, src1, start, end, c0_broadcast, c1_broadcast, dst);
        } else if (src1_shape[dim_idx] == 1) { // broadcast src1
            arithmetic_broadcast_lastdim_src1_broadcast_n16cx_fp32_avx<_op, fuse_relu>(
                src0, src1, start, end, c0_broadcast, c1_broadcast, dst);
        }
    } else {
        for (block->idx[dim_idx] = start; block->idx[dim_idx] <= end; block->idx[dim_idx]++) {
            int64_t i = block->idx[dim_idx];
            arithmetic_broadcast_recursive_n16cx_fp32_avx<_op, fuse_relu>(
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

template <arithmetic_op_type_t _op, bool fuse_relu>
static ppl::common::RetCode arithmetic_broadcast_n16cx_fp32_avx(
    const ppl::common::TensorShape *src0_shape,
    const ppl::common::TensorShape *src1_shape,
    const ppl::common::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const int64_t c_dim_idx,
    float *dst)
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
        bool cur_dim_input0_need_broadcast =
            real_src0_shape[i] != real_src1_shape[i] && real_src0_shape[i] == 1;
        bool cur_dim_input1_need_broadcast =
            real_src0_shape[i] != real_src1_shape[i] && real_src1_shape[i] == 1;
        bool prev_dim_input0_need_broadcast =
            real_src0_shape[i - 1] != real_src1_shape[i - 1] && real_src0_shape[i - 1] == 1;
        bool prev_dim_input1_need_broadcast =
            real_src0_shape[i - 1] != real_src1_shape[i - 1] && real_src1_shape[i - 1] == 1;

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

    int64_t inc0[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t inc1[PPL_X86_TENSOR_MAX_DIMS()] = {0};
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

    const int64_t task_len = 16;
    std::vector<int64_t> loop_iter(1, max<int64_t>(dst_shape->CalcElementsIncludingPadding() / task_len, 1));
    auto pc = select_single_parallel_loop(
        loop_iter,
        ppl::common::ISA_X86_AVX,
        task_len * 2 * sizeof(float),
        task_len * sizeof(float),
        task_len * sizeof(float),
        1);

    // split task for each thread
    const int64_t num_threads = pc.num_threads;
    const int64_t total_len   = dst_shape->CalcElementsIncludingPadding() /
                              16; // because C dim has been divided by 16, len should also div 16
    const int64_t len_per_thread = div_up(total_len, num_threads);

    std::vector<parallel_block> blocks(num_threads);
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
        arithmetic_broadcast_recursive_n16cx_fp32_avx<_op, fuse_relu>(
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
            &blocks[i],
            dst);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_AVX_ARITHMETIC_BROADCAST_N16CX_FP32_AVX_H_
