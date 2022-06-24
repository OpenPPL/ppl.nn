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

#ifndef __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_MAX6D_SSE_ARITHMETIC_FP32_SSE_COMMON_H_
#define __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_MAX6D_SSE_ARITHMETIC_FP32_SSE_COMMON_H_

#include <nmmintrin.h>
#include <math.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/common/arithmetic/arithmetic_common.h"

namespace ppl { namespace kernel { namespace x86 {

template <arithmetic_op_type_t _op, int32_t broadcast_side>
void arithmetic_binary_op_ndarray_6d_broadcast_fp32_sse(
    const int64_t lhs_strides[5],
    const int64_t rhs_strides[5],
    const int64_t dst_strides[5],
    const int64_t dst_dims[6],
    const float *lhs,
    const float *rhs,
    float *dst)
{
#ifdef PPL_USE_X86_OMP
    const int64_t num_threads = omp_get_max_threads();
#else
    const int64_t num_threads = 1;
#endif
    const int64_t simd_w            = 4;
    const int64_t unroll_len        = simd_w * 4;
    const int64_t inner_max_threads = 8;
    const int64_t inner_threads     = min<int64_t>(inner_max_threads, num_threads);
    const int64_t inner_blk_align   = unroll_len;
    const int64_t inner_blk =
        round_up(max<int64_t>(1, dst_dims[5] / inner_threads), inner_blk_align);

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(6)
#endif
    for (int64_t d0 = 0; d0 < dst_dims[0]; ++d0) {
        for (int64_t d1 = 0; d1 < dst_dims[1]; ++d1) {
            for (int64_t d2 = 0; d2 < dst_dims[2]; ++d2) {
                for (int64_t d3 = 0; d3 < dst_dims[3]; ++d3) {
                    for (int64_t d4 = 0; d4 < dst_dims[4]; ++d4) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_PARALLEL_FOR()
#endif
                        for (int64_t ib = 0; ib < dst_dims[5]; ib += inner_blk) {
                            const float *l_lhs = lhs +
                                                 d0 * lhs_strides[0] +
                                                 d1 * lhs_strides[1] +
                                                 d2 * lhs_strides[2] +
                                                 d3 * lhs_strides[3] +
                                                 d4 * lhs_strides[4];
                            const float *l_rhs = rhs + d0 * rhs_strides[0] +
                                                 d1 * rhs_strides[1] +
                                                 d2 * rhs_strides[2] +
                                                 d3 * rhs_strides[3] +
                                                 d4 * rhs_strides[4];
                            float *l_dst = dst + d0 * dst_strides[0] +
                                           d1 * dst_strides[1] +
                                           d2 * dst_strides[2] +
                                           d3 * dst_strides[3] +
                                           d4 * dst_strides[4] +
                                           ib;
                            const float *broadcast_src = broadcast_side == 0 ? l_lhs : l_rhs;
                            const float *plain_src     = broadcast_side == 0 ? l_rhs + ib : l_lhs + ib;
                            const int64_t inner_eff    = min<int64_t>(dst_dims[5] - ib, inner_blk);
                            int64_t unroll_body        = round(inner_eff, unroll_len);
                            if (_op == ARITHMETIC_POW) {
                                unroll_body = 0;
                            }
                            if (unroll_body) {
                                __m128 mm_broadcast = _mm_set1_ps(broadcast_src[0]);
                                for (int64_t i = 0; i < unroll_body; i += unroll_len) {
                                    __m128 mm_src0 = _mm_loadu_ps(plain_src + i + 0 * simd_w);
                                    __m128 mm_src1 = _mm_loadu_ps(plain_src + i + 1 * simd_w);
                                    __m128 mm_src2 = _mm_loadu_ps(plain_src + i + 2 * simd_w);
                                    __m128 mm_src3 = _mm_loadu_ps(plain_src + i + 3 * simd_w);
                                    if (_op == ARITHMETIC_ADD) {
                                        mm_src0 = _mm_add_ps(mm_src0, mm_broadcast);
                                        mm_src1 = _mm_add_ps(mm_src1, mm_broadcast);
                                        mm_src2 = _mm_add_ps(mm_src2, mm_broadcast);
                                        mm_src3 = _mm_add_ps(mm_src3, mm_broadcast);
                                    } else if (_op == ARITHMETIC_MUL) {
                                        mm_src0 = _mm_mul_ps(mm_src0, mm_broadcast);
                                        mm_src1 = _mm_mul_ps(mm_src1, mm_broadcast);
                                        mm_src2 = _mm_mul_ps(mm_src2, mm_broadcast);
                                        mm_src3 = _mm_mul_ps(mm_src3, mm_broadcast);
                                    }
                                    if (broadcast_side == 0) {
                                        if (_op == ARITHMETIC_DIV) {
                                            mm_src0 = _mm_div_ps(mm_broadcast, mm_src0);
                                            mm_src1 = _mm_div_ps(mm_broadcast, mm_src1);
                                            mm_src2 = _mm_div_ps(mm_broadcast, mm_src2);
                                            mm_src3 = _mm_div_ps(mm_broadcast, mm_src3);
                                        } else if (_op == ARITHMETIC_SUB) {
                                            mm_src0 = _mm_sub_ps(mm_broadcast, mm_src0);
                                            mm_src1 = _mm_sub_ps(mm_broadcast, mm_src1);
                                            mm_src2 = _mm_sub_ps(mm_broadcast, mm_src2);
                                            mm_src3 = _mm_sub_ps(mm_broadcast, mm_src3);
                                        }
                                    } else if (broadcast_side == 1) {
                                        if (_op == ARITHMETIC_DIV) {
                                            mm_src0 = _mm_div_ps(mm_src0, mm_broadcast);
                                            mm_src1 = _mm_div_ps(mm_src1, mm_broadcast);
                                            mm_src2 = _mm_div_ps(mm_src2, mm_broadcast);
                                            mm_src3 = _mm_div_ps(mm_src3, mm_broadcast);
                                        } else if (_op == ARITHMETIC_SUB) {
                                            mm_src0 = _mm_sub_ps(mm_src0, mm_broadcast);
                                            mm_src1 = _mm_sub_ps(mm_src1, mm_broadcast);
                                            mm_src2 = _mm_sub_ps(mm_src2, mm_broadcast);
                                            mm_src3 = _mm_sub_ps(mm_src3, mm_broadcast);
                                        }
                                    }
                                    _mm_storeu_ps(l_dst + i + 0 * simd_w, mm_src0);
                                    _mm_storeu_ps(l_dst + i + 1 * simd_w, mm_src1);
                                    _mm_storeu_ps(l_dst + i + 2 * simd_w, mm_src2);
                                    _mm_storeu_ps(l_dst + i + 3 * simd_w, mm_src3);
                                }
                            }

                            for (int64_t i = unroll_body; i < inner_eff; ++i) {
                                if (_op == ARITHMETIC_ADD) {
                                    l_dst[i] = broadcast_src[0] + plain_src[i];
                                } else if (_op == ARITHMETIC_MUL) {
                                    l_dst[i] = broadcast_src[0] * plain_src[i];
                                }
                                if (broadcast_side == 0) {
                                    if (_op == ARITHMETIC_DIV) {
                                        l_dst[i] = broadcast_src[0] / plain_src[i];
                                    } else if (_op == ARITHMETIC_SUB) {
                                        l_dst[i] = broadcast_src[0] - plain_src[i];
                                    } else if (_op == ARITHMETIC_POW) {
                                        l_dst[i] = powf(broadcast_src[0], plain_src[i]);
                                    }
                                } else if (broadcast_side == 1) {
                                    if (_op == ARITHMETIC_DIV) {
                                        l_dst[i] = plain_src[i] / broadcast_src[0];
                                    } else if (_op == ARITHMETIC_SUB) {
                                        l_dst[i] = plain_src[i] - broadcast_src[0];
                                    } else if (_op == ARITHMETIC_POW) {
                                        l_dst[i] = powf(plain_src[i], broadcast_src[0]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <arithmetic_op_type_t _op>
void arithmetic_binary_op_ndarray_6d_eltwise_fp32_sse(
    const int64_t lhs_strides[5],
    const int64_t rhs_strides[5],
    const int64_t dst_strides[5],
    const int64_t dst_dims[6],
    const float *lhs,
    const float *rhs,
    float *dst)
{
#ifdef PPL_USE_X86_OMP
    const int64_t num_threads = omp_get_max_threads();
#else
    const int64_t num_threads = 1;
#endif
    const int64_t simd_w            = 4;
    const int64_t unroll_len        = simd_w * 4;
    const int64_t inner_max_threads = 8;
    const int64_t inner_threads     = min<int64_t>(inner_max_threads, num_threads);
    const int64_t inner_blk_align   = unroll_len;
    const int64_t inner_blk =
        round_up(max<int64_t>(1, dst_dims[5] / inner_threads), inner_blk_align);

#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(6)
#endif
    for (int64_t d0 = 0; d0 < dst_dims[0]; ++d0) {
        for (int64_t d1 = 0; d1 < dst_dims[1]; ++d1) {
            for (int64_t d2 = 0; d2 < dst_dims[2]; ++d2) {
                for (int64_t d3 = 0; d3 < dst_dims[3]; ++d3) {
                    for (int64_t d4 = 0; d4 < dst_dims[4]; ++d4) {
#ifndef PPL_USE_X86_OMP_COLLAPSE
                        PRAGMA_OMP_PARALLEL_FOR()
#endif
                        for (int64_t ib = 0; ib < dst_dims[5]; ib += inner_blk) {
                            const float *l_lhs = lhs +
                                                 d0 * lhs_strides[0] +
                                                 d1 * lhs_strides[1] +
                                                 d2 * lhs_strides[2] +
                                                 d3 * lhs_strides[3] +
                                                 d4 * lhs_strides[4] +
                                                 ib;
                            const float *l_rhs = rhs +
                                                 d0 * rhs_strides[0] +
                                                 d1 * rhs_strides[1] +
                                                 d2 * rhs_strides[2] +
                                                 d3 * rhs_strides[3] +
                                                 d4 * rhs_strides[4] +
                                                 ib;
                            float *l_dst = dst +
                                           d0 * dst_strides[0] +
                                           d1 * dst_strides[1] +
                                           d2 * dst_strides[2] +
                                           d3 * dst_strides[3] +
                                           d4 * dst_strides[4] +
                                           ib;
                            const int64_t inner_eff = min<int64_t>(dst_dims[5] - ib, inner_blk);
                            int64_t unroll_body     = round(inner_eff, unroll_len);
                            if (_op == ARITHMETIC_POW) {
                                unroll_body = 0;
                            }
                            for (int64_t i = 0; i < unroll_body; i += unroll_len) {
                                __m128 mm_src0 = _mm_loadu_ps(l_rhs + i + 0 * simd_w);
                                __m128 mm_src1 = _mm_loadu_ps(l_rhs + i + 1 * simd_w);
                                __m128 mm_src2 = _mm_loadu_ps(l_rhs + i + 2 * simd_w);
                                __m128 mm_src3 = _mm_loadu_ps(l_rhs + i + 3 * simd_w);
                                if (_op == ARITHMETIC_ADD) {
                                    mm_src0 = _mm_add_ps(_mm_loadu_ps(l_lhs + i + 0 * simd_w), mm_src0);
                                    mm_src1 = _mm_add_ps(_mm_loadu_ps(l_lhs + i + 1 * simd_w), mm_src1);
                                    mm_src2 = _mm_add_ps(_mm_loadu_ps(l_lhs + i + 2 * simd_w), mm_src2);
                                    mm_src3 = _mm_add_ps(_mm_loadu_ps(l_lhs + i + 3 * simd_w), mm_src3);
                                } else if (_op == ARITHMETIC_SUB) {
                                    mm_src0 = _mm_sub_ps(_mm_loadu_ps(l_lhs + i + 0 * simd_w), mm_src0);
                                    mm_src1 = _mm_sub_ps(_mm_loadu_ps(l_lhs + i + 1 * simd_w), mm_src1);
                                    mm_src2 = _mm_sub_ps(_mm_loadu_ps(l_lhs + i + 2 * simd_w), mm_src2);
                                    mm_src3 = _mm_sub_ps(_mm_loadu_ps(l_lhs + i + 3 * simd_w), mm_src3);
                                } else if (_op == ARITHMETIC_MUL) {
                                    mm_src0 = _mm_mul_ps(_mm_loadu_ps(l_lhs + i + 0 * simd_w), mm_src0);
                                    mm_src1 = _mm_mul_ps(_mm_loadu_ps(l_lhs + i + 1 * simd_w), mm_src1);
                                    mm_src2 = _mm_mul_ps(_mm_loadu_ps(l_lhs + i + 2 * simd_w), mm_src2);
                                    mm_src3 = _mm_mul_ps(_mm_loadu_ps(l_lhs + i + 3 * simd_w), mm_src3);
                                } else if (_op == ARITHMETIC_DIV) {
                                    mm_src0 = _mm_div_ps(_mm_loadu_ps(l_lhs + i + 0 * simd_w), mm_src0);
                                    mm_src1 = _mm_div_ps(_mm_loadu_ps(l_lhs + i + 1 * simd_w), mm_src1);
                                    mm_src2 = _mm_div_ps(_mm_loadu_ps(l_lhs + i + 2 * simd_w), mm_src2);
                                    mm_src3 = _mm_div_ps(_mm_loadu_ps(l_lhs + i + 3 * simd_w), mm_src3);
                                }
                                _mm_storeu_ps(l_dst + i + 0 * simd_w, mm_src0);
                                _mm_storeu_ps(l_dst + i + 1 * simd_w, mm_src1);
                                _mm_storeu_ps(l_dst + i + 2 * simd_w, mm_src2);
                                _mm_storeu_ps(l_dst + i + 3 * simd_w, mm_src3);
                            }
                            for (int64_t i = unroll_body; i < inner_eff; ++i) {
                                if (_op == ARITHMETIC_ADD) {
                                    l_dst[i] = l_lhs[i] + l_rhs[i];
                                } else if (_op == ARITHMETIC_SUB) {
                                    l_dst[i] = l_lhs[i] - l_rhs[i];
                                } else if (_op == ARITHMETIC_MUL) {
                                    l_dst[i] = l_lhs[i] * l_rhs[i];
                                } else if (_op == ARITHMETIC_DIV) {
                                    l_dst[i] = l_lhs[i] / l_rhs[i];
                                } else if (_op == ARITHMETIC_POW) {
                                    l_dst[i] = powf(l_lhs[i], l_rhs[i]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

template <arithmetic_op_type_t _op>
ppl::common::RetCode arithmetic_binary_op_ndarray_fp32_sse(
    const ppl::nn::TensorShape *lhs_shape,
    const ppl::nn::TensorShape *rhs_shape,
    const float *lhs,
    const float *rhs,
    float *dst)
{
    bool eltwise = lhs_shape->GetRealDimCount() == rhs_shape->GetRealDimCount();
    if (eltwise) {
        for (uint32_t i = 0; i < lhs_shape->GetRealDimCount(); ++i) {
            if (lhs_shape->GetDim(i) != rhs_shape->GetDim(i)) {
                eltwise = false;
                break;
            }
        }
    }
    if (eltwise) {
        int64_t zero_strides[5] = {0, 0, 0, 0, 0};
        int64_t dst_dims[6]     = {1, 1, 1, 1, 1, (int64_t)lhs_shape->CalcElementsIncludingPadding()};
        arithmetic_binary_op_ndarray_6d_eltwise_fp32_sse<_op>(
            zero_strides, zero_strides, zero_strides, dst_dims, lhs, rhs, dst);
        return ppl::common::RC_SUCCESS;
    }

    const int32_t ndims = 6;
    if (lhs_shape->GetRealDimCount() > ndims || rhs_shape->GetRealDimCount() > ndims) {
        return ppl::common::RC_UNSUPPORTED;
    }

    int64_t out_dims[ndims] = {1, 1, 1, 1, 1, 1};
    int64_t lhs_dims[ndims] = {1, 1, 1, 1, 1, 1};
    int64_t rhs_dims[ndims] = {1, 1, 1, 1, 1, 1};
    int32_t lhs_off         = ndims - lhs_shape->GetRealDimCount();
    for (int32_t i = lhs_off; i < ndims; ++i) {
        lhs_dims[i] = lhs_shape->GetDim(i - lhs_off);
    }
    int32_t rhs_off = ndims - rhs_shape->GetRealDimCount();
    for (int32_t i = rhs_off; i < ndims; ++i) {
        rhs_dims[i] = rhs_shape->GetDim(i - rhs_off);
    }
    for (int32_t i = 0; i < ndims; ++i) {
        if (lhs_dims[i] != rhs_dims[i] && lhs_dims[i] != 1 && rhs_dims[i] != 1) {
            return ppl::common::RC_UNSUPPORTED;
        } else {
            out_dims[i] = max(lhs_dims[i], rhs_dims[i]);
        }
    }

    int32_t suffix = 0;
    int32_t prefix = 0;
    for (int32_t i = ndims - suffix - 1; i >= prefix; --i) {
        if (out_dims[i] == 1) {
            ++suffix;
        } else {
            break;
        }
    }
    for (int32_t i = prefix; i < ndims - suffix; ++i) {
        if (out_dims[i] == 1) {
            ++prefix;
        } else {
            break;
        }
    }

    const bool simd_broadcast    = lhs_dims[ndims - suffix - 1] != rhs_dims[ndims - suffix - 1];
    const int32_t broadcast_side = lhs_dims[ndims - suffix - 1] == 1 ? 0 : 1;
    // prod simd dims
    if (simd_broadcast) {
        int64_t *broadcast_dims = broadcast_side == 0 ? lhs_dims : rhs_dims;
        for (int32_t i = ndims - suffix - 2; i >= prefix; --i) {
            if (broadcast_dims[i] == 1) {
                out_dims[i] *= out_dims[i + 1];
                lhs_dims[i] *= lhs_dims[i + 1];
                rhs_dims[i] *= rhs_dims[i + 1];
                out_dims[i + 1] = 1;
                lhs_dims[i + 1] = 1;
                rhs_dims[i + 1] = 1;
                ++suffix;
            } else {
                break;
            }
        }
    } else {
        for (int32_t i = ndims - suffix - 2; i >= prefix; --i) {
            if (lhs_dims[i] == rhs_dims[i]) {
                out_dims[i] *= out_dims[i + 1];
                lhs_dims[i] *= lhs_dims[i + 1];
                rhs_dims[i] *= rhs_dims[i + 1];
                out_dims[i + 1] = 1;
                lhs_dims[i + 1] = 1;
                rhs_dims[i + 1] = 1;
                ++suffix;
            } else {
                break;
            }
        }
    }

    const int32_t dim_unused      = prefix + suffix;
    const int32_t dim_used        = ndims - dim_unused;
    bool broadcast_lhs[ndims - 1] = {false, false, false, false, false};
    bool broadcast_rhs[ndims - 1] = {false, false, false, false, false};
    for (int32_t i = 0; i < dim_used - 1; ++i) {
        if (lhs_dims[prefix + i] != rhs_dims[prefix + i]) {
            if (lhs_dims[prefix + i] == 1) {
                broadcast_lhs[dim_unused + i] = true;
            } else if (rhs_dims[prefix + i] == 1) {
                broadcast_rhs[dim_unused + i] = true;
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
        }
    }

    // stride of dims which will be broadcasted is 0
    int64_t lhs_strides[ndims - 1] = {0, 0, 0, 0, 0};
    int64_t rhs_strides[ndims - 1] = {0, 0, 0, 0, 0};
    int64_t dst_strides[ndims - 1] = {0, 0, 0, 0, 0};
    int64_t dst_dims[ndims]        = {1, 1, 1, 1, 1, 1};
    const int32_t end_dim          = prefix + dim_used;

    // we should not use the last stride to get the current stride, because it may be 0
    if (dim_used >= 1) {
        dst_dims[ndims - 1] = out_dims[end_dim - 1];
    }
    if (dim_used >= 2) {
        dst_dims[ndims - 2]    = out_dims[end_dim - 2];
        lhs_strides[ndims - 2] = broadcast_lhs[ndims - 2] ? 0 : lhs_dims[end_dim - 1];
        rhs_strides[ndims - 2] = broadcast_rhs[ndims - 2] ? 0 : rhs_dims[end_dim - 1];
        dst_strides[ndims - 2] = dst_dims[ndims - 1];
    }
    if (dim_used >= 3) {
        dst_dims[ndims - 3]    = out_dims[end_dim - 3];
        lhs_strides[ndims - 3] = broadcast_lhs[ndims - 3] ? 0 : (lhs_dims[end_dim - 1] * lhs_dims[end_dim - 2]);
        rhs_strides[ndims - 3] = broadcast_rhs[ndims - 3] ? 0 : (rhs_dims[end_dim - 1] * rhs_dims[end_dim - 2]);
        dst_strides[ndims - 3] = dst_dims[ndims - 1] * dst_dims[ndims - 2];
    }
    if (dim_used >= 4) {
        dst_dims[ndims - 4]    = out_dims[end_dim - 4];
        lhs_strides[ndims - 4] = broadcast_lhs[ndims - 4] ? 0 : (lhs_dims[end_dim - 1] * lhs_dims[end_dim - 2] * lhs_dims[end_dim - 3]);
        rhs_strides[ndims - 4] = broadcast_rhs[ndims - 4] ? 0 : (rhs_dims[end_dim - 1] * rhs_dims[end_dim - 2] * rhs_dims[end_dim - 3]);
        dst_strides[ndims - 4] = dst_dims[ndims - 1] * dst_dims[ndims - 2] * dst_dims[ndims - 3];
    }
    if (dim_used >= 5) {
        dst_dims[ndims - 5]    = out_dims[end_dim - 5];
        lhs_strides[ndims - 5] = broadcast_lhs[ndims - 5] ? 0 : (lhs_dims[end_dim - 1] * lhs_dims[end_dim - 2] * lhs_dims[end_dim - 3] * lhs_dims[end_dim - 4]);
        rhs_strides[ndims - 5] = broadcast_rhs[ndims - 5] ? 0 : (rhs_dims[end_dim - 1] * rhs_dims[end_dim - 2] * rhs_dims[end_dim - 3] * rhs_dims[end_dim - 4]);
        dst_strides[ndims - 5] = dst_dims[ndims - 1] * dst_dims[ndims - 2] * dst_dims[ndims - 3] * dst_dims[ndims - 4];
    }
    if (dim_used >= 6) {
        dst_dims[ndims - 6]    = out_dims[end_dim - 6];
        lhs_strides[ndims - 6] = broadcast_lhs[ndims - 6] ? 0 : (lhs_dims[end_dim - 1] * lhs_dims[end_dim - 2] * lhs_dims[end_dim - 3] * lhs_dims[end_dim - 4] * lhs_dims[end_dim - 5]);
        rhs_strides[ndims - 6] = broadcast_rhs[ndims - 6] ? 0 : (rhs_dims[end_dim - 1] * rhs_dims[end_dim - 2] * rhs_dims[end_dim - 3] * rhs_dims[end_dim - 4] * rhs_dims[end_dim - 5]);
        dst_strides[ndims - 6] = dst_dims[ndims - 1] * dst_dims[ndims - 2] * dst_dims[ndims - 3] * dst_dims[ndims - 4] * dst_dims[ndims - 5];
    }

    if (simd_broadcast) {
        if (broadcast_side == 0) {
            arithmetic_binary_op_ndarray_6d_broadcast_fp32_sse<_op, 0>(
                lhs_strides, rhs_strides, dst_strides, dst_dims, lhs, rhs, dst);
        } else {
            arithmetic_binary_op_ndarray_6d_broadcast_fp32_sse<_op, 1>(
                lhs_strides, rhs_strides, dst_strides, dst_dims, lhs, rhs, dst);
        }
    } else {
        arithmetic_binary_op_ndarray_6d_eltwise_fp32_sse<_op>(lhs_strides, rhs_strides, dst_strides, dst_dims, lhs, rhs, dst);
    }

    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_FP32_ARITHMETIC_MAX6D_SSE_ARITHMETIC_FP32_SSE_COMMON_H_
