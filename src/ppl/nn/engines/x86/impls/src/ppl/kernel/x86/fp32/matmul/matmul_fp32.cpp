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

#include <deque>
#include <memory>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace kernel { namespace x86 {

static ppl::common::RetCode matmul_ndarray_fp32_recursive(
    const ppl::common::isa_t isa,
    const float *A,
    const float *B,
    const int64_t *A_strides,
    const int64_t *B_strides,
    const int64_t *Y_strides,
    const int64_t *Y_dims,
    const int64_t dim_count,
    const int64_t dim_idx,
    const int32_t M,
    const int32_t N,
    const int32_t K,
    float *Y)
{
    if (dim_idx >= dim_count - 2) {
        return gemm_fp32(
            isa, A, B, nullptr, nullptr,
            gemm_m_type::NOTRANS, gemm_m_type::NOTRANS,
            gemm_v_type::EMPTY, gemm_m_type::EMPTY,
            M, N, K, K, N, N, 0, 1.0f, 0.0f, 0.0f, 0.0f,
            gemm_post::NONE, Y);
    } else {
        const int64_t length = Y_dims[dim_idx];
        for (int64_t i = 0; i < length; i++) {
            auto ret = matmul_ndarray_fp32_recursive(
                isa,
                A + i * A_strides[dim_idx],
                B + i * B_strides[dim_idx],
                A_strides, B_strides,
                Y_strides, Y_dims,
                dim_count, dim_idx + 1,
                M, N, K,
                Y + i * Y_strides[dim_idx]);
            if (ret != ppl::common::RC_SUCCESS) {
                return ret;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode matmul_ndarray_fp32(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *A_shape,
    const ppl::nn::TensorShape *B_shape,
    const ppl::nn::TensorShape *Y_shape,
    const float *A,
    const float *B,
    float *Y)
{
    const int64_t max_dim_count = max(A_shape->GetDimCount(), B_shape->GetDimCount());
    std::deque<int64_t> A_dims(A_shape->GetDims(), A_shape->GetDims() + A_shape->GetDimCount());
    std::deque<int64_t> B_dims(B_shape->GetDims(), B_shape->GetDims() + B_shape->GetDimCount());

    if (A_dims.size() == 1) A_dims.push_front(1);
    if (B_dims.size() == 1) B_dims.push_back(1);

    while ((int64_t)A_dims.size() < max_dim_count) A_dims.push_front(1);
    while ((int64_t)B_dims.size() < max_dim_count) B_dims.push_front(1);

    bool is_single_gemm = A_dims.size() == 2 && B_dims.size() == 2;

    const int64_t K = A_dims[max_dim_count - 1];
    const int64_t N = B_dims[max_dim_count - 1];

    if (B_shape->GetElementsExcludingPadding() / (N * K) == 1) {
        for (int64_t i = max_dim_count - 3; i >= 0; i--) {
            A_dims[max_dim_count - 2] *= A_dims[i];
            A_dims[i] = 1;
        }
        is_single_gemm = true;
    }

    const int64_t M = A_dims[max_dim_count - 2];

    if (is_single_gemm) {
        return gemm_fp32(
            isa, A, B, nullptr, nullptr,
            gemm_m_type::NOTRANS, gemm_m_type::NOTRANS,
            gemm_v_type::EMPTY, gemm_m_type::EMPTY,
            M, N, K, K, N, N, 0, 1.0f, 0.0f, 0.0f, 0.0f,
            gemm_post::NONE, Y);
    }

    int64_t Y_dims[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    Y_dims[max_dim_count - 2] = M;
    Y_dims[max_dim_count - 1] = N;
    for (int64_t i = 0; i < max_dim_count - 2; i++) {
        Y_dims[i] = A_dims[i] == B_dims[i] ? A_dims[i] : A_dims[i] * B_dims[i]; // assuming that can broadcast
    }

    int64_t A_strides[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t B_strides[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t Y_strides[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    A_strides[max_dim_count - 1] = 1;
    B_strides[max_dim_count - 1] = 1;
    Y_strides[max_dim_count - 1]  = 1;
    for (int64_t i = max_dim_count - 2; i >= 0; i--) {
        A_strides[i] = A_strides[i + 1] * A_dims[i + 1];
        B_strides[i] = B_strides[i + 1] * B_dims[i + 1];
        Y_strides[i]  = Y_strides[i + 1] * Y_dims[i + 1];
    }
    for (int64_t i = 0; i < max_dim_count - 2; i++) {
        A_strides[i] = A_dims[i] == 1 ? 0 : A_strides[i];
        B_strides[i] = B_dims[i] == 1 ? 0 : B_strides[i];
    }

    return matmul_ndarray_fp32_recursive(
        isa, A, B,
        A_strides, B_strides,
        Y_strides, Y_dims,
        max_dim_count, 0,
        M, N, K, Y);
}

}}}; // namespace ppl::kernel::x86
