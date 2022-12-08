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

#include <vector>
#include <deque>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gemm.h"

namespace ppl { namespace kernel { namespace x86 {

static void matmul_ndarray_fp32_mat_list_construct(
    const float *A,
    const float *B,
    float *Y,
    const int64_t *Y_dims,
    const int64_t *A_strides,
    const int64_t *B_strides,
    const int64_t *Y_strides,
    const int64_t dim_index,
    const int64_t dim_count,
    std::vector<const float *> &A_list,
    std::vector<const float *> &B_list,
    std::vector<float *> &Y_list)
{
    if (dim_index >= dim_count - 2) {
        A_list.push_back(A);
        B_list.push_back(B);
        Y_list.push_back(Y);
    } else {
        const int64_t length = Y_dims[dim_index];
        for (int64_t i = 0; i < length; i++) {
            matmul_ndarray_fp32_mat_list_construct(
                A + i * A_strides[dim_index],
                B + i * B_strides[dim_index],
                Y + i * Y_strides[dim_index],
                Y_dims, A_strides, B_strides,
                Y_strides, dim_index + 1,
                dim_count, A_list, B_list, Y_list);
        }
    }
}

ppl::common::RetCode matmul_ndarray_fp32(
    const ppl::common::isa_t isa,
    const ppl::nn::TensorShape *A_shape,
    const ppl::nn::TensorShape *B_shape,
    const ppl::nn::TensorShape *Y_shape,
    const float *A,
    const float *B,
    const bool packedB,
    float *Y)
{
    const int64_t dim_count = Y_shape->GetDimCount();
    std::vector<int64_t> A_dims;
    std::vector<int64_t> B_dims;
    {
        std::deque<int64_t> A_dims_for_arrange(A_shape->GetDims(), A_shape->GetDims() + A_shape->GetDimCount());
        std::deque<int64_t> B_dims_for_arrange(B_shape->GetDims(), B_shape->GetDims() + B_shape->GetDimCount());

        if (A_dims_for_arrange.size() == 1) A_dims_for_arrange.push_front(1);
        if (B_dims_for_arrange.size() == 1) B_dims_for_arrange.push_back(1);

        while ((int64_t)A_dims_for_arrange.size() < dim_count) A_dims_for_arrange.push_front(1);
        while ((int64_t)B_dims_for_arrange.size() < dim_count) B_dims_for_arrange.push_front(1);

        A_dims.assign(A_dims_for_arrange.begin(), A_dims_for_arrange.end());
        B_dims.assign(B_dims_for_arrange.begin(), B_dims_for_arrange.end());
    }

    const int64_t K = A_dims[dim_count - 1];
    const int64_t N = B_dims[dim_count - 1];
    const bool is_single_gemm = B_shape->CalcElementsExcludingPadding() / (N * K) == 1;
    if (is_single_gemm) {
        for (int64_t i = dim_count - 3; i >= 0; i--) {
            A_dims[dim_count - 2] *= A_dims[i];
            A_dims[i] = 1;
        }
    }
    const int64_t M = A_dims[dim_count - 2];

    if (is_single_gemm) {
        return gemm_fp32(
            isa, A, B, nullptr, nullptr,
            gemm_m_type::NOTRANS, packedB ? gemm_m_type::PACKED : gemm_m_type::NOTRANS,
            gemm_v_type::EMPTY, gemm_m_type::EMPTY,
            M, N, K, K, N, N, 0, 1.0f, 0.0f, 0.0f, 0.0f,
            gemm_post::NONE, Y);
    }

    if (packedB) {
        return ppl::common::RC_UNSUPPORTED;
    }

    int64_t batch_y = 1;
    std::vector<int64_t> Y_dims(dim_count);
    Y_dims[dim_count - 2] = M;
    Y_dims[dim_count - 1] = N;
    for (int64_t i = 0; i < dim_count - 2; ++i) {
        Y_dims[i] = A_dims[i] == B_dims[i] ? A_dims[i] : A_dims[i] * B_dims[i];
        batch_y *= Y_dims[i];
    }

    std::vector<int64_t> A_strides(dim_count, 0);
    std::vector<int64_t> B_strides(dim_count, 0);
    std::vector<int64_t> Y_strides(dim_count, 0);
    A_strides[dim_count - 1] = 1;
    B_strides[dim_count - 1] = 1;
    Y_strides[dim_count - 1] = 1;
    for (int64_t i = dim_count - 2; i >= 0; i--) {
        A_strides[i] = A_strides[i + 1] * A_dims[i + 1];
        B_strides[i] = B_strides[i + 1] * B_dims[i + 1];
        Y_strides[i] = Y_strides[i + 1] * Y_dims[i + 1];
    }
    for (int64_t i = 0; i < dim_count - 2; i++) {
        A_strides[i] = A_dims[i] == 1 ? 0 : A_strides[i];
        B_strides[i] = B_dims[i] == 1 ? 0 : B_strides[i];
    }

    std::vector<const float*> A_list;
    std::vector<const float*> B_list;
    std::vector<float*>       Y_list;
    A_list.reserve(batch_y);
    B_list.reserve(batch_y);
    Y_list.reserve(batch_y);

    matmul_ndarray_fp32_mat_list_construct(
        A, B, Y,
        Y_dims.data(),
        A_strides.data(),
        B_strides.data(),
        Y_strides.data(),
        0, dim_count,
        A_list, B_list, Y_list);

    return batch_gemm_fp32(
        isa, A_list.data(), B_list.data(), nullptr, nullptr,
        gemm_m_type::NOTRANS, gemm_m_type::NOTRANS,
        gemm_v_type::EMPTY, gemm_m_type::EMPTY,
        batch_y, M, N, K, K, N, N, 0, 1.0f, 0.0f, 0.0f, 0.0f,
        gemm_post::NONE, Y_list.data());
}

}}}; // namespace ppl::kernel::x86
