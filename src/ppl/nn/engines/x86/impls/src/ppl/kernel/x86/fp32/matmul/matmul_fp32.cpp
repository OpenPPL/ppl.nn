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
#include "ppl/kernel/x86/fp32/gemm_v2.h"

namespace ppl { namespace kernel { namespace x86 {

uint64_t matmul_ndarray_fp32_get_buffer_bytes(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::common::isa_t isa_flag)
{
    const int64_t max_dim_count = max(src0_shape->GetDimCount(), src1_shape->GetDimCount());
    std::deque<int64_t> src0_dims(src0_shape->GetDims(), src0_shape->GetDims() + src0_shape->GetDimCount());
    std::deque<int64_t> src1_dims(src1_shape->GetDims(), src1_shape->GetDims() + src1_shape->GetDimCount());

    if (src0_dims.size() == 1) {
        src0_dims.push_front(1);
    }
    if (src1_dims.size() == 1) {
        src1_dims.push_back(1);
    }

    while ((int64_t)src0_dims.size() < max_dim_count) {
        src0_dims.push_front(1);
    }
    while ((int64_t)src1_dims.size() < max_dim_count) {
        src1_dims.push_front(1);
    }

    const int32_t m = src0_dims[max_dim_count - 2];
    const int32_t k = src0_dims[max_dim_count - 1];
    const int32_t n = src1_dims[max_dim_count - 1];

    gemm_v2_param_fp32 param;
    param.M        = m;
    param.N        = n;
    param.K        = k;
    param.lda      = k;
    param.ldb      = n;
    param.ldy      = n;
    param.isa_flag = isa_flag;

    auto executor = std::unique_ptr<gemm_v2_executor_fp32>(create_gemm_v2_executor_fp32(param));
    if (!executor) {
        return 0;
    }
    return executor->get_buffer_bytes();
}

static ppl::common::RetCode matmul_ndarray_recursive_fp32(
    const float *src0,
    const float *src1,
    gemm_v2_executor_fp32 *executor,
    const int64_t *src0_strides,
    const int64_t *src1_strides,
    const int64_t *dst_strides,
    const int64_t *dst_dims,
    const int64_t dim_count,
    const int64_t dim_idx,
    const int32_t m,
    const int32_t n,
    const int32_t k,
    float *dst)
{
    if (dim_idx >= dim_count - 2) {
        executor->get_param_mutable().src_A = src0;
        executor->get_param_mutable().src_B = src1;
        executor->get_param_mutable().dst_Y = dst;
        return executor->execute();
    } else {
        const int64_t length = dst_dims[dim_idx];
        for (int64_t i = 0; i < length; i++) {
            ppl::common::RetCode ret = matmul_ndarray_recursive_fp32(
                src0 + i * src0_strides[dim_idx], src1 + i * src1_strides[dim_idx], executor,
                src0_strides, src1_strides, dst_strides, dst_dims,
                dim_count, dim_idx + 1, m, n, k, dst + i * dst_strides[dim_idx]);
            if (ret != ppl::common::RC_SUCCESS) {
                return ret;
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode matmul_ndarray_fp32(
    const ppl::nn::TensorShape *src0_shape,
    const ppl::nn::TensorShape *src1_shape,
    const ppl::nn::TensorShape *dst_shape,
    const float *src0,
    const float *src1,
    const ppl::common::isa_t isa_flag,
    void *temp_buffer,
    float *dst)
{
    const int64_t max_dim_count = max(src0_shape->GetDimCount(), src1_shape->GetDimCount());
    std::deque<int64_t> src0_dims(src0_shape->GetDims(), src0_shape->GetDims() + src0_shape->GetDimCount());
    std::deque<int64_t> src1_dims(src1_shape->GetDims(), src1_shape->GetDims() + src1_shape->GetDimCount());

    if (src0_dims.size() == 1) {
        src0_dims.push_front(1);
    }
    if (src1_dims.size() == 1) {
        src1_dims.push_back(1);
    }

    while ((int64_t)src0_dims.size() < max_dim_count) {
        src0_dims.push_front(1);
    }
    while ((int64_t)src1_dims.size() < max_dim_count) {
        src1_dims.push_front(1);
    }

    const int32_t m = src0_dims[max_dim_count - 2];
    const int32_t k = src0_dims[max_dim_count - 1];
    const int32_t n = src1_dims[max_dim_count - 1];

    gemm_v2_param_fp32 param;
    param.M        = m;
    param.N        = n;
    param.K        = k;
    param.lda      = k;
    param.ldb      = n;
    param.ldy      = n;
    param.isa_flag = isa_flag; // other param use default value

    auto executor = std::unique_ptr<gemm_v2_executor_fp32>(create_gemm_v2_executor_fp32(param));
    if (!executor) {
        return ppl::common::RC_UNSUPPORTED;
    }
    executor->set_temp_buffer(temp_buffer);

    if (src0_dims.size() == 2 && src1_dims.size() == 2) { // normal gemm
        executor->get_param_mutable().src_A = src0;
        executor->get_param_mutable().src_B = src1;
        executor->get_param_mutable().dst_Y = dst;
        return executor->execute();
    }

    int64_t dst_dims[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    dst_dims[max_dim_count - 2] = m;
    dst_dims[max_dim_count - 1] = n;
    for (int64_t i = 0; i < max_dim_count - 2; i++) {
        dst_dims[i] = src0_dims[i] == src1_dims[i] ? src0_dims[i] : src0_dims[i] * src1_dims[i]; // assuming that can broadcast
    }

    int64_t src0_strides[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t src1_strides[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    int64_t dst_strides[PPL_X86_TENSOR_MAX_DIMS()] = {0};
    src0_strides[max_dim_count - 1] = 1;
    src1_strides[max_dim_count - 1] = 1;
    dst_strides[max_dim_count - 1]  = 1;
    for (int64_t i = max_dim_count - 2; i >= 0; i--) {
        src0_strides[i] = src0_strides[i + 1] * src0_dims[i + 1];
        src1_strides[i] = src1_strides[i + 1] * src1_dims[i + 1];
        dst_strides[i]  = dst_strides[i + 1] * dst_dims[i + 1];
    }
    for (int64_t i = 0; i < max_dim_count - 2; i++) {
        src0_strides[i] = src0_dims[i] == 1 ? 0 : src0_strides[i];
        src1_strides[i] = src1_dims[i] == 1 ? 0 : src1_strides[i];
    }

    return matmul_ndarray_recursive_fp32(
        src0, src1, executor.get(),
        src0_strides, src1_strides,
        dst_strides, dst_dims,
        max_dim_count, 0, m, n, k, dst);
}

}}}; // namespace ppl::kernel::x86
