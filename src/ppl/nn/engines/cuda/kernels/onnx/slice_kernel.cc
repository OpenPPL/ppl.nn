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

#include "ppl/nn/engines/cuda/kernels/onnx/slice_kernel.h"

#include "cudakernel/memory/slice.h"
namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode SliceKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    SliceKernelParam kernel_param;

    const TensorShape& in_shape0 = ctx->GetInput<TensorImpl>(0)->GetShape();
    int dim_count = in_shape0.GetDimCount();
    int input_count = ctx->GetInputCount();
    { // starts
        auto input = ctx->GetInput<TensorImpl>(1);
        auto status = input->CopyToHost(kernel_param.starts);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy starts failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }
    { // ends
        auto input = ctx->GetInput<TensorImpl>(2);
        auto status = input->CopyToHost(kernel_param.ends);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy ends failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }
    if (input_count >= 4) { // axes
        auto input = ctx->GetInput<TensorImpl>(3);
        auto status = input->CopyToHost(kernel_param.axes);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy axes failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
        kernel_param.axes_num = input->GetShape().GetElementsIncludingPadding();
    } else {
        for (int it = 0; it < dim_count; ++it) {
            kernel_param.axes[it] = it;
        }
        kernel_param.axes_num = dim_count;
    }
    if (input_count >= 5) { // steps
        auto input = ctx->GetInput<TensorImpl>(4);
        auto status = input->CopyToHost(kernel_param.steps);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy steps failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    } else {
        for (int it = 0; it < dim_count; ++it) {
            kernel_param.steps[it] = 1;
        }
    }
    for (int it = 0; it < kernel_param.axes_num; ++it) {
        int64_t axis = kernel_param.axes[it];
        int64_t start_val = kernel_param.starts[it];
        int64_t end_val = kernel_param.ends[it];
        // int step_val = kernel_param.steps[it];
        kernel_param.axes[it] = (axis + dim_count) % dim_count;
        int cur_dim_size = in_shape0.GetDim((axis + dim_count) % dim_count);
        if (start_val == INT_MIN) {
            start_val = 0;
        }
        if (start_val == INT_MAX || start_val > cur_dim_size) {
            start_val = cur_dim_size;
        }
        if (start_val < 0) {
            start_val = cur_dim_size + start_val;
        }
        if (end_val == INT_MIN) {
            end_val = 0;
        }
        if (end_val == INT_MAX || end_val > cur_dim_size) {
            end_val = cur_dim_size;
        }
        if (end_val < 0) {
            end_val = cur_dim_size + end_val;
        }
        kernel_param.starts[it] = start_val;
        kernel_param.ends[it] = end_val;
    }

    ppl::common::RetCode status = ppl::common::RC_SUCCESS;
    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        input->GetShape().GetElementsIncludingPadding() == output->GetShape().GetElementsIncludingPadding()) {
        output->TransferBufferFrom(input);
    } else {
        status = PPLCUDASliceForwardImp(GetStream(), kernel_param, &input->GetShape(), input->GetBufferPtr(),
                                        &output->GetShape(), output->GetBufferPtr());
    }

    return status;
}

}}} // namespace ppl::nn::cuda
