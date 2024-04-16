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

#include "slice_kernel.h"

#include "cudakernel/memory/slice.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace onnx {


ppl::common::RetCode SliceKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(starts, 1);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(ends, 2);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(axes, 3);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(steps, 4);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [starts]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(starts);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [ends]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(ends);
    if (axes) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [axes]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(axes);
    }
    if (steps) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [steps]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(steps);
    }

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    SliceKernelParam kernel_param;
    kernel_param.axes_num = starts->GetShape()->CalcElementsIncludingPadding();

    if (param_->starts.empty() && kernel_param.axes_num > 0) {
        auto status = starts->CopyToHost(kernel_param.starts);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "starts->CopyToHost() failed: "
                << ppl::common::GetRetCodeStr(status);
            return status;
        }
    } else {
        for (int64_t i = 0; i < kernel_param.axes_num; ++i) {
            kernel_param.starts[i] = param_->starts[i];
        }
    }

    if (param_->ends.empty() && kernel_param.axes_num > 0) {
        auto status = ends->CopyToHost(kernel_param.ends);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "ends->CopyToHost() failed: "
                << ppl::common::GetRetCodeStr(status);
            return status;
        }
    } else {
        for (int64_t i = 0; i < kernel_param.axes_num; ++i) {
            kernel_param.ends[i] = param_->ends[i];
        }
    }

    if (param_->axes.empty() && kernel_param.axes_num > 0) {
        if (axes) {
            auto status = axes->CopyToHost(kernel_param.axes);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "axes->CopyToHost() failed: "
                    << ppl::common::GetRetCodeStr(status);
                return status;
            }
        } else {
            for (int64_t i = 0; i < kernel_param.axes_num; ++i) {
                kernel_param.axes[i] = i;
            }
        }
    } else {
        for (int64_t i = 0; i < kernel_param.axes_num; ++i) {
            kernel_param.axes[i] = param_->axes[i];
        }
    }

    if (param_->steps.empty() && kernel_param.axes_num > 0) {
        if (steps) {
            auto status = steps->CopyToHost(kernel_param.steps);
            if (status != ppl::common::RC_SUCCESS) {
                LOG(ERROR) << "steps->CopyToHost() failed: "
                    << ppl::common::GetRetCodeStr(status);
                return status;
            }
        } else {
            for (int64_t i = 0; i < kernel_param.axes_num; ++i) {
                kernel_param.steps[i] = 1;
            }
        }
    } else {
        for (int64_t i = 0; i < kernel_param.axes_num; ++i) {
            kernel_param.steps[i] = param_->steps[i];
        }
    }

    bool can_trans = ctx->IsLastConsumerOfInput(0)
        && input->GetType() == TENSORTYPE_NORMAL
        && input->GetShape()->CalcElementsIncludingPadding() == ctx->GetOutput<TensorImpl>(0)->GetShape()->CalcElementsIncludingPadding();

    const int64_t dim_count = input->GetShape()->GetDimCount();
    for (int64_t i = 0; i < kernel_param.axes_num; ++i) {
        int64_t axis_val = kernel_param.axes[i];
        int64_t start_val = kernel_param.starts[i];
        int64_t end_val = kernel_param.ends[i];

        if (axis_val < 0) {
            axis_val += dim_count;
        }

        const int64_t axis_dim = input->GetShape()->GetDim(axis_val);

        if (start_val == INT_MIN
            || start_val == LLONG_MIN) {
            start_val = 0;
        }
        if (start_val == INT_MAX
            || start_val == LLONG_MAX
            || start_val > axis_dim) {
            start_val = axis_dim;
        }
        if (start_val < 0) {
            start_val = axis_dim + start_val;
        }
        if (end_val == INT_MIN
            || end_val == LLONG_MIN) {
            end_val = 0;
        }
        if (end_val == INT_MAX
            || end_val == LLONG_MAX
            || end_val > axis_dim) {
            end_val = axis_dim;
        }
        if (end_val < 0) {
            end_val = axis_dim + end_val;
        }

        kernel_param.axes[i] = axis_val;
        kernel_param.starts[i] = start_val;
        kernel_param.ends[i] = end_val;

        if (start_val != 0 || end_val != axis_dim) {
            can_trans = false;
        }
    }

    if (can_trans) {
        output->TransferBufferFrom(input);
    } else {
        PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    if (!can_trans) {
        return PPLCUDASliceForwardImp(
            GetStream(), kernel_param,
            input->GetShape(),
            input->GetBufferPtr(),
            output->GetShape(),
            output->GetBufferPtr());
    }

    return ppl::common::RC_SUCCESS;
}

}}}}} // namespace ppl::nn::llm::cuda::pmx
