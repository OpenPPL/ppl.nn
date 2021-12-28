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

#include "ppl/nn/engines/cuda/kernels/onnx/resize_kernel.h"

#include "cudakernel/nn/resize.h"

namespace ppl { namespace nn { namespace cuda {

bool ResizeKernel::CanDoExecute(const KernelExecContext& ctx) const {
    auto& X = ctx.GetInput<TensorImpl>(0)->GetShape();
    if (X.GetBytesIncludingPadding() == 0) {
        return false;
    }

    auto& scales = ctx.GetInput<TensorImpl>(2)->GetShape();
    if (ctx.GetInputCount() == 3 && scales.GetBytesIncludingPadding() == 0) {
        return false;
    }

    if (ctx.GetInputCount() >= 4) {
        auto& sizes = ctx.GetInput<TensorImpl>(3)->GetShape();
        if (scales.GetBytesIncludingPadding() == 0 && sizes.GetBytesIncludingPadding() == 0) {
            return false;
        }
    }
    return true;
}

ppl::common::RetCode ResizeKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    // same size, just transfer
    if (input->GetEdge()->CalcConsumerCount() == 1 && input->GetType() == TENSORTYPE_NORMAL &&
        input->GetShape().GetDimCount() == 4 && input->GetShape().GetDim(2) == output->GetShape().GetDim(2) &&
        input->GetShape().GetDim(3) == output->GetShape().GetDim(3)) {
        output->TransferBufferFrom(input);
        return ppl::common::RC_SUCCESS;
    }

    // deal with pre-set h_scale&w_scale
    bool scale_pre_set = false;
    float h_scale = 0.f, w_scale = 0.f;
    std::vector<float> scales_data;
    if (!ctx->GetInput<TensorImpl>(2)->GetShape().IsEmpty()) {
        scale_pre_set = true;
        auto shape = ctx->GetInput<TensorImpl>(2)->GetShape();
        scales_data.resize(shape.GetElementsIncludingPadding());
        auto status = ctx->GetInput<TensorImpl>(2)->CopyToHost(scales_data.data());
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy scales data failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
        h_scale = 1.f / scales_data[2];
        w_scale = 1.f / scales_data[3];
    }    
    auto input_id = input->GetEdge()->GetId();
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input_id);
    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    auto status = PPLCUDAResizeForwardImp(
        GetStream(), &input->GetShape(), input->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr(),
        scale_pre_set, h_scale, w_scale, param_->coord_trans_mode, param_->mode, param_->cubic_coeff_a, input_quant.scale[0], output_quant.scale[0]);
    return status;
}

}}} // namespace ppl::nn::cuda
