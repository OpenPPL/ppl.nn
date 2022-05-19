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

#include "ppl/nn/engines/cuda/kernels/onnx/avepooling_kernel.h"

#include "cudakernel/nn/pooling_ave.h"
#include "cudakernel/nn/global_pooling_ave.h"

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode AvePoolingKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ppl::common::RetCode status = ppl::common::RC_UNSUPPORTED;
    auto input_id = input->GetEdge()->GetId();
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input_id);
    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    if (param_->global_pooling) {
        status =
            PPLCUDAGlobalAvePoolingForwardImp(GetStream(), input->GetShape(), input->GetBufferPtr(), output->GetShape(),
                                              output->GetBufferPtr(), input_quant.scale[0], output_quant.scale[0]);
    } else {
        int32_t kernel_h = param_->kernel_shape[0];
        int32_t kernel_w = param_->kernel_shape[1];
        int32_t stride_h = param_->strides[0];
        int32_t stride_w = param_->strides[1];
        int32_t pad_h = param_->pads.size() >= 1 ? param_->pads[0] : 0;
        int32_t pad_w = param_->pads.size() >= 2 ? param_->pads[1] : 0;
        // 1*1 pooling, just transfer
        int32_t kernel_dims = param_->kernel_shape.size();
        bool pooling_1x1 = true;
        for (int32_t i = 0; i < kernel_dims; ++i) {
            int32_t pad_i = param_->pads.size() >= (uint32_t)(i + 1) ? param_->pads[i] : 0;
            if (param_->kernel_shape[i] != 1 || param_->strides[i] != 1 || pad_i != 0)
                pooling_1x1 = false;
        }
        if (pooling_1x1) {
            output->TransferBufferFrom(input);
            return ppl::common::RC_SUCCESS;
        }
        TensorShape in_shape(*input->GetShape());
        TensorShape out_shape(*output->GetShape());
        // 3d pooling not supported now, try to convert to 2d ones, if can not, return "unsupported"
        if (kernel_dims == 3) {
            int32_t pad_3 = param_->pads.size() >= 3 ? param_->pads[2] : 0;
            in_shape.SetDimCount(4); out_shape.SetDimCount(4);
            if (param_->kernel_shape[0] == 1 && param_->strides[0] == 1 && pad_h == 0) {
                kernel_h = param_->kernel_shape[1]; kernel_w = param_->kernel_shape[2];
                stride_h = param_->strides[1]; stride_w = param_->strides[2];
                pad_h = param_->pads.size() >= 2 ? param_->pads[1] : 0;
                pad_w = param_->pads.size() >= 3 ? param_->pads[2] : 0;
                in_shape.SetDim(2, input->GetShape()->GetDim(2) * input->GetShape()->GetDim(3));
                out_shape.SetDim(2, output->GetShape()->GetDim(2) * output->GetShape()->GetDim(3));
                in_shape.SetDim(3, input->GetShape()->GetDim(4));
                out_shape.SetDim(3, output->GetShape()->GetDim(4));
            } else if (param_->kernel_shape[1] == 1 && param_->strides[1] == 1 && pad_w == 0) {
                kernel_w = param_->kernel_shape[2]; stride_w = param_->strides[2];
                pad_w = param_->pads.size() >= 3 ? param_->pads[2] : 0;
                in_shape.SetDim(3, input->GetShape()->GetDim(3) * input->GetShape()->GetDim(4));
                out_shape.SetDim(3, output->GetShape()->GetDim(3) * output->GetShape()->GetDim(4));
            } else if (param_->kernel_shape[2] == 1 && param_->strides[2] == 1 && pad_3 == 0 && input->GetShape()->GetDim(4) == 1) {
            } else {
                return ppl::common::RC_UNSUPPORTED;
            }
        }

        int32_t if_excluding_padding = 1;
        if (param_->mode == ppl::nn::onnx::PoolingParam::POOLING_AVERAGE_INCLUDE) {
            if_excluding_padding = 0;
        }
        status = PPLCUDAAvePoolingForwardImp(GetStream(), &in_shape, input->GetBufferPtr(), &out_shape,
                                             output->GetBufferPtr(), kernel_h, kernel_w, stride_h, stride_w, pad_h,
                                             pad_w, if_excluding_padding, input_quant.scale[0], output_quant.scale[0]);
    }
    return status;
}

}}} // namespace ppl::nn::cuda
