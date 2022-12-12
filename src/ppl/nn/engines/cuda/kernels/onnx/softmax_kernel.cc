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

#include "ppl/nn/engines/cuda/kernels/onnx/softmax_kernel.h"
#include "ppl/nn/engines/cuda/params/quant_param_cuda.h"
#include "ppl/common/destructor.h"
#include "cudakernel/nn/softmax.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t SoftmaxKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto input = ctx.GetInput<TensorImpl>(0);
    return PPLSoftmaxGetTempBufferSize(input->GetShape(), param_->axis);
}

ppl::common::RetCode SoftmaxKernel::DoExecute(KernelExecContext* ctx) {
    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto input_shape = input->GetShape();
    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input->GetEdge()->GetId());
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output->GetEdge()->GetId());
    if (input_shape->GetDataType() == ppl::common::DATATYPE_INT8) {
        QuantKernelParamCuda qparam(input_quant.zero_point[0], output_quant.zero_point[0], input_quant.scale[0], output_quant.scale[0]);
        auto status = PPLCUDASoftmaxForwardImpInt8(GetStream(), input->GetShape(), input->GetBufferPtr(), output->GetShape(),
                                          output->GetBufferPtr(), nullptr, param_->axis, &qparam);
        return status;
    } else if (input_shape->GetDimCount() == 4 && param_->axis == 3 && input_shape->GetDim(2) == input_shape->GetDim(3)) {
        return PPLCUDAFastSoftmax(GetStream(), input->GetShape(), input->GetBufferPtr(), output->GetShape(),
                                  output->GetBufferPtr(), nullptr, 1);
    } else {
        BufferDesc tmp_buffer_desc;
        auto tmp_buffer_bytes = CalcTmpBufferSize(*ctx);
        auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                       << "] failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
        ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
            GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
        });
        auto tmp_buffer = tmp_buffer_desc.addr;
        status = PPLCUDASoftmaxForwardImp(GetStream(), input->GetShape(), input->GetBufferPtr(), output->GetShape(),
                                          output->GetBufferPtr(), tmp_buffer, param_->axis);
        return status;
    }
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
