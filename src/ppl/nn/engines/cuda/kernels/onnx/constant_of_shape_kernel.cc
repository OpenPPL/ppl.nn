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

#include "ppl/nn/engines/cuda/kernels/onnx/constant_of_shape_kernel.h"
#include "ppl/nn/common/logger.h"
#include "cudakernel/memory/constant_of_shape.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t ConstantOfShapeKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto output = ctx.GetOutput<TensorImpl>(0);
    return ppl::common::GetSizeOfDataType(output->GetShape().GetDataType());
}

ppl::common::RetCode ConstantOfShapeKernel::DoExecute(KernelExecContext* ctx) {
    auto cuda_device = GetCudaDevice();
    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = CalcTmpBufferSize(*ctx);
    auto status = cuda_device->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    BufferDescGuard __tmp_buffer_guard(&tmp_buffer_desc, [&cuda_device](BufferDesc* buffer) -> void {
        cuda_device->FreeTmpBuffer(buffer);
    });

    auto output = ctx->GetOutput<TensorImpl>(0);
    status = cuda_device->CopyFromHost(&tmp_buffer_desc, param_->data.data(), tmp_buffer_bytes);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "copy data failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }

    auto tmp_buffer = tmp_buffer_desc.addr;
    status = PPLCUDAConstantOfShapeForwardImp(GetStream(), tmp_buffer, &output->GetShape(), output->GetBufferPtr());
    return status;
}

}}} // namespace ppl::nn::cuda
