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

#include "ppl/nn/engines/cuda/kernels/onnx/non_max_suppression_kernel.h"

#include "cudakernel/nn/nms.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t NonMaxSuppressionKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto scores = ctx.GetInput<TensorImpl>(1);
    return PPLNMSGetTempBufferSize(&scores->GetShape());
}

ppl::common::RetCode NonMaxSuppressionKernel::DoExecute(KernelExecContext* ctx) {
    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = CalcTmpBufferSize(*ctx);
    auto status = GetCudaDevice()->AllocTmpBuffer(tmp_buffer_bytes, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << tmp_buffer_bytes << "] for kernel[" << GetName()
                   << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    BufferDescGuard __tmp_buffer_guard(&tmp_buffer_desc, [this](BufferDesc* buffer) -> void {
        GetCudaDevice()->FreeTmpBuffer(buffer);
    });
    auto tmp_buffer = tmp_buffer_desc.addr;

    auto boxes = ctx->GetInput<TensorImpl>(0);
    auto scores = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    uint32_t max_output_boxes_per_class = 0;
    if (ctx->GetInputCount() >= 3) {
        auto status = ctx->GetInput<TensorImpl>(2)->CopyToHost(&max_output_boxes_per_class);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy max output boxes failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }
    uint32_t num_boxes = boxes->GetShape().GetDim(1);
    max_output_boxes_per_class = std::min(max_output_boxes_per_class, num_boxes);

    float iou_threshold = 0.f;
    if (ctx->GetInputCount() >= 4) {
        auto status = ctx->GetInput<TensorImpl>(3)->CopyToHost(&iou_threshold);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy iou threshold failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }

    float score_threshold = -FLT_MAX;
    if (ctx->GetInputCount() >= 5) {
        auto status = ctx->GetInput<TensorImpl>(4)->CopyToHost(&score_threshold);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "Copy score threshold failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }
    }

    int device_id = GetCudaDevice()->GetDeviceId();
    status = PPLCUDANMSForwardImp(GetStream(), &boxes->GetShape(), boxes->GetBufferPtr(), &scores->GetShape(),
                                  scores->GetBufferPtr(), &output->GetShape(), output->GetBufferPtr<int64_t>(),
                                  tmp_buffer, tmp_buffer_bytes, device_id, param_->center_point_box,
                                  max_output_boxes_per_class, iou_threshold, score_threshold);

    return status;
}

}}} // namespace ppl::nn::cuda
