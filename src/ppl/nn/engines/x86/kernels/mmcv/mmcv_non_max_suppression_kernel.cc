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

#include "ppl/nn/engines/x86/kernels/mmcv/mmcv_non_max_suppression_kernel.h"

#include "ppl/kernel/x86/fp32/mmcv_nms.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode MMCVNonMaxSuppressionKernel::DoExecute(KernelExecContext* ctx) {
    auto boxes = ctx->GetInput<TensorImpl>(0);
    auto scores = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [boxes]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(boxes);
    PPLNN_X86_DEBUG_TRACE("Input [scores]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(scores);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("iou_threshold: %f\n", param_->iou_threshold);
    PPLNN_X86_DEBUG_TRACE("offset: %ld\n", param_->offset);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    int64_t real_num_boxes_output = 0;
    auto ret = kernel::x86::mmcv_nms_ndarray_fp32(
        boxes->GetBufferPtr<const float>(), scores->GetBufferPtr<const float>(), boxes->GetShape().GetDim(0),
        param_->iou_threshold, param_->offset, output->GetBufferPtr<int64_t>(), &real_num_boxes_output);
    if (ret != ppl::common::RC_SUCCESS) {
        ctx->GetOutput<TensorImpl>(0)->GetShape().Reshape({0});
        return ret;
    }

    ctx->GetOutput<TensorImpl>(0)->GetShape().Reshape({real_num_boxes_output});
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
