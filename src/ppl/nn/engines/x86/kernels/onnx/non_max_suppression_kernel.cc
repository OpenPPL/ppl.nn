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

#include "ppl/nn/engines/x86/kernels/onnx/non_max_suppression_kernel.h"
#include "ppl/kernel/x86/fp32/nms.h"
#include <float.h>

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode NonMaxSuppressionKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_X86_REQUIRED_INPUT(boxes, 0);
    PPLNN_X86_REQUIRED_INPUT(scores, 1);
    PPLNN_X86_OPTIONAL_INPUT(max_output_boxes_per_class_tensor, 2);
    PPLNN_X86_OPTIONAL_INPUT(iou_threshold_tensor, 3);
    PPLNN_X86_OPTIONAL_INPUT(score_threshold_tensor, 4);
    PPLNN_X86_REQUIRED_OUTPUT(output, 0);
    
    const int64_t max_output_boxes_per_class = max_output_boxes_per_class_tensor ? max_output_boxes_per_class_tensor->GetBufferPtr<int64_t>()[0] : 0;
    const float iou_threshold = iou_threshold_tensor ? (iou_threshold_tensor->GetBufferPtr<float>())[0] : 0;
    const float score_threshold = score_threshold_tensor ? (score_threshold_tensor->GetBufferPtr<float>())[0] : -FLT_MAX;

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [boxes]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(boxes);
    PPLNN_X86_DEBUG_TRACE("Input [scores]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(scores);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("center_point_box: %d\n", param_->center_point_box);
    PPLNN_X86_DEBUG_TRACE("max_output_boxes_per_class: %ld\n", max_output_boxes_per_class);
    PPLNN_X86_DEBUG_TRACE("iou_threshold: %f\n", iou_threshold);
    PPLNN_X86_DEBUG_TRACE("score_threshold: %f\n", score_threshold);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    int64_t real_num_boxes_output = 0;

    auto ret = kernel::x86::nms_ndarray_fp32(boxes->GetBufferPtr<const float>(), scores->GetBufferPtr<const float>(),
                                             boxes->GetShape().GetDim(1), boxes->GetShape().GetDim(0),
                                             scores->GetShape().GetDim(1), param_->center_point_box != 0,
                                             max_output_boxes_per_class, iou_threshold, score_threshold,
                                             output->GetBufferPtr<int64_t>(), &real_num_boxes_output);
    if (ret != ppl::common::RC_SUCCESS) {
        ctx->GetOutput<TensorImpl>(0)->GetShape().Reshape({0, 3});
        return ret;
    }

    ctx->GetOutput<TensorImpl>(0)->GetShape().Reshape({real_num_boxes_output, 3});
    // TODO: this will cause output data shape changed according to result, but never exceed max output shape
    PPLNN_X86_DEBUG_TRACE("Output [output] after forward:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::x86
