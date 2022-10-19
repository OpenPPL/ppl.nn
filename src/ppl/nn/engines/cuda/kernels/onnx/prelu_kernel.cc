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

#include "ppl/nn/engines/cuda/kernels/onnx/prelu_kernel.h"

#include "cudakernel/arithmetic/arithmetic.h"

namespace ppl { namespace nn { namespace cuda {

static void ppl_pad_second_shape(const ppl::nn::TensorShape *tensor_shape0,
                          const ppl::nn::TensorShape *tensor_shape1,
                          ppl::nn::TensorShape *pad_tensor_shape) {
    int max_dims = tensor_shape0->GetDimCount();
    pad_tensor_shape->SetDimCount(max_dims);
    // pad 1 to shape_min_pad's higher dim
    int offset = max_dims - tensor_shape1->GetDimCount();
    for (int i = 0; i < offset; i++) {
        pad_tensor_shape->SetDim(i, 1);
    }
    for (int i = offset; i < max_dims; i++) {
        pad_tensor_shape->SetDim(i, tensor_shape1->GetDim(i - offset));
    }
}

ppl::common::RetCode PReluKernel::DoExecute(KernelExecContext* ctx) {
    auto input0 = ctx->GetInput<TensorImpl>(0);
    auto input1 = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);
    auto input_id0 = input0->GetEdge()->GetId();
    auto input_id1 = input1->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);
    auto input_quant1 = GetCommonParam()->cuda_tensor_info->at(input_id1);
    auto output_id = output->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);

    // pad second input shape to enable broadcast
    auto input1_shape = *input1->GetShape();
    if (input1_shape.GetDimCount() > 1 && input1_shape.GetDimCount() < input0->GetShape()->GetDimCount())
        ppl_pad_second_shape(input0->GetShape(), input1->GetShape(), &input1_shape);
    ppl::common::RetCode status =
        PPLCUDAArithMeticPReluForwardImp(GetStream(), input0->GetShape(), input0->GetBufferPtr(), &input1_shape,
                                       input1->GetBufferPtr(), output->GetShape(), output->GetBufferPtr(), input_quant0.scale[0], input_quant1.scale[0], output_quant.scale[0]);
    return status;
}

}}} // namespace ppl::nn::cuda
