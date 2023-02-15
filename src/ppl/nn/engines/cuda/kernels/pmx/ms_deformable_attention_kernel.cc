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

#include "ppl/nn/engines/cuda/kernels/pmx/ms_deformable_attention_kernel.h"
#include "ppl/common/destructor.h"

#include <numeric>

#include "cudakernel/nn/ms_deformable_attention.h"

namespace ppl { namespace nn { namespace cuda {

/* uint64_t MSDeformAttnKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const { */
    /* auto y = ctx.GetOutput<TensorImpl>(0); */
    /* if (y->GetShape()->GetDataType() == ppl::common::DATATYPE_INT8) { */
        /* return sizeof(float) * y->GetShape()->CalcElementsExcludingPadding(); */
    /* } else { */
        /* return 0; */
    /* } */
/* } */

ppl::common::RetCode MSDeformAttnKernel::DoExecute(KernelExecContext* ctx) {

    auto data = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);

    const TensorShape* input_shape = data->GetShape();
    const TensorShape* output_shape = output->GetShape();

    // auto input_quant = GetCommonParam()->cuda_tensor_info->at(input->GetEdge()->GetId());
    // auto output_quant = GetCommonParam()->cuda_tensor_info->at(output->GetEdge()->GetId());
    // QuantKernelParamCuda qparam(input_quant.zero_point[0], output_quant.zero_point[0], input_quant.scale[0], output_quant.scale[0]);

    auto spatial_shapes = ctx->GetInput<TensorImpl>(1);
    auto level_start_index = ctx->GetInput<TensorImpl>(2);
    auto sampling_loc = ctx->GetInput<TensorImpl>(3);
    auto attn_weight = ctx->GetInput<TensorImpl>(4);

    const int batch = input_shape->GetDim(0);
    const int spatial_size = input_shape->GetDim(1);
    const int num_heads = input_shape->GetDim(2);
    const int channels = input_shape->GetDim(3);

    const int num_levels = spatial_shapes->GetShape()->GetDim(0);

    const int num_query = sampling_loc->GetShape()->GetDim(1);
    const int num_point = sampling_loc->GetShape()->GetDim(4);

    const int im2col_step_ = std::min(batch, param_->im2col_step);

    auto status =
        PPLCUDAMSDeformAttnForwardImp(
                GetStream(),
                input_shape,
                output_shape,
                data->GetBufferPtr(),
                spatial_shapes->GetBufferPtr(),
                level_start_index->GetBufferPtr(),
                sampling_loc->GetBufferPtr(),
                attn_weight->GetBufferPtr(),
                output->GetBufferPtr(), 
                batch,
                im2col_step_, 
                spatial_size, 
                num_heads, 
                channels, 
                num_levels, 
                num_query, 
                num_point);
    
    return status;

}

}}} // namespace ppl::nn::cuda
