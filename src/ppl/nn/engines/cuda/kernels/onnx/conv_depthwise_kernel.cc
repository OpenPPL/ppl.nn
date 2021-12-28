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

#include "ppl/nn/engines/cuda/kernels/onnx/conv_depthwise_kernel.h"

#include <cuda_fp16.h>
#include "cuda_runtime.h" 

namespace ppl { namespace nn { namespace cuda {

ppl::common::RetCode ConvDepthwiseKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != ppl::common::RC_SUCCESS) {
        return status;
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        auto device = GetCudaDevice();
        auto concat_edge_id = param_->extra_param.fuse_info.concat_edge_id;
        if (param_->extra_param.fuse_info.channel_offset >= 0) {
            auto edge2buffer = device->GetEdge2Buffer();
            auto ptr = edge2buffer->find(concat_edge_id);
            if (ptr == edge2buffer->end()) {
                BufferDesc buffer;
                auto concat_shape = tensor->GetShape();
                concat_shape.SetDim(1, param_->extra_param.fuse_info.channel_size);
                status = device->Realloc(concat_shape, &buffer);
                if (status != RC_SUCCESS) {
                    LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
                    return status;
                }
                tensor->SetBuffer(buffer);
                edge2buffer->emplace(concat_edge_id, std::move(buffer));
            } else {
                tensor->SetBuffer(ptr->second);
            }
        } else {
            status = tensor->ReallocBuffer();
        }
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed.";
            return status;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode ConvDepthwiseKernel::DoExecute(KernelExecContext* ctx) {
    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;

    auto shape_in0 = ctx->GetInput<TensorImpl>(0)->GetShape();
    auto shape_in1 = ctx->GetInput<TensorImpl>(1)->GetShape();
    auto shape_out = ctx->GetOutput<TensorImpl>(0)->GetShape();

    auto input_id0 = ctx->GetInput<TensorImpl>(0)->GetEdge()->GetId();
    auto input_id1 = ctx->GetInput<TensorImpl>(1)->GetEdge()->GetId();
    auto input_quant0 = GetCommonParam()->cuda_tensor_info->at(input_id0);
    auto input_quant1 = GetCommonParam()->cuda_tensor_info->at(input_id1);
    auto output_id = ctx->GetOutput<TensorImpl>(0)->GetEdge()->GetId();
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output_id);
    auto d_weight_scale = ctx->GetInput<TensorImpl>(ctx->GetInputCount() - 1)->GetBufferPtr();
    // auto paddingc = (shape_in1.GetDim(0) + 15) / 16 * 16;
    // cudaMalloc((void**)&d_weight_scale, paddingc*sizeof(float));
    // cudaMemcpy(d_weight_scale, input_quant1.scale.data(), paddingc*sizeof(float), cudaMemcpyHostToDevice);


    ConvertToForwardConvParam(shape_in0, shape_in1, shape_out, param_->param, temp_conv_param);
    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    if (!param_->extra_param.algo_info.is_initializer_weight) {
        auto newshape = shape_in1;
        newshape.SetDim(0, (newshape.GetDim(0) + 15) / 16 * 16);

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDADepthwiseConvertFilter(stream, ctx->GetInput<TensorImpl>(1)->GetBufferPtr(), weight_buffer.addr,
                                      temp_conv_param, shape_out.GetDataType());
    }
    BufferDescGuard __tmp_buffer_guard__(&weight_buffer, [this](BufferDesc* buffer) {
        GetDevice()->Free(buffer);
    });


    auto stream = GetStream();
    PPLCUDADepthwiseForwardCudaImp(
        stream, param_->extra_param.algo_info.kid, ctx->GetInput<TensorImpl>(0)->GetBufferPtr(),
        param_->extra_param.algo_info.is_initializer_weight ? ctx->GetInput<TensorImpl>(1)->GetBufferPtr()
                                                            : weight_buffer.addr,
        param_->param.bias_term ? ctx->GetInput<TensorImpl>(2)->GetBufferPtr() : nullptr, temp_conv_param,
        temp_fuse_param, ctx->GetOutput<TensorImpl>(0)->GetBufferPtr(), shape_out.GetDataType(), input_quant0.scale[0], (float*)d_weight_scale, output_quant.scale[0]);


    LOG(DEBUG) << "Excute Depthwise conv with kernel id:" << param_->extra_param.algo_info.kid;
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
