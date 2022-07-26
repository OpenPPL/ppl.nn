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

#include "ppl/nn/engines/cuda/kernels/onnx/gemm_kernel.h"
#include "ppl/common/destructor.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/gemm/gemm.h"

namespace ppl { namespace nn { namespace cuda {

bool GemmKernel::CanDoExecute(const KernelExecContext& ctx) const {
    const TensorShape& input = *ctx.GetInput<TensorImpl>(0)->GetShape();
    const TensorShape& weight = *ctx.GetInput<TensorImpl>(1)->GetShape();
    if (input.CalcBytesIncludingPadding() == 0) {
        return false;
    }
    if (input.GetDim(1) != weight.GetDim(1)) {
        return false;
    }

    return true;
}

uint64_t GemmKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    return 0;
}


ppl::common::RetCode GemmKernel::DoExecute(KernelExecContext* ctx) {
    conv_param_t temp_conv_param;
    fuse_param_t temp_fuse_param;

    auto input = ctx->GetInput<TensorImpl>(0);
    auto weight = ctx->GetInput<TensorImpl>(1);
    auto output = ctx->GetOutput<TensorImpl>(0);

    const TensorShape& shape_in0 = *input->GetShape();
    const TensorShape& shape_in1 = *weight->GetShape();
    // const TensorShape& shape_out = *output->GetShape();

    temp_conv_param.in_num  = param_->param.transA ? shape_in0.GetDim(1) : shape_in0.GetDim(0);
    temp_conv_param.num_chl = param_->param.transB ? shape_in1.GetDim(1) : shape_in1.GetDim(0);
    temp_conv_param.num_flt = param_->param.transB ? shape_in1.GetDim(0) : shape_in1.GetDim(1);
    temp_conv_param.in_height = 1;
    temp_conv_param.in_width = 1;
    temp_conv_param.flt_height = 1;
    temp_conv_param.flt_width = 1;
    temp_conv_param.out_height = 1;
    temp_conv_param.out_width = 1;
    temp_conv_param.pad_height = 1;
    temp_conv_param.pad_width = 1;
    temp_conv_param.stride_height = 1;
    temp_conv_param.stride_width = 1;
    temp_conv_param.hole_height = 1;
    temp_conv_param.hole_width = 1;
    temp_conv_param.num_grp = 1;

    ConvertToForwardFuseParam(ctx, GetCudaDevice(), param_->extra_param.fuse_info, temp_fuse_param);

    struct algo_param_t algo_param;
    algo_param = param_->extra_param.algo_info;

    BufferDesc tmp_buffer_desc;
    auto tmp_buffer_bytes = PPLGemmCUDAGetRuntimeBufSize(&shape_in0, temp_conv_param, algo_param.splitk,
            algo_param.splitf, param_->param.transA);

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

    // convert filter only if the filter tensor is an output of another kernel
    BufferDesc weight_buffer;
    auto newshape = *weight->GetShape();
    if (!param_->extra_param.is_initializer_weight) {
        auto align_size = 8;
        newshape.SetDim(0, (newshape.GetDim(0) + align_size - 1) / align_size * align_size);

        auto status = GetCudaDevice()->Realloc(newshape, &weight_buffer);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "alloc buffer for constant failed: " << GetRetCodeStr(status);
            return status;
        }
        auto stream = GetStream();
        PPLCUDAGemmModifyWeights(stream, &newshape, weight->GetBufferPtr(), weight_buffer.addr, &param_->param);
    }
    ppl::common::Destructor __tmp_buffer_guard__([this, &weight_buffer]() -> void {
        GetCudaDevice()->Free(&weight_buffer);
    });

    TensorShape bias_shape;
    void* bias = nullptr;
    if (ctx->GetInputCount() >= 3) {
        bias_shape = *ctx->GetInput<TensorImpl>(2)->GetShape();
        bias = ctx->GetInput<TensorImpl>(2)->GetBufferPtr();
    }

    auto stream = GetStream();
    int device_id = GetDeviceId();
    CUDAModule* module = static_cast<CUDAModule*>(this->GetCommonParam()->module);

    if (shape_in0.GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        status = PPLCUDAGemmForwardImp(device_id, stream, module, input->GetShape(), input->GetBufferPtr(),
                                   weight->GetShape(), weight->GetBufferPtr(), bias,
                                   output->GetShape(), output->GetBufferPtr(),
                                   param_->param, tmp_buffer, temp_fuse_param, param_->extra_param.algo_info);
    } else if (shape_in0.GetDataType() == ppl::common::DATATYPE_INT8) {
        quant_param_t temp_quant_param;
        auto input_quant = GetCommonParam()->cuda_tensor_info->at(input->GetEdge()->GetId());
        auto output_quant = GetCommonParam()->cuda_tensor_info->at(output->GetEdge()->GetId());
        auto input_scale = input_quant.scale[0];
        auto output_scale = output_quant.scale[0];
        auto d_weight_scale = ctx->GetInput<TensorImpl>(ctx->GetInputCount() - 1)->GetBufferPtr();
        temp_quant_param.in_scale = input_scale;
        temp_quant_param.out_scale = 1 / output_scale;
        temp_quant_param.d_flt_scale = d_weight_scale;
        if (temp_fuse_param.has_elt) {
            auto tps = param_->extra_param.fuse_info.types;
            auto ret = std::find(tps.begin(), tps.end(), "Add");
            if (ret == tps.end())
                LOG(ERROR) << "fuse_info types error: no add op";
            int id = ret - tps.begin();
            auto elt_index = param_->extra_param.fuse_info.input_ind[id];
            auto elt = ctx->GetInput<TensorImpl>(elt_index);
            auto elt_quant = GetCommonParam()->cuda_tensor_info->at(elt->GetEdge()->GetId());
            temp_quant_param.pre_scale = elt_quant.scale[0];
        }
        if (param_->extra_param.fuse_info.channel_offset >= 0) {
            temp_quant_param.out_scale =
                1 / GetCommonParam()->cuda_tensor_info->at(param_->extra_param.fuse_info.concat_edge_id).scale[0];
        }

        status = PPLCUDAGemmForwardImpInt8(device_id, stream, module, input->GetShape(), input->GetBufferPtr(),
                                           weight->GetShape(), weight->GetBufferPtr(), bias,
                                           output->GetShape(), output->GetBufferPtr(),
                                           param_->param, tmp_buffer, temp_quant_param,
                                           temp_fuse_param, param_->extra_param.algo_info);
    }

    return status;
}

}}} // namespace ppl::nn::cuda
