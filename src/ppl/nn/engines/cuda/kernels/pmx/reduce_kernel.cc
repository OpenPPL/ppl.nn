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

#include "ppl/nn/engines/cuda/kernels/pmx/reduce_kernel.h"
#include "ppl/common/destructor.h"

#include <numeric>

#include "cudakernel/reduce/reduce.h"

namespace ppl { namespace nn { namespace cuda {

uint64_t ReduceKernel::CalcTmpBufferSize(const KernelExecContext& ctx) const {
    auto y = ctx.GetOutput<TensorImpl>(0);
    if (y->GetShape()->GetDataType() == ppl::common::DATATYPE_INT8) {
        return sizeof(float) * y->GetShape()->CalcElementsExcludingPadding();
    } else {
        return 0;
    }
}

ppl::common::RetCode ReduceKernel::DoExecute(KernelExecContext* ctx) {
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

    auto input = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    ReduceParam param;
    const TensorShape& input_shape = *input->GetShape();
    uint32_t n_outer = 1, n_reduce = 1, n_inner = 1;

    switch (param_->type)
    {
    case ReduceMax:
        param = ReduceMax;
        break;
    case ReduceMean:
        param = ReduceMean;
        break;
    case ReduceMin:
        param = ReduceMin;
        break;
    case ReduceSum:
        param = ReduceSum;
        break;
    case ReduceProd:
        param = ReduceProd;
        break;
    default:
        return ppl::common::RC_UNSUPPORTED;
    }

    const uint32_t dim_count = input_shape.GetDimCount();
    if (param_->axes.empty()) { // empty axes means reduce all dims
        n_reduce =
            accumulate(input_shape.GetDims(), input_shape.GetDims() + dim_count, n_reduce, std::multiplies<uint32_t>());
    } else {
        std::vector<uint32_t> real_axis(param_->axes.size());

        for (uint32_t i = 0; i < param_->axes.size(); ++i) {
            real_axis[i] = (param_->axes[i] + dim_count) % dim_count;
            if (i > 0 && real_axis[i] != real_axis[i - 1] + 1) {
                return ppl::common::RC_UNSUPPORTED;
            }
            n_reduce *= input_shape.GetDim(real_axis[i]);
        }
        n_outer = accumulate(input_shape.GetDims(), input_shape.GetDims() + real_axis[0], n_outer,
                             std::multiplies<uint32_t>());
        n_inner = accumulate(input_shape.GetDims() + real_axis[param_->axes.size() - 1] + 1,
                             input_shape.GetDims() + dim_count, n_inner, std::multiplies<uint32_t>());
    }

    auto input_quant = GetCommonParam()->cuda_tensor_info->at(input->GetEdge()->GetId());
    auto output_quant = GetCommonParam()->cuda_tensor_info->at(output->GetEdge()->GetId());
    QuantParamCuda qparam(input_quant.zero_point[0], output_quant.zero_point[0], input_quant.scale[0], output_quant.scale[0]);
    PPLReduceDimDes des(n_inner, n_reduce, n_outer);
    status =
        PPLCUDAReduceForwardImp(GetStream(), param, des, input->GetShape(), input->GetBufferPtr(), output->GetShape(),
                                output->GetBufferPtr(), tmp_buffer, &qparam);
    return status;
}

}}} // namespace ppl::nn::cuda
