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

#include "ppl/nn/engines/cuda/kernels/onnx/concat_kernel.h"

#include "cudakernel/memory/concat.h"
#include "ppl/nn/engines/cuda/macros.h"

namespace ppl { namespace nn { namespace cuda {

bool ConcatKernel::CanDoExecute(const KernelExecContext& ctx) const {
    bool all_empty = true;
    for (uint32_t i = 0; i < ctx.GetInputCount(); i++) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor) {
            return false;
        }
        if (tensor->GetShape().GetBytesIncludingPadding() != 0) {
            all_empty = false;
        }
    }
    return !all_empty;
}

ppl::common::RetCode ConcatKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != ppl::common::RC_SUCCESS) {
        return status;
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        auto device = GetCudaDevice();
        auto edge2buffer = device->GetEdge2Buffer();
        auto ptr = edge2buffer->find(GetNode()->GetOutput(i));
        if (ptr != edge2buffer->end()) {
            tensor->SetBuffer(ptr->second, nullptr, true);
            edge2buffer->erase(GetNode()->GetOutput(i));
        } else {
            status = tensor->GetBufferPtr() != nullptr ? ppl::common::RC_SUCCESS : tensor->ReallocBuffer();
        }
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed.";
            return status;
        }
    }

    return ppl::common::RC_SUCCESS;
}

ppl::common::RetCode ConcatKernel::DoExecute(KernelExecContext* ctx) {
    auto output = ctx->GetOutput<TensorImpl>(0);
    int32_t dim_count = output->GetShape().GetDimCount();

    std::vector<std::vector<int>> src_dims(ctx->GetInputCount());
    std::vector<void*> src_list(ctx->GetInputCount());
    std::vector<std::vector<int>> src_padded_dims(ctx->GetInputCount());

    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto input = ctx->GetInput<TensorImpl>(i);
        src_list[i] = input->GetBufferPtr();
        auto shape = input->GetShape();
        for (int32_t it = 0; it < dim_count; ++it) {
            src_dims[i].push_back(shape.GetDim(it));
            src_padded_dims[i].push_back(shape.GetDim(it) + shape.GetPadding0(it) + shape.GetPadding1(it));
        }
    }

    typedef int32_t* pint;
    std::unique_ptr<pint[]> input_dims(new pint[ctx->GetInputCount()]);
    std::unique_ptr<pint[]> input_padded_dims(new pint[ctx->GetInputCount()]);
    for (uint32_t it = 0; it < ctx->GetInputCount(); ++it) {
        input_dims[it] = src_dims[it].data();
        input_padded_dims[it] = src_padded_dims[it].data();
    }
    int mask = param_->extra_param.mask;
    int axis = param_->param.axis;
    if (axis < 0)
        axis += dim_count;
    ppl::common::RetCode status = PPLCUDAConcatForwardImp(
        GetStream(), axis, ctx->GetInputCount(), (int**)input_dims.get(), (int**)input_padded_dims.get(),
        (const void**)src_list.data(), &output->GetShape(), output->GetBufferPtr(), mask);

    return status;
}

}}} // namespace ppl::nn::cuda
