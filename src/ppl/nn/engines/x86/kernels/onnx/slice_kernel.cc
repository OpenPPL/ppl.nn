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

#include "ppl/nn/engines/x86/kernels/onnx/slice_kernel.h"

#include "ppl/kernel/x86/fp32/slice.h"
#include "ppl/kernel/x86/int64/slice.h"

namespace ppl { namespace nn { namespace x86 {

ppl::common::RetCode SliceKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto output = ctx->GetOutput<TensorImpl>(0);
    const int axes_num = ctx->GetInput<TensorImpl>(1)->GetShape().GetDim(0);

    PPLNN_X86_DEBUG_TRACE("Op: %s\n", GetName().c_str());
    PPLNN_X86_DEBUG_TRACE("Input [data]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(data);
    PPLNN_X86_DEBUG_TRACE("Output [output]:\n");
    PPL_X86_TENSOR_PRINT_DEBUG_MSG(output);
    PPLNN_X86_DEBUG_TRACE("isa: %u\n", GetISA());

    // prepare starts, axes, steps
    auto starts = ctx->GetInput<TensorImpl>(1)->GetBufferPtr<int64_t>();

    const int64_t* axes = nullptr;
    std::vector<int64_t> axes_vec;
    if (ctx->GetInputCount() >= 4) {
        axes = ctx->GetInput<TensorImpl>(3)->GetBufferPtr<int64_t>();
    } else {
        axes_vec.resize(axes_num);
        for (int i = 0; i < axes_num; i++) {
            axes_vec[i] = i;
        }
        axes = axes_vec.data();
    }

    std::vector<int64_t> steps_vec;
    const int64_t* steps = nullptr;
    if (ctx->GetInputCount() >= 5) {
        steps = ctx->GetInput<TensorImpl>(4)->GetBufferPtr<int64_t>();
    } else {
        steps_vec.resize(axes_num, 1);
        steps = steps_vec.data();
    }

    const ppl::common::datatype_t data_type = data->GetShape().GetDataType();
    if (data_type == ppl::common::DATATYPE_FLOAT32) {
        return kernel::x86::slice_ndarray_fp32(&data->GetShape(), &output->GetShape(), data->GetBufferPtr<float>(),
                                               starts, steps, axes, axes_num, output->GetBufferPtr<float>());
    } else if (data_type == ppl::common::DATATYPE_INT64) {
        return kernel::x86::slice_ndarray_int64(&data->GetShape(), &output->GetShape(), data->GetBufferPtr<int64_t>(),
                                                starts, steps, axes, axes_num, output->GetBufferPtr<int64_t>());
    }

    return ppl::common::RC_UNSUPPORTED;
}

}}} // namespace ppl::nn::x86
