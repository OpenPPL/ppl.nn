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

#include "ppl/nn/engines/common/pmx/converter_kernel.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace pmx {

RetCode ConverterKernel::DoExecute(KernelExecContext* ctx) {
    if (ctx->GetInputCount() != ctx->GetOutputCount()) {
        LOG(ERROR) << "input count [" << ctx->GetInputCount() << "] != output count [" << ctx->GetOutputCount() << "]";
        return RC_INVALID_VALUE;
    }

    auto kernel_device = GetDevice();
    utils::GenericCpuDevice tmp_cpu_device;
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto src = ctx->GetInput<TensorImpl>(i);
        auto dst = ctx->GetOutput<TensorImpl>(i);

        dst->SetDevice(kernel_device);
        *dst->GetShape() = *src->GetShape();
        dst->GetShape()->SetDataFormat(DATAFORMAT_NDARRAY);

        auto status = utils::CopyTensorBuffer(*src, dst, &tmp_cpu_device);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy tensor from [" << src->GetName() << "] to [" << dst->GetName()
                       << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::pmx
