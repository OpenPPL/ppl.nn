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

#include "ppl/nn/engines/x86/kernel.h"
using namespace std;
using namespace ppl::common;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
#include "ppl/nn/utils/cpu_timing_guard.h"
#endif

namespace ppl { namespace nn { namespace x86 {

RetCode X86Kernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "reshape kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        status = tensor->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

bool X86Kernel::CanDoExecute(const KernelExecContext& ctx) const {
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape().GetBytesIncludingPadding() == 0) {
            return false;
        }
    }
    return true;
}

RetCode X86Kernel::Execute(KernelExecContext* ctx) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    utils::CpuTimingGuard __timing_guard__(&begin_ts_, &end_ts_, ctx->IsProfilingEnabled());
#endif

    auto status = BeforeExecute(ctx);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "BeforeExecute() of kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    if (CanDoExecute(*ctx)) {
        status = DoExecute(ctx);
    }

    return status;
}

}}} // namespace ppl::nn::x86
