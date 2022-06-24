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

#include "ppl/nn/engines/riscv/kernel.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/utils/cpu_timing_guard.h"

// #define RISCV_PERLAYER_DEBUG
#ifdef RISCV_PERLAYER_DEBUG
#include <fstream>
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode RiscvKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "reshape kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        tensor->SetDevice(GetRiscvDevice());
        status = tensor->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

bool RiscvKernel::CanDoExecute(const KernelExecContext& ctx) const {
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape()->CalcBytesIncludingPadding() == 0) {
            return false;
        }
    }
    return true;
}

RetCode RiscvKernel::Execute(KernelExecContext* ctx) {
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

#ifdef RISCV_PERLAYER_DEBUG
        for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
            auto tensor = ctx->GetOutput<TensorImpl>(i);
            TensorShape& dst_shape = *(tensor->GetShape());
            auto bytes = dst_shape.CalcBytesIncludingPadding();
            vector<char> buffer(dst_shape.CalcElementsExcludingPadding());

            string shape_out = "-";
            for (int64_t n = 0; n < dst_shape.GetDimCount(); n++) {
                string dim_info = (n == 0) ? to_string(dst_shape.GetDim(n)) : ("_" + to_string(dst_shape.GetDim(n)));
                shape_out += dim_info;
            }

            string out_file_name =
                "pplnn_out-" + GetName() + "-" + ppl::common::GetDataFormatStr(dst_shape.GetDataFormat()) + "-";
            out_file_name += ppl::common::GetDataTypeStr(dst_shape.GetDataType()) + shape_out + ".dat";
            ofstream ofs(out_file_name, ios_base::out | ios_base::binary | ios_base::trunc);
            if (!ofs.is_open()) {
                LOG(ERROR) << "open output file[" << out_file_name << "]";
                return false;
            }
            ofs.write(tensor->GetBufferPtr<char>(), bytes);
        }
#endif
    }

    return status;
}

}}} // namespace ppl::nn::riscv
