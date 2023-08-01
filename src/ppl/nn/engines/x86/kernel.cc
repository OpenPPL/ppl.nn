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

#include <fstream>
#include <cctype>

#include "ppl/nn/engines/x86/kernel.h"
using namespace ppl::common;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
#include "ppl/common/destructor.h"
#include <chrono>
#endif

namespace ppl { namespace nn { namespace x86 {

RetCode X86Kernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "reshape kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

bool X86Kernel::CanDoExecute(const KernelExecContext& ctx) const {
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape()->CalcBytesIncludingPadding() == 0) {
            return false;
        }
    }
    return true;
}

RetCode X86Kernel::DumpOutputTensors(KernelExecContext* ctx) {
    auto get_dim_str = [](const TensorShape* shape) {
        if (shape->IsScalar()) {
            return std::string("scalar");
        }

        if (shape->GetRealDimCount() == 0) {
            return std::string("none");
        }

        std::string res = std::to_string(shape->GetDim(0));
        for (uint32_t i = 1; i < shape->GetDimCount(); ++i) {
            res += "_" + std::to_string(shape->GetDim(i));
        }

        return res;
    };

    auto get_dt_str = [](const TensorShape* shape) {
        std::string res = GetDataTypeStr(shape->GetDataType());
        std::transform(res.begin(), res.end(), res.begin(),
            [](const char c) { return std::tolower(c); });
        return res;
    };

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        auto shape = tensor->GetShape();
        std::string tensor_name = tensor->GetName();
        for (size_t s = 0; s < tensor_name.length(); ++s) {
            if (tensor_name[s] == '/')
                tensor_name[s] = '.';
        }
        const std::string out_file_name = engine_config_->debug_data_dir
                                + "/pplnn_dbg_tensor-" + tensor_name
                                + "-" + get_dim_str(shape)
                                + "-" + get_dt_str(shape)
                                + ".dat";
        std::ofstream ofs(out_file_name, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
        if (!ofs.is_open()) {
            LOG(ERROR) << "open output file[" << out_file_name << "] failed";
            return RC_OTHER_ERROR;
        }

        std::vector<char> cvt_buffer;
        auto output_buffer_ptr = tensor->GetBufferPtr<char>();
        auto bytes = shape->CalcBytesExcludingPadding();

        if (shape->GetDataFormat() != DATAFORMAT_NDARRAY) {
            TensorShape dst_desc = *shape;
            dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);

            cvt_buffer.resize(bytes);
            auto status = tensor->ConvertToHost(cvt_buffer.data(), dst_desc);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "convert data of tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
                return status;
            }

            output_buffer_ptr = cvt_buffer.data();
        }

        ofs.write(output_buffer_ptr, bytes);
    }

    return RC_SUCCESS;
}

RetCode X86Kernel::Execute(KernelExecContext* ctx) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    begin_ts_ = std::chrono::system_clock::now();
    auto is_profiling_enabled = ctx->IsProfilingEnabled();
    ppl::common::Destructor __timing_guard__([is_profiling_enabled, this]() -> void {
        if (is_profiling_enabled) {
            end_ts_ = std::chrono::system_clock::now();
        }
    });
#endif

    auto status = BeforeExecute(ctx);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "BeforeExecute() of kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }

    if (CanDoExecute(*ctx)) {
        status = DoExecute(ctx);
    } else {
        // TODO: discard the boundary case of conv/pool/deconv, and try to remove this thing
        for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
            auto tensor = ctx->GetOutput<TensorImpl>(i);
            tensor->SetDevice(GetX86Device());
            status = tensor->ReallocBuffer();
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
                return status;
            }
        }
    }

    if (engine_config_->enable_tensor_debug) {
        status = DumpOutputTensors(ctx);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "DumpOutputTensors() of kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return status;
}

}}} // namespace ppl::nn::x86
