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

#include "ppl/nn/engines/cuda/kernel.h"

#include <chrono>
#include <memory>
#include "ppl/common/allocator.h"
#include "ppl/nn/common/logger.h"
using namespace ppl::common;

#ifdef CUDA_DUMP_OUTPUT_TENSOR
#include <fstream>
#include <string.h>
#include <string>
#include <functional>
#include <algorithm>
using namespace std;
#endif

namespace ppl { namespace nn { namespace cuda {

#ifdef CUDA_DUMP_OUTPUT_TENSOR
ppl::common::RetCode DumpOutputTensors(KernelExecContext* ctx, const std::string &debug_data_dir) {
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
        const std::string out_file_name = debug_data_dir
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

        TensorShape dst_desc = *shape;
        dst_desc.SetDataFormat(DATAFORMAT_NDARRAY);

        cvt_buffer.resize(bytes);
        auto status = tensor->ConvertToHost(cvt_buffer.data(), dst_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "convert data of tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
        output_buffer_ptr = cvt_buffer.data();

        ofs.write(output_buffer_ptr, bytes);
    }

    return RC_SUCCESS;
}
#endif

CudaKernel::~CudaKernel() {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    if (exec_begin_event_) {
        cudaEventDestroy(exec_begin_event_);
    }
    if (exec_end_event_) {
        cudaEventDestroy(exec_end_event_);
    }
#endif
}

RetCode CudaKernel::Init() {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    auto err = cudaEventCreate(&exec_begin_event_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaEventCreate failed: " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }

    err = cudaEventCreate(&exec_end_event_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaEventCreate failed: " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
#endif

    return RC_SUCCESS;
}

RetCode CudaKernel::BeforeExecute(KernelExecContext* ctx) {
    auto status = Reshape(ctx);
    if (status != RC_SUCCESS) {
        return status;
    }

    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        tensor->SetDevice(GetCudaDevice());
        status = tensor->ReallocBuffer();
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "ReallocBuffer for tensor[" << tensor->GetName() << "] failed: " << GetRetCodeStr(status);
            return status;
        }
    }

    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
class CudaTimingGuard final {
public:
    CudaTimingGuard(cudaStream_t stream, cudaEvent_t* begin_event, cudaEvent_t* end_event, bool is_profiling_enabled)
        : is_profiling_enabled_(is_profiling_enabled), end_event_(end_event), stream_(stream) {
        if (is_profiling_enabled) {
            cudaEventRecord(*begin_event, stream);
            stream_ = stream;
        }
    }
    ~CudaTimingGuard() {
        if (is_profiling_enabled_) {
            cudaEventRecord(*end_event_, stream_);
        }
    }

private:
    bool is_profiling_enabled_;
    cudaEvent_t* end_event_;
    cudaStream_t stream_;
};
#endif

bool CudaKernel::CanDoExecute(const KernelExecContext& ctx) const {
    for (uint32_t i = 0; i < ctx.GetInputCount(); ++i) {
        auto tensor = ctx.GetInput<TensorImpl>(i);
        if (!tensor || tensor->GetShape()->CalcBytesIncludingPadding() == 0) {
            LOG(WARNING) << "Cannot execute [" << GetName() << "]";
            if(!tensor) {
                LOG(DEBUG) << "Tensor(" << i << ") is NULL";
            } else {
                if(tensor->GetShape()->CalcBytesIncludingPadding() == 0) {
                    LOG(DEBUG) << "Tensor(" << i << ")[" << tensor->GetName() << "] is EMPTY";
                }
            }
            return false;
        }
    }
    return true;
}

RetCode CudaKernel::Execute(KernelExecContext* ctx) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    CudaTimingGuard __timing_guard__(GetCudaDevice()->GetStream(), &exec_begin_event_, &exec_end_event_,
                                     ctx->IsProfilingEnabled());
#endif

    auto status = BeforeExecute(ctx);
    if (status != RC_SUCCESS) {
        return status;
    }
    
#ifndef NDEBUG
    uint32_t total_size = 0;
    LOG(INFO) << "Before [" << GetName() << "]";
    LOG(DEBUG) << "Input Tensor Info:";
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto tensor = ctx->GetInput<TensorImpl>(i);
        if (!tensor) {
            continue;
        }
        auto tensor_size = tensor->GetShape()->CalcBytesIncludingPadding();
        auto tensor_dim_count = tensor->GetShape()->GetDimCount();
        std::string tensor_dims = "";
        for (uint32_t j = 0; j < tensor_dim_count; ++j) {
            tensor_dims += std::to_string(tensor->GetShape()->GetDim(j)) + " ";
        }
        LOG(DEBUG) << "Input Tensor(" << i << ") Name [" << tensor->GetName() << "]";
        LOG(DEBUG) << " |-Buffer Ptr: " << tensor->GetBufferPtr();
        LOG(DEBUG) << " |-Size: " << tensor_size;
        LOG(DEBUG) << " |-DataType: " << ppl::common::GetDataTypeStr(tensor->GetShape()->GetDataType());
        LOG(DEBUG) << " |-DataFormat: " << ppl::common::GetDataFormatStr(tensor->GetShape()->GetDataFormat());
        LOG(DEBUG) << " |-Dimcount: " << tensor_dim_count;
        LOG(DEBUG) << " |-Dim(s): " << tensor_dims;
        const int64_t elem_count = tensor->GetShape()->CalcElementsExcludingPadding();
        if (tensor->GetShape()->GetDataType() == ppl::common::DATATYPE_INT64 && elem_count <= 10) {
            std::vector<int64_t> vals(elem_count, -7890);
            if (ppl::common::RC_SUCCESS != tensor->CopyToHost(vals.data())) {
                LOG(ERROR) << "[" << tensor->GetName() << "] CopyToHost FAILED";
            } else {
                std::string val_str = "";
                for (uint32_t j = 0; j < elem_count; ++j) {
                    val_str += std::to_string(vals[j]) + " ";
                }
                LOG(DEBUG) << " |-Value(s): " << val_str;
            }
        }
        total_size += tensor_size;
    }
    LOG(DEBUG) << "Output Tensor Info before execute:";
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        auto tensor_size = tensor->GetShape()->CalcBytesIncludingPadding();
        auto tensor_dim_count = tensor->GetShape()->GetDimCount();
        std::string tensor_dims = "";
        for (uint32_t j = 0; j < tensor_dim_count; ++j) {
            tensor_dims += std::to_string(tensor->GetShape()->GetDim(j)) + " ";
        }
        LOG(DEBUG) << "Output Tensor(" << i << ") Name [" << tensor->GetName() << "]";
        LOG(DEBUG) << " |-Buffer Ptr: " << tensor->GetBufferPtr();
        LOG(DEBUG) << " |-Size: " << tensor_size;
        LOG(DEBUG) << " |-DataType: " << ppl::common::GetDataTypeStr(tensor->GetShape()->GetDataType());
        LOG(DEBUG) << " |-DataFormat: " << ppl::common::GetDataFormatStr(tensor->GetShape()->GetDataFormat());
        LOG(DEBUG) << " |-Dimcount: " << tensor_dim_count;
        LOG(DEBUG) << " |-Dims: " << tensor_dims;
    }

    auto run_begin_ts = std::chrono::system_clock::now();
#endif

    if (CanDoExecute(*ctx)) {
        status = DoExecute(ctx);
    }

#ifndef NDEBUG
    auto run_end_ts = std::chrono::system_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(run_end_ts - run_begin_ts);
    LOG(DEBUG) << "Output Tensor Info after execute:";
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        auto tensor_size = tensor->GetShape()->CalcBytesIncludingPadding();
        auto tensor_dim_count = tensor->GetShape()->GetDimCount();
        std::string tensor_dims = "";
        for (uint32_t j = 0; j < tensor_dim_count; ++j) {
            tensor_dims += std::to_string(tensor->GetShape()->GetDim(j)) + " ";
        }
        LOG(DEBUG) << "Output Tensor(" << i << ") Name [" << tensor->GetName() << "]";
        LOG(DEBUG) << " |-Buffer Ptr: " << tensor->GetBufferPtr();
        LOG(DEBUG) << " |-Size: " << tensor_size;
        LOG(DEBUG) << " |-DataType: " << ppl::common::GetDataTypeStr(tensor->GetShape()->GetDataType());
        LOG(DEBUG) << " |-DataFormat: " << ppl::common::GetDataFormatStr(tensor->GetShape()->GetDataFormat());
        LOG(DEBUG) << " |-Dimcount: " << tensor_dim_count;
        LOG(DEBUG) << " |-Dims: " << tensor_dims;
        total_size += tensor_size;
    }
    LOG(INFO) << "After [" << GetName() << "], Time(ms): " << (float)diff.count() << " Mem(bytes) " << total_size;
#endif

#ifdef CUDA_DUMP_OUTPUT_TENSOR
    std::string dump_dir = ".";
    status = DumpOutputTensors(ctx, dump_dir);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "DumpOutputTensors() of kernel[" << GetName() << "] failed: " << GetRetCodeStr(status);
        return status;
    }
#endif

    return status;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
void CudaKernel::GetProfilingInfo(InternalProfilingInfo* info) const {
    cudaEventSynchronize(exec_end_event_);
    float ms = 0.0;
    cudaEventElapsedTime(&ms, exec_begin_event_, exec_end_event_);
    info->exec_microseconds = static_cast<uint64_t>(ms * 1000);
}
#endif

}}} // namespace ppl::nn::cuda
