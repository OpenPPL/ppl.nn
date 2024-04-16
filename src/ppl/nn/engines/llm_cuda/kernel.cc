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

#include "kernel.h"

#ifdef PPLNN_LLM_CUDA_DUMP_OUTPUT_TENSORS
#include <fstream>
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

#ifdef PPLNN_LLM_CUDA_DUMP_OUTPUT_TENSORS
static RetCode LlmCudaDumpOutputTensors(KernelExecContext* ctx, const std::string &debug_data_dir) {
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
                                + "/pplnn_llm_cuda_dbg_tensor-" + tensor_name
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

RetCode LlmCudaKernel::Init() {
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

void LlmCudaKernel::Destroy() {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    if (exec_begin_event_) {
        auto err = cudaEventDestroy(exec_begin_event_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaEventDestroy failed: " << cudaGetErrorString(err);
        }
    }
    if (exec_end_event_) {
        auto err = cudaEventDestroy(exec_end_event_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaEventDestroy failed: " << cudaGetErrorString(err);
        }
    }
#endif
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
class CudaTimingGuard final {
public:
    CudaTimingGuard(cudaStream_t stream, cudaEvent_t* begin_event, cudaEvent_t* end_event, bool is_profiling_enabled)
        : is_profiling_enabled_(is_profiling_enabled), end_event_(end_event), stream_(stream) {
        if (is_profiling_enabled) {
            auto err = cudaEventRecord(*begin_event, stream);
            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaEventRecord failed: " << cudaGetErrorString(err);
            }
            stream_ = stream;
        }
    }
    ~CudaTimingGuard() {
        if (is_profiling_enabled_) {
            auto err = cudaEventRecord(*end_event_, stream_);
            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaEventRecord failed: " << cudaGetErrorString(err);
            }
        }
    }

private:
    bool is_profiling_enabled_;
    cudaEvent_t* end_event_;
    cudaStream_t stream_;
};
#endif

RetCode LlmCudaKernel::Execute(KernelExecContext* ctx) {
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
    CudaTimingGuard __timing_guard__(GetCudaDevice()->GetStream(), &exec_begin_event_, &exec_end_event_,
                                     ctx->IsProfilingEnabled());
#endif
    RetCode rc = RC_SUCCESS;

    rc = DoExecute(ctx);
    if (RC_SUCCESS != rc) {
        LOG(ERROR) << "DoExecute kernel [" << GetName() << "] failed: " << GetRetCodeStr(rc);
        return rc;
    }

#if !defined(NDEBUG) || defined(DEBUG)
    {
        auto err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaDeviceSynchronize failed: " << (int)err << ", " <<  cudaGetErrorString(err);
            return ppl::common::RC_DEVICE_RUNTIME_ERROR;
        }
    }
#endif

#ifdef PPLNN_LLM_CUDA_DUMP_OUTPUT_TENSORS
    std::string dump_dir = "./rank_" + std::to_string(GetCudaDevice()->GetTensorParallelNcclParam()->rank);
    rc = LlmCudaDumpOutputTensors(ctx, dump_dir);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "LlmCudaDumpOutputTensors() of kernel[" << GetName() << "] failed: " << GetRetCodeStr(rc);
        return rc;
    }
#endif

    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
void LlmCudaKernel::GetProfilingInfo(InternalProfilingInfo* info) const {
    auto err = cudaEventSynchronize(exec_end_event_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaEventSynchronize failed: " << cudaGetErrorString(err);
    }
    float ms = 0.0;
    err = cudaEventElapsedTime(&ms, exec_begin_event_, exec_end_event_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaEventElapsedTime failed: " << cudaGetErrorString(err);
    }
    info->exec_microseconds = static_cast<uint64_t>(ms * 1000);
}
#endif

}}}} // namespace ppl::nn::llm::cuda
