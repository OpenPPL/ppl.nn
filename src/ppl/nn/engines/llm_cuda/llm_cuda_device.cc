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

#include "llm_cuda_device.h"
#include "ppl/common/cuda/cuda_env.h"
#include <stdarg.h>
#include <cuda.h>

#include "ppl/kernel/llm/cuda/pmx/i4f16/gemm.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

LlmCudaDevice::LlmCudaDevice() {
    *(uint64_t*)(type_.str) = 0;
    type_.str[0] = 'c';
    type_.str[1] = 'u';
    type_.str[2] = 'd';
    type_.str[3] = 'a';
}

LlmCudaDevice::~LlmCudaDevice() {
    DoDestroy();

    if (own_stream_ && stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }

    if (cublas_handle_) {
        cublasLtDestroy(cublas_handle_);
    }

    if (cublas_workspace_) {
        cudaFree(cublas_workspace_);
    }

    if (cudnn_handle_) {
        cudnnDestroy(cudnn_handle_);
    }

    if (i4f16_gemm_handle_) {
        ppl::kernel::llm::cuda::pmx::i4f16::destory_gemm_handle(i4f16_gemm_handle_);
    }

    if (device_id_ != -1) {
        DestroyCudaEnv(device_id_);
    }
}

RetCode LlmCudaDevice::Init(int device_id, bool init_cublas, NcclParam* tensor_parallel_nccl_param,
                            DeviceStreamFlag flag, cudaStream_t stream) {
    auto rc = InitCudaEnv(device_id);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "init cuda env failed: " << GetRetCodeStr(rc);
        return rc;
    }

    auto err = cudaGetDeviceProperties(&device_prop_, device_id);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaGetDeviceProperties failed: " << cudaGetErrorString(err);
        return RC_UNSUPPORTED;
    }

    if (!cublas_handle_ && init_cublas) {
        auto cu_status = cublasLtCreate(&cublas_handle_);
        if (cu_status != CUBLAS_STATUS_SUCCESS) {
            LOG(ERROR) << "cublasLtCreate failed: " << cublasLtGetStatusName(cu_status);
            return RC_INTERNAL_ERROR;
        }

        /* refer to https://developer.nvidia.com/blog/new-cublas-12-0-features-and-matrix-multiplication-performance-on-nvidia-hopper-gpus/
         NV said:
            NVIDIA Hopper architecture workspace requirements
            H100 native kernels have increased the need for workspace size.
            It is therefore highly recommended to provide at least 32 MiB (33554432 B)
            of workspace for cuBLASLt calls or when using cublasSetWorkspace.
        */
        auto err = cudaMalloc(&cublas_workspace_, 32 * 1024 * 1024);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMalloc cublas_workspace for 32MiB failed: " << cudaGetErrorString(err);
            return RC_OUT_OF_MEMORY;
        }
        cublas_workspace_size_ = 32 * 1024 * 1024;
    }

    if (!stream_) {
        if (flag == NEW) {
            auto cu_status = cudaStreamCreate(&stream_);
            if (cu_status != cudaSuccess) {
                LOG(ERROR) << "cudaStreamCreate failed: " << cudaGetErrorString(cu_status);
                return RC_INTERNAL_ERROR;
            }
            own_stream_ = true;
        } else if (flag == SHARE) {
            stream_ = stream;
        }
    }

    device_id_ = device_id;
    tensor_parallel_nccl_param_ = tensor_parallel_nccl_param;

    return DoInit();
}

RetCode LlmCudaDevice::CopyFromHostAsync(BufferDesc* dst, const void* src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst->addr, src, bytes, cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync failed: " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

RetCode LlmCudaDevice::CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const {
    auto rc = CopyFromHostAsync(dst, src, bytes);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CopyFromHostAsync failed";
        return RC_OTHER_ERROR;
    }

    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize faild: " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

RetCode LlmCudaDevice::CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHost(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmCudaDevice::CopyFromHostAsync(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHostAsync(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmCudaDevice::CopyToHostAsync(void* dst, const BufferDesc& src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst, src.addr, bytes, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync failed: " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

RetCode LlmCudaDevice::CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const {
    auto rc = CopyToHostAsync(dst, src, bytes);
    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "CopyToHostAsync failed";
        return RC_OTHER_ERROR;
    }

    cudaError_t err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize faild: " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

RetCode LlmCudaDevice::CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHost(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmCudaDevice::CopyToHostAsync(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHostAsync(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmCudaDevice::Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst->addr, src.addr, bytes, cudaMemcpyDeviceToDevice, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync failed: " << (int)err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode LlmCudaDevice::Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const {
    return Copy(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode LlmCudaDevice::ConvertToHostCommon(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                           const TensorShape& src_desc, const void* src_custom_info,
                                           const function<RetCode(void*, const BufferDesc&, const TensorShape&)>& copy_fn) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        // TODO release dst
        return RC_SUCCESS;
    }

    if (dst_desc.GetDataFormat() == src_desc.GetDataFormat() && dst_desc.GetDataType() == src_desc.GetDataType()) {
        auto status = copy_fn(dst, src, src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy dst data to Host failed: " << GetRetCodeStr(status);
        }
        return status;
    }

    LOG(ERROR) << "only support same dataformat and datatype convert"
        << ", src: [" << GetDataTypeStr(src_desc.GetDataType()) << ", " << GetDataFormatStr(src_desc.GetDataFormat()) << "]"
        << ", dst: [" << GetDataTypeStr(dst_desc.GetDataType()) << ", " << GetDataFormatStr(dst_desc.GetDataFormat()) << "]";
    return RC_UNSUPPORTED;
}

RetCode LlmCudaDevice::ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                     const TensorShape& src_desc, const void* src_custom_info) {
    return ConvertToHostCommon(dst, dst_desc, src, src_desc, src_custom_info,
                               [this](void* dst, const BufferDesc& src, const TensorShape& src_desc) -> RetCode {
                                   return CopyToHost(dst, src, src_desc);
                               });
}

RetCode LlmCudaDevice::ConvertToHostAsync(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                          const TensorShape& src_desc, const void* src_custom_info) {
    return ConvertToHostCommon(dst, dst_desc, src, src_desc, src_custom_info,
                               [this](void* dst, const BufferDesc& src, const TensorShape& src_desc) -> RetCode {
                                   return CopyToHostAsync(dst, src, src_desc);
                               });
}

RetCode LlmCudaDevice::ConvertFromHostCommon(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                             const TensorShape& src_desc, const void* dst_custom_info,
                                             const function<RetCode(BufferDesc*, const void*, const TensorShape&)>& copy_fn) {
    if (src_desc.CalcBytesExcludingPadding() == 0) {
        Free(dst);
        return RC_SUCCESS;
    }

    if (dst_desc.GetDataFormat() == src_desc.GetDataFormat() && dst_desc.GetDataType() == src_desc.GetDataType()) {
        auto status = copy_fn(dst, src, src_desc);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy src data from host failed: " << GetRetCodeStr(status);
        }
        return status;
    }

    LOG(ERROR) << "only support same dataformat and datatype convert"
        << ", src: [" << GetDataTypeStr(src_desc.GetDataType()) << ", " << GetDataFormatStr(src_desc.GetDataFormat()) << "]"
        << ", dst: [" << GetDataTypeStr(dst_desc.GetDataType()) << ", " << GetDataFormatStr(dst_desc.GetDataFormat()) << "]";
    return RC_UNSUPPORTED;
}

RetCode LlmCudaDevice::ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                       const TensorShape& src_desc, const void* dst_custom_info) {
    return ConvertFromHostCommon(dst, dst_desc, src, src_desc, dst_custom_info,
                                 [this](BufferDesc* dst, const void* src, const TensorShape& src_desc) -> RetCode {
                                     return CopyFromHost(dst, src, src_desc);
                                 });
}

RetCode LlmCudaDevice::ConvertFromHostAsync(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                            const TensorShape& src_desc, const void* dst_custom_info) {
    return ConvertFromHostCommon(dst, dst_desc, src, src_desc, dst_custom_info,
                                 [this](BufferDesc* dst, const void* src, const TensorShape& src_desc) -> RetCode {
                                     return CopyFromHostAsync(dst, src, src_desc);
                                 });
}

RetCode LlmCudaDevice::Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                               const TensorShape& src_desc, const void* dst_custom_info,
                               const void* src_custom_info) {
    LOG(ERROR) << "do not support custom convert";
    return RC_UNSUPPORTED;
}

RetCode LlmCudaDevice::Synchronize() {
    auto cu_ret = cudaStreamSynchronize(stream_);
    if (cu_ret != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize failed: " << (int)cu_ret << ", " << cudaGetErrorString(cu_ret);
        return RC_DEVICE_RUNTIME_ERROR;
    }
    return RC_SUCCESS;
}

/* ------------------------------------------------------------------------- */

RetCode LlmCudaDevice::ConfGetDeviceId(LlmCudaDevice* dev, va_list args) {
    auto did = va_arg(args, int*);
    *did = dev->device_id_;
    return RC_SUCCESS;
}

RetCode LlmCudaDevice::ConfGetStream(LlmCudaDevice* dev, va_list args) {
    auto stream = va_arg(args, cudaStream_t*);
    *stream = dev->stream_;
    return RC_SUCCESS;
}

LlmCudaDevice::ConfHandlerFunc LlmCudaDevice::conf_handlers_[] = {
    LlmCudaDevice::ConfGetDeviceId,
    LlmCudaDevice::ConfGetStream,
};

RetCode LlmCudaDevice::Configure(uint32_t option, ...) {
    if (option >= DEV_CONF_MAX) {
        LOG(ERROR) << "invalid option[" << option << "] >= [" << (uint32_t)DEV_CONF_MAX << "]";
        return RC_INVALID_VALUE;
    }

    va_list args;
    va_start(args, option);
    auto status = conf_handlers_[option](this, args);
    va_end(args);

    return status;
}

}}}} // namespace ppl::nn::llm::cuda
