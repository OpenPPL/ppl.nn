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

#ifndef _ST_HPC_PPL_NN_LLM_CUDA_DEVICE_H_
#define _ST_HPC_PPL_NN_LLM_CUDA_DEVICE_H_

#include "ppl/nn/engines/llm_cuda/options.h"
#include "ppl/nn/common/device.h"
#include "ppl/common/cuda/nccl_utils.h"

#include "ppl/kernel/llm/cuda/cublas/gemm_algo.h"

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cudnn.h>

#include <functional>

namespace ppl { namespace nn { namespace llm { namespace cuda {

enum DeviceStreamFlag {
    NONE, // do not use stream
    NEW, // create a new stream for this device
    SHARE, // use the specified stream
};

class LlmCudaDevice : public Device {
public:
    LlmCudaDevice();
    virtual ~LlmCudaDevice();

    ppl::common::RetCode Init(int device_id, bool init_cublas, ppl::common::NcclParam* tensor_parallel_nccl_param,
                              DeviceStreamFlag flag, cudaStream_t stream = 0);

    using Device::Realloc;
    ppl::common::RetCode Realloc(const TensorShape& shape, BufferDesc* buffer) override final {
        return Realloc(shape.CalcBytesIncludingPadding(), buffer);
    }

    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyFromHostAsync(BufferDesc* dst, const void* src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const override final;
    ppl::common::RetCode CopyFromHostAsync(BufferDesc* dst, const void* src, const TensorShape& shape) const override final;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyToHostAsync(void* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const override final;
    ppl::common::RetCode CopyToHostAsync(void* dst, const BufferDesc& src, const TensorShape& shape) const override final;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const override final;

    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                       const TensorShape& src_desc,
                                       const void* src_custom_info = nullptr) override;
    ppl::common::RetCode ConvertToHostAsync(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                            const TensorShape& src_desc,
                                            const void* src_custom_info = nullptr) override;

    ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc,
                                         const void* dst_custom_info = nullptr) override;
    ppl::common::RetCode ConvertFromHostAsync(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc,
                                         const void* dst_custom_info = nullptr) override;

    ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                 const TensorShape& src_desc, const void* dst_custom_info = nullptr,
                                 const void* src_custom_info = nullptr) override;

    const Type& GetType() const override final {
        return type_;
    }

    ppl::common::RetCode Synchronize() override final;

    ppl::common::RetCode Configure(uint32_t, ...) override;

    cudaStream_t GetStream() const {
        return stream_;
    }

    const cudaDeviceProp& GetDeviceProp() const {
        return device_prop_;
    }

    const int GetSMVersion() const {
        return device_prop_.major * 10 + device_prop_.minor;
    }

    cublasLtHandle_t GetCublasHandle() const {
        return cublas_handle_;
    }

    void* GetCublasWorkspace() {
        return cublas_workspace_;
    }

    const int GetCublasWorkspaceSize() const {
        return cublas_workspace_size_;
    }

    void* GetI4F16GemmHandle() const {
        return i4f16_gemm_handle_;
    }

    ppl::kernel::llm::cuda::cublas::AlgoCache* GetCublasAlgoCache() {
        return &cublas_algo_cache_;
    }

    const ppl::kernel::llm::cuda::cublas::AlgoCache* GetCublasAlgoCache() const {
        return &cublas_algo_cache_;
    }

    void SetCudnnHandle(cudnnHandle_t cudnn_handle) {
        cudnn_handle_ = cudnn_handle;
    };

    cudnnHandle_t GetCudnnHandle() const {
        return cudnn_handle_;
    }

    ppl::common::NcclParam* GetTensorParallelNcclParam() const {
        return tensor_parallel_nccl_param_;
    }

    int GetDeviceId() const {
        return device_id_;
    }

    virtual ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
        return Realloc(bytes, buffer);
    }
    virtual void FreeTmpBuffer(BufferDesc* buffer) {
        Free(buffer);
    }

    inline bool Equal(const Device *dev) const {
        if (dev->GetType() == GetType()) {
            return reinterpret_cast<const LlmCudaDevice*>(dev)->GetDeviceId() == GetDeviceId();
        }
        return false;
    }

private:
    ppl::common::RetCode ConvertToHostCommon(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                             const TensorShape& src_desc, const void* src_custom_info,
                                             const std::function<ppl::common::RetCode(void*, const BufferDesc&, const TensorShape&)>& copy_fn);
    ppl::common::RetCode ConvertFromHostCommon(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                               const TensorShape& src_desc, const void* dst_custom_info,
                                               const std::function<ppl::common::RetCode(BufferDesc*, const void*, const TensorShape&)>& copy_fn);

private:
    static ppl::common::RetCode ConfGetDeviceId(LlmCudaDevice*, va_list);
    static ppl::common::RetCode ConfGetStream(LlmCudaDevice*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(LlmCudaDevice*, va_list);
    static ConfHandlerFunc conf_handlers_[DEV_CONF_MAX];

protected:
    /** last call in Init() */
    virtual ppl::common::RetCode DoInit() = 0;
    /** first call in destructor */
    virtual void DoDestroy() {}

protected:
    Type type_;

    int device_id_ = -1;
    cudaStream_t stream_ = nullptr;
    bool own_stream_ = false;
    cudaDeviceProp device_prop_;
    ppl::common::NcclParam* tensor_parallel_nccl_param_ = nullptr;

    cublasLtHandle_t cublas_handle_ = nullptr;
    void* cublas_workspace_ = nullptr;
    int cublas_workspace_size_ = 0;
    ppl::kernel::llm::cuda::cublas::AlgoCache cublas_algo_cache_;

    cudnnHandle_t cudnn_handle_ = nullptr;

    void* i4f16_gemm_handle_ = nullptr;
};

}}}} // namespace ppl::nn::llm::cuda

#endif
