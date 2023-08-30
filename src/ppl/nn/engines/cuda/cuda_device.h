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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_DEVICE_H_

#include "ppl/nn/common/device.h"
#include <cuda_runtime.h>
#include <cublasLt.h>

#include <map>
#include <random>
#include <functional>

#include "ppl/nn/engines/cuda/engine_options.h"
#include "ppl/nn/engines/cuda/cuda_common_param.h"
#include "ppl/common/cuda/nccl_utils.h"

namespace ppl { namespace nn { namespace cuda {

class CudaDevice : public Device {
public:
    CudaDevice() {
        *(uint64_t*)(type_.str) = 0;
        type_.str[0] = 'c';
        type_.str[1] = 'u';
        type_.str[2] = 'd';
        type_.str[3] = 'a';
    }
    virtual ~CudaDevice();

    ppl::common::RetCode Init(int device_id, ppl::common::NcclParam* tp_nccl_param, bool enable_cuda_graph = false);

    using Device::Realloc;
    ppl::common::RetCode Realloc(const TensorShape& shape, BufferDesc* buffer) override final {
        return Realloc(shape.CalcBytesIncludingPadding(), buffer);
    }

    virtual ppl::common::RetCode ReallocWithRandomValue(uint64_t bytes, BufferDesc* buffer) {
        auto status = Realloc(bytes, buffer);
        if (status != ppl::common::RC_SUCCESS) {
            return status;
        }
        std::default_random_engine eng;
        std::uniform_real_distribution<float> dis(-2.64f, 2.64f);
        std::vector<float> host_random_data(bytes / sizeof(float));
        for (size_t i = 0; i < bytes / sizeof(float); ++i) {
            host_random_data[i] = dis(eng);
        }
        cudaMemcpyAsync(buffer->addr, host_random_data.data(), bytes, cudaMemcpyHostToDevice, stream_);
        return status;
    }

    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyFromHostAsync(BufferDesc* dst, const void* src, uint64_t bytes) const override final;

    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, const TensorShape&) const override final;
    ppl::common::RetCode CopyFromHostAsync(BufferDesc* dst, const void* src, const TensorShape&) const override final;

    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyToHostAsync(void* dst, const BufferDesc& src, uint64_t bytes) const override final;

    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, const TensorShape&) const override final;
    ppl::common::RetCode CopyToHostAsync(void* dst, const BufferDesc& src, const TensorShape&) const override final;

    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape&) const override final;

    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                       const TensorShape& src_desc,
                                       const void* src_custom_info = nullptr) override final;
    ppl::common::RetCode ConvertToHostAsync(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                            const TensorShape& src_desc,
                                            const void* src_custom_info = nullptr) override final;

    ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc,
                                         const void* dst_custom_info = nullptr) override final;
    ppl::common::RetCode ConvertFromHostAsync(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                              const TensorShape& src_desc,
                                              const void* dst_custom_info = nullptr) override final;

    ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                 const TensorShape& src_desc, const void* dst_custom_info = nullptr,
                                 const void* src_custom_info = nullptr) override final;

    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const CudaTensorQuant& dst_quant,
                                       const BufferDesc& src, const TensorShape& src_desc,
                                       const CudaTensorQuant& src_quant);

    ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const CudaTensorQuant& dst_quant,
                                         const void* src, const TensorShape& src_desc,
                                         const CudaTensorQuant& src_quant);

    ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const CudaTensorQuant& dst_quant,
                                 const BufferDesc& src, const TensorShape& src_desc, const CudaTensorQuant& src_quant);

    ppl::common::RetCode Synchronize() override final;

    const Type& GetType() const override final {
        return type_;
    }

    ppl::common::RetCode Configure(uint32_t, ...) override;

    virtual ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
        return Realloc(bytes, buffer);
    }
    virtual void FreeTmpBuffer(BufferDesc* buffer) {
        Free(buffer);
    }

    ppl::common::RetCode SyncStream();

    const cudaDeviceProp& GetDeviceProp() const {
        return device_prop_;
    }

    cudaStream_t GetStream() const {
        return stream_;
    }

    cublasLtHandle_t GetCublasHandle() const {
        return cublas_handle_;
    }

    ppl::common::NcclParam* GetTpNcclParam() const {
        return tp_nccl_param_;
    }

    int GetDeviceId() const {
        return device_id_;
    }

    std::map<edgeid_t, BufferDesc>* GetEdge2Buffer() {
        return &edge2buffer_;
    }

private:
    ppl::common::RetCode ConvertToHostCommon(
        void* dst, const TensorShape& dst_desc, const BufferDesc& src, const TensorShape& src_desc,
        const void* src_info,
        const std::function<ppl::common::RetCode(void*, const BufferDesc&, const TensorShape&)>& copy_fn);
    ppl::common::RetCode ConvertFromHostCommon(
        BufferDesc* dst, const TensorShape& dst_desc, const void* src, const TensorShape& src_desc,
        const void* dst_info,
        const std::function<ppl::common::RetCode(BufferDesc*, const void*, const TensorShape&)>& copy_fn);

private:
    Type type_;
    int device_id_ = INT_MAX;
    bool enable_cuda_graph_ = false;
    cudaStream_t stream_ = nullptr;
    cublasLtHandle_t cublas_handle_ = nullptr;
    cudaDeviceProp device_prop_;
    std::map<edgeid_t, BufferDesc> edge2buffer_;
    ppl::common::NcclParam* tp_nccl_param_ = nullptr;

private:
    static ppl::common::RetCode ConfGetDeviceId(CudaDevice*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(CudaDevice*, va_list);
    static ConfHandlerFunc conf_handlers_[DEV_CONF_MAX];

    cudaError_t CheckCaptureStreamSync(cudaStream_t stream) const;
};

}}} // namespace ppl::nn::cuda

#endif
