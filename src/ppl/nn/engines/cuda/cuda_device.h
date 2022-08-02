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

#include <map>
#include <random>

#include "ppl/nn/engines/cuda/data_converter.h"
#include "ppl/nn/engines/cuda/engine_options.h"

namespace ppl { namespace nn { namespace cuda {

class CudaDevice : public Device {
public:
    virtual ~CudaDevice();

    ppl::common::RetCode Init(int device_id);

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
    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, const TensorShape&) const override final;

    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, const TensorShape&) const override final;

    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape&) const override final;

    ppl::common::RetCode Sync() override final;

    const DataConverter* GetDataConverter() const override final {
        return &data_converter_;
    }

    const char* GetType() const override final {
        return "cuda";
    }

    ppl::common::RetCode Configure(uint32_t, ...) override {
        return ppl::common::RC_UNSUPPORTED;
    }

    virtual ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
        return Realloc(bytes, buffer);
    }
    virtual void FreeTmpBuffer(BufferDesc* buffer) {
        Free(buffer);
    }

    ppl::common::RetCode SyncStream();

    cudaStream_t GetStream() const {
        return stream_;
    }
    int GetDeviceId() const {
        return device_id_;
    }

    std::map<edgeid_t, BufferDesc>* GetEdge2Buffer() {
        return &edge2buffer_;
    }

private:
    int device_id_ = INT_MAX;
    cudaStream_t stream_ = nullptr;
    CudaDataConverter data_converter_;
    std::map<edgeid_t, BufferDesc> edge2buffer_;
};

}}} // namespace ppl::nn::cuda

#endif
