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

#include "ppl/nn/engines/cuda/cuda_device.h"

#include <stddef.h>
#include <cuda_runtime.h>

#include "ppl/nn/common/logger.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

CudaDevice::~CudaDevice() {
    cudaStreamDestroy(context_.stream);
}

void CudaDevice::Init(uint32_t device_id) {
    context_.device_id = device_id;
    cudaSetDevice(context_.device_id);
    if (!(context_.stream)) {
        cudaStreamCreate(&(context_.stream));
    }

    data_converter_.SetDevice(this);
}

// Copy from host
RetCode CudaDevice::CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst->addr, src, bytes, cudaMemcpyHostToDevice, context_.stream);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaDevice::CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHost(dst, src, shape.GetBytesIncludingPadding());
}

// Copy to host
RetCode CudaDevice::CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst, src.addr, bytes, cudaMemcpyDeviceToHost, context_.stream);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    err = cudaStreamSynchronize(context_.stream);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}
RetCode CudaDevice::CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHost(dst, src, shape.GetBytesIncludingPadding());
}

RetCode CudaDevice::Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst->addr, src.addr, bytes, cudaMemcpyDeviceToDevice, context_.stream);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaDevice::Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const {
    return Copy(dst, src, shape.GetBytesIncludingPadding());
}

}}} // namespace ppl::nn::cuda
