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
#include "ppl/nn/common/logger.h"
#include <cuda.h>

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

static RetCode InitDriverEnv(int device_id) {
    const char* errmsg = nullptr;
    auto cu_status = cuInit(0);
    if (cu_status != CUDA_SUCCESS) {
        cuGetErrorString(cu_status, &errmsg);
        LOG(ERROR) << "cuInit failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    CUcontext cu_context;
    cu_status = cuDevicePrimaryCtxRetain(&cu_context, device_id);
    if (cu_status != CUDA_SUCCESS) {
        cuGetErrorString(cu_status, &errmsg);
        LOG(ERROR) << "cuDevicePrimaryCtxRetain failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    cu_status = cuCtxSetCurrent(cu_context);
    if (cu_status != CUDA_SUCCESS) {
        cuGetErrorString(cu_status, &errmsg);
        LOG(ERROR) << "cuCtxSetCurrent failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    return RC_SUCCESS;
}

static RetCode DestroyDriverEnv(int device_id) {
    auto rc = cuDevicePrimaryCtxRelease(device_id);
    if (rc != CUDA_SUCCESS) {
        const char* errmsg = nullptr;
        cuGetErrorString(rc, &errmsg);
        LOG(ERROR) << "cuDevicePrimaryCtxRelease failed: " << errmsg;
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

CudaDevice::~CudaDevice() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
    }
    if (device_id_ != INT_MAX) {
        DestroyDriverEnv(device_id_);
    }
}

RetCode CudaDevice::Init(int device_id) {
    auto status = InitDriverEnv(device_id);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "InitDriverEnv failed: " << GetRetCodeStr(status);
        return status;
    }

    if (!stream_) {
        cudaStreamCreate(&stream_);
    }

    device_id_ = device_id;
    data_converter_.SetDevice(this);

    return RC_SUCCESS;
}

RetCode CudaDevice::SyncStream() {
    if (stream_) {
        auto rc = cudaStreamSynchronize(stream_);
        if (rc != cudaSuccess) {
            LOG(ERROR) << "sync stream failed: " << cudaGetErrorString(rc);
            return RC_OTHER_ERROR;
        }
    }
    return RC_SUCCESS;
}

// Copy from host
RetCode CudaDevice::CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst->addr, src, bytes, cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaDevice::CopyFromHost(BufferDesc* dst, const void* src, const TensorShape& shape) const {
    return CopyFromHost(dst, src, shape.CalcBytesIncludingPadding());
}

// Copy to host
RetCode CudaDevice::CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst, src.addr, bytes, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}
RetCode CudaDevice::CopyToHost(void* dst, const BufferDesc& src, const TensorShape& shape) const {
    return CopyToHost(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode CudaDevice::Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const {
    cudaError_t err = cudaMemcpyAsync(dst->addr, src.addr, bytes, cudaMemcpyDeviceToDevice, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaDevice::Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape& shape) const {
    return Copy(dst, src, shape.CalcBytesIncludingPadding());
}

RetCode CudaDevice::Sync() {
    auto err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize " << err << ", " << cudaGetErrorString(err);
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
