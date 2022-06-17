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

#include "ppl/nn/engines/cuda/engine_context.h"
#include "ppl/nn/engines/cuda/default_cuda_device.h"
#include "ppl/nn/engines/cuda/buffered_cuda_device.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CudaEngineContext::Init(const EngineOptions& options) {
    const char* errmsg = nullptr;

    cu_device_ = options.device_id;

    auto cu_status = cuInit(0);
    if (cu_status != CUDA_SUCCESS) {
        cuGetErrorString(cu_status, &errmsg);
        LOG(ERROR) << "cuInit failed: " << errmsg;
        return RC_OTHER_ERROR;
    }

    CUcontext cu_context;
    cu_status = cuDevicePrimaryCtxRetain(&cu_context, options.device_id);
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

    auto status = device_.Init(options.device_id, options.mm_policy);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init BufferedCudaDevice failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

CudaEngineContext::~CudaEngineContext() {
    cuDevicePrimaryCtxRelease(cu_device_);
}

}}} // namespace ppl::nn::cuda
