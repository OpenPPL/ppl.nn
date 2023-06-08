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

#include "graph_runner.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/cuda/cuda_device.h"

using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

CudaGraphRunner::CudaGraphRunner() : built_(false), graph_exec_(nullptr), stream_(nullptr) {}

RetCode CudaGraphRunner::GenStreamDepandency(cudaStream_t source, cudaStream_t destination) {
    cudaEvent_t event;
    const char* errmsg = nullptr;
    auto err = cudaEventCreate(&event);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda event create failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
    err = cudaEventRecord(event, source);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda event record failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
    err = cudaStreamWaitEvent(destination, event, 0);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda event wait failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
    err = cudaEventDestroy(event);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda event destroy failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

RetCode CudaGraphRunner::ExecInit() {
    if (IsGraphBuilt()) {
        return RC_SUCCESS;
    }

    const char* errmsg = nullptr;
    auto err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda stream begin capture failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
    for (auto dev : depandency_devices_) {
        auto status = GenStreamDepandency(stream_, dev->GetStream());
        if (status != RC_SUCCESS) {
            return status;
        }
    }
    return RC_SUCCESS;
}

RetCode CudaGraphRunner::ExecEnd() {
    if (IsGraphBuilt()) {
        return RC_SUCCESS;
    }

    for (auto dev : depandency_devices_) {
        auto status = GenStreamDepandency(dev->GetStream(), stream_);
        if (status != RC_SUCCESS) {
            return status;
        }
    }

    const char* errmsg = nullptr;
    cudaGraph_t graph;
    auto err = cudaStreamEndCapture(stream_, &graph);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda stream end capture failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
    // graph_exec is not null,means that exist old graph_exec
    if (graph_exec_) {
        cudaGraphExecDestroy(graph_exec_);
    }
#if PPLNN_CUDACC_VER_MAJOR * 1000 + PPLNN_CUDACC_VER_MINOR * 10 >= 11040
    err = cudaGraphInstantiateWithFlags(&graph_exec_, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda graph instantiate failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
#else
    LOG(ERROR) << "Duw to lower CUDA version, Graph mode is not supported,please update cuda to 11.4 or above.";
    return RC_UNSUPPORTED;
#endif
    err = cudaGraphDestroy(graph);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda graph destroy failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
    built_ = true;
    return RC_SUCCESS;
}

RetCode CudaGraphRunner::TrueExec() {
    if (!graph_exec_) {
        return RC_INVALID_VALUE;
    }
    const char* errmsg = nullptr;
    auto err = cudaGraphLaunch(graph_exec_, stream_);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda graph launch failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        errmsg = cudaGetErrorString(err);
        LOG(ERROR) << "cuda synchronize after launch failed:" << errmsg;
        return RC_OTHER_ERROR;
    }
    return RC_SUCCESS;
}

void CudaGraphRunner::AddDevice(const CudaDevice* dev) {
    if (!stream_) {
        stream_ = dev->GetStream();
        return;
    }
    depandency_devices_.emplace_back(dev);
    return;
}

CudaGraphRunner::~CudaGraphRunner() {
    if (IsGraphBuilt()) {
        cudaGraphExecDestroy(graph_exec_);
    }
}
}}} // namespace ppl::nn::cuda