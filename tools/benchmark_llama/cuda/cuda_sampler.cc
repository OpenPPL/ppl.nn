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

#include "cuda_sampler.h"

#include "ppl/common/log.h"
#include "ppl/kernel/llm/cuda/pmx/sample.h"

ppl::common::RetCode CudaSampler::Clear() {
    if (output_device_) {
        auto err = cudaFreeAsync(output_device_, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaFreeAsync failed: " << cudaGetErrorString(err);
            return ppl::common::RC_DEVICE_MEMORY_ERROR;
        }
        err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);
            return ppl::common::RC_DEVICE_RUNTIME_ERROR;
        }
        output_device_ = nullptr;
    }
    return ppl::common::RC_SUCCESS;
}


ppl::common::RetCode CudaSampler::SampleArgMax(
    const float* logits_device,
    const int32_t batch,
    const int32_t vocab_size,
    const int32_t batch_stride,
    int32_t* output_host)
{
    if (batch > max_batch_) {
        max_batch_ = batch;
        auto err = cudaFreeAsync(output_device_, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaFreeAsync failed: " << cudaGetErrorString(err);
            return ppl::common::RC_DEVICE_MEMORY_ERROR;
        }
        err = cudaMallocAsync(&output_device_, max_batch_ * sizeof(int32_t), stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMallocAsync failed: " << cudaGetErrorString(err);
            return ppl::common::RC_OUT_OF_MEMORY;
        }
    }

    ppl::common::RetCode rc = ppl::kernel::llm::cuda::pmx::sample_argmax(
        stream_, logits_device,
        batch, vocab_size,
        batch_stride, output_device_);
    if (rc != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "sampling kernel failed: " << ppl::common::GetRetCodeStr(rc);
        return rc;
    }

    cudaError_t err = cudaMemcpyAsync(
        output_host, output_device_, batch * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync output failed: " << cudaGetErrorString(err);
        return ppl::common::RC_DEVICE_MEMORY_ERROR;
    }
    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);
        return ppl::common::RC_DEVICE_RUNTIME_ERROR;
    }

    return ppl::common::RC_SUCCESS;
}
