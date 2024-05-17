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

#include "sampler.h"

#include "ppl/nn/engines/llm_cuda/options.h"
#include "ppl/common/log.h"
#include "ppl/kernel/llm/cuda/pmx/sample.h"

using namespace ppl::common;

namespace ppl { namespace llm { namespace cuda {

void Sampler::Clear() {
    if (cu_output_) {
        auto err = cudaFreeAsync(cu_output_, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaFreeAsync failed: " << cudaGetErrorString(err);
        }
        err = cudaStreamSynchronize(stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);
        }
        cu_output_ = nullptr;
    }
}

RetCode Sampler::SampleTopPTopK(const float* logits_device, const float* temperatures_host, int32_t batch,
                                int32_t vocab_size, int32_t batch_stride, float top_p, float top_k, int32_t* output_host) {
    const int64_t output_size = batch * sizeof(int32_t);
    int32_t output_offset = 0;
    int64_t needed_output_size = output_size;
    cudaError_t err;

    if (top_k != 1 || top_p != 0.0) {
        LOG(ERROR) << "currently only support top_k == 1, top_p == 0";
        return RC_UNSUPPORTED;
    }

    if (needed_output_size > cu_output_size_) {
        if (cu_output_) {
            err = cudaFreeAsync(cu_output_, stream_);
            if (err != cudaSuccess) {
                LOG(ERROR) << "cudaFreeAsync failed: " << cudaGetErrorString(err);
                return RC_DEVICE_MEMORY_ERROR;
            }
        }
        err = cudaMallocAsync(&cu_output_, needed_output_size, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMallocAsync failed: " << cudaGetErrorString(err);
            return RC_OUT_OF_MEMORY;
        }
        cu_output_size_ = needed_output_size;
    }

    RetCode rc;
    rc = ppl::kernel::llm::cuda::pmx::sample_argmax(stream_, logits_device, batch, vocab_size, batch_stride, cu_output_);

    if (rc != RC_SUCCESS) {
        LOG(ERROR) << "sampling kernel failed: " << GetRetCodeStr(rc);
        return rc;
    }

    err = cudaMemcpyAsync(output_host, cu_output_ + output_offset, output_size, cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaMemcpyAsync output failed: " << cudaGetErrorString(err);
        return RC_DEVICE_MEMORY_ERROR;
    }

    err = cudaStreamSynchronize(stream_);
    if (err != cudaSuccess) {
        LOG(ERROR) << "cudaStreamSynchronize failed: " << cudaGetErrorString(err);
        return RC_DEVICE_RUNTIME_ERROR;
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::llm::cuda
