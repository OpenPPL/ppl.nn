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
                                int32_t vocab_size, float top_p, float top_k, int32_t* output_host) {
    const int64_t output_size = batch * sizeof(int32_t);
    int32_t pad_vocab_size = 0;
    int32_t output_offset = 0;
    int64_t needed_output_size = output_size;
    cudaError_t err;

    if (top_k != 1) {
        LOG(ERROR) << "currently only support top_k == 1";
        return RC_UNSUPPORTED;
    }

    if (top_p != 0.0) {
        pad_vocab_size = ppl::kernel::llm::cuda::pmx::flash_sample_top_p_get_pad_vocab_size(vocab_size);
        needed_output_size += (int64_t)batch * pad_vocab_size * sizeof(int32_t);
        needed_output_size += (int64_t)batch * pad_vocab_size * sizeof(float);
        needed_output_size += batch * sizeof(float);
        output_offset = batch + batch * pad_vocab_size * 2;
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
    if (top_p == 0.0) {
        rc = ppl::kernel::llm::cuda::pmx::sample_argmax(stream_, logits_device, batch, vocab_size, cu_output_);
    } else {
        float* sorted_value = (float*)cu_output_;
        int32_t* sorted_index = (int32_t*)sorted_value + (int64_t)batch * pad_vocab_size;
        float* temperatures_device = (float*)sorted_index + (int64_t)batch * pad_vocab_size;

        err = cudaMemcpyAsync(temperatures_device, temperatures_host, output_size, cudaMemcpyHostToDevice, stream_);
        if (err != cudaSuccess) {
            LOG(ERROR) << "cudaMemcpyAsync temperatures failed: " << cudaGetErrorString(err);
            return RC_DEVICE_MEMORY_ERROR;
        }

        // TODO: currently it has some bug
        rc = ppl::kernel::llm::cuda::pmx::flash_sample_top_p(stream_, logits_device, batch, vocab_size,
                                                             temperatures_device, top_p, sorted_value, sorted_index,
                                                             cu_output_ + output_offset);
    }

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
