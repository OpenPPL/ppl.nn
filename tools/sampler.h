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

#ifndef __PPL_LLM_CUDA_SAMPLER_H__
#define __PPL_LLM_CUDA_SAMPLER_H__

#include "ppl/common/retcode.h"

#include <cuda_runtime.h>

namespace ppl { namespace llm { namespace cuda {

class Sampler final {
public:
    Sampler(cudaStream_t stream) : stream_(stream) {}
    virtual ~Sampler() {
        Clear();
    }

    ppl::common::RetCode SampleTopPTopK(const float* logits_device, const float* temperatures_host, int32_t batch,
                                        int32_t vocab_size, int32_t batch_stride, float top_p, float top_k, int32_t* output_host);

private:
    void Clear();

private:
    cudaStream_t stream_ = 0;
    int32_t* cu_output_ = nullptr;
    int64_t cu_output_size_ = 0;
};

}}}; // namespace ppl::llm::cuda

#endif
