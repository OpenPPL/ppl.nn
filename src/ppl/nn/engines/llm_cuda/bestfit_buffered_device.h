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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_BESTFIT_BUFFERED_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_BESTFIT_BUFFERED_DEVICE_H_

#include "llm_cuda_device.h"

#include "ppl/common/cuda/cuda_plain_async_allocator.h"
#include "ppl/nn/utils/stack_buffer_manager.h"

namespace ppl { namespace nn { namespace llm { namespace cuda {

class BestFitBufferedDevice final : public LlmCudaDevice {
public:
    BestFitBufferedDevice() : mgr_(&ar_, true) {}

    using LlmCudaDevice::Realloc;
    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buf) override {
        return mgr_.Realloc(bytes, buf);
    }
    void Free(BufferDesc* buf) override {
        mgr_.Free(buf);
    }

protected:
    ppl::common::RetCode DoInit() override {
        ar_.Init(stream_);
        return ppl::common::RC_SUCCESS;
    }

    void DoDestroy() override {}

private:
    ppl::common::CudaPlainAsyncAllocator ar_;
    ppl::nn::utils::StackBufferManager mgr_;
};

}}}} // namespace ppl::nn::llm::cuda

#endif
