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

#ifndef _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_LLM_CUDA_KERNEL_H_

#include "llm_cuda_device.h"
#include "engine_context.h"

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/engines/llm_cuda/macros.h"

#include <functional>
#ifdef PPLNN_ENABLE_KERNEL_PROFILING
#include <cuda_runtime.h>
#endif

namespace ppl { namespace nn { namespace llm { namespace cuda {

class LlmCudaKernel : public KernelImpl {
public:
    LlmCudaKernel(const ir::Node* node) : KernelImpl(node) {}
    LlmCudaKernel(LlmCudaKernel&&) = default;
    virtual ~LlmCudaKernel() {
        Destroy();
    }

    cudaStream_t GetStream() const {
        return GetCudaDevice()->GetStream();
    }

    cublasLtHandle_t GetCublasHandle() const {
        return GetCudaDevice()->GetCublasHandle();
    }

    ppl::common::NcclParam* GetTensorParallelNcclParam() const {
        return GetCudaDevice()->GetTensorParallelNcclParam();
    }

    int GetDeviceId() const {
        return GetCudaDevice()->GetDeviceId();
    }

    ppl::common::RetCode Init();

    void SetReshapeFunc(const std::function<ppl::common::RetCode(InputOutputInfo*)>& f) {
        reshape_func_ = f;
    }

    ppl::common::RetCode Reshape(InputOutputInfo* info) const override final {
        return reshape_func_(info);
    }

    ppl::common::RetCode Execute(KernelExecContext*) override final;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
public:
    void GetProfilingInfo(InternalProfilingInfo*) const override final;

private:
    cudaEvent_t exec_begin_event_ = nullptr, exec_end_event_ = nullptr;
#endif

protected:
    LlmCudaDevice* GetCudaDevice() const {
        return dynamic_cast<LlmCudaDevice*>(GetEngineContext()->GetDevice());
    }

    const EngineOptions& GetEngineOptions() const {
        return reinterpret_cast<LlmCudaEngineContext*>(GetEngineContext())->GetEngineOptions();
    }

    virtual ppl::common::RetCode DoExecute(KernelExecContext*) = 0;

private:
    void Destroy();

private:
    std::function<ppl::common::RetCode(InputOutputInfo*)> reshape_func_;
};

}}}} // namespace ppl::nn::llm::cuda

#endif
