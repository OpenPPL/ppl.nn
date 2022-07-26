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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNEL_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/engines/cuda/macros.h"
#include "ppl/nn/engines/cuda/cuda_device.h"
#include "ppl/nn/engines/cuda/cuda_common_param.h"
#include "ppl/common/sys.h"

namespace ppl { namespace nn { namespace cuda {

class CudaKernel : public KernelImpl {
public:
    CudaKernel(const ir::Node* node) : KernelImpl(node) {}
    virtual ~CudaKernel();

    ppl::common::RetCode Init();

    cudaStream_t GetStream() const {
        auto cuda_device = static_cast<const CudaDevice*>(GetDevice());
        return cuda_device->GetStream();
    }

    int GetDeviceId() const {
        auto cuda_device = static_cast<const CudaDevice*>(GetDevice());
        return cuda_device->GetDeviceId();
    }

    void SetCommonParam(const CudaCommonParam* p) {
        common_param_ = p;
    }

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
    cudaEvent_t exec_begin_event_, exec_end_event_;
#endif

protected:
    virtual bool CanDoExecute(const KernelExecContext&) const;
    virtual ppl::common::RetCode DoExecute(KernelExecContext*) = 0;
    virtual ppl::common::RetCode BeforeExecute(KernelExecContext*);

    virtual uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const {
        return 0;
    }

    CudaDevice* GetCudaDevice() {
        return reinterpret_cast<CudaDevice*>(GetDevice());
    }

    const CudaCommonParam* GetCommonParam() {
        return common_param_;
    }

private:
    const CudaCommonParam* common_param_ = nullptr;
    std::function<ppl::common::RetCode(InputOutputInfo*)> reshape_func_;
};

}}} // namespace ppl::nn::cuda

#endif
