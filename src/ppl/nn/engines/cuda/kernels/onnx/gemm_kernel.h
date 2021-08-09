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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_GEMM_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNELS_ONNX_GEMM_KERNEL_H_

#include "ppl/nn/engines/cuda/kernel.h"
#include "ppl/nn/engines/cuda/params/gemm_extra_param.h"

namespace ppl { namespace nn { namespace cuda {

class GemmKernel : public CudaKernel {
public:
    GemmKernel(const ir::Node* node) : CudaKernel(node) {}

    void SetParam(const CudaGemmParam* p) {
        param_ = p;
    }

private:
    uint64_t CalcTmpBufferSize(const KernelExecContext&) const override;
    ppl::common::RetCode DoExecute(KernelExecContext*) override;
    bool CanDoExecute(const KernelExecContext&) const override;

private:
    const CudaGemmParam* param_ = nullptr;
};

}}} // namespace ppl::nn::cuda

#endif
