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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_MODULE_OP_COMPILE_MANAGER_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_MODULE_OP_COMPILE_MANAGER_H_

#include "ppl/nn/engines/cuda/module/op_compiler.h"
#include "ppl/nn/engines/cuda/module/conv_compiler.h"
#include "ppl/nn/engines/cuda/module/convtranspose_compiler.h"
#include "ppl/nn/engines/cuda/module/gemm_compiler.h"
#include "ppl/nn/engines/cuda/module/normal_compiler.h"

namespace ppl { namespace nn { namespace cuda {

class OpCompilerManager {
public:
    static OpCompilerManager* Instance() {
        static OpCompilerManager mgr;
        return &mgr;
    }
    OpCompiler* FindCompiler(const std::string& kernel_type) const;
    template <typename T>
    void Register(const std::string& kernel_type, T& fusion_type);
    void Remove(const std::string& kernel_type);

private:
    OpCompilerManager();

private:
    std::map<std::string, OpCompiler*> type2compiler_;
    ConvTransposeCompiler convtranspose_;
    ConvCompiler conv_;
    GemmCompiler gemm_;
    NormalCompiler normal_;
};

}}} // namespace ppl::nn::cuda

#endif
