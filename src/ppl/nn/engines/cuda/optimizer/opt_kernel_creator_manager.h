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

#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPT_KERNEL_CREATOR_MANAGER_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_OPTIMIZER_OPT_KERNEL_CREATOR_MANAGER_H_

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"
#include "ppl/nn/utils/op_info_manager.h"

namespace ppl { namespace nn { namespace cuda {

typedef CudaOptKernel* (*OptKernelCreator)(const ir::Node*);

class OptKernelCreatorManager {
public:
    static OptKernelCreatorManager* Instance() {
        static OptKernelCreatorManager mgr;
        return &mgr;
    }

    ppl::common::RetCode Register(const std::string& domain, const std::string& type, const utils::VersionRange&,
                                  OptKernelCreator);
    OptKernelCreator Find(const std::string& domain, const std::string& type, uint64_t version) const;
    void Remove(const std::string& domain, const std::string& type);

private:
    utils::OpInfoManager<OptKernelCreator> mgr_;

private:
    OptKernelCreatorManager();
};

}}} // namespace ppl::nn::cuda

#endif
