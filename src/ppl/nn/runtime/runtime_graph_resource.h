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

#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_RESOURCE_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_RESOURCE_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include <vector>

namespace ppl { namespace nn {

/**
   @class RuntimeGraphResource
   @brief resource used in runtime stage
*/
struct RuntimeGraphResource final {
    void Clear() {
        tensors.clear();
        edgeid2object.clear();
        nodeid2kernel.clear();
    }

    /** union of inputs/extra_inputs/constants/outputs */
    std::map<edgeid_t, TensorImpl> tensors;

    /** objects that are used during Run() */
    std::vector<EdgeObject*> edgeid2object;

    /** kernels list where the subscriptor is KernelImpl::GetNode()::GetId() */
    std::vector<std::unique_ptr<KernelImpl>> nodeid2kernel;
};

}} // namespace ppl::nn

#endif
