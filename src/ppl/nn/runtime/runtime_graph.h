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

#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/barrier.h"
#include <vector>

namespace ppl { namespace nn {

/**
   @class RuntimeGraph
   @brief data used in runtime stage
*/
struct RuntimeGraph {
    void Clear() {
        inputs.clear();
        extra_inputs.clear();
        constants.clear();
        outputs.clear();
        tensors.clear();
        nodeid2kernel.clear();
    }

    /** input tensors of the graph except constants */
    std::vector<TensorImpl*> inputs;

    /** extra inputs used by this graph */
    std::vector<TensorImpl*> extra_inputs;

    /** constant tensors */
    std::vector<TensorImpl*> constants;

    /** output tensors of the graph */
    std::vector<TensorImpl*> outputs;

    /** union of inputs/extra_inputs/constants/outputs */
    std::map<edgeid_t, TensorImpl> tensors;

    /** kernels list where the subscriptor is KernelImpl::GetNode()::GetId() */
    std::vector<std::unique_ptr<KernelImpl>> nodeid2kernel;

    /** whether a kernel needs to be synchronized before getting its outputs */
    std::vector<bool> kernel_barrier_flag;

    /** barriers for EdgeObjects before getting their contents */
    std::vector<std::shared_ptr<Barrier>> edgeid2barrier;
};

}} // namespace ppl::nn

#endif
