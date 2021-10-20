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

#ifndef _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_COMMON_H_
#define _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_COMMON_H_

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/runtime_graph.h"
#include "ppl/nn/runtime/profiler.h"
#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace utils {

/** @brief calculate each object's reference count in `topo`. */
std::vector<uint32_t> InitObjectRefcount(const ir::GraphTopo* topo);

/** @brief put inputs/extra_inputs/outputs/constants into a vector */
std::vector<EdgeObject*> InitObjectInUse(const ir::GraphTopo* topo, RuntimeGraph* graph);

ppl::common::RetCode ExecuteKernel(KernelImpl*, KernelExecContext*,
                                   const std::function<ppl::common::RetCode(EdgeObject*)>& release_object_func,
                                   Profiler*);

}}} // namespace ppl::nn::utils

#endif
