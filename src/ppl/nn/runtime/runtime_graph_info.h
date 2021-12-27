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

#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_INFO_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_INFO_H_

#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/runtime/runtime_constant_info.h"
#include <vector>
#include <map>

namespace ppl { namespace nn {

class EngineImpl;

struct RuntimeGraphInfo final {
    struct Partition final {
        // vs2019 requires these constructors
        Partition() {}
        Partition(Partition&&) = default;
        Partition& operator=(Partition&&) = default;
        Partition(const Partition&) = delete;
        Partition& operator=(const Partition&) = delete;

        EngineImpl* engine = nullptr;
        std::vector<std::unique_ptr<OptKernel>> ops;
        std::map<edgeid_t, RuntimeConstantInfo> constants;
    };
    std::map<edgeid_t, TensorShape> shapes;
    std::vector<Partition> partitions;
};

}} // namespace ppl::nn

#endif
