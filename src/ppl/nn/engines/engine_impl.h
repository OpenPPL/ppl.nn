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

#ifndef _ST_HPC_PPL_NN_ENGINES_ENGINE_IMPL_H_
#define _ST_HPC_PPL_NN_ENGINES_ENGINE_IMPL_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/engine.h"
#include "ppl/nn/engines/engine_context.h"

namespace ppl { namespace nn {

namespace utils {
struct SharedResource;
}

struct RuntimePartitionInfo;

/**
   @class EngineImpl
   @brief engine implementation interface
*/
class EngineImpl : public Engine {
public:
    /** @param name engine's name */
    EngineImpl(const std::string& name) : name_(name) {}

    virtual ~EngineImpl() {}

    const char* GetName() const override final {
        return name_.c_str();
    }

    /** @brief create a `Device` instance for `Runtime` instances */
    virtual EngineContext* CreateEngineContext() = 0;

    /** @brief tells whether this engine implements `node`. */
    virtual bool Supports(const ir::Node* node) const = 0;

    /**
       @brief optimize the compute graph `graph` and fill `info`
       @param graph graph to be optimized and can be modified
       @note DO NOT modify input and output edges
    */
    virtual ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph* graph, RuntimePartitionInfo* info) = 0;

private:
    const std::string name_;
};

}} // namespace ppl::nn

#endif
