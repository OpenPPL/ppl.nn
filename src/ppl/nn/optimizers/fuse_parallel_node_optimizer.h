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

#ifndef _ST_HPC_PPL_NN_OPTIMIZERS_FUSE_PARALLEL_NODE_OPTIMIZER_H_
#define _ST_HPC_PPL_NN_OPTIMIZERS_FUSE_PARALLEL_NODE_OPTIMIZER_H_

#include "ppl/nn/optimizers/graph_optimizer.h"

namespace ppl { namespace nn {

class FuseParallelNodeOptimizer : public GraphOptimizer {
public:
    FuseParallelNodeOptimizer() : GraphOptimizer("FuseParallelNodeOptimizer") {}
    virtual ~FuseParallelNodeOptimizer() {}
    ppl::common::RetCode Optimize(ir::Graph*) const override;
};

}} // namespace ppl::nn

#endif
