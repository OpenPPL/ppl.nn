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

#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"
#include "ppl/common/sys.h"
#include "ppl/common/log.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

X86OptKernel::X86OptKernel(const ir::Node* node) : OptKernel(node), engine_config_(nullptr) {
    common_param_.output_formats.resize(node->GetOutputCount(), DATAFORMAT_NDARRAY);
}

ppl::common::RetCode X86OptKernel::Init(const OptKernelOptions& options) {
    if (options.config == nullptr) {
        LOG(ERROR) << "EngineConfig must not be NULL.";
        return RC_INVALID_VALUE;
    }

    engine_config_ = options.config;
    return DoInit(options);
}

}}} // namespace ppl::nn::x86
