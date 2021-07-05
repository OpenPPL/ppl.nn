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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/split_to_sequence_op.h"
#include "ppl/nn/common/logger.h"
#include <cstring>
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode SplitToSequenceOp::Init(const OptKernelOptions& options) {
    auto node = GetNode();
    auto graph_data = options.graph_data;
    auto attr_ref = graph_data->attrs.find(node->GetId());
    if (attr_ref == graph_data->attrs.end()) {
        LOG(ERROR) << "cannot find attr for SplitToSequenceOp[" << node->GetName() << "]";
        return RC_NOT_FOUND;
    }

    auto param = static_cast<common::SplitToSequenceParam*>(attr_ref->second.get());
    op_.Init(param->axis, param->keepdims, common::SplitToSequenceOp::GenericSplitFunc);
    return RC_SUCCESS;
}

KernelImpl* SplitToSequenceOp::CreateKernelImpl() const {
    return op_.CreateKernelImpl();
}

}}} // namespace ppl::nn::x86
