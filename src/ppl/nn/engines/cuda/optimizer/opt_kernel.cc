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

#include "ppl/nn/engines/cuda/optimizer/opt_kernel.h"

#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace cuda {

RetCode CudaOptKernel::SetCommonParam(const OptKernelOptions& options) {
    auto node = GetNode();

    common_param_.output_tensor_info.resize(node->GetOutputCount());
    for (uint32_t i = 0; i < node->GetOutputCount(); ++i) {
        auto edge_id = node->GetOutput(i);
        auto iter = options.tensors->find(edge_id);
        if (iter == options.tensors->end()) {
            LOG(ERROR) << "can not find edge " << edge_id;
            return RC_NOT_FOUND;
        }
        common_param_.output_tensor_info[i].data_format = iter->second->GetShape().GetDataFormat();
        common_param_.output_tensor_info[i].data_type = iter->second->GetShape().GetDataType();
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::cuda
