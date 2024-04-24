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

#include "ppl/nn/models/onnx/parsers/opmx/parse_parallel_embedding_param.h"
#include "ppl/nn/common/logger.h"
#include "ppl/nn/models/onnx/utils.h"
using namespace std;
using namespace ppl::common;
using namespace ppl::nn::opmx;

namespace ppl { namespace nn { namespace opmx {

RetCode ParseParallelEmbeddingParam(const ::onnx::NodeProto& pb_node, const onnx::ParamParserExtraArgs& args, ir::Node* node,
                                 ir::Attr* arg) {
    auto param = static_cast<ParallelEmbeddingParam*>(arg);

    if (!onnx::utils::GetNodeAttr(pb_node, "num_embeddings", &param->num_embeddings, -1)) {
        LOG(ERROR) << node->GetName() << ": missing num_embeddings";
        return RC_INVALID_VALUE;
    }

    if (!onnx::utils::GetNodeAttr(pb_node, "embedding_dims", &param->embedding_dims, -1)) {
        LOG(ERROR) << node->GetName() << ": missing embedding_dims";
        return RC_INVALID_VALUE;
    }

    onnx::utils::GetNodeAttr(pb_node, "padding_idx", &param->padding_idx, -1);
    onnx::utils::GetNodeAttr(pb_node, "max_norm", &param->max_norm, 0.0f);
    onnx::utils::GetNodeAttr(pb_node, "norm_type", &param->norm_type, 2.0f);

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::opmx
