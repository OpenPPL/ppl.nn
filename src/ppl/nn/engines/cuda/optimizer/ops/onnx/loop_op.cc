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

#include "ppl/nn/engines/cuda/optimizer/ops/onnx/loop_op.h"
#include "ppl/nn/common/tensor_buffer_info.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace cuda {

static RetCode ConcatOutputs(const vector<TensorBufferInfo>& outputs, BufferDesc* buf) {
    auto buf_cursor = *buf;
    for (auto it = outputs.begin(); it != outputs.end(); ++it) {
        auto device = it->GetDevice();
        const uint32_t bytes = it->GetShape().GetBytesIncludingPadding();
        auto status = device->Copy(&buf_cursor, it->GetBufferDesc(), bytes);
        if (status != RC_SUCCESS) {
            LOG(ERROR) << "copy data failed: " << GetRetCodeStr(status);
            return status;
        }

        buf_cursor.addr = (char*)(buf_cursor.addr) + bytes;
    }

    return RC_SUCCESS;
}

RetCode LoopOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = &info->GetOutput<TensorImpl>(i)->GetShape();
            if (out_shape->GetDataFormat() == DATAFORMAT_UNKNOWN) {
                out_shape->Reshape({1, 3, 128, 128});
                out_shape->SetDataFormat(DATAFORMAT_NDARRAY);
            }
        }
        return RC_SUCCESS;
    };
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t) -> RetCode {
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto out_shape = &info->GetOutput<TensorImpl>(i)->GetShape();
            out_shape->SetDataType(DATATYPE_UNKNOWN);
        }
        return RC_SUCCESS;
    };

    auto node = GetNode();
    auto graph_data = options.graph->data.get();
    auto attr_ref = graph_data->attrs.find(node->GetId());
    if (attr_ref == graph_data->attrs.end()) {
        LOG(ERROR) << "cannot find attr for loop kernel[" << node->GetName() << "]";
        return RC_NOT_FOUND;
    }

    auto loop_param = static_cast<LoopParam*>(attr_ref->second.get());
    return op_.Init(options.resource, loop_param, ConcatOutputs);
}

KernelImpl* LoopOp::CreateKernelImpl() const {
    return op_.CreateKernelImpl();
}

}}} // namespace ppl::nn::cuda
