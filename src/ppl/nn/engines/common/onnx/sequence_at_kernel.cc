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

#include "ppl/nn/engines/common/onnx/sequence_at_kernel.h"
#include "ppl/nn/engines/utils.h"
#include "ppl/nn/runtime/tensor_sequence.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace common {

RetCode SequenceAtKernel::DoExecute(KernelExecContext* ctx) {
    auto seq = ctx->GetInput<TensorSequence>(0);
    auto pos = ctx->GetInput<TensorImpl>(1);

    int64_t idx;
    switch (pos->GetShape().GetDataType()) {
        case DATATYPE_INT32: {
            int32_t temp;
            auto status = pos->CopyToHost(&temp);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "get index failed: " << GetRetCodeStr(status);
                return status;
            }
            idx = temp;
            break;
        }
        case DATATYPE_INT64: {
            auto status = pos->CopyToHost(&idx);
            if (status != RC_SUCCESS) {
                LOG(ERROR) << "get index failed: " << GetRetCodeStr(status);
                return status;
            }
            break;
        }
        default:
            LOG(ERROR) << "unspoorted data type [" << GetDataTypeStr(pos->GetShape().GetDataType()) << "]";
            return RC_INVALID_VALUE;
    }

    auto nr_tensor = seq->GetElementCount();

    if (idx < 0) {
        idx += nr_tensor;
    }

    if (idx < 0 || idx >= nr_tensor) {
        LOG(ERROR) << "invalid position[" << idx << "]. tensor count[" << nr_tensor << "]";
        return RC_INVALID_VALUE;
    }

    auto src = seq->GetElement(idx);
    auto dst = ctx->GetOutput<TensorImpl>(0);

    dst->GetShape() = src->GetShape();

    auto status = utils::CopyBuffer(src->GetBufferDesc(), src->GetShape(), src->GetDevice(), dst);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "copy pos[" << idx << "] of tensor sequence [" << seq->GetEdge()->GetName() << "] to tensor["
                   << dst->GetName() << "] failed: " << GetRetCodeStr(status);
    }

    return status;
}

}}} // namespace ppl::nn::common
