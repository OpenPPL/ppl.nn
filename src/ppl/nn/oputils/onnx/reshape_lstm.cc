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

#include "ppl/nn/oputils/onnx/reshape_lstm.h"
#include "ppl/nn/runtime/tensor_impl.h"
using namespace ppl::common;
using namespace ppl::nn::common;

namespace ppl { namespace nn { namespace oputils {

RetCode ReshapeLSTM(InputOutputInfo* info, const void* arg) {
    auto param = (const LSTMParam*)arg;
    const TensorShape& in_shape = *info->GetInput<TensorImpl>(0)->GetShape();
    const int64_t seq_len = in_shape.GetDim(0);
    const int64_t batch = in_shape.GetDim(1);
    const int64_t num_directions = param->direction == LSTMParam::DIR_BIDIRECTIONAL ? 2 : 1;

    if (info->GetOutputCount() > 0) {
        info->GetOutput<TensorImpl>(0)->GetShape()->Reshape({seq_len, num_directions, batch, param->hidden_size});
    }
    if (info->GetOutputCount() > 1) {
        info->GetOutput<TensorImpl>(1)->GetShape()->Reshape({num_directions, batch, param->hidden_size});
    }
    if (info->GetOutputCount() > 2) {
        info->GetOutput<TensorImpl>(2)->GetShape()->Reshape({num_directions, batch, param->hidden_size});
    }

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::oputils
