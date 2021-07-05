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

#include "demo_kernel.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace demo {

RetCode DemoKernel::Execute(KernelExecContext* ctx) {
    auto& type = GetType();
    LOG(INFO) << "Execute kernel: name[" << GetName() << "], type[" << type.domain << ":" << type.name << "]";

    string input_names_str = "input[";
    for (uint32_t i = 0; i < ctx->GetInputCount(); ++i) {
        auto tensor = ctx->GetInput<TensorImpl>(i);
        input_names_str += string(tensor->GetName()) + ", ";
    }
    input_names_str.resize(input_names_str.size() - 2);
    input_names_str += "]";

    string output_names_str = "output[";
    for (uint32_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto tensor = ctx->GetOutput<TensorImpl>(i);
        output_names_str += string(tensor->GetName()) + ", ";
    }
    output_names_str.resize(output_names_str.size() - 2);
    output_names_str += "]";

    LOG(INFO) << "    " << input_names_str.c_str();
    LOG(INFO) << "    " << output_names_str.c_str();

    return RC_SUCCESS;
}

}}} // namespace ppl::nn::demo
