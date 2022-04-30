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

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/range_op.h"
#include "ppl/nn/engines/riscv/kernels/onnx/range_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_range.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

RetCode RangeOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeRange(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    return RC_SUCCESS;
}

RetCode RangeOp::SelectFormat(const InputOutputInfo& info, vector<dataformat_t>* selected_input_formats,
                              vector<dataformat_t>* selected_output_formats) {
    selected_input_formats->at(0) = DATAFORMAT_NDARRAY;
    selected_input_formats->at(1) = DATAFORMAT_NDARRAY;
    selected_input_formats->at(2) = DATAFORMAT_NDARRAY;
    selected_output_formats->at(0) = DATAFORMAT_NDARRAY;

    return RC_SUCCESS;
}

KernelImpl* RangeOp::CreateKernelImpl() const {
    return CreateKernelImplWithoutParam<RangeKernel>();
}

}}} // namespace ppl::nn::riscv