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

#include "ppl/nn/engines/common/ppl/shape_operation_kernel.h"
#include "ppl/nn/runtime/tensor_impl.h"

namespace ppl { namespace nn { namespace common {

ppl::common::RetCode PPLShapeOperationKernel::DoExecute(KernelExecContext* ctx) {
    auto data = ctx->GetInput<TensorImpl>(0);
    auto input_dim_size = ctx->GetInput<TensorImpl>(0)->GetShape().GetRealDimCount();
    for (size_t i = 0; i < ctx->GetOutputCount(); ++i) {
        auto shape = ctx->GetOutput<TensorImpl>(i);
        auto edge = shape->GetEdge();
        auto pair = param_->alpha.find(edge->GetId());
        if (pair == param_->alpha.end()) {
            pair = param_->alpha.begin();
        }
        auto& matrix = pair->second;
        auto dim_size = matrix.real_dim < 0 ? input_dim_size : matrix.real_dim;
        shape->GetShape().SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        shape->GetShape().SetDataType(ppl::common::DATATYPE_INT64);
        if (matrix.scalar) {
            shape->GetShape().ReshapeAsScalar();
        } else {
            shape->GetShape().Reshape({dim_size});
        }
        shape->ReallocBuffer();

        std::unique_ptr<int64_t[]> shape_host(new int64_t[dim_size]);
        for (uint32_t j = 0; j < dim_size; ++j) {
            int64_t numer = matrix.numerator[j][ppl::nn::common::ShapeMatrix::MAXDIMSIZE];
            int64_t denom = matrix.denominator[j][ppl::nn::common::ShapeMatrix::MAXDIMSIZE];
            for (uint32_t k = 0; k < data->GetShape().GetDimCount(); ++k) {
                if (matrix.numerator[j][k]) {
                    numer = numer * matrix.denominator[j][k] + denom * data->GetShape().GetDim(k) * matrix.numerator[j][k];
                    denom = denom * matrix.denominator[j][k];
                }
            }
            shape_host[j] = numer / denom;
        }
        shape->CopyFromHost(shape_host.get());
    }
    return ppl::common::RC_SUCCESS;
}

}}} // namespace ppl::nn::common
