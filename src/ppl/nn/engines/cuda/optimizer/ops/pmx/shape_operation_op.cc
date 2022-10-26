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

#include "ppl/nn/engines/cuda/optimizer/ops/pmx/shape_operation_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/common/pmx/shape_operation_kernel.h"

using namespace std;
using namespace ppl::common;
using namespace ppl::nn::pmx;

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/pmx/ppl_shape_operation.h"
#endif

namespace ppl { namespace nn { namespace cuda {

RetCode PPLShapeOperationOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ShapeOperationParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

PPLShapeOperationOp::PPLShapeOperationOp(const ir::Node* node) : CudaOptKernel(node), op_(node) {
    infer_type_func_ = [](InputOutputInfo* info, std::vector<CudaTensorQuant>* quant, datatype_t type) -> RetCode {
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto shape = info->GetOutput<TensorImpl>(i)->GetShape();
            shape->SetDataType(ppl::common::DATATYPE_INT64);
        }
        return RC_SUCCESS;
    };

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        auto param = (ShapeOperationParam*)this->GetParam();
        auto input_dim_size = info->GetInput<TensorImpl>(0)->GetShape()->GetRealDimCount();
        for (uint32_t i = 0; i < info->GetOutputCount(); ++i) {
            auto output_shape = info->GetOutput<TensorImpl>(i)->GetShape();
            auto edge = info->GetOutput<TensorImpl>(i)->GetEdge();
            auto pair = param->alpha.find(edge->GetId());
            if (pair == param->alpha.end()) {
                LOG(ERROR) << "Can not find param for edge[" << edge->GetName() << "].";
                return RC_NOT_FOUND;
            }
            auto& matrix = pair->second;
            output_shape->Reshape({matrix.real_dim < 0 ? input_dim_size : matrix.real_dim});
        }
        return RC_SUCCESS;
    };
}

RetCode PPLShapeOperationOp::Finalize(const OptKernelOptions& options) {
    auto status = SetCommonParam(options);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load common param failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

KernelImpl* PPLShapeOperationOp::CreateKernelImpl() const {
    auto kernel = op_.CreateKernelImpl();
    ((ppl::nn::pmx::ShapeOperationKernel*)kernel)->SetParam(&param_);
    return kernel;
}

#ifdef PPLNN_ENABLE_PMX_MODEL
    ppl::common::RetCode PPLShapeOperationOp::SerializeData(const pmx::SerializationContext& ctx, utils::DataStream* ds) const {
        flatbuffers::FlatBufferBuilder builder;
        auto fb_param = pmx::pmx::SerializeShapeOperationParam(ctx.eid2seq, param_, &builder);
        auto fb_op_param = pmx::pmx::CreateOpParam(builder, pmx::pmx::OpParamType_ShapeOperationParam, fb_param.Union());
        pmx::pmx::FinishOpParamBuffer(builder, fb_op_param);
        return ds->Write(builder.GetBufferPointer(), builder.GetSize());
    }
    ppl::common::RetCode PPLShapeOperationOp::DeserializeData(const pmx::DeserializationContext&, const void* base, uint64_t size) {
        auto fb_op_param = pmx::pmx::GetOpParam(base);
        auto fb_pplshapeoperaion_param = fb_op_param->value_as_ShapeOperationParam();
        pmx::pmx::DeserializeShapeOperationParam(*fb_pplshapeoperaion_param, &param_);
        return ppl::common::RC_SUCCESS;
    }
#endif

}}} // namespace ppl::nn::cuda
