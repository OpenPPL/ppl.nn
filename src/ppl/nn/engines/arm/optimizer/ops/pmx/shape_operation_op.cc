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

#include "ppl/nn/engines/arm/optimizer/ops/pmx/shape_operation_op.h"

#include "ppl/nn/common/logger.h"
#include "ppl/nn/engines/common/pmx/shape_operation_kernel.h"

#ifdef PPLNN_ENABLE_PMX_MODEL
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/engines/arm/pmx/generated/arm_op_params_generated.h"
#endif

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

ShapeOperationOp::ShapeOperationOp(const ir::Node* node) : ArmOptKernel(node), op_(node) {
    infer_type_func_ = GenericInferType;
    infer_dims_func_ = GenericInferDims;
}

RetCode ShapeOperationOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam<ppl::nn::pmx::ShapeOperationParam>(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    return RC_SUCCESS;
}

#ifdef PPLNN_ENABLE_PMX_MODEL

ppl::common::RetCode ShapeOperationOp::SerializeData(const ::ppl::nn::pmx::SerializationContext& ctx, utils::DataStream* ds) const {
    const std::vector<edgeid_t>& eid2seq = ctx.eid2seq;

    flatbuffers::FlatBufferBuilder shape_op_param_builder;
    std::vector<flatbuffers::Offset<ppl::nn::pmx::arm::ShapeMatrixP>> alpha;
    for (auto const & item : param_->alpha) {
        uint32_t edge_id = item.first < eid2seq.size() ? eid2seq[item.first] : -1;
        LOG(ERROR) << item.first;
        LOG(ERROR) << edge_id;
        const ppl::nn::pmx::ShapeMatrix & matrix = item.second;
        LOG(ERROR) << "real_dim";
        LOG(ERROR) << matrix.real_dim;
        LOG(ERROR) << "scalar";
        LOG(ERROR) << matrix.scalar;

        std::vector<int64_t> numerator(81);
        std::vector<int64_t> denominator(81);
        memcpy(numerator.data(),   &matrix.numerator[0][0],   81 * sizeof(int64_t));
        memcpy(denominator.data(), &matrix.denominator[0][0], 81 * sizeof(int64_t));
        auto fb_shape_matrix = ppl::nn::pmx::arm::CreateShapeMatrixPDirect(shape_op_param_builder, 
            edge_id, &numerator, &denominator, matrix.real_dim, (int8_t)(matrix.scalar ? 1 : 0));
        alpha.push_back(fb_shape_matrix);
    }
    auto fb_shape_op_param = ppl::nn::pmx::arm::CreateShapeOperationParamDirect(shape_op_param_builder, &alpha);
    auto fp_pmx_op_data = ppl::nn::pmx::arm::CreatePmxOpData(shape_op_param_builder, ppl::nn::pmx::arm::PmxOpType_ShapeOperationParam, fb_shape_op_param.Union());

    auto fp_output_info = ppl::nn::pmx::arm::CreateOutputInfoDirect(shape_op_param_builder, &common_param_.output_types, &common_param_.output_formats);
    auto fb_op_data = ppl::nn::pmx::arm::CreateOpData(shape_op_param_builder, fp_output_info, ppl::nn::pmx::arm::PrivateDataType_PmxOpData, fp_pmx_op_data.Union());
    ppl::nn::pmx::arm::FinishOpDataBuffer(shape_op_param_builder, fb_op_data);

    flatbuffers::FlatBufferBuilder op_builder;
    auto fb_data = op_builder.CreateVector(shape_op_param_builder.GetBufferPointer(), shape_op_param_builder.GetSize());
    auto fb_root = ppl::nn::pmx::onnx::CreateOpParam(op_builder, ppl::nn::pmx::onnx::OpParamType_NONE, 0, fb_data);
    ppl::nn::pmx::onnx::FinishOpParamBuffer(op_builder, fb_root);
    return ds->Write(op_builder.GetBufferPointer(), op_builder.GetSize());
    return 0;
}

ppl::common::RetCode ShapeOperationOp::DeserializeData(const ::ppl::nn::pmx::DeserializationContext& ctx, const void* base, uint64_t size) {
    // std::vector<edgeid_t>& seq2eid = ctx.seq2eid;

    auto fb_op_param = ppl::nn::pmx::onnx::GetOpParam(base);
    param_ = std::make_shared<ppl::nn::pmx::ShapeOperationParam>();

    auto arm_op_data = ppl::nn::pmx::arm::GetOpData(fb_op_param->data_()->data());
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_op_data->output_info()->dtype(), &common_param_.output_types);
    ppl::nn::pmx::utils::Fbvec2Stdvec(arm_op_data->output_info()->dformat(), &common_param_.output_formats);

    auto arm_shape_op_param = arm_op_data->value_as_PmxOpData()->value_as_ShapeOperationParam();
    std::vector<flatbuffers::Offset<ppl::nn::pmx::arm::ShapeMatrixP>> alpha;
    auto fb_matrix_vec = arm_shape_op_param->shape_matrix();
    for (uint32_t i = 0; i < fb_matrix_vec->size(); ++i) {
        ppl::nn::pmx::ShapeMatrix matrix;
        auto fb_matrix = fb_matrix_vec->Get(i);
        memcpy(&matrix.numerator[0][0],   fb_matrix->numerator()->data(),   81 * sizeof(int64_t));
        memcpy(&matrix.denominator[0][0], fb_matrix->denominator()->data(), 81 * sizeof(int64_t));
        matrix.real_dim = fb_matrix->real_dim();
        matrix.scalar = (fb_matrix->scalar() == 1);
 
        LOG(ERROR) << fb_matrix->edge();
        LOG(ERROR) << "real_dim";
        LOG(ERROR) << matrix.real_dim;
        LOG(ERROR) << "scalar";
        LOG(ERROR) << matrix.scalar;
        param_->alpha[fb_matrix->edge()] = matrix;
    }

    op_ = ppl::nn::pmx::ShapeOperationOp(GetNode());
    return RC_SUCCESS;
}

#endif

KernelImpl* ShapeOperationOp::CreateKernelImpl() const {
    auto kernel = op_.CreateKernelImpl();
    ((ppl::nn::pmx::ShapeOperationKernel*)kernel)->SetParam(param_.get());
    return kernel;
}

}}} // namespace ppl::nn::arm
