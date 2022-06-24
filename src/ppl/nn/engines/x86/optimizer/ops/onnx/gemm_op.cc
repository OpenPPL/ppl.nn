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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/gemm_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_gemm.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

GemmOp::~GemmOp() {
    if (aux_param_.packed_b) ppl::common::AlignedFree(aux_param_.packed_b);
}

RetCode GemmOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }

    auto node = GetNode();
    auto graph_data = options.graph_data;

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeGemm(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    auto b_data_it = graph_data->constants.find(node->GetInput(1));
    const float* b_data = nullptr;
    if (b_data_it != graph_data->constants.end()) {
        b_data = (const float*)b_data_it->second.data.GetData();
    }

    aux_param_.trans_a = param_->transA;
    aux_param_.trans_b = param_->transB;
    aux_param_.alpha = param_->alpha;
    aux_param_.beta = param_->beta;

    if (b_data != nullptr) {
        auto& b_shape = graph_data->shapes.find(node->GetInput(1))->second;
        auto K = b_shape.dims[0 + aux_param_.trans_b];
        auto N = b_shape.dims[1 - aux_param_.trans_b];

        auto isa = options.device->GetISA();
        auto type_b = aux_param_.trans_b ? ppl::kernel::x86::gemm_m_type::TRANS : ppl::kernel::x86::gemm_m_type::NOTRANS;

        auto packed_b_bytes = ppl::kernel::x86::gemm_fp32_get_packed_b_bytes(isa, N, K);
        aux_param_.packed_b = (float*)ppl::common::AlignedAlloc(packed_b_bytes, 64);
        if (ppl::common::RC_SUCCESS != ppl::kernel::x86::gemm_fp32_pack_b(
                isa, b_data, type_b, N, K, b_shape.dims[1], aux_param_.packed_b)) {
            LOG(WARNING) << "\"" << node->GetName() << "\" gemm pack matrix-B failed, will use non-packed gemm.";
            ppl::common::AlignedFree(aux_param_.packed_b);
            aux_param_.packed_b = nullptr;
            return RC_SUCCESS;
        }
    }

    return RC_SUCCESS;
}

RetCode GemmOp::OmitConstantsData(std::map<edgeid_t, int64_t>* constants_data_refcount) {
    if (aux_param_.packed_b) {
        auto b_id = GetNode()->GetInput(1);
        auto it = constants_data_refcount->find(b_id);
        if (it != constants_data_refcount->end()) {
            it->second--;
        }
    }
    return RC_SUCCESS;
}

bool GemmOp::TryFuseReLU() {
    aux_param_.post = ppl::kernel::x86::gemm_post::RELU;
    return true;
}

KernelImpl* GemmOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<GemmKernel>(&aux_param_);
}

}}} // namespace ppl::nn::x86
