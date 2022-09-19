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

#include <float.h>
#include "ppl/nn/engines/x86/optimizer/ops/onnx/gru_op.h"
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/kernel/x86/common/rnn_common.h"
#include "ppl/nn/engines/x86/kernels/onnx/gru_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_gru.h"
#include "ppl/nn/common/logger.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

GRUOp::~GRUOp() {
    if (aux_param_.packed_W[0])
        ppl::common::AlignedFree(aux_param_.packed_W[0]);
    if (aux_param_.packed_W[1])
        ppl::common::AlignedFree(aux_param_.packed_W[1]);
    if (aux_param_.packed_Rzr[0])
        ppl::common::AlignedFree(aux_param_.packed_Rzr[0]);
    if (aux_param_.packed_Rzr[1])
        ppl::common::AlignedFree(aux_param_.packed_Rzr[1]);
    if (aux_param_.packed_Rh[0])
        ppl::common::AlignedFree(aux_param_.packed_Rh[0]);
    if (aux_param_.packed_Rh[1])
        ppl::common::AlignedFree(aux_param_.packed_Rh[1]);
}

RetCode GRUOp::Init(const OptKernelOptions& options) {
    auto status = GenericLoadParam(options, &param_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "load param failed: " << GetRetCodeStr(status);
        return status;
    }
    aux_param_.param = param_.get();
    if (param_->activations.size() || param_->activation_alpha.size() || param_->activation_beta.size()) {
        LOG(ERROR) << "GRU dose not support customize activations and parameters";
        return ppl::common::RC_UNSUPPORTED;
    }

    if (param_->clip != FLT_MAX) {
        LOG(ERROR) << "GRU dose not support clip";
        return ppl::common::RC_UNSUPPORTED;
    }

    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        return ppl::nn::onnx::ReshapeGRU(info, param_.get());
    };

    infer_type_func_ = GenericInferType;

    auto isa = options.device->GetISA();
    auto node = GetNode();
    auto graph_data = options.graph_data;
    auto type_b = ppl::kernel::x86::gemm_m_type::TRANS;
    bool bidirection = param_->direction == ppl::nn::onnx::GRUParam::DIR_BIDIRECTIONAL ? true : false;

    auto w_data_it = graph_data->constants.find(node->GetInput(1));
    const float* w_data = nullptr;
    if (w_data_it != graph_data->constants.end()) {
        w_data = (const float*)w_data_it->second.data.GetData();
    }
    auto& w_shape = graph_data->shapes.find(node->GetInput(1))->second;
    auto K_w = w_shape.dims[2]; // trans
    auto N_w = w_shape.dims[1];
    if (w_data != nullptr) {
        auto packed_w_bytes = ppl::kernel::x86::gemm_fp32_get_packed_b_bytes(isa, N_w, K_w);
        aux_param_.packed_W[0] = (float*)ppl::common::AlignedAlloc(packed_w_bytes, 64);
        if (ppl::common::RC_SUCCESS !=
            ppl::kernel::x86::gemm_fp32_pack_b(isa, w_data, type_b, N_w, K_w, w_shape.dims[2],
                                               aux_param_.packed_W[0])) {
            LOG(WARNING) << "\"" << node->GetName() << "\" gru pack matrix w[0] failed, will use non-packed gemm.";
            ppl::common::AlignedFree(aux_param_.packed_W[0]);
            aux_param_.packed_W[0] = nullptr;
        }
        if (bidirection) {
            aux_param_.packed_W[1] = (float*)ppl::common::AlignedAlloc(packed_w_bytes, 64);
            if (ppl::common::RC_SUCCESS !=
                ppl::kernel::x86::gemm_fp32_pack_b(isa, w_data + K_w * N_w, type_b, N_w, K_w, w_shape.dims[2],
                                                   aux_param_.packed_W[1])) {
                LOG(WARNING) << "\"" << node->GetName() << "\" gru pack matrix w[1] failed, will use non-packed gemm.";
                ppl::common::AlignedFree(aux_param_.packed_W[1]);
                aux_param_.packed_W[1] = nullptr;
            }
        }
    }

    auto& r_shape = graph_data->shapes.find(node->GetInput(2))->second; // [num_directions, 3*hidden_size, hidden_size].
    auto K_r = r_shape.dims[2]; // trans
    auto N_r = r_shape.dims[1] / 3; //  3*hidden_size
    auto r_data_it = graph_data->constants.find(node->GetInput(2));
    const float* r_data = nullptr;
    if (r_data_it != graph_data->constants.end()) {
        r_data = (const float*)r_data_it->second.data.GetData();
    }
    if (r_data != nullptr) {
        auto packed_zr_bytes = ppl::kernel::x86::gemm_fp32_get_packed_b_bytes(isa, N_r * 2, K_r);
        aux_param_.packed_Rzr[0] = (float*)ppl::common::AlignedAlloc(packed_zr_bytes, 64);
        auto packed_h_bytes = ppl::kernel::x86::gemm_fp32_get_packed_b_bytes(isa, N_r, K_r);
        aux_param_.packed_Rh[0] = (float*)ppl::common::AlignedAlloc(packed_h_bytes, 64);
        if (ppl::common::RC_SUCCESS !=
            ppl::kernel::x86::gemm_fp32_pack_b(isa, r_data, type_b, N_r * 2, K_r, r_shape.dims[2],
                                               aux_param_.packed_Rzr[0])) {
            LOG(WARNING) << "\"" << node->GetName() << "\" gru pack matrix Rzr[0] failed, will use non-packed gemm.";
            ppl::common::AlignedFree(aux_param_.packed_Rzr[0]);
            aux_param_.packed_Rzr[0] = nullptr;
        }
        if (ppl::common::RC_SUCCESS !=
            ppl::kernel::x86::gemm_fp32_pack_b(isa, r_data + (2 * N_r * K_r), type_b, N_r, K_r, r_shape.dims[2],
                                               aux_param_.packed_Rh[0])) {
            LOG(WARNING) << "\"" << node->GetName() << "\" gru pack matrix Rh[0] failed, will use non-packed gemm.";
            ppl::common::AlignedFree(aux_param_.packed_Rh[0]);
            aux_param_.packed_Rh[0] = nullptr;
        }
        if (bidirection) {
            aux_param_.packed_Rzr[1] = (float*)ppl::common::AlignedAlloc(packed_zr_bytes, 64);
            aux_param_.packed_Rh[1] = (float*)ppl::common::AlignedAlloc(packed_h_bytes, 64);
            if (ppl::common::RC_SUCCESS !=
                ppl::kernel::x86::gemm_fp32_pack_b(isa, r_data + 3 * K_r * N_r, type_b, N_r * 2, K_r, r_shape.dims[2],
                                                   aux_param_.packed_Rzr[1])) {
                LOG(WARNING) << "\"" << node->GetName()
                             << "\" gru pack matrix Rzr[1] failed, will use non-packed gemm.";
                ppl::common::AlignedFree(aux_param_.packed_Rzr[1]);
                aux_param_.packed_Rzr[1] = nullptr;
            }
            if (ppl::common::RC_SUCCESS !=
                ppl::kernel::x86::gemm_fp32_pack_b(isa, r_data + (5 * N_r * K_r), type_b, N_r, K_r, r_shape.dims[2],
                                                   aux_param_.packed_Rh[1])) {
                LOG(WARNING) << "\"" << node->GetName() << "\" gru pack matrix Rh[1] failed, will use non-packed gemm.";
                ppl::common::AlignedFree(aux_param_.packed_Rh[1]);
                aux_param_.packed_Rh[1] = nullptr;
            }
        }
    }

    return RC_SUCCESS;
}

KernelImpl* GRUOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<GRUKernel>(&aux_param_);
}

}}} // namespace ppl::nn::x86
