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

#include "ppl/nn/engines/x86/optimizer/ops/onnx/matmul_op.h"
#include "ppl/nn/engines/x86/kernels/onnx/matmul_kernel.h"
#include "ppl/nn/oputils/onnx/reshape_matmul.h"
#include "ppl/kernel/x86/fp32/gemm.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode MatMulOp::DoInit(const OptKernelOptions& options) {
    infer_dims_func_ = [](InputOutputInfo* info) -> RetCode {
        return onnx::ReshapeMatMul(info, nullptr);
    };

    infer_type_func_ = GenericInferType;

    auto node = GetNode();
    auto graph_data = options.graph_data;

    auto b_data_it = graph_data->constants.find(node->GetInput(1));
    const float* b_data = nullptr;
    if (b_data_it != graph_data->constants.end()) {
        b_data = (const float*)b_data_it->second.data.GetData();
    }

    if (b_data != nullptr) {
        auto& b_shape = graph_data->shapes.find(node->GetInput(1))->second;
        bool can_pack = b_shape.dims.size() == 2;
        int dim_count = b_shape.dims.size();
        if (b_shape.dims.size() > 2) {
            can_pack = true;
            for (int d = 0; d < dim_count - 2; ++d) {
                if (b_shape.dims[d] != 1) {
                    can_pack = false;
                    break;
                }
            }
        }

        if (can_pack) {
            auto K = b_shape.dims[dim_count - 2];
            auto N = b_shape.dims[dim_count - 1];

            auto isa = options.device->GetISA();
            auto packed_b_bytes = ppl::kernel::x86::gemm_fp32_get_packed_b_bytes(isa, N, K);
            aux_param_.packed_b = (float*)ppl::common::AlignedAlloc(packed_b_bytes, 64);
            if (aux_param_.packed_b == nullptr) {
                return ppl::common::RC_OUT_OF_MEMORY;
            }
            if (ppl::common::RC_SUCCESS != ppl::kernel::x86::gemm_fp32_pack_b(
                    isa, b_data, ppl::kernel::x86::gemm_m_type::NOTRANS, N, K, b_shape.dims[dim_count - 1], aux_param_.packed_b)) {
                LOG(WARNING) << "\"" << node->GetName() << "\" gemm pack matrix-B failed, will use non-packed gemm.";
                ppl::common::AlignedFree(aux_param_.packed_b);
                aux_param_.packed_b = nullptr;
                return RC_SUCCESS;
            }
        }
    }

    return RC_SUCCESS;
}

RetCode MatMulOp::OmitConstantsData(std::map<edgeid_t, int64_t>* constants_data_refcount) {
    if (aux_param_.packed_b) {
        auto b_id = GetNode()->GetInput(1);
        auto it = constants_data_refcount->find(b_id);
        if (it != constants_data_refcount->end()) {
            it->second--;
        }
    }
    return RC_SUCCESS;
}

KernelImpl* MatMulOp::CreateKernelImpl() const {
    return CreateKernelImplWithParam<MatMulKernel>(&aux_param_);
}

}}} // namespace ppl::nn::x86
