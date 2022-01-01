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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_OPS_ONNX_GEMM_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_OPS_ONNX_GEMM_OP_H_

#include <vector>

#include "ppl/nn/params/onnx/gemm_param.h"
#include "ppl/nn/engines/riscv/params/fc_param.h"
#include "ppl/nn/engines/riscv/optimizer/opt_kernel.h"
#include "ppl/nn/engines/riscv/utils/fp16fp32_cvt.h"
#include "ppl/nn/oputils/onnx/reshape_gemm.h"
#include "ppl/nn/common/logger.h"
#include "ppl/kernel/riscv/fp16/fc.h"
#include "ppl/kernel/riscv/fp32/fc.h"

namespace ppl { namespace nn { namespace riscv {

class GemmOp final : public RiscvOptKernel {
public:
    GemmOp(const ir::Node* node) : RiscvOptKernel(node), fc_param_(nullptr) {}
    ~GemmOp();
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    ppl::common::RetCode SelectDataType(const InputOutputInfo& info,
                                        std::vector<ppl::common::datatype_t>* selected_input_data_types,
                                        std::vector<ppl::common::datatype_t>* selected_output_data_types) override;
    KernelImpl* CreateKernelImpl() const override;
    bool TryFuseReLU();

private:
    FCParam* fc_param_;
    std::shared_ptr<ppl::nn::common::GemmParam> param_;
    bool gemm_fuse_relu_ = false;

    template <typename T>
    void gemm_op_graph_data_cvt(const float* graph_data, std::vector<T>& cvt_data, int32_t data_len) {
        auto data_bytes = data_len * sizeof(T);
        cvt_data.resize(data_bytes);
        if (typeid(T) == typeid(__fp16)) {
            CvtFp32ToFp16(data_len, graph_data, cvt_data.data());
        } else if (typeid(T) == typeid(float)) {
            memcpy(cvt_data.data(), graph_data, data_bytes);
        } else {
            memcpy(cvt_data.data(), graph_data, data_bytes);
        }
    }

    template <typename T>
    ppl::kernel::riscv::fc_manager<T>* gemm_op_gen_fc_algo(const ppl::kernel::riscv::fc_common_param& param,
                                                           const ppl::kernel::riscv::fc_common_algo_info& algo_info,
                                                           ppl::common::Allocator* allocator) {
        ppl::kernel::riscv::fc_base_manager* offline_manager;
        if (typeid(T) == typeid(__fp16)) {
            offline_manager = ppl::kernel::riscv::fc_algo_selector_fp16::gen_algo(param, algo_info, allocator);
        } else if (typeid(T) == typeid(float)) {
            offline_manager = ppl::kernel::riscv::fc_algo_selector_fp32::gen_algo(param, algo_info, allocator);
        } else {
            offline_manager = nullptr;
        }
        return (ppl::kernel::riscv::fc_manager<T>*)(offline_manager);
    }

    template <typename T>
    ppl::kernel::riscv::fc_common_algo_info gemm_op_select_fc_algo(const ppl::common::dataformat_t& src_format,
                                                                   const ppl::kernel::riscv::fc_common_param& param) {
        if (typeid(T) == typeid(__fp16)) {
            return ppl::kernel::riscv::fc_algo_selector_fp16::select_algo(src_format, param);
        } else if (typeid(T) == typeid(float)) {
            return ppl::kernel::riscv::fc_algo_selector_fp32::select_algo(src_format, param);
        } else {
            return {ppl::kernel::riscv::fc_common_algo::unknown};
        }
    }

    template <typename T>
    ppl::common::RetCode InitFC(const OptKernelOptions& options) {
        auto status = GenericLoadParam(options, &param_);
        if (status != ppl::common::RC_SUCCESS) {
            LOG(ERROR) << "load param failed: " << ppl::common::GetRetCodeStr(status);
            return status;
        }

        auto node = GetNode();
        auto graph_data = options.graph_data;

        auto weight_data_it = graph_data->constants.find(node->GetInput(1));
        const float* weight_data = nullptr;
        std::vector<T> weight_cvt;
        if (weight_data_it != graph_data->constants.end()) {
            weight_data = (const float*)weight_data_it->second.data.data();
            int64_t weight_len = weight_data_it->second.data.size() / sizeof(float);
            gemm_op_graph_data_cvt<T>(weight_data, weight_cvt, weight_len);
        }

        const float* bias_data = nullptr;
        std::vector<T> bias_cvt;
        if (node->GetInputCount() == 3) {
            auto bias_data_it = graph_data->constants.find(node->GetInput(2));
            if (bias_data_it != graph_data->constants.end()) {
                bias_data = (const float*)bias_data_it->second.data.data();
                int64_t bias_len = bias_data_it->second.data.size() / sizeof(float);
                gemm_op_graph_data_cvt<T>(bias_data, bias_cvt, bias_len);
            }
        }

        if (!param_->transA && param_->transB && weight_data != nullptr) {
            if (!fc_param_) {
                fc_param_ = new FCParam;
            }
            if (!fc_param_) {
                return ppl::common::RC_OUT_OF_MEMORY;
            }

            const ir::Shape& weight_shape = graph_data->shapes.find(node->GetInput(1))->second;
            fc_param_->param.num_output = weight_shape.dims[0];
            fc_param_->param.channels = weight_shape.dims[1];
            fc_param_->param.fuse_flag = 0;

            fc_param_->algo_info = gemm_op_select_fc_algo<T>(ppl::common::DATAFORMAT_NDARRAY, fc_param_->param);
            if (fc_param_->algo_info.algo_type == ppl::kernel::riscv::fc_common_algo::unknown) {
                LOG(DEBUG) << "FC select algorithm failed";
                return ppl::common::RC_UNSUPPORTED;
            } else {
                auto mgr =
                    gemm_op_gen_fc_algo<T>(fc_param_->param, fc_param_->algo_info, options.device->GetAllocator());
                if (bias_data != nullptr) {
                    mgr->gen_cvt_weights(weight_cvt.data(), bias_cvt.data());
                } else {
                    std::vector<T> zero_bias(weight_shape.dims[0], 0.0f);
                    mgr->gen_cvt_weights(weight_cvt.data(), zero_bias.data());
                }
                fc_param_->mgr = mgr;
            }
        }

        infer_dims_func_ = [this](InputOutputInfo* info) -> ppl::common::RetCode {
            return oputils::ReshapeGemm(info, param_.get());
        };

        infer_type_func_ = GenericInferType;

        return ppl::common::RC_SUCCESS;
    }
};

}}} // namespace ppl::nn::riscv

#endif