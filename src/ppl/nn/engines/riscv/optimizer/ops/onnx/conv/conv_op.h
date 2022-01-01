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

#ifndef _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_ONNX_OPS_ONNX_CONV_OP_H_
#define _ST_HPC_PPL_NN_ENGINES_RISCV_OPTIMIZER_ONNX_OPS_ONNX_CONV_OP_H_

#include <cstring>
#include <vector>

#include "ppl/nn/engines/riscv/impls/include/ppl/kernel/riscv/common/conv2d.h"
#include "ppl/nn/engines/riscv/impls/include/ppl/kernel/riscv/fp16/conv2d.h"
#include "ppl/nn/engines/riscv/impls/include/ppl/kernel/riscv/fp32/conv2d.h"
#include "ppl/nn/params/onnx/convolution_param.h"
#include "ppl/nn/engines/riscv/params/conv_param.h"
#include "ppl/nn/engines/riscv/optimizer/opt_kernel.h"
#include "ppl/nn/engines/riscv/utils/fp16fp32_cvt.h"

namespace ppl { namespace nn { namespace riscv {

class ConvOp final : public RiscvOptKernel {
public:
    ConvOp(const ir::Node* node) : RiscvOptKernel(node), conv2d_param_(nullptr){};
    ~ConvOp();
    ppl::common::RetCode Init(const OptKernelOptions& options) override;
    KernelImpl* CreateKernelImpl() const override;
    ppl::common::RetCode SelectFormat(const InputOutputInfo& info,
                                      std::vector<ppl::common::dataformat_t>* selected_input_formats,
                                      std::vector<ppl::common::dataformat_t>* selected_output_formats) override;
    ppl::common::RetCode SelectDataType(const InputOutputInfo& info,
                                        std::vector<ppl::common::datatype_t>* selected_input_data_types,
                                        std::vector<ppl::common::datatype_t>* selected_output_data_types) override;
    ppl::common::RetCode SelectAlgorithm(const InputOutputInfo& info, const OptKernelOptions& options) override;

private:
    template <typename T>
    void conv_op_graph_data_cvt(float* graph_data, std::vector<T>& cvt_data, int32_t data_len) {
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
    ppl::kernel::riscv::conv2d_offline_manager<T>* conv_op_gen_algo(
        const ppl::kernel::riscv::conv2d_common_param& param,
        const ppl::kernel::riscv::conv2d_common_algo_info& algo_info, ppl::common::Allocator* allocator) {
        ppl::kernel::riscv::conv2d_base_offline_manager* offline_manager;
        if (typeid(T) == typeid(__fp16)) {
            offline_manager = ppl::kernel::riscv::conv2d_fp16_algo_selector::gen_algo(param, algo_info, allocator);
        } else if (typeid(T) == typeid(float)) {
            offline_manager = ppl::kernel::riscv::conv2d_fp32_algo_selector::gen_algo(param, algo_info, allocator);
        } else {
            offline_manager = ppl::kernel::riscv::conv2d_fp32_algo_selector::gen_algo(param, algo_info, allocator);
        }
        return (ppl::kernel::riscv::conv2d_offline_manager<T>*)(offline_manager);
    }

    template <typename T>
    ppl::kernel::riscv::conv2d_common_algo_info conv_op_select_algo(
        const ppl::nn::TensorShape& input_shape, const ppl::kernel::riscv::conv2d_common_param& param) {
        if (typeid(T) == typeid(__fp16)) {
            return ppl::kernel::riscv::conv2d_fp16_algo_selector::select_algo(input_shape, param);
        } else if (typeid(T) == typeid(float)) {
            return ppl::kernel::riscv::conv2d_fp32_algo_selector::select_algo(input_shape, param);
        } else {
            return ppl::kernel::riscv::conv2d_fp32_algo_selector::select_algo(input_shape, param);
        }
    }

    template <typename T>
    ppl::common::RetCode SelectAlgorithmGeneric(const InputOutputInfo& info, const OptKernelOptions& options) {
        auto node = GetNode();
        auto graph_data = options.graph_data;

        auto weight_data_it = graph_data->constants.find(node->GetInput(1));
        if (weight_data_it == graph_data->constants.end()) {
            LOG(DEBUG) << "ConvOp constant weight not found, will use conv runtime.";
            return ppl::common::RC_SUCCESS;
        }

        std::vector<T> weight_data_cvt;
        float* weight_data = (float*)weight_data_it->second.data.data();
        int64_t weight_len = weight_data_it->second.data.size() / sizeof(float);
        conv_op_graph_data_cvt<T>(weight_data, weight_data_cvt, weight_len);

        std::vector<T> bias_data_cvt;
        float* bias_data = nullptr;
        if (node->GetInputCount() == 3) {
            auto bias_data_it = graph_data->constants.find(node->GetInput(2));
            if (bias_data_it == graph_data->constants.end()) {
                LOG(DEBUG) << "ConvOp constant weight not found, will use conv runtime.";
                return ppl::common::RC_SUCCESS;
            }
            bias_data = (float*)bias_data_it->second.data.data();
            int64_t bias_len = bias_data_it->second.data.size() / sizeof(float);
            conv_op_graph_data_cvt<T>(bias_data, bias_data_cvt, bias_len);
        }

        const ir::Shape& weight_shape = graph_data->shapes.find(node->GetInput(1))->second;
        const int64_t kernel_dims = weight_shape.dims.size() - 2;

        param_->bias_term = (node->GetInputCount() == 3) ? 1 : 0;
        param_->num_output = weight_shape.dims[0];
        param_->channels = weight_shape.dims[1] * param_->group;

        // Check Param
        {
            const ppl::nn::common::ConvolutionParam& conv_param = *param_.get();
            for (int64_t i = 0; i < kernel_dims; ++i) {
                if (conv_param.pads[i] != conv_param.pads[i + kernel_dims]) {
                    return ppl::common::RC_UNSUPPORTED;
                }
            }

            if (kernel_dims == 2) {
                if (!conv2d_param_) {
                    conv2d_param_ = new Convolution2DParam;
                }
                if (!conv2d_param_) {
                    return ppl::common::RC_OUT_OF_MEMORY;
                }

                ppl::kernel::riscv::conv2d_common_param& conv2d_kernel_param = conv2d_param_->param;
                conv2d_kernel_param.kernel_h = conv_param.kernel_shape[0];
                conv2d_kernel_param.kernel_w = conv_param.kernel_shape[1];
                conv2d_kernel_param.stride_h = conv_param.strides[0];
                conv2d_kernel_param.stride_w = conv_param.strides[1];
                conv2d_kernel_param.pad_h = conv_param.pads[0];
                conv2d_kernel_param.pad_w = conv_param.pads[1];
                conv2d_kernel_param.dilation_h = conv_param.dilations[0];
                conv2d_kernel_param.dilation_w = conv_param.dilations[1];
                conv2d_kernel_param.group = conv_param.group;
                conv2d_kernel_param.num_output = conv_param.num_output;
                conv2d_kernel_param.channels = conv_param.channels;
                // conv2d_kernel_param.fuse_flag = 0;

                auto algo_info = conv_op_select_algo<T>(*info.GetInput<TensorImpl>(0)->GetShape(), conv2d_param_->param);

                if (algo_info.algo_type == ppl::kernel::riscv::conv2d_common_algo::unknown) {
                    LOG(ERROR) << "Conv select algorithm failed, use fallback kernel";
                    return ppl::common::RC_UNSUPPORTED;
                }

                ppl::kernel::riscv::conv2d_offline_manager<T>* mgr =
                    conv_op_gen_algo<T>(conv2d_param_->param, algo_info, options.device->GetAllocator());
                {
                    if (nullptr == mgr) {
                        return ppl::common::RC_UNSUPPORTED;
                    }

                    mgr->fast_init_tunning_param();
                    if (options.engine_options->tune_param_flag) {
                        auto& src_shape = *info.GetInput<TensorImpl>(0)->GetShape();
                        auto& dst_shape = *info.GetOutput<TensorImpl>(0)->GetShape();
                        std::vector<T> tunning_src, tunning_dst;
                        tunning_src.resize(src_shape.GetElementsIncludingPadding());
                        tunning_dst.resize(dst_shape.GetElementsIncludingPadding());
                        mgr->pick_best_tunning_param(tunning_src.data(), weight_data_cvt.data(), tunning_dst.data(),
                                                     src_shape, dst_shape);
                    }

                    if (bias_data != nullptr) {
                        mgr->gen_cvt_weights(weight_data_cvt.data(), bias_data_cvt.data());
                    } else {
                        std::vector<T> zero_bias(conv2d_kernel_param.num_output, 0.0f);
                        mgr->gen_cvt_weights(weight_data_cvt.data(), zero_bias.data());
                    }
                }
                conv2d_param_->algo_info = algo_info;
                conv2d_param_->mgr = mgr;

            } else {
                LOG(ERROR) << "Unsupported kernel dim: " << kernel_dims;
                return ppl::common::RC_UNSUPPORTED;
            }
        }

        return ppl::common::RC_SUCCESS;
    }

private:
    Convolution2DParam* conv2d_param_;
    std::shared_ptr<ppl::nn::common::ConvolutionParam> param_;
};

}}} // namespace ppl::nn::riscv

#endif
