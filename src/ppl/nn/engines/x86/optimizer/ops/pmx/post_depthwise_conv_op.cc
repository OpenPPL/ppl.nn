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

#include "ppl/nn/engines/x86/optimizer/ops/pmx/post_depthwise_conv_op.h"
#include "ppl/nn/engines/x86/kernels/pmx/post_depthwise_conv2d_kernel.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

PostDepthwiseConvOp::~PostDepthwiseConvOp() {
    if (pd_conv2d_param_ != nullptr) {
        if (pd_conv2d_param_->mgr != nullptr) {
            pd_conv2d_param_->mgr->release_cvt_weights();
        }
        if (pd_conv2d_param_->conv2d_param != nullptr) {
            delete pd_conv2d_param_->conv2d_param;
        }
        if (pd_conv2d_param_->depthwise_conv2d_param != nullptr) {
            delete pd_conv2d_param_->depthwise_conv2d_param;
        }
        delete pd_conv2d_param_;
    }
}

PostDepthwiseConv2dParam* PostDepthwiseConvOp::TryMakePostDepthwiseConv2dParam(ConvOp *conv_op, ConvOp *post_conv_op) {
    if (!conv_op->conv2d_param_ || !post_conv_op->conv2d_param_) {
        return nullptr;
    }
    if (conv_op->conv2d_param_->fallback_mgr || post_conv_op->conv2d_param_->fallback_mgr) {
        return nullptr;
    }

    auto pd_c2d_algo_info = ppl::kernel::x86::pd_conv2d_algo_selector::select_algo(
        conv_op->conv2d_param_->algo_info,
        post_conv_op->conv2d_param_->algo_info,
        conv_op->conv2d_param_->param,
        post_conv_op->conv2d_param_->param);

    if (pd_c2d_algo_info.algo_type == ppl::kernel::x86::pd_conv2d_fp32_algo::UNKNOWN) {
        return nullptr;
    }

    auto pd_c2d_mgr = ppl::kernel::x86::pd_conv2d_algo_selector::gen_algo(
        pd_c2d_algo_info, conv_op->conv2d_param_->mgr, post_conv_op->conv2d_param_->mgr);
    if (!pd_c2d_mgr) {
        return nullptr;
    }

    PostDepthwiseConv2dParam *pd_c2d_param = new PostDepthwiseConv2dParam;
    pd_c2d_param->conv2d_param = conv_op->conv2d_param_;
    pd_c2d_param->depthwise_conv2d_param = post_conv_op->conv2d_param_;
    pd_c2d_param->algo_info = pd_c2d_algo_info;
    pd_c2d_param->mgr = pd_c2d_mgr;

    // release
    conv_op->conv2d_param_ = nullptr;
    post_conv_op->conv2d_param_ = nullptr;

    return pd_c2d_param;
}

RetCode PostDepthwiseConvOp::Init(const OptKernelOptions& options) {
    infer_dims_func_ = [this](InputOutputInfo* info) -> RetCode {
        if (!pd_conv2d_param_ || pd_conv2d_param_->algo_info.algo_type == ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
            return RC_INVALID_VALUE;
        }

        auto cv_p = &pd_conv2d_param_->conv2d_param->param;
        auto dw_p = &pd_conv2d_param_->depthwise_conv2d_param->param;
        auto x = info->GetInput<TensorImpl>(0)->GetShape();
        auto y = info->GetOutput<TensorImpl>(0)->GetShape();
        auto num_output = dw_p->num_output;

        const int64_t kernel_h_eff = (cv_p->kernel_h - 1) * cv_p->dilation_h + 1;
        const int64_t kernel_w_eff = (cv_p->kernel_w - 1) * cv_p->dilation_w + 1;
        const int64_t inter_h = (x->GetDim(2) + 2 * cv_p->pad_h - kernel_h_eff) / cv_p->stride_h + 1;
        const int64_t inter_w = (x->GetDim(3) + 2 * cv_p->pad_w - kernel_w_eff) / cv_p->stride_w + 1;

        const int64_t dw_kernel_h_eff = (dw_p->kernel_h - 1) * dw_p->dilation_h + 1;
        const int64_t dw_kernel_w_eff = (dw_p->kernel_w - 1) * dw_p->dilation_w + 1;
        const int64_t dst_h = (inter_h + 2 * dw_p->pad_h - dw_kernel_h_eff) / dw_p->stride_h + 1;
        const int64_t dst_w = (inter_w + 2 * dw_p->pad_w - dw_kernel_w_eff) / dw_p->stride_w + 1;

        y->SetDimCount(x->GetDimCount());
        y->SetDim(0, x->GetDim(0));
        y->SetDim(1, num_output);
        y->SetDim(2, dst_h);
        y->SetDim(3, dst_w);
        y->CalcPadding();

        return RC_SUCCESS;
    };

    infer_type_func_ = GenericInferType;
    return RC_SUCCESS;
}

RetCode PostDepthwiseConvOp::SelectFormat(
    const InputOutputInfo& info,
    vector<dataformat_t>* selected_input_formats,
    vector<dataformat_t>* selected_output_formats) {
    if (pd_conv2d_param_ && pd_conv2d_param_->algo_info.algo_type != ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
        selected_input_formats->at(0) = pd_conv2d_param_->algo_info.input_format;
        selected_output_formats->at(0) = pd_conv2d_param_->algo_info.output_format;
        return RC_SUCCESS;
    }
    return RC_INVALID_VALUE;
}

KernelImpl* PostDepthwiseConvOp::CreateKernelImpl() const {
    if (pd_conv2d_param_ && pd_conv2d_param_->algo_info.algo_type != ppl::kernel::x86::conv2d_fp32_algo::UNKNOWN) {
        return CreateKernelImplWithParam<PostDepthwiseConv2dKernel>(pd_conv2d_param_);
    }
    return nullptr;
}

}}} // namespace ppl::nn::x86
