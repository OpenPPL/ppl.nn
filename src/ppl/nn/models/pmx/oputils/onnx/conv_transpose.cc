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

#include "ppl/nn/params/onnx/auto_pad_type.h"
#include "ppl/nn/models/pmx/utils.h"
#include "ppl/nn/models/pmx/oputils/onnx/conv_transpose.h"
using namespace flatbuffers;

namespace ppl { namespace nn { namespace pmx { namespace onnx {

Offset<ConvTransposeParam> SerializeConvTransposeParam(const ppl::nn::onnx::ConvTransposeParam& param,
                                                       FlatBufferBuilder* builder) {
    auto fb_dilations = builder->CreateVector(param.dilations);
    auto fb_kernel_shape = builder->CreateVector(param.kernel_shape);
    auto fb_pads = builder->CreateVector(param.pads);
    auto fb_strides = builder->CreateVector(param.strides);
    auto fb_output_padding = builder->CreateVector(param.output_padding);
    auto fb_output_shape = builder->CreateVector(param.output_shape);
    AutoPadType tmp_auto_pad = AutoPadType_NOTSET;
    if (param.auto_pad == ppl::nn::onnx::AUTO_PAD_SAME_UPPER) {
        tmp_auto_pad = AutoPadType_SAME_UPPER;
    } else if (param.auto_pad == ppl::nn::onnx::AUTO_PAD_SAME_LOWER) {
        tmp_auto_pad = AutoPadType_SAME_LOWER;
    } else if (param.auto_pad == ppl::nn::onnx::AUTO_PAD_VALID) {
        tmp_auto_pad = AutoPadType_VALID;
    }
    return CreateConvTransposeParam(*builder, static_cast<AutoPadType>(tmp_auto_pad), param.group, fb_dilations,
                                    fb_kernel_shape, fb_output_padding, fb_output_shape, fb_pads, fb_strides);
}

void DeserializeConvTransposeParam(const ConvTransposeParam& fb_param, ppl::nn::onnx::ConvTransposeParam* param) {
    switch (fb_param.auto_pad()) {
        case AutoPadType_NOTSET:
            param->auto_pad = ppl::nn::onnx::AUTO_PAD_NOTSET;
            break;
        case AutoPadType_SAME_UPPER:
            param->auto_pad = ppl::nn::onnx::AUTO_PAD_SAME_UPPER;
            break;
        case AutoPadType_SAME_LOWER:
            param->auto_pad = ppl::nn::onnx::AUTO_PAD_SAME_LOWER;
            break;
        case AutoPadType_VALID:
            param->auto_pad = ppl::nn::onnx::AUTO_PAD_VALID;
            break;
        default:
            break;
    }
    param->group = fb_param.group();
    utils::Fbvec2Stdvec(fb_param.dilations(), &param->dilations);
    utils::Fbvec2Stdvec(fb_param.kernel_shape(), &param->kernel_shape);
    utils::Fbvec2Stdvec(fb_param.output_padding(), &param->output_padding);
    utils::Fbvec2Stdvec(fb_param.output_shape(), &param->output_shape);
    utils::Fbvec2Stdvec(fb_param.pads(), &param->pads);
    utils::Fbvec2Stdvec(fb_param.strides(), &param->strides);
}

}}}} // namespace ppl::nn::pmx::onnx
