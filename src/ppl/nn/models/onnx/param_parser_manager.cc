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

#include "ppl/nn/models/onnx/param_parser_manager.h"
#include "ppl/nn/models/op_info_manager.h"

// NOTE: sorted in alphabet order
#include "ppl/nn/models/onnx/parsers/parse_argmax_param.h"
#include "ppl/nn/models/onnx/parsers/parse_batchnormalization_param.h"
#include "ppl/nn/models/onnx/parsers/parse_cast_param.h"
#include "ppl/nn/models/onnx/parsers/parse_concat_param.h"
#include "ppl/nn/models/onnx/parsers/parse_constant_param.h"
#include "ppl/nn/models/onnx/parsers/parse_constant_of_shape_param.h"
#include "ppl/nn/models/onnx/parsers/parse_convolution_param.h"
#include "ppl/nn/models/onnx/parsers/parse_convtranspose_param.h"
#include "ppl/nn/models/onnx/parsers/parse_depth_to_space_param.h"
#include "ppl/nn/models/onnx/parsers/parse_flatten_param.h"
#include "ppl/nn/models/onnx/parsers/parse_gather_param.h"
#include "ppl/nn/models/onnx/parsers/parse_gather_nd_param.h"
#include "ppl/nn/models/onnx/parsers/parse_gemm_param.h"
#include "ppl/nn/models/onnx/parsers/parse_if_param.h"
#include "ppl/nn/models/onnx/parsers/parse_leaky_relu_param.h"
#include "ppl/nn/models/onnx/parsers/parse_loop_param.h"
#include "ppl/nn/models/onnx/parsers/parse_maxunpool_param.h"
#include "ppl/nn/models/onnx/parsers/parse_non_max_suppression_param.h"
#include "ppl/nn/models/onnx/parsers/parse_pad_param.h"
#include "ppl/nn/models/onnx/parsers/parse_pooling_param.h"
#include "ppl/nn/models/onnx/parsers/parse_reduce_param.h"
#include "ppl/nn/models/onnx/parsers/parse_resize_param.h"
#include "ppl/nn/models/onnx/parsers/parse_roialign_param.h"
#include "ppl/nn/models/onnx/parsers/parse_scatter_elements_param.h"
#include "ppl/nn/models/onnx/parsers/parse_softmax_param.h"
#include "ppl/nn/models/onnx/parsers/parse_split_param.h"
#include "ppl/nn/models/onnx/parsers/parse_split_to_sequence_param.h"
#include "ppl/nn/models/onnx/parsers/parse_squeeze_param.h"
#include "ppl/nn/models/onnx/parsers/parse_topk_param.h"
#include "ppl/nn/models/onnx/parsers/parse_transpose_param.h"
#include "ppl/nn/models/onnx/parsers/parse_unsqueeze_param.h"
#include "ppl/nn/models/onnx/parsers/parse_lrn_param.h"

#include "ppl/nn/models/onnx/parsers/parse_mmcv_gridsample_param.h"
#include "ppl/nn/models/onnx/parsers/parse_mmcv_modulated_deform_conv2d_param.h"
#include "ppl/nn/models/onnx/parsers/parse_mmcv_nonmaxsupression_param.h"
#include "ppl/nn/models/onnx/parsers/parse_mmcv_roialign_param.h"

#include "ppl/nn/models/onnx/parsers/parse_ppl_channel_shuffle_param.h"

using namespace std;

namespace ppl { namespace nn { namespace onnx {

void ParamParserManager::Register(const string& domain, const string& type, const ParserInfo& info) {
    domain_type_parser_[domain][type] = info;
}

const ParserInfo* ParamParserManager::Find(const string& domain, const string& op_type) const {
    auto type_parser_ref = domain_type_parser_.find(domain);
    if (type_parser_ref != domain_type_parser_.end()) {
        auto parser_ref = type_parser_ref->second.find(op_type);
        if (parser_ref != type_parser_ref->second.end()) {
            return &(parser_ref->second);
        }
    }
    return nullptr;
}

template <typename T>
void* CreateParam() {
    return new T();
}

template <typename T>
void DeleteParam(void* ptr) {
    delete static_cast<T*>(ptr);
}

template <typename T>
bool ParamEqual(void* param_0, void* param_1) {
    return *static_cast<T*>(param_0) == *static_cast<T*>(param_1);
}

#define PPL_REGISTER_OP_WITH_PARAM(domain, type, param_type, parse_param_func) \
    do {                                                                       \
        ParserInfo parse_info;                                                 \
        parse_info.create_param = CreateParam<param_type>;                     \
        parse_info.parse_param = parse_param_func;                             \
        parse_info.destroy_param = DeleteParam<param_type>;                    \
        domain_type_parser_[domain][type] = parse_info;                        \
                                                                               \
        OpInfo info;                                                           \
        info.param_equal = ParamEqual<param_type>;                             \
        OpInfoManager::Instance()->Register(domain, type, info);               \
    } while (0)

#define PPL_REGISTER_OP_WITHOUT_PARAM(domain, type)     \
    do {                                                \
        ParserInfo parse_info;                          \
        parse_info.create_param = nullptr;              \
        parse_info.parse_param = nullptr;               \
        parse_info.destroy_param = nullptr;             \
        domain_type_parser_[domain][type] = parse_info; \
    } while (0)

// NOTE: sorted in alphabet order
ParamParserManager::ParamParserManager() {
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Add");
    PPL_REGISTER_OP_WITH_PARAM("", "ArgMax", ppl::nn::common::ArgMaxParam, ParseArgMaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "AveragePool", ppl::nn::common::PoolingParam, ParsePoolingParam);
    PPL_REGISTER_OP_WITH_PARAM("", "BatchNormalization", ppl::nn::common::BatchNormalizationParam,
                               ParseBatchNormalizationParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Cast", ppl::nn::common::CastParam, ParseCastParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Clip");
    PPL_REGISTER_OP_WITH_PARAM("", "Concat", ppl::nn::common::ConcatParam, ParseConcatParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Constant", ppl::nn::common::ConstantParam, ParseConstantParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConstantOfShape", ppl::nn::common::ConstantOfShapeParam, ParseConstantOfShapeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Conv", ppl::nn::common::ConvolutionParam, ParseConvolutionParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConvTranspose", ppl::nn::common::ConvTransposeParam, ParseConvTransposeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "DepthToSpace", ppl::nn::common::DepthToSpaceParam, ParseDepthToSpaceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Div");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Equal");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Exp");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Expand");
    PPL_REGISTER_OP_WITH_PARAM("", "Flatten", ppl::nn::common::FlattenParam, ParseFlattenParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Floor");
    PPL_REGISTER_OP_WITH_PARAM("", "Gather", ppl::nn::common::GatherParam, ParseGatherParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GatherND", ppl::nn::common::GatherNDParam, ParseGatherNDParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Gemm", ppl::nn::common::GemmParam, ParseGemmParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GlobalAveragePool", ppl::nn::common::PoolingParam, ParsePoolingParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Greater");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Identity");
    PPL_REGISTER_OP_WITH_PARAM("", "If", ppl::nn::common::IfParam, ParseIfParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Less");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Log");
    PPL_REGISTER_OP_WITH_PARAM("", "Loop", ppl::nn::common::LoopParam, ParseLoopParam);
    PPL_REGISTER_OP_WITH_PARAM("", "LeakyRelu", ppl::nn::common::LeakyReLUParam, ParseLeakyReLUParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "MatMul");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Max");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Min");
    PPL_REGISTER_OP_WITH_PARAM("", "MaxPool", ppl::nn::common::PoolingParam, ParsePoolingParam);
    PPL_REGISTER_OP_WITH_PARAM("", "MaxUnpool", ppl::nn::common::MaxUnpoolParam, ParseMaxUnpoolParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Mul");
    PPL_REGISTER_OP_WITH_PARAM("", "NonMaxSuppression", ppl::nn::common::NonMaxSuppressionParam,
                               ParseNonMaxSuppressionParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "NonZero");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Not");
    PPL_REGISTER_OP_WITH_PARAM("", "Pad", ppl::nn::common::PadParam, ParsePadParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Pow");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Range");
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMax", ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMean", ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMin", ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceSum", ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceProd", ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Relu");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Reshape");
    PPL_REGISTER_OP_WITH_PARAM("", "Resize", ppl::nn::common::ResizeParam, ParseResizeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "RoiAlign", ppl::nn::common::ROIAlignParam, ParseROIAlignParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ScatterElements", ppl::nn::common::ScatterElementsParam, ParseScatterElementsParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "ScatterND");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "SequenceAt");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Shape");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sigmoid");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Slice");
    PPL_REGISTER_OP_WITH_PARAM("", "Softmax", ppl::nn::common::SoftmaxParam, ParseSoftmaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Split", ppl::nn::common::SplitParam, ParseSplitParam);
    PPL_REGISTER_OP_WITH_PARAM("", "SplitToSequence", ppl::nn::common::SplitToSequenceParam, ParseSplitToSequenceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sqrt");
    PPL_REGISTER_OP_WITH_PARAM("", "Squeeze", ppl::nn::common::SqueezeParam, ParseSqueezeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sub");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sum");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tanh");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tile");
    PPL_REGISTER_OP_WITH_PARAM("", "TopK", ppl::nn::common::TopKParam, ParseTopKParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Transpose", ppl::nn::common::TransposeParam, ParseTransposeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Unsqueeze", ppl::nn::common::UnsqueezeParam, ParseUnsqueezeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Where");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Ceil");
    PPL_REGISTER_OP_WITHOUT_PARAM("", "And");
    PPL_REGISTER_OP_WITH_PARAM("", "LRN", ppl::nn::common::LRNParam, ParseLRNParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Dropout");

    // mmcv op param parser
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "NonMaxSuppression", ppl::nn::common::MMCVNMSParam, ParseMMCVNMSParam);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVRoiAlign", ppl::nn::common::MMCVROIAlignParam, ParseMMCVROIAlignParam);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "grid_sampler", ppl::nn::common::MMCVGridSampleParam, ParseMMCVGridSampleParam);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVModulatedDeformConv2d", ppl::nn::common::MMCVModulatedDeformConv2dParam,
                               ParseMMCVModulatedDeformConv2dParam);

    // ppl op param parser
    PPL_REGISTER_OP_WITH_PARAM("ppl", "ChannelShuffle", ppl::nn::common::ChannelShuffleParam, ParseChannelShuffleParam);
}

}}} // namespace ppl::nn::onnx
