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
#include "ppl/nn/params/param_utils_manager.h"

// NOTE: sorted in alphabet order
#include "ppl/nn/models/onnx/parsers/onnx/parse_argmax_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_batchnormalization_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_cast_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_concat_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_constant_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_constant_of_shape_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_convolution_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_convtranspose_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_depth_to_space_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_flatten_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gather_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gather_nd_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gemm_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_if_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_leaky_relu_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_loop_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_maxunpool_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_non_max_suppression_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_pad_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_pooling_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_reduce_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_resize_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_roialign_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_scatter_elements_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_softmax_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_split_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_split_to_sequence_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_squeeze_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_topk_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_transpose_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_unsqueeze_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_lrn_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_lstm_param.h"

#include "ppl/nn/models/onnx/parsers/mmcv/parse_mmcv_gridsample_param.h"
#include "ppl/nn/models/onnx/parsers/mmcv/parse_mmcv_modulated_deform_conv2d_param.h"
#include "ppl/nn/models/onnx/parsers/mmcv/parse_mmcv_nonmaxsupression_param.h"
#include "ppl/nn/models/onnx/parsers/mmcv/parse_mmcv_roialign_param.h"

#include "ppl/nn/models/onnx/parsers/ppl/parse_ppl_channel_shuffle_param.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RetCode ParamParserManager::Register(const string& domain, const string& type, const utils::VersionRange& ver,
                                     const ParserInfo& info) {
    return mgr_.Register(domain, type, ver, info);
}

const ParserInfo* ParamParserManager::Find(const string& domain, const string& type, uint64_t version) const {
    return mgr_.Find(domain, type, version);
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
bool ParamEqual(const void* param_0, const void* param_1) {
    return *static_cast<const T*>(param_0) == *static_cast<const T*>(param_1);
}

#define PPL_REGISTER_OP_WITH_PARAM(domain, type, first_version, last_version, param_type, parse_param_func)         \
    do {                                                                                                            \
        ParserInfo parse_info;                                                                                      \
        parse_info.create_param = CreateParam<param_type>;                                                          \
        parse_info.parse_param = parse_param_func;                                                                  \
        parse_info.destroy_param = DeleteParam<param_type>;                                                         \
        Register(domain, type, utils::VersionRange(first_version, last_version), parse_info);                       \
                                                                                                                    \
        ParamUtils u;                                                                                               \
        u.equal = ParamEqual<param_type>;                                                                           \
        ParamUtilsManager::Instance()->Register(domain, type, utils::VersionRange(first_version, last_version), u); \
    } while (0)

#define PPL_REGISTER_OP_WITHOUT_PARAM(domain, type, first_version, last_version)              \
    do {                                                                                      \
        ParserInfo parse_info;                                                                \
        parse_info.create_param = nullptr;                                                    \
        parse_info.parse_param = nullptr;                                                     \
        parse_info.destroy_param = nullptr;                                                   \
        Register(domain, type, utils::VersionRange(first_version, last_version), parse_info); \
    } while (0)

// NOTE: sorted in alphabet order
ParamParserManager::ParamParserManager() {
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Add", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "ArgMax", 11, 11, ppl::nn::common::ArgMaxParam, ParseArgMaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "AveragePool", 11, 11, ppl::nn::common::PoolingParam, ParsePoolingParam);
    PPL_REGISTER_OP_WITH_PARAM("", "BatchNormalization", 11, 11, ppl::nn::common::BatchNormalizationParam,
                               ParseBatchNormalizationParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Cast", 11, 11, ppl::nn::common::CastParam, ParseCastParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Clip", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "Concat", 11, 11, ppl::nn::common::ConcatParam, ParseConcatParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Constant", 11, 11, ppl::nn::common::ConstantParam, ParseConstantParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConstantOfShape", 11, 11, ppl::nn::common::ConstantOfShapeParam,
                               ParseConstantOfShapeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Conv", 11, 11, ppl::nn::common::ConvolutionParam, ParseConvolutionParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConvTranspose", 11, 11, ppl::nn::common::ConvTransposeParam,
                               ParseConvTransposeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "DepthToSpace", 11, 11, ppl::nn::common::DepthToSpaceParam, ParseDepthToSpaceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Div", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Equal", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Exp", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Expand", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "Flatten", 11, 11, ppl::nn::common::FlattenParam, ParseFlattenParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Floor", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "Gather", 11, 11, ppl::nn::common::GatherParam, ParseGatherParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GatherND", 11, 11, ppl::nn::common::GatherNDParam, ParseGatherNDParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Gemm", 11, 11, ppl::nn::common::GemmParam, ParseGemmParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GlobalAveragePool", 11, 11, ppl::nn::common::PoolingParam, ParsePoolingParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Greater", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Identity", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "If", 1, 13, ppl::nn::common::IfParam, ParseIfParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Less", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Log", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "Loop", 1, 13, ppl::nn::common::LoopParam, ParseLoopParam);
    PPL_REGISTER_OP_WITH_PARAM("", "LeakyRelu", 11, 11, ppl::nn::common::LeakyReLUParam, ParseLeakyReLUParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "MatMul", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Max", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Min", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "MaxPool", 11, 11, ppl::nn::common::PoolingParam, ParsePoolingParam);
    PPL_REGISTER_OP_WITH_PARAM("", "MaxUnpool", 11, 11, ppl::nn::common::MaxUnpoolParam, ParseMaxUnpoolParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Mul", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "NonMaxSuppression", 11, 11, ppl::nn::common::NonMaxSuppressionParam,
                               ParseNonMaxSuppressionParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "NonZero", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Not", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "Pad", 11, 11, ppl::nn::common::PadParam, ParsePadParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Pow", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Range", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMax", 11, 11, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMean", 11, 11, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMin", 11, 11, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceSum", 11, 11, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceProd", 11, 11, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Relu", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Reshape", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "Resize", 11, 11, ppl::nn::common::ResizeParam, ParseResizeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "RoiAlign", 11, 11, ppl::nn::common::ROIAlignParam, ParseROIAlignParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ScatterElements", 11, 11, ppl::nn::common::ScatterElementsParam,
                               ParseScatterElementsParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "ScatterND", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "SequenceAt", 1, 13);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Shape", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sigmoid", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Slice", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "Softmax", 11, 11, ppl::nn::common::SoftmaxParam, ParseSoftmaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Split", 11, 11, ppl::nn::common::SplitParam, ParseSplitParam);
    PPL_REGISTER_OP_WITH_PARAM("", "SplitToSequence", 1, 13, ppl::nn::common::SplitToSequenceParam,
                               ParseSplitToSequenceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sqrt", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "Squeeze", 11, 11, ppl::nn::common::SqueezeParam, ParseSqueezeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sub", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sum", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tanh", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tile", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "TopK", 11, 11, ppl::nn::common::TopKParam, ParseTopKParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Transpose", 11, 11, ppl::nn::common::TransposeParam, ParseTransposeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Unsqueeze", 11, 11, ppl::nn::common::UnsqueezeParam, ParseUnsqueezeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Where", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Ceil", 11, 11);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "And", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "LRN", 11, 11, ppl::nn::common::LRNParam, ParseLRNParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Dropout", 11, 11);
    PPL_REGISTER_OP_WITH_PARAM("", "LSTM", 11, 11, ppl::nn::common::LSTMParam, ParseLSTMParam);

    // mmcv op param parser
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "NonMaxSuppression", 1, 1, ppl::nn::common::MMCVNMSParam, ParseMMCVNMSParam);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVRoiAlign", 1, 1, ppl::nn::common::MMCVROIAlignParam,
                               ParseMMCVROIAlignParam);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "grid_sampler", 1, 1, ppl::nn::common::MMCVGridSampleParam,
                               ParseMMCVGridSampleParam);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVModulatedDeformConv2d", 1, 1,
                               ppl::nn::common::MMCVModulatedDeformConv2dParam, ParseMMCVModulatedDeformConv2dParam);

    // ppl op param parser
    PPL_REGISTER_OP_WITH_PARAM("ppl", "ChannelShuffle", 1, 1, ppl::nn::common::ChannelShuffleParam,
                               ParseChannelShuffleParam);
}

}}} // namespace ppl::nn::onnx
