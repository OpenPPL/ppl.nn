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
#include "ppl/nn/models/onnx/parsers/onnx/parse_clip_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_concat_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_constant_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_constant_of_shape_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_conv_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_convtranspose_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_cumsum_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_depth_to_space_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_flatten_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gather_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gather_nd_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_gemm_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_if_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_leaky_relu_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_loop_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_lrn_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_lstm_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_maxunpool_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_non_max_suppression_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_pad_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_pooling_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_reduce_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_resize_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_roialign_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_scatter_elements_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_slice_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_softmax_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_split_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_split_to_sequence_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_squeeze_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_topk_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_transpose_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_unsqueeze_param.h"

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

#define PPL_REGISTER_OP_WITH_PARAM(domain, type, first_version, last_version, param_type, parse_param_func)    \
    do {                                                                                                       \
        if (last_version < first_version) {                                                                    \
            LOG(ERROR) << "register op[" << domain << ":" << type << "] failed: last_version[" << last_version \
                       << "] < first_version[" << first_version << "]";                                        \
            exit(-1);                                                                                          \
        }                                                                                                      \
                                                                                                               \
        ParserInfo parse_info;                                                                                 \
        parse_info.create_param = CreateParam<param_type>;                                                     \
        parse_info.parse_param = parse_param_func;                                                             \
        parse_info.destroy_param = DeleteParam<param_type>;                                                    \
        auto status = Register(domain, type, utils::VersionRange(first_version, last_version), parse_info);    \
        if (status != RC_SUCCESS) {                                                                            \
            exit(-1);                                                                                          \
        }                                                                                                      \
                                                                                                               \
        ParamUtils u;                                                                                          \
        u.equal = ParamEqual<param_type>;                                                                      \
        status = ParamUtilsManager::Instance()->Register(domain, type,                                         \
                                                         utils::VersionRange(first_version, last_version), u); \
        if (status != RC_SUCCESS) {                                                                            \
            exit(-1);                                                                                          \
        }                                                                                                      \
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
    // A
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Add", 7, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "And", 7, 16);
    PPL_REGISTER_OP_WITH_PARAM("", "ArgMax", 1, 11, ppl::nn::common::ArgMaxParam, ParseArgMaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "AveragePool", 1, 16, ppl::nn::common::PoolingParam, ParsePoolingParam);
    // B
    PPL_REGISTER_OP_WITH_PARAM("", "BatchNormalization", 9, 13, ppl::nn::common::BatchNormalizationParam,
                               ParseBatchNormalizationParam);
    // C
    PPL_REGISTER_OP_WITH_PARAM("", "Cast", 9, 16, ppl::nn::common::CastParam, ParseCastParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Ceil", 6, 16);
    PPL_REGISTER_OP_WITH_PARAM("", "Clip", 6, 16, ppl::nn::common::ClipParam, ParseClipParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Concat", 4, 16, ppl::nn::common::ConcatParam, ParseConcatParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Constant", 9, 16, ppl::nn::common::ConstantParam, ParseConstantParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConstantOfShape", 9, 16, ppl::nn::common::ConstantOfShapeParam,
                               ParseConstantOfShapeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Conv", 1, 16, ppl::nn::common::ConvParam, ParseConvParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConvTranspose", 1, 16, ppl::nn::common::ConvTransposeParam,
                               ParseConvTransposeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Cos", 7, 16);
    PPL_REGISTER_OP_WITH_PARAM("", "CumSum", 11, 16, ppl::nn::common::CumSumParam, ParseCumSumParam);
    // D
    PPL_REGISTER_OP_WITH_PARAM("", "DepthToSpace", 1, 16, ppl::nn::common::DepthToSpaceParam, ParseDepthToSpaceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Div", 7, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Dropout", 1, 16); // will be skip
    // E
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Equal", 7, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Erf", 9, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Exp", 6, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Expand", 8, 16);
    // F
    PPL_REGISTER_OP_WITH_PARAM("", "Flatten", 1, 16, ppl::nn::common::FlattenParam, ParseFlattenParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Floor", 6, 16);
    // G
    PPL_REGISTER_OP_WITH_PARAM("", "Gather", 1, 16, ppl::nn::common::GatherParam, ParseGatherParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GatherND", 11, 16, ppl::nn::common::GatherNDParam, ParseGatherNDParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Gemm", 9, 16, ppl::nn::common::GemmParam, ParseGemmParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GlobalAveragePool", 1, 16, ppl::nn::common::PoolingParam, ParsePoolingParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Greater", 7, 16);
    // I
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Identity", 1, 13);
    PPL_REGISTER_OP_WITH_PARAM("", "If", 1, 12, ppl::nn::common::IfParam, ParseIfParam);
    // L
    PPL_REGISTER_OP_WITH_PARAM("", "LeakyRelu", 6, 16, ppl::nn::common::LeakyReluParam, ParseLeakyReluParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Less", 7, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Log", 6, 16);
    PPL_REGISTER_OP_WITH_PARAM("", "Loop", 1, 12, ppl::nn::common::LoopParam, ParseLoopParam);
    PPL_REGISTER_OP_WITH_PARAM("", "LRN", 1, 16, ppl::nn::common::LRNParam, ParseLRNParam);
    PPL_REGISTER_OP_WITH_PARAM("", "LSTM", 7, 13, ppl::nn::common::LSTMParam, ParseLSTMParam);
    // M
    PPL_REGISTER_OP_WITHOUT_PARAM("", "MatMul", 1, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Max", 6, 16);
    PPL_REGISTER_OP_WITH_PARAM("", "MaxPool", 1, 16, ppl::nn::common::PoolingParam, ParsePoolingParam);
    PPL_REGISTER_OP_WITH_PARAM("", "MaxUnpool", 9, 16, ppl::nn::common::MaxUnpoolParam, ParseMaxUnpoolParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Min", 6, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Mul", 7, 16);
    // N
    PPL_REGISTER_OP_WITH_PARAM("", "NonMaxSuppression", 10, 16, ppl::nn::common::NonMaxSuppressionParam,
                               ParseNonMaxSuppressionParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "NonZero", 9, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Not", 1, 16);
    // P
    PPL_REGISTER_OP_WITH_PARAM("", "Pad", 2, 16, ppl::nn::common::PadParam, ParsePadParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Pow", 7, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "PRelu", 6, 16);
    // R
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Range", 11, 16);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMax", 1, 16, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMean", 1, 16, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMin", 1, 16, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceProd", 1, 16, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceSum", 1, 16, ppl::nn::common::ReduceParam, ParseReduceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Relu", 6, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Reshape", 5, 13);
    PPL_REGISTER_OP_WITH_PARAM("", "Resize", 11, 16, ppl::nn::common::ResizeParam, ParseResizeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "RoiAlign", 10, 15, ppl::nn::common::RoiAlignParam, ParseRoiAlignParam);
    // S
    PPL_REGISTER_OP_WITH_PARAM("", "ScatterElements", 11, 15, ppl::nn::common::ScatterElementsParam,
                               ParseScatterElementsParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "ScatterND", 11, 15);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "SequenceAt", 11, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Shape", 1, 14);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sigmoid", 6, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sin", 7, 16);
    PPL_REGISTER_OP_WITH_PARAM("", "Slice", 1, 16, ppl::nn::common::SliceParam, ParseSliceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Softmax", 1, 16, ppl::nn::common::SoftmaxParam, ParseSoftmaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Split", 2, 12, ppl::nn::common::SplitParam, ParseSplitParam);
    PPL_REGISTER_OP_WITH_PARAM("", "SplitToSequence", 11, 16, ppl::nn::common::SplitToSequenceParam,
                               ParseSplitToSequenceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sqrt", 6, 16);
    PPL_REGISTER_OP_WITH_PARAM("", "Squeeze", 1, 12, ppl::nn::common::SqueezeParam, ParseSqueezeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sub", 7, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sum", 6, 16);
    // T
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tanh", 6, 16);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tile", 6, 16);
    PPL_REGISTER_OP_WITH_PARAM("", "TopK", 1, 16, ppl::nn::common::TopKParam, ParseTopKParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Transpose", 1, 16, ppl::nn::common::TransposeParam, ParseTransposeParam);
    // U
    PPL_REGISTER_OP_WITH_PARAM("", "Unsqueeze", 1, 12, ppl::nn::common::UnsqueezeParam, ParseUnsqueezeParam);
    // W
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Where", 9, 16);

    // mmcv op param parser
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "grid_sampler", 1, 1, ppl::nn::common::MMCVGridSampleParam,
                               ParseMMCVGridSampleParam);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVRoiAlign", 1, 1, ppl::nn::common::MMCVRoiAlignParam,
                               ParseMMCVRoiAlignParam);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVModulatedDeformConv2d", 1, 1,
                               ppl::nn::common::MMCVModulatedDeformConv2dParam, ParseMMCVModulatedDeformConv2dParam);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "NonMaxSuppression", 1, 1, ppl::nn::common::MMCVNMSParam, ParseMMCVNMSParam);

    // ppl op param parser
    PPL_REGISTER_OP_WITH_PARAM("ppl", "ChannelShuffle", 1, 1, ppl::nn::common::ChannelShuffleParam,
                               ParseChannelShuffleParam);
}

}}} // namespace ppl::nn::onnx
