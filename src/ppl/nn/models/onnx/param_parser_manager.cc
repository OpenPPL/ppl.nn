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
#include "ppl/nn/models/onnx/parsers/onnx/parse_hard_sigmoid_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_if_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_instancenormalization_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_leaky_relu_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_loop_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_lrn_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_lstm_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_maxunpool_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_non_max_suppression_param.h"
#include "ppl/nn/models/onnx/parsers/onnx/parse_one_hot_param.h"
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

#include "ppl/nn/models/onnx/parsers/pmx/parse_ppl_channel_shuffle_param.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

template <typename T>
shared_ptr<ir::Attr> CreateParam() {
    return make_shared<T>();
}

#define PPL_REGISTER_OP_WITH_PARAM(domain, type, first_version, last_version, param_type, parse_param_func,    \
                                   pack_param_func)                                                            \
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
        parse_info.pack_param = pack_param_func;                                                               \
        auto status = Register(domain, type, utils::VersionRange(first_version, last_version), parse_info);    \
        if (status != RC_SUCCESS) {                                                                            \
            exit(-1);                                                                                          \
        }                                                                                                      \
    } while (0)

#define PPL_REGISTER_OP_WITHOUT_PARAM(domain, type, first_version, last_version, parse_param_func) \
    do {                                                                                           \
        ParserInfo parse_info;                                                                     \
        parse_info.create_param = nullptr;                                                         \
        parse_info.parse_param = parse_param_func;                                                 \
        parse_info.pack_param = nullptr;                                                           \
        Register(domain, type, utils::VersionRange(first_version, last_version), parse_info);      \
    } while (0)

// NOTE: sorted in alphabet order
ParamParserManager::ParamParserManager() {
    // A
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Abs", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Add", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "And", 7, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "ArgMax", 1, 11, ArgMaxParam, ParseArgMaxParam, PackArgMaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "AveragePool", 1, 16, PoolingParam, ParsePoolingParam, PackPoolingParam);
    // B
    PPL_REGISTER_OP_WITH_PARAM("", "BatchNormalization", 9, 13, BatchNormalizationParam, ParseBatchNormalizationParam,
                               PackBatchNormalizationParam);
    // C
    PPL_REGISTER_OP_WITH_PARAM("", "Cast", 9, 16, CastParam, ParseCastParam, PackCastParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Ceil", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Clip", 6, 16, ParseClipParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Concat", 4, 16, ConcatParam, ParseConcatParam, PackConcatParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Constant", 9, 16, ConstantParam, ParseConstantParam, PackConstantParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConstantOfShape", 9, 16, ConstantOfShapeParam, ParseConstantOfShapeParam,
                               PackConstantOfShapeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Conv", 1, 16, ConvParam, ParseConvParam, PackConvParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ConvTranspose", 1, 16, ConvTransposeParam, ParseConvTransposeParam,
                               PackConvTransposeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Cos", 7, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "CumSum", 11, 16, CumSumParam, ParseCumSumParam, PackCumSumParam);
    // D
    PPL_REGISTER_OP_WITH_PARAM("", "DepthToSpace", 1, 16, DepthToSpaceParam, ParseDepthToSpaceParam,
                               PackDepthToSpaceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Div", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Dropout", 1, 16, nullptr); // will be skip
    // E
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Equal", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Erf", 9, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Exp", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Expand", 8, 16, nullptr);
    // F
    PPL_REGISTER_OP_WITH_PARAM("", "Flatten", 1, 16, FlattenParam, ParseFlattenParam, PackFlattenParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Floor", 6, 16, nullptr);
    // G
    PPL_REGISTER_OP_WITH_PARAM("", "Gather", 1, 16, GatherParam, ParseGatherParam, PackGatherParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GatherND", 11, 16, GatherNDParam, ParseGatherNDParam, PackGatherNDParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Gemm", 9, 16, GemmParam, ParseGemmParam, PackGemmParam);
    PPL_REGISTER_OP_WITH_PARAM("", "GlobalAveragePool", 1, 16, PoolingParam, ParsePoolingParam, PackPoolingParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Greater", 7, 16, nullptr);
    // H
    PPL_REGISTER_OP_WITH_PARAM("", "HardSigmoid", 6, 16, HardSigmoidParam, ParseHardSigmoidParam, PackHardSigmoidParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "HardSwish", 14, 16, nullptr);
    // I
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Identity", 1, 13, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "If", 1, 12, IfParam, ParseIfParam, PackIfParam);
    PPL_REGISTER_OP_WITH_PARAM("", "InstanceNormalization", 6, 13, InstanceNormalizationParam,
                               ParseInstanceNormalizationParam, PackInstanceNormalizationParam);
    // L
    PPL_REGISTER_OP_WITH_PARAM("", "LeakyRelu", 6, 16, LeakyReluParam, ParseLeakyReluParam, PackLeakyReluParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Less", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Log", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Loop", 1, 12, LoopParam, ParseLoopParam, PackLoopParam);
    PPL_REGISTER_OP_WITH_PARAM("", "LRN", 1, 16, LRNParam, ParseLRNParam, PackLRNParam);
    PPL_REGISTER_OP_WITH_PARAM("", "LSTM", 7, 13, LSTMParam, ParseLSTMParam, PackLSTMParam);
    // M
    PPL_REGISTER_OP_WITHOUT_PARAM("", "MatMul", 1, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Max", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "MaxPool", 1, 16, PoolingParam, ParsePoolingParam, PackPoolingParam);
    PPL_REGISTER_OP_WITH_PARAM("", "MaxUnpool", 9, 16, MaxUnpoolParam, ParseMaxUnpoolParam, PackMaxUnpoolParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Min", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Mul", 7, 16, nullptr);
    // N
    PPL_REGISTER_OP_WITH_PARAM("", "NonMaxSuppression", 10, 16, NonMaxSuppressionParam, ParseNonMaxSuppressionParam,
                               PackNonMaxSuppressionParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "NonZero", 9, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Not", 1, 16, nullptr);
    // O
    PPL_REGISTER_OP_WITH_PARAM("", "OneHot", 9, 11, OneHotParam, ParseOneHotParam, PackOneHotParam);
    // P
    PPL_REGISTER_OP_WITH_PARAM("", "Pad", 2, 16, PadParam, ParsePadParam, PackPadParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Pow", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "PRelu", 6, 16, nullptr);
    // R
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Range", 11, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceL2", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMax", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMean", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceMin", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceProd", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "ReduceSum", 1, 16, ReduceParam, ParseReduceParam, PackReduceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Relu", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Reshape", 5, 13, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Resize", 11, 16, ResizeParam, ParseResizeParam, PackResizeParam);
    PPL_REGISTER_OP_WITH_PARAM("", "RoiAlign", 10, 15, RoiAlignParam, ParseRoiAlignParam, PackRoiAlignParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Round", 11, 16, nullptr);
    // // S
    PPL_REGISTER_OP_WITH_PARAM("", "ScatterElements", 11, 15, ScatterElementsParam, ParseScatterElementsParam,
                               PackScatterElementsParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "ScatterND", 11, 15, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "SequenceAt", 11, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Shape", 1, 14, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sigmoid", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sign", 9, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sin", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Slice", 1, 16, ParseSliceParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Softmax", 1, 16, SoftmaxParam, ParseSoftmaxParam, PackSoftmaxParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Split", 2, 12, SplitParam, ParseSplitParam, PackSplitParam);
    PPL_REGISTER_OP_WITH_PARAM("", "SplitToSequence", 11, 16, SplitToSequenceParam, ParseSplitToSequenceParam,
                               PackSplitToSequenceParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sqrt", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "Squeeze", 1, 16, SqueezeParam, ParseSqueezeParam, PackSqueezeParam);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sub", 7, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Sum", 6, 16, nullptr);
    // T
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tanh", 6, 16, nullptr);
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Tile", 6, 16, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("", "TopK", 1, 16, TopKParam, ParseTopKParam, PackTopKParam);
    PPL_REGISTER_OP_WITH_PARAM("", "Transpose", 1, 16, TransposeParam, ParseTransposeParam, PackTransposeParam);
    // U
    PPL_REGISTER_OP_WITH_PARAM("", "Unsqueeze", 1, 16, UnsqueezeParam, ParseUnsqueezeParam, PackUnsqueezeParam);
    // W
    PPL_REGISTER_OP_WITHOUT_PARAM("", "Where", 9, 16, nullptr);

    // mmcv op param parser
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "grid_sampler", 1, 1, ppl::nn::mmcv::MMCVGridSampleParam,
                               ParseMMCVGridSampleParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVRoiAlign", 1, 1, ppl::nn::mmcv::MMCVRoiAlignParam, ParseMMCVRoiAlignParam,
                               nullptr);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "MMCVModulatedDeformConv2d", 1, 1, ppl::nn::mmcv::MMCVModulatedDeformConv2dParam,
                               ParseMMCVModulatedDeformConv2dParam, nullptr);
    PPL_REGISTER_OP_WITH_PARAM("mmcv", "NonMaxSuppression", 1, 1, ppl::nn::mmcv::MMCVNMSParam, ParseMMCVNMSParam,
                               nullptr);

    // ppl op param parser
    PPL_REGISTER_OP_WITH_PARAM("pmx", "ChannelShuffle", 1, 1, ppl::nn::pmx::ChannelShuffleParam,
                               ParseChannelShuffleParam, nullptr);
}

}}} // namespace ppl::nn::onnx
