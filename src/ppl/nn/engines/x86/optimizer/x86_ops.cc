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

#include "ppl/nn/engines/x86/optimizer/opt_kernel_creator_manager.h"
using namespace std;
using namespace ppl::common;

#include "ppl/nn/engines/x86/optimizer/ops/onnx/conv_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/abs_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/add_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/and_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/argmax_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/average_pool_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/batch_normalization_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/cast_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/ceil_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/convtranspose_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/clip_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/constant_of_shape_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/cos_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/cumsum_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/depth_to_space_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/div_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/equal_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/erf_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/exp_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/expand_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/flatten_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/floor_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/gather_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/gather_nd_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/greater_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/hard_sigmoid_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/hard_swish_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/identity_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/if_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/leaky_relu_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/less_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/log_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/loop_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/lstm_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/matmul_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/max_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/max_pool_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/max_unpool_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/min_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/mul_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/non_max_suppression_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/non_zero_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/not_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/roialign_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/one_hot_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/pad_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/pow_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/prelu_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/range_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/reduce_max_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/reduce_min_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/reduce_mean_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/reduce_prod_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/reduce_sum_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/relu_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/reshape_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/resize_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/scatter_elements_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/scatter_nd_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/sequence_at_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/shape_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/sigmoid_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/sign_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/sin_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/slice_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/softmax_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/split_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/split_to_sequence_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/sqrt_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/squeeze_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/sub_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/sum_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/tanh_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/tile_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/topk_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/transpose_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/unsqueeze_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/where_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/mmcv/mmcv_gridsample_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/mmcv/mmcv_modulated_deform_conv2d_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/mmcv/mmcv_non_max_suppression_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/mmcv/mmcv_roialign_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/pmx/reorder_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/pmx/channel_shuffle_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/pmx/shape_operation_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/pmx/swish_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/pmx/post_depthwise_conv_op.h"

namespace ppl { namespace nn { namespace x86 {

template <typename T>
static X86OptKernel* GenericCreateOptKernel(const ir::Node* node) {
    return new T(node);
}

template <typename T>
static void RegisterOptKernelCreator(const string& domain, const string& type, uint64_t first_version,
                                     uint64_t last_version) {
    if (last_version < first_version) {
        LOG(ERROR) << "register op[" << domain << ":" << type << "] failed: last_version[" << last_version
                   << "] < first_version[" << first_version << "]";
        exit(-1);
    }
    OptKernelCreatorManager::GetInstance()->Register(domain, type, utils::VersionRange(first_version, last_version),
                                                     GenericCreateOptKernel<T>);
}

// NOTE: sorted in alphabet order
void RegisterBuiltinOpImpls() {
    static bool ops_are_registered = false;
    if (ops_are_registered) {
        return;
    }
    ops_are_registered = true;

    // onnx op's default domain is ""
    // A
    RegisterOptKernelCreator<AbsOp>("", "Abs", 6, 16);
    RegisterOptKernelCreator<AddOp>("", "Add", 7, 16);
    RegisterOptKernelCreator<AndOp>("", "And", 7, 16);
    RegisterOptKernelCreator<ArgmaxOp>("", "ArgMax", 1, 11);
    RegisterOptKernelCreator<AveragePoolOp>("", "AveragePool", 1, 16);
    // B
    RegisterOptKernelCreator<BatchNormalizationOp>("", "BatchNormalization", 9, 13);
    // C
    RegisterOptKernelCreator<CastOp>("", "Cast", 9, 16);
    RegisterOptKernelCreator<CeilOp>("", "Ceil", 6, 16);
    RegisterOptKernelCreator<ClipOp>("", "Clip", 6, 16);
    RegisterOptKernelCreator<ConcatOp>("", "Concat", 4, 16);
    RegisterOptKernelCreator<ConstantOfShapeOp>("", "ConstantOfShape", 9, 16);
    RegisterOptKernelCreator<ConvOp>("", "Conv", 1, 16);
    RegisterOptKernelCreator<ConvTransposeOp>("", "ConvTranspose", 1, 16);
    RegisterOptKernelCreator<CosOp>("", "Cos", 7, 16);
    RegisterOptKernelCreator<CumSumOp>("", "CumSum", 11, 16);
    // D
    RegisterOptKernelCreator<DepthToSpaceOp>("", "DepthToSpace", 1, 16);
    RegisterOptKernelCreator<DivOp>("", "Div", 7, 16);
    // E
    RegisterOptKernelCreator<EqualOp>("", "Equal", 7, 16);
    RegisterOptKernelCreator<ErfOp>("", "Erf", 9, 16);
    RegisterOptKernelCreator<ExpOp>("", "Exp", 6, 16);
    RegisterOptKernelCreator<ExpandOp>("", "Expand", 8, 16);
    // F
    RegisterOptKernelCreator<FlattenOp>("", "Flatten", 1, 16);
    RegisterOptKernelCreator<FloorOp>("", "Floor", 6, 16);
    // G
    RegisterOptKernelCreator<GatherOp>("", "Gather", 1, 16);
    RegisterOptKernelCreator<GatherNDOp>("", "GatherND", 11, 11);
    RegisterOptKernelCreator<GemmOp>("", "Gemm", 9, 16);
    RegisterOptKernelCreator<AveragePoolOp>("", "GlobalAveragePool", 1, 16);
    RegisterOptKernelCreator<GreaterOp>("", "Greater", 7, 16);
    // H
    RegisterOptKernelCreator<HardSigmoidOp>("", "HardSigmoid", 6, 16);
    RegisterOptKernelCreator<HardSwishOp>("", "HardSwish", 14, 16);
    // I
    RegisterOptKernelCreator<IdentityOp>("", "Identity", 1, 13);
    RegisterOptKernelCreator<IfOp>("", "If", 1, 12);
    // L
    RegisterOptKernelCreator<LeakyReluOp>("", "LeakyRelu", 6, 16);
    RegisterOptKernelCreator<LessOp>("", "Less", 7, 16);
    RegisterOptKernelCreator<LogOp>("", "Log", 6, 16);
    RegisterOptKernelCreator<LoopOp>("", "Loop", 1, 12);
    RegisterOptKernelCreator<LSTMOp>("", "LSTM", 7, 13);
    // M
    RegisterOptKernelCreator<MatMulOp>("", "MatMul", 1, 16);
    RegisterOptKernelCreator<MaxOp>("", "Max", 6, 16);
    RegisterOptKernelCreator<MaxPoolOp>("", "MaxPool", 1, 16);
    RegisterOptKernelCreator<MaxUnPoolOp>("", "MaxUnpool", 9, 16);
    RegisterOptKernelCreator<MinOp>("", "Min", 6, 16);
    RegisterOptKernelCreator<MulOp>("", "Mul", 7, 16);
    // N
    RegisterOptKernelCreator<NonMaxSupressionOp>("", "NonMaxSuppression", 10, 16);
    RegisterOptKernelCreator<NonZeroOp>("", "NonZero", 9, 12);
    RegisterOptKernelCreator<NotOp>("", "Not", 1, 16);
    // O
    RegisterOptKernelCreator<OneHotOp>("", "OneHot", 9, 11);
    // P
    RegisterOptKernelCreator<PadOp>("", "Pad", 2, 16);
    RegisterOptKernelCreator<PowOp>("", "Pow", 7, 16);
    RegisterOptKernelCreator<PReluOp>("", "PRelu", 6, 16);
    // R
    RegisterOptKernelCreator<RangeOp>("", "Range", 11, 16);
    RegisterOptKernelCreator<ReduceMaxOp>("", "ReduceMax", 1, 16);
    RegisterOptKernelCreator<ReduceMeanOp>("", "ReduceMean", 1, 16);
    RegisterOptKernelCreator<ReduceMinOp>("", "ReduceMin", 1, 16);
    RegisterOptKernelCreator<ReduceProdOp>("", "ReduceProd", 1, 16);
    RegisterOptKernelCreator<ReduceSumOp>("", "ReduceSum", 1, 16);
    RegisterOptKernelCreator<ReluOp>("", "Relu", 6, 16);
    RegisterOptKernelCreator<ReshapeOp>("", "Reshape", 5, 13);
    RegisterOptKernelCreator<ResizeOp>("", "Resize", 11, 16);
    RegisterOptKernelCreator<ROIAlignOp>("", "RoiAlign", 10, 15);
    // S
    RegisterOptKernelCreator<ScatterElementsOp>("", "ScatterElements", 11, 15);
    RegisterOptKernelCreator<ScatterNDOp>("", "ScatterND", 11, 15);
    RegisterOptKernelCreator<SequenceAtOp>("", "SequenceAt", 11, 16);
    RegisterOptKernelCreator<ShapeOp>("", "Shape", 1, 14);
    RegisterOptKernelCreator<SigmoidOp>("", "Sigmoid", 6, 16);
    RegisterOptKernelCreator<SignOp>("", "Sign", 9, 16);
    RegisterOptKernelCreator<SinOp>("", "Sin", 7, 16);
    RegisterOptKernelCreator<SliceOp>("", "Slice", 1, 16);
    RegisterOptKernelCreator<SoftmaxOp>("", "Softmax", 1, 16);
    RegisterOptKernelCreator<SplitOp>("", "Split", 2, 12);
    RegisterOptKernelCreator<SplitToSequenceOp>("", "SplitToSequence", 11, 16);
    RegisterOptKernelCreator<SqrtOp>("", "Sqrt", 6, 16);
    RegisterOptKernelCreator<SqueezeOp>("", "Squeeze", 1, 12);
    RegisterOptKernelCreator<SubOp>("", "Sub", 7, 16);
    RegisterOptKernelCreator<SumOp>("", "Sum", 6, 16);
    // T
    RegisterOptKernelCreator<TanhOp>("", "Tanh", 6, 16);
    RegisterOptKernelCreator<TileOp>("", "Tile", 6, 16);
    RegisterOptKernelCreator<TopKOp>("", "TopK", 1, 16);
    RegisterOptKernelCreator<TransposeOp>("", "Transpose", 1, 16);
    // U
    RegisterOptKernelCreator<UnsqueezeOp>("", "Unsqueeze", 1, 12);
    // W
    RegisterOptKernelCreator<WhereOp>("", "Where", 9, 16);

    // mmcv custom op
    RegisterOptKernelCreator<MMCVGridSampleOp>("mmcv", "grid_sampler", 1, 1);
    RegisterOptKernelCreator<MMCVNonMaxSuppressionOp>("mmcv", "NonMaxSuppression", 1, 1);
    RegisterOptKernelCreator<MMCVROIAlignOp>("mmcv", "MMCVRoiAlign", 1, 1);
    RegisterOptKernelCreator<MMCVModulatedDeformConv2dOp>("mmcv", "MMCVModulatedDeformConv2d", 1, 1);

    // pmx
    RegisterOptKernelCreator<ChannelShuffleOp>("pmx", "ChannelShuffle", 1, 1);
    RegisterOptKernelCreator<ReorderOp>("pmx", "Reorder", 1, 1);
    RegisterOptKernelCreator<ShapeOperationOp>("pmx", "Shape", 1, 1);
    RegisterOptKernelCreator<SwishOp>("pmx", "Swish", 1, 1);
    RegisterOptKernelCreator<PostDepthwiseConvOp>("pmx", "PostDepthwiseConv", 1, 1);
}

}}} // namespace ppl::nn::x86
