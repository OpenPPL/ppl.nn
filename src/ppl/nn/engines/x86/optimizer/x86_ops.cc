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
#include "ppl/nn/engines/x86/optimizer/ops/ppl/reorder_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/channel_shuffle_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/shape_operation_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/swish_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/post_depthwise_conv_op.h"

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
    auto status = OptKernelCreatorManager::GetInstance()->Register(
        domain, type, utils::VersionRange(first_version, last_version), GenericCreateOptKernel<T>);
    if (status != RC_SUCCESS) {
        exit(-1);
    }
}

// NOTE: sorted in alphabet order
void RegisterBuiltinOpImpls() {
    // onnx op's default domain is ""
    // A
    RegisterOptKernelCreator<AddOp>("", "Add", 7, 12);
    RegisterOptKernelCreator<AndOp>("", "And", 7, 16);
    RegisterOptKernelCreator<ArgmaxOp>("", "ArgMax", 11, 11);
    RegisterOptKernelCreator<AveragePoolOp>("", "AveragePool", 11, 16);
    // B
    RegisterOptKernelCreator<BatchNormalizationOp>("", "BatchNormalization", 9, 13);
    // C
    RegisterOptKernelCreator<CastOp>("", "Cast", 9, 12);
    RegisterOptKernelCreator<CeilOp>("", "Ceil", 6, 12);
    RegisterOptKernelCreator<ClipOp>("", "Clip", 11, 11);
    RegisterOptKernelCreator<ConcatOp>("", "Concat", 11, 12);
    RegisterOptKernelCreator<ConstantOfShapeOp>("", "ConstantOfShape", 9, 16);
    RegisterOptKernelCreator<ConvOp>("", "Conv", 1, 16);
    RegisterOptKernelCreator<ConvTransposeOp>("", "ConvTranspose", 11, 16);
    RegisterOptKernelCreator<CosOp>("", "Cos", 1, 16);
    RegisterOptKernelCreator<CumSumOp>("", "CumSum", 1, 16);
    // D
    RegisterOptKernelCreator<DepthToSpaceOp>("", "DepthToSpace", 11, 12);
    RegisterOptKernelCreator<DivOp>("", "Div", 7, 12);
    // E
    RegisterOptKernelCreator<EqualOp>("", "Equal", 11, 12);
    RegisterOptKernelCreator<ErfOp>("", "Erf", 1, 16);
    RegisterOptKernelCreator<ExpOp>("", "Exp", 6, 12);
    RegisterOptKernelCreator<ExpandOp>("", "Expand", 8, 12);
    // F
    RegisterOptKernelCreator<FlattenOp>("", "Flatten", 11, 12);
    RegisterOptKernelCreator<FloorOp>("", "Floor", 6, 12);
    // G
    RegisterOptKernelCreator<GatherOp>("", "Gather", 11, 12);
    RegisterOptKernelCreator<GatherNDOp>("", "GatherND", 11, 11);
    RegisterOptKernelCreator<GemmOp>("", "Gemm", 11, 12);
    RegisterOptKernelCreator<AveragePoolOp>("", "GlobalAveragePool", 1, 16);
    RegisterOptKernelCreator<GreaterOp>("", "Greater", 9, 12);
    // I
    RegisterOptKernelCreator<IdentityOp>("", "Identity", 1, 12);
    RegisterOptKernelCreator<IfOp>("", "If", 11, 12);
    // L
    RegisterOptKernelCreator<LeakyReluOp>("", "LeakyRelu", 6, 16);
    RegisterOptKernelCreator<LessOp>("", "Less", 9, 12);
    RegisterOptKernelCreator<LogOp>("", "Log", 6, 12);
    RegisterOptKernelCreator<LoopOp>("", "Loop", 11, 12);
    RegisterOptKernelCreator<LSTMOp>("", "LSTM", 7, 13);
    // M
    RegisterOptKernelCreator<MatMulOp>("", "MatMul", 9, 12);
    RegisterOptKernelCreator<MaxOp>("", "Max", 8, 11);
    RegisterOptKernelCreator<MaxPoolOp>("", "MaxPool", 10, 16);
    RegisterOptKernelCreator<MaxUnPoolOp>("", "MaxUnpool", 11, 16);
    RegisterOptKernelCreator<MinOp>("", "Min", 8, 11);
    RegisterOptKernelCreator<MulOp>("", "Mul", 7, 12);
    // N
    RegisterOptKernelCreator<NonMaxSupressionOp>("", "NonMaxSuppression", 11, 16);
    RegisterOptKernelCreator<NonZeroOp>("", "NonZero", 9, 12);
    RegisterOptKernelCreator<NotOp>("", "Not", 1, 16);
    // P
    RegisterOptKernelCreator<PadOp>("", "Pad", 11, 12);
    RegisterOptKernelCreator<PowOp>("", "Pow", 7, 11);
    RegisterOptKernelCreator<PReluOp>("", "PRelu", 9, 16);
    // R
    RegisterOptKernelCreator<RangeOp>("", "Range", 11, 16);
    RegisterOptKernelCreator<ReduceMaxOp>("", "ReduceMax", 1, 16);
    RegisterOptKernelCreator<ReduceMeanOp>("", "ReduceMean", 1, 16);
    RegisterOptKernelCreator<ReduceMinOp>("", "ReduceMin", 1, 16);
    RegisterOptKernelCreator<ReduceProdOp>("", "ReduceProd", 1, 16);
    RegisterOptKernelCreator<ReduceSumOp>("", "ReduceSum", 1, 16);
    RegisterOptKernelCreator<ReluOp>("", "Relu", 6, 12);
    RegisterOptKernelCreator<ReshapeOp>("", "Reshape", 5, 12);
    RegisterOptKernelCreator<ResizeOp>("", "Resize", 11, 12);
    RegisterOptKernelCreator<ROIAlignOp>("", "RoiAlign", 10, 15);
    // S
    RegisterOptKernelCreator<ScatterElementsOp>("", "ScatterElements", 11, 12);
    RegisterOptKernelCreator<ScatterNDOp>("", "ScatterND", 11, 12);
    RegisterOptKernelCreator<SequenceAtOp>("", "SequenceAt", 11, 16);
    RegisterOptKernelCreator<ShapeOp>("", "Shape", 1, 12);
    RegisterOptKernelCreator<SigmoidOp>("", "Sigmoid", 6, 12);
    RegisterOptKernelCreator<SinOp>("", "Sin", 1, 16);
    RegisterOptKernelCreator<SliceOp>("", "Slice", 11, 12);
    RegisterOptKernelCreator<SoftmaxOp>("", "Softmax", 1, 12);
    RegisterOptKernelCreator<SplitOp>("", "Split", 2, 12);
    RegisterOptKernelCreator<SplitToSequenceOp>("", "SplitToSequence", 11, 16);
    RegisterOptKernelCreator<SqrtOp>("", "Sqrt", 6, 12);
    RegisterOptKernelCreator<SqueezeOp>("", "Squeeze", 11, 12);
    RegisterOptKernelCreator<SubOp>("", "Sub", 7, 12);
    RegisterOptKernelCreator<SumOp>("", "Sum", 8, 12);
    // T
    RegisterOptKernelCreator<TanhOp>("", "Tanh", 6, 12);
    RegisterOptKernelCreator<TileOp>("", "Tile", 6, 12);
    RegisterOptKernelCreator<TopKOp>("", "TopK", 11, 16);
    RegisterOptKernelCreator<TransposeOp>("", "Transpose", 1, 12);
    // U
    RegisterOptKernelCreator<UnsqueezeOp>("", "Unsqueeze", 11, 12);
    // W
    RegisterOptKernelCreator<WhereOp>("", "Where", 9, 15);

    // mmcv custom op
    RegisterOptKernelCreator<MMCVGridSampleOp>("mmcv", "grid_sampler", 1, 1);
    RegisterOptKernelCreator<MMCVNonMaxSuppressionOp>("mmcv", "NonMaxSuppression", 1, 1);
    RegisterOptKernelCreator<MMCVROIAlignOp>("mmcv", "MMCVRoiAlign", 1, 1);
    RegisterOptKernelCreator<MMCVModulatedDeformConv2dOp>("mmcv", "MMCVModulatedDeformConv2d", 1, 1);

    // ppl
    RegisterOptKernelCreator<ChannelShuffleOp>("ppl", "ChannelShuffle", 1, 1);
    RegisterOptKernelCreator<ReorderOp>("ppl", "Reorder", 1, 1);
    RegisterOptKernelCreator<PPLShapeOperationOp>("ppl", "Shape", 1, 1);
    RegisterOptKernelCreator<SwishOp>("ppl", "Swish", 1, 1);
    RegisterOptKernelCreator<PostDepthwiseConvOp>("ppl", "PostDepthwiseConv", 1, 1);
}

}}} // namespace ppl::nn::x86
