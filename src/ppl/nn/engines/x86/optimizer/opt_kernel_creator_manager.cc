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
#include "ppl/nn/engines/x86/optimizer/ops/onnx/constant_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/constant_of_shape_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/depth_to_space_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/div_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/onnx/equal_op.h"
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
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode OptKernelCreatorManager::Register(const string& domain, const string& type, const utils::VersionRange& ver,
                                          OptKernelCreator creator) {
    return mgr_.Register(domain, type, ver, creator);
}

OptKernelCreator OptKernelCreatorManager::Find(const string& domain, const string& type, uint64_t version) const {
    return *mgr_.Find(domain, type, version);
}

void OptKernelCreatorManager::Remove(const string& domain, const string& type) {
    mgr_.Remove(domain, type);
}

template <typename T>
static X86OptKernel* GenericCreateOptKernel(const ir::Node* node) {
    return new T(node);
}

#define REGISTER_OPT_KERNEL_CREATOR(domain, type, first_version, last_version, classname) \
    Register(domain, type, utils::VersionRange(first_version, last_version), GenericCreateOptKernel<classname>)

OptKernelCreatorManager::OptKernelCreatorManager() {
    // onnx op default domain is ""
    REGISTER_OPT_KERNEL_CREATOR("", "Conv", 11, 11, ConvOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Add", 11, 11, AddOp);
    REGISTER_OPT_KERNEL_CREATOR("", "And", 11, 11, AndOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ArgMax", 11, 11, ArgmaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "AveragePool", 11, 11, AveragePoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "GlobalAveragePool", 11, 11, AveragePoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "BatchNormalization", 11, 11, BatchNormalizationOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Cast", 11, 11, CastOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Ceil", 11, 11, CeilOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ConvTranspose", 11, 11, ConvTransposeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Clip", 11, 11, ClipOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Concat", 11, 11, ConcatOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Constant", 11, 11, ConstantOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ConstantOfShape", 11, 11, ConstantOfShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "DepthToSpace", 11, 11, DepthToSpaceOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Div", 11, 11, DivOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Equal", 11, 11, EqualOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Exp", 11, 11, ExpOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Expand", 11, 11, ExpandOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Flatten", 11, 11, FlattenOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Floor", 11, 11, FloorOp);
    REGISTER_OPT_KERNEL_CREATOR("", "GatherND", 11, 11, GatherNDOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Gather", 11, 11, GatherOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Gemm", 11, 11, GemmOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Greater", 11, 11, GreaterOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Identity", 11, 11, IdentityOp);
    REGISTER_OPT_KERNEL_CREATOR("", "If", 1, 13, IfOp);
    REGISTER_OPT_KERNEL_CREATOR("", "LeakyRelu", 11, 11, LeakyReluOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Less", 11, 11, LessOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Log", 11, 11, LogOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Loop", 1, 13, LoopOp);
    REGISTER_OPT_KERNEL_CREATOR("", "LSTM", 11, 11, LSTMOp);
    REGISTER_OPT_KERNEL_CREATOR("", "MatMul", 11, 11, MatMulOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Max", 11, 11, MaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "MaxPool", 11, 11, MaxPoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "MaxUnpool", 11, 11, MaxUnPoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Min", 11, 11, MinOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Mul", 11, 11, MulOp);
    REGISTER_OPT_KERNEL_CREATOR("", "NonMaxSuppression", 11, 11, NonMaxSupressionOp);
    REGISTER_OPT_KERNEL_CREATOR("", "NonZero", 11, 11, NonZeroOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Not", 11, 11, NotOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Pad", 11, 11, PadOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Pow", 11, 11, PowOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Range", 11, 11, RangeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMax", 11, 11, ReduceMaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMin", 11, 11, ReduceMinOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMean", 11, 11, ReduceMeanOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceProd", 11, 11, ReduceProdOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceSum", 11, 11, ReduceSumOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Relu", 11, 11, ReluOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Reshape", 11, 11, ReshapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Resize", 11, 11, ResizeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "RoiAlign", 11, 11, ROIAlignOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ScatterElements", 11, 11, ScatterElementsOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ScatterND", 11, 11, ScatterNDOp);
    REGISTER_OPT_KERNEL_CREATOR("", "SequenceAt", 1, 13, SequenceAtOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Shape", 11, 11, ShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sigmoid", 11, 11, SigmoidOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Slice", 11, 11, SliceOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Softmax", 11, 11, SoftmaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Split", 11, 11, SplitOp);
    REGISTER_OPT_KERNEL_CREATOR("", "SplitToSequence", 1, 13, SplitToSequenceOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sqrt", 11, 11, SqrtOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Squeeze", 11, 11, SqueezeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sub", 11, 11, SubOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sum", 11, 11, SumOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Tanh", 11, 11, TanhOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Tile", 11, 11, TileOp);
    REGISTER_OPT_KERNEL_CREATOR("", "TopK", 11, 11, TopKOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Transpose", 11, 11, TransposeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Unsqueeze", 11, 11, UnsqueezeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Where", 11, 11, WhereOp);

    // mmcv custom op
    REGISTER_OPT_KERNEL_CREATOR("mmcv", "grid_sampler", 1, 1, MMCVGridSampleOp);
    REGISTER_OPT_KERNEL_CREATOR("mmcv", "NonMaxSuppression", 1, 1, MMCVNonMaxSuppressionOp);
    REGISTER_OPT_KERNEL_CREATOR("mmcv", "MMCVRoiAlign", 1, 1, MMCVROIAlignOp);
    REGISTER_OPT_KERNEL_CREATOR("mmcv", "MMCVModulatedDeformConv2d", 1, 1, MMCVModulatedDeformConv2dOp);

    // ppl
    REGISTER_OPT_KERNEL_CREATOR("ppl", "ChannelShuffle", 1, 1, ChannelShuffleOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Reorder", 1, 1, ReorderOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Shape", 1, 1, PPLShapeOperationOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Swish", 1, 1, SwishOp);
}

}}} // namespace ppl::nn::x86
