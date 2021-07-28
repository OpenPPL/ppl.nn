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
#include "ppl/nn/engines/x86/optimizer/ops/mmcv/mmcv_non_max_suppression_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/mmcv/mmcv_roialign_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/reorder_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/channel_shuffle_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/shape_op.h"
#include "ppl/nn/engines/x86/optimizer/ops/ppl/swish_op.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace x86 {

RetCode OptKernelCreatorManager::Register(const string& domain, const string& type, OptKernelCreator creator) {
    auto domain_ret = domain_type_creator_.insert(make_pair(domain, map<string, OptKernelCreator>()));
    auto type_ret = domain_ret.first->second.insert(make_pair(type, creator));
    return type_ret.second ? RC_SUCCESS : RC_EXISTS;
}

OptKernelCreator OptKernelCreatorManager::Find(const string& domain, const string& type) {
    auto type_creator_ref = domain_type_creator_.find(domain);
    if (type_creator_ref != domain_type_creator_.end()) {
        auto creator_ref = type_creator_ref->second.find(type);
        if (creator_ref != type_creator_ref->second.end()) {
            return creator_ref->second;
        }
    }
    return nullptr;
}

template <typename T>
static X86OptKernel* GenericCreateOptKernel(const ir::Node* node) {
    return new T(node);
}

#define REGISTER_OPT_KERNEL_CREATOR(domain, type, classname) \
    domain_type_creator_[domain].insert(make_pair(type, GenericCreateOptKernel<classname>))

OptKernelCreatorManager::OptKernelCreatorManager() {
    // onnx op default domain is ""
    REGISTER_OPT_KERNEL_CREATOR("", "Conv", ConvOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Add", AddOp);
    REGISTER_OPT_KERNEL_CREATOR("", "And", AndOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ArgMax", ArgmaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "AveragePool", AveragePoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "GlobalAveragePool", AveragePoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "BatchNormalization", BatchNormalizationOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Cast", CastOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Ceil", CeilOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ConvTranspose", ConvTransposeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Clip", ClipOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Concat", ConcatOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Constant", ConstantOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ConstantOfShape", ConstantOfShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "DepthToSpace", DepthToSpaceOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Div", DivOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Equal", EqualOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Exp", ExpOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Expand", ExpandOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Flatten", FlattenOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Floor", FloorOp);
    REGISTER_OPT_KERNEL_CREATOR("", "GatherND", GatherNDOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Gather", GatherOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Gemm", GemmOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Greater", GreaterOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Identity", IdentityOp);
    REGISTER_OPT_KERNEL_CREATOR("", "If", IfOp);
    REGISTER_OPT_KERNEL_CREATOR("", "LeakyRelu", LeakyReluOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Less", LessOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Log", LogOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Loop", LoopOp);
    REGISTER_OPT_KERNEL_CREATOR("", "MatMul", MatMulOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Max", MaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "MaxPool", MaxPoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "MaxUnpool", MaxUnPoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Min", MinOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Mul", MulOp);
    REGISTER_OPT_KERNEL_CREATOR("", "NonMaxSuppression", NonMaxSupressionOp);
    REGISTER_OPT_KERNEL_CREATOR("", "NonZero", NonZeroOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Not", NotOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Pad", PadOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Pow", PowOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Range", RangeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMax", ReduceMaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMin", ReduceMinOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMean", ReduceMeanOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceProd", ReduceProdOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceSum", ReduceSumOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Relu", ReluOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Reshape", ReshapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Resize", ResizeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "RoiAlign", ROIAlignOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ScatterElements", ScatterElementsOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ScatterND", ScatterNDOp);
    REGISTER_OPT_KERNEL_CREATOR("", "SequenceAt", SequenceAtOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Shape", ShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sigmoid", SigmoidOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Slice", SliceOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Softmax", SoftmaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Split", SplitOp);
    REGISTER_OPT_KERNEL_CREATOR("", "SplitToSequence", SplitToSequenceOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sqrt", SqrtOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Squeeze", SqueezeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sub", SubOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sum", SumOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Tanh", TanhOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Tile", TileOp);
    REGISTER_OPT_KERNEL_CREATOR("", "TopK", TopKOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Transpose", TransposeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Unsqueeze", UnsqueezeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Where", WhereOp);

    // mmcv custom op
    REGISTER_OPT_KERNEL_CREATOR("mmcv", "grid_sampler", MMCVGridSampleOp);
    REGISTER_OPT_KERNEL_CREATOR("mmcv", "NonMaxSuppression", MMCVNonMaxSuppressionOp);
    REGISTER_OPT_KERNEL_CREATOR("mmcv", "MMCVRoiAlign", MMCVROIAlignOp);

    // ppl
    REGISTER_OPT_KERNEL_CREATOR("ppl", "ChannelShuffle", ChannelShuffleOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Reorder", ReorderOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Shape", PPLShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Swish", SwishOp);
}

}}} // namespace ppl::nn::x86
