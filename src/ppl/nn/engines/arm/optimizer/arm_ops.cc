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

#include "ppl/nn/engines/arm/optimizer/opt_kernel_creator_manager.h"
using namespace std;
using namespace ppl::common;

#include "ppl/nn/engines/arm/optimizer/ops/onnx/conv_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/add_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/argmax_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/avepool_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/batch_normalization_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/cast_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/constant_of_shape_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/clip_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/div_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/equal_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/exp_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/expand_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/flatten_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/gather_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/less_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/log_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/mul_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/not_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/range_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/flatten_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/gather_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/leaky_relu_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/maxpool_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/mul_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/pad_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/reduce_max_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/reduce_mean_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/reduce_min_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/reduce_prod_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/reduce_sum_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/relu_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/reshape_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/resize_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/shape_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/scatter_nd_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/sigmoid_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/slice_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/softmax_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/split_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/sqrt_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/squeeze_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/sub_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/tile_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/topk_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/transpose_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/unsqueeze_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/onnx/where_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/pmx/shape_operation_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/pmx/channel_shuffle_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/pmx/reorder_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/pmx/shape_operation_op.h"

namespace ppl { namespace nn { namespace arm {

template <typename T>
static ArmOptKernel* GenericCreateOptKernel(const ir::Node* node) {
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
    // onnx op default domain is ""
    // A
    RegisterOptKernelCreator<AddOp>("", "Add", 7, 16);
    RegisterOptKernelCreator<ArgMaxOp>("", "ArgMax", 1, 11);
    RegisterOptKernelCreator<AvePoolOp>("", "AveragePool", 1, 16);
    // B
    RegisterOptKernelCreator<BatchNormalizationOp>("", "BatchNormalization", 9, 13);
    // C
    RegisterOptKernelCreator<CastOp>("", "Cast", 9, 16);
    RegisterOptKernelCreator<ClipOp>("", "Clip", 6, 16);
    RegisterOptKernelCreator<ConcatOp>("", "Concat", 4, 16);
    RegisterOptKernelCreator<ConstantOfShapeOp>("", "ConstantOfShape", 9, 16);
    RegisterOptKernelCreator<ConvOp>("", "Conv", 1, 16);
    // D
    RegisterOptKernelCreator<DivOp>("", "Div", 7, 16);
    // E
    RegisterOptKernelCreator<EqualOp>("", "Equal", 7, 16);
    RegisterOptKernelCreator<ExpOp>("", "Exp", 6, 16);
    RegisterOptKernelCreator<ExpandOp>("", "Expand", 8, 16);
    // F
    RegisterOptKernelCreator<FlattenOp>("", "Flatten", 1, 16);
    // G
    RegisterOptKernelCreator<GatherOp>("", "Gather", 1, 16);
    RegisterOptKernelCreator<GemmOp>("", "Gemm", 9, 16);
    RegisterOptKernelCreator<AvePoolOp>("", "GlobalAveragePool", 1, 16);
    RegisterOptKernelCreator<MaxPoolOp>("", "GlobalMaxPool", 1, 16);
    // L
    RegisterOptKernelCreator<LeakyReLUOp>("", "LeakyRelu", 6, 16);
    RegisterOptKernelCreator<LessOp>("", "Less", 7, 16);
    RegisterOptKernelCreator<LogOp>("", "Log", 6, 16);
    // M
    RegisterOptKernelCreator<MaxPoolOp>("", "MaxPool", 1, 16);
    RegisterOptKernelCreator<MulOp>("", "Mul", 7, 16);
    // N
    RegisterOptKernelCreator<NotOp>("", "Not", 1, 16);
    // P
    RegisterOptKernelCreator<PadOp>("", "Pad", 2, 16);
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
    // S
    RegisterOptKernelCreator<ScatterNDOp>("", "ScatterND", 11, 15);
    RegisterOptKernelCreator<ShapeOp>("", "Shape", 1, 14);
    RegisterOptKernelCreator<SigmoidOp>("", "Sigmoid", 6, 16);
    RegisterOptKernelCreator<SliceOp>("", "Slice", 1, 16);
    RegisterOptKernelCreator<SoftmaxOp>("", "Softmax", 1, 16);
    RegisterOptKernelCreator<SplitOp>("", "Split", 2, 12);
    RegisterOptKernelCreator<SqrtOp>("", "Sqrt", 6, 16);
    RegisterOptKernelCreator<SqueezeOp>("", "Squeeze", 1, 12);
    RegisterOptKernelCreator<SubOp>("", "Sub", 7, 16);
    // T
    RegisterOptKernelCreator<TileOp>("", "Tile", 6, 16);
    RegisterOptKernelCreator<TopKOp>("", "TopK", 1, 16);
    RegisterOptKernelCreator<TransposeOp>("", "Transpose", 1, 16);
    // U
    RegisterOptKernelCreator<UnsqueezeOp>("", "Unsqueeze", 1, 12);
    // W
    RegisterOptKernelCreator<WhereOp>("", "Where", 9, 16);

    // pmx
    RegisterOptKernelCreator<ChannelShuffleOp>("pmx", "ChannelShuffle", 1, 1);
    RegisterOptKernelCreator<ReorderOp>("pmx", "Reorder", 1, 1);
    RegisterOptKernelCreator<ShapeOperationOp>("pmx", "Shape", 1, 1);
}

}}} // namespace ppl::nn::arm
