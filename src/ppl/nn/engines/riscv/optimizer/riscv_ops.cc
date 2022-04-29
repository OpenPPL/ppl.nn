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

#include "ppl/nn/engines/riscv/optimizer/opt_kernel_creator_manager.h"
using namespace std;
using namespace ppl::common;

#include "ppl/nn/engines/riscv/optimizer/ops/onnx/conv/conv_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/add_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/sub_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/mul_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/div_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/relu_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reshape_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/shape_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/sigmoid_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/split_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/unsqueeze_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/max_pool_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/average_pool_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/flatten_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/softmax_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/gemm_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/clip_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reduce_mean_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reduce_max_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reduce_min_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/reduce_sum_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/concat_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/transpose_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/gather_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/slice_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/argmax_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/resize_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/leaky_relu_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/equal_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/less_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/topk_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/where_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/constant_of_shape_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/tile_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/squeeze_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/range_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/expand_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/not_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/sqrt_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/log_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/floor_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/exp_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/cast_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/scatter_nd_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/non_max_suppression_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/conv_transpose_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/onnx/roialign_op.h"

#include "ppl/nn/engines/riscv/optimizer/ops/mmcv/mmcv_gridsample_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/mmcv/mmcv_roialign_op.h"

#include "ppl/nn/engines/riscv/optimizer/ops/pmx/shape_operation_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/pmx/reorder_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/pmx/channel_shuffle_op.h"

namespace ppl { namespace nn { namespace riscv {

template <typename T>
static RiscvOptKernel* GenericCreateOptKernel(const ir::Node* node) {
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
    RegisterOptKernelCreator<AddOp>("", "Add", 7, 12);
    RegisterOptKernelCreator<ArgmaxOp>("", "ArgMax", 1, 11);
    RegisterOptKernelCreator<AveragePoolOp>("", "AveragePool", 1, 16);

    RegisterOptKernelCreator<CastOp>("", "Cast", 9, 12);
    RegisterOptKernelCreator<ClipOp>("", "Clip", 6, 16);
    RegisterOptKernelCreator<ConcatOp>("", "Concat", 4, 16);
    RegisterOptKernelCreator<ConstantOfShapeOp>("", "ConstantOfShape", 9, 16);
    RegisterOptKernelCreator<ConvOp>("", "Conv", 1, 16);
    RegisterOptKernelCreator<ConvTransposeOp>("", "ConvTranspose", 1, 16);

    RegisterOptKernelCreator<DivOp>("", "Div", 7, 12);

    RegisterOptKernelCreator<EqualOp>("", "Equal", 7, 16);
    RegisterOptKernelCreator<ExpandOp>("", "Expand", 8, 12);
    RegisterOptKernelCreator<ExpOp>("", "Exp", 6, 16);

    RegisterOptKernelCreator<FlattenOp>("", "Flatten", 1, 16);
    RegisterOptKernelCreator<FloorOp>("", "Floor", 6, 16);

    RegisterOptKernelCreator<GatherOp>("", "Gather", 1, 16);
    RegisterOptKernelCreator<GemmOp>("", "Gemm", 9, 16);
    RegisterOptKernelCreator<AveragePoolOp>("", "GlobalAveragePool", 1, 16);

    RegisterOptKernelCreator<LeakyReLUOp>("", "LeakyRelu", 6, 16);
    RegisterOptKernelCreator<LessOp>("", "Less", 9, 12);
    RegisterOptKernelCreator<LogOp>("", "Log", 6, 16);

    RegisterOptKernelCreator<MaxPoolOp>("", "MaxPool", 1, 16);
    RegisterOptKernelCreator<MulOp>("", "Mul", 7, 12);

    RegisterOptKernelCreator<NotOp>("", "Not", 1, 16);
    RegisterOptKernelCreator<NonMaxSupressionOp>("", "NonMaxSuppression", 10, 16);

    RegisterOptKernelCreator<RangeOp>("", "Range", 11, 16);
    RegisterOptKernelCreator<ReduceMeanOp>("", "ReduceMean", 1, 16);
    RegisterOptKernelCreator<ReduceMaxOp>("", "ReduceMax", 1, 16);
    RegisterOptKernelCreator<ReduceMinOp>("", "ReduceMin", 1, 16);
    RegisterOptKernelCreator<ReduceSumOp>("", "ReduceSum", 1, 16);
    RegisterOptKernelCreator<ReluOp>("", "Relu", 6, 12);
    RegisterOptKernelCreator<ReshapeOp>("", "Reshape", 5, 12);
    RegisterOptKernelCreator<ResizeOp>("", "Resize", 11, 12);
    RegisterOptKernelCreator<RoiAlignOp>("", "RoiAlign", 10, 15);

    RegisterOptKernelCreator<ScatterNDOp>("", "ScatterND", 11, 15);
    RegisterOptKernelCreator<ShapeOp>("", "Shape", 1, 12);
    RegisterOptKernelCreator<SigmoidOp>("", "Sigmoid", 6, 12);
    RegisterOptKernelCreator<SoftmaxOp>("", "Softmax", 1, 16);
    RegisterOptKernelCreator<SplitOp>("", "Split", 2, 12);
    RegisterOptKernelCreator<SqueezeOp>("", "Squeeze", 1, 12);
    RegisterOptKernelCreator<SliceOp>("", "Slice", 1, 16);
    RegisterOptKernelCreator<SubOp>("", "Sub", 7, 12);
    RegisterOptKernelCreator<SqrtOp>("", "Sqrt", 6, 16);

    RegisterOptKernelCreator<TileOp>("", "Tile", 6, 12);
    RegisterOptKernelCreator<TopKOp>("", "TopK", 1, 16);
    RegisterOptKernelCreator<TransposeOp>("", "Transpose", 1, 12);

    RegisterOptKernelCreator<UnsqueezeOp>("", "Unsqueeze", 1, 12);

    RegisterOptKernelCreator<WhereOp>("", "Where", 9, 15);

    // mmcv custom op
    RegisterOptKernelCreator<MMCVGridSampleOp>("mmcv", "grid_sampler", 1, 1);
    RegisterOptKernelCreator<MMCVROIAlignOp>("mmcv", "MMCVRoiAlign", 1, 1);

    // pmx
    RegisterOptKernelCreator<ShapeOperationOp>("pmx", "Shape", 1, 1);
    RegisterOptKernelCreator<ReorderOp>("pmx", "Reorder", 1, 1);
    RegisterOptKernelCreator<ChannelShuffleOp>("pmx", "ChannelShuffle", 1, 1);
}

}}} // namespace ppl::nn::riscv
