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
#include "ppl/nn/engines/arm/optimizer/ops/ppl/shape_operation_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/ppl/channel_shuffle_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/ppl/reorder_op.h"
#include "ppl/nn/engines/arm/optimizer/ops/ppl/shape_operation_op.h"

#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace arm {

RetCode OptKernelCreatorManager::Register(const string& domain, const string& type, const utils::VersionRange& ver,
                                          OptKernelCreator creator) {
    return mgr_.Register(domain, type, ver, creator);
}

OptKernelCreator OptKernelCreatorManager::Find(const string& domain, const string& type, uint64_t version) const {
    auto ret = mgr_.Find(domain, type, version);
    if (ret) {
        return *ret;
    }
    return nullptr;
}

void OptKernelCreatorManager::Remove(const string& domain, const string& type) {
    mgr_.Remove(domain, type);
}

template <typename T>
static ArmOptKernel* GenericCreateOptKernel(const ir::Node* node) {
    return new T(node);
}

#define REGISTER_OPT_KERNEL_CREATOR(domain, type, first_version, last_version, classname)                      \
    do {                                                                                                       \
        if (last_version < first_version) {                                                                    \
            LOG(ERROR) << "register op[" << domain << ":" << type << "] failed: last_version[" << last_version \
                       << "] < first_version[" << first_version << "]";                                        \
            exit(-1);                                                                                          \
        }                                                                                                      \
        auto status = Register(domain, type, utils::VersionRange(first_version, last_version),                 \
                               GenericCreateOptKernel<classname>);                                             \
        if (status != RC_SUCCESS) {                                                                            \
            exit(-1);                                                                                          \
        }                                                                                                      \
    } while (0)

OptKernelCreatorManager::OptKernelCreatorManager() {
    // onnx op default domain is ""
    // A
    REGISTER_OPT_KERNEL_CREATOR("", "Add", 7, 16, AddOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ArgMax", 11, 11, ArgMaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "AveragePool", 11, 16, AvePoolOp);
    // B
    REGISTER_OPT_KERNEL_CREATOR("", "BatchNormalization", 9, 13, BatchNormalizationOp);
    // C
    REGISTER_OPT_KERNEL_CREATOR("", "Cast", 9, 16, CastOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Clip", 11, 16, ClipOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Concat", 4, 16, ConcatOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ConstantOfShape", 9, 16, ConstantOfShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Conv", 1, 16, ConvOp);
    // D
    REGISTER_OPT_KERNEL_CREATOR("", "Div", 7, 16, DivOp);
    // E
    REGISTER_OPT_KERNEL_CREATOR("", "Equal", 11, 16, EqualOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Exp", 6, 16, ExpOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Expand", 8, 16, ExpandOp);
    // F
    REGISTER_OPT_KERNEL_CREATOR("", "Flatten", 1, 16, FlattenOp);
    // G
    REGISTER_OPT_KERNEL_CREATOR("", "Gather", 11, 16, GatherOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Gemm", 11, 16, GemmOp);
    REGISTER_OPT_KERNEL_CREATOR("", "GlobalAveragePool", 1, 16, AvePoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "GlobalMaxPool", 1, 16, MaxPoolOp);
    // L
    REGISTER_OPT_KERNEL_CREATOR("", "LeakyRelu", 6, 16, LeakyReLUOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Less", 9, 16, LessOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Log", 6, 16, LogOp);
    // M
    REGISTER_OPT_KERNEL_CREATOR("", "MaxPool", 10, 16, MaxPoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Mul", 7, 16, MulOp);
    // N
    REGISTER_OPT_KERNEL_CREATOR("", "Not", 1, 16, NotOp);
    // P
    REGISTER_OPT_KERNEL_CREATOR("", "Pad", 11, 16, PadOp);
    // R
    REGISTER_OPT_KERNEL_CREATOR("", "Range", 11, 16, RangeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMax", 1, 16, ReduceMaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMean", 1, 16, ReduceMeanOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMin", 1, 16, ReduceMinOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceProd", 1, 16, ReduceProdOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceSum", 1, 16, ReduceSumOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Relu", 6, 16, ReluOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Reshape", 5, 13, ReshapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Resize", 11, 12, ResizeOp);
    // S
    REGISTER_OPT_KERNEL_CREATOR("", "ScatterND", 11, 15, ScatterNDOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Shape", 1, 14, ShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sigmoid", 1, 16, SigmoidOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Slice", 10, 16, SliceOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Softmax", 11, 12, SoftmaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Split", 2, 12, SplitOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sqrt", 6, 16, SqrtOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Squeeze", 1, 12, SqueezeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sub", 7, 16, SubOp);
    // T
    REGISTER_OPT_KERNEL_CREATOR("", "Tile", 6, 16, TileOp);
    REGISTER_OPT_KERNEL_CREATOR("", "TopK", 11, 16, TopKOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Transpose", 1, 16, TransposeOp);
    // U
    REGISTER_OPT_KERNEL_CREATOR("", "Unsqueeze", 1, 12, UnsqueezeOp);
    // W
    REGISTER_OPT_KERNEL_CREATOR("", "Where", 9, 16, WhereOp);
    // mmcv custom op

    // ppl
    REGISTER_OPT_KERNEL_CREATOR("ppl", "ChannelShuffle", 1, 1, ChannelShuffleOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Reorder", 1, 1, ReorderOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Shape", 1, 1, PPLShapeOperationOp);
}

}}} // namespace ppl::nn::x86
