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

#include "ppl/nn/engines/riscv/optimizer/ops/ppl/shape_operation_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/ppl/reorder_op.h"
#include "ppl/nn/engines/riscv/optimizer/ops/ppl/channel_shuffle_op.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace riscv {

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
static RiscvOptKernel* GenericCreateOptKernel(const ir::Node* node) {
    return new T(node);
}

// #define REGISTER_OPT_KERNEL_CREATOR(domain, type, classname) \
//     domain_type_creator_[domain].insert(make_pair(type, GenericCreateOptKernel<classname>))

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
    REGISTER_OPT_KERNEL_CREATOR("", "Add", 7, 12, AddOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ArgMax", 11, 11, ArgmaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "AveragePool", 11, 16, AveragePoolOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Cast", 9, 12, CastOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Clip", 11, 11, ClipOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Concat", 11, 12, ConcatOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ConstantOfShape", 11, 12, ConstantOfShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Conv", 1, 16, ConvOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ConvTranspose", 11, 16, ConvTransposeOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Div", 7, 12, DivOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Equal", 11, 12, EqualOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Expand", 8, 12, ExpandOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Exp", 6, 16, ExpOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Flatten", 11, 12, FlattenOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Floor", 6, 12, FloorOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Gather", 11, 12, GatherOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Gemm", 11, 12, GemmOp);
    REGISTER_OPT_KERNEL_CREATOR("", "GlobalAveragePool", 1, 16, AveragePoolOp);

    REGISTER_OPT_KERNEL_CREATOR("", "LeakyRelu", 6, 16, LeakyReLUOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Less", 9, 12, LessOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Log", 6, 16, LogOp);

    REGISTER_OPT_KERNEL_CREATOR("", "MaxPool", 11, 11, MaxPoolOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Mul", 7, 12, MulOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Not", 1, 16, NotOp);
    REGISTER_OPT_KERNEL_CREATOR("", "NonMaxSuppression", 11, 16, NonMaxSupressionOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Range", 11, 16, RangeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMean", 11, 12, ReduceMeanOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMax", 11, 11, ReduceMaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceMin", 11, 11, ReduceMinOp);
    REGISTER_OPT_KERNEL_CREATOR("", "ReduceSum", 11, 12, ReduceSumOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Relu", 6, 12, ReluOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Reshape", 5, 12, ReshapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Resize", 11, 12, ResizeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "RoiAlign", 10, 15, RoiAlignOp);

    REGISTER_OPT_KERNEL_CREATOR("", "ScatterND", 11, 12, ScatterNDOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Shape", 1, 12, ShapeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sigmoid", 6, 12, SigmoidOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Softmax", 11, 12, SoftmaxOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Split", 11, 12, SplitOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Squeeze", 11, 12, SqueezeOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Slice", 11, 12, SliceOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sub", 7, 12, SubOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Sqrt", 6, 16, SqrtOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Tile", 6, 12, TileOp);
    REGISTER_OPT_KERNEL_CREATOR("", "TopK", 11, 16, TopKOp);
    REGISTER_OPT_KERNEL_CREATOR("", "Transpose", 1, 12, TransposeOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Unsqueeze", 11, 12, UnsqueezeOp);

    REGISTER_OPT_KERNEL_CREATOR("", "Where", 9, 15, WhereOp);

    // mmcv custom op
    REGISTER_OPT_KERNEL_CREATOR("mmcv", "grid_sampler", 1, 1, MMCVGridSampleOp);
    REGISTER_OPT_KERNEL_CREATOR("mmcv", "MMCVRoiAlign", 1, 1, MMCVROIAlignOp);

    // ppl
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Shape", 1, 1, PPLShapeOperationOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "Reorder", 1, 1, ReorderOp);
    REGISTER_OPT_KERNEL_CREATOR("ppl", "ChannelShuffle", 1, 1, ChannelShuffleOp);
}

}}} // namespace ppl::nn::riscv
