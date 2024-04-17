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

#include "opt_kernel_creator_manager.h"

#include "ppl/nn/common/logger.h"

#include "ppl/nn/engines/llm_cuda/ops/onnx/add_op.h"
#include "ppl/nn/engines/llm_cuda/ops/onnx/cast_op.h"
#include "ppl/nn/engines/llm_cuda/ops/onnx/gather_op.h"
#include "ppl/nn/engines/llm_cuda/ops/onnx/mul_op.h"
#include "ppl/nn/engines/llm_cuda/ops/onnx/reshape_op.h"
#include "ppl/nn/engines/llm_cuda/ops/onnx/slice_op.h"
#include "ppl/nn/engines/llm_cuda/ops/onnx/split_op.h"
#include "ppl/nn/engines/llm_cuda/ops/onnx/sub_op.h"

#include "ppl/nn/engines/llm_cuda/ops/pmx/column_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/geglu_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/gelu_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/key_value_cache_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/layer_norm_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/moe_column_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/moe_reduce_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/moe_row_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/moe_select_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/multi_head_attention_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/parallel_embedding_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/rms_norm_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/rotary_position_embedding_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/rotary_2d_position_embedding_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/row_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/silu_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/swiglu_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/swish_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/vision_embedding_op.h"

#include "ppl/nn/engines/llm_cuda/ops/pmx/dynamic_batching/key_value_cache_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/dynamic_batching/multi_head_attention_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/dynamic_batching/multi_head_cache_attention_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/dynamic_batching/position_index_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/dynamic_batching/rotary_position_embedding_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/dynamic_batching/rotary_2d_position_embedding_op.h"

#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/column_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/online_dequantize_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/online_quantize_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/online_quantize_rms_norm_op.h"
#include "ppl/nn/engines/llm_cuda/ops/pmx/i8i8/row_parallel_linear_op.h"

using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace llm { namespace cuda {

template <typename T>
static LlmCudaOptKernel* GenericCreateOptKernel(const ir::Node* node) {
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

void RegisterBuiltinOpImpls() {
    static bool ops_are_registered = false;
    if (ops_are_registered) {
        return;
    }
    ops_are_registered = true;

    /*                                                                        */
    /*                                 ONNX                                   */
    /*                                                                        */
    // A
    RegisterOptKernelCreator<onnx::AddOp>("", "Add", 7, 16);
    // B
    // C
    RegisterOptKernelCreator<onnx::CastOp>("", "Cast", 9, 16);
    // D
    // E
    // F
    // G
    RegisterOptKernelCreator<onnx::GatherOp>("", "Gather", 1, 16);
    // H
    // I
    // J
    // K
    // L
    // M
    RegisterOptKernelCreator<onnx::MulOp>("", "Mul", 1, 16);

    // N
    // O
    // P
    // Q
    // R
    RegisterOptKernelCreator<onnx::ReshapeOp>("", "Reshape", 5, 16);
    // S
    RegisterOptKernelCreator<onnx::SliceOp>("", "Slice", 1, 16);
    RegisterOptKernelCreator<onnx::SplitOp>("", "Split", 2, 16);
    RegisterOptKernelCreator<onnx::SubOp>("", "Sub", 7, 16);
    // T
    // U
    // V
    // W
    // X
    // Y
    // Z

    /*                                                                        */
    /*                                 PMX                                    */
    /*                                                                        */
    // A
    // B
    // C
    RegisterOptKernelCreator<pmx::ColumnParallelLinearOp>("pmx", "ColumnParallelLinear", 1, 1);
    // D
    // E
    // F
    // G
    RegisterOptKernelCreator<pmx::GeGLUOp>("pmx", "GeGLU", 1, 1);
    RegisterOptKernelCreator<pmx::GELUOp>("pmx", "GELU", 1, 1);
    // H
    // I
    // J
    // K
    RegisterOptKernelCreator<pmx::KeyValueCacheOp>("pmx", "KeyValueCache", 1, 1);
    // L
    RegisterOptKernelCreator<pmx::LayerNormOp>("pmx", "LayerNorm", 1, 1);
    RegisterOptKernelCreator<pmx::LinearOp>("pmx", "Linear", 1, 1);

    // M
    RegisterOptKernelCreator<pmx::MoeColumnParallelLinearOp>("pmx", "MoeColumnParallelLinear", 1, 1);
    RegisterOptKernelCreator<pmx::MoeReduceOp>("pmx", "MoeReduce", 1, 1);
    RegisterOptKernelCreator<pmx::MoeRowParallelLinearOp>("pmx", "MoeRowParallelLinear", 1, 1);
    RegisterOptKernelCreator<pmx::MoeSelectOp>("pmx", "MoeSelect", 1, 1);
    // N
    RegisterOptKernelCreator<pmx::MultiHeadAttentionOp>("pmx", "MultiHeadAttention", 1, 1);

    // N
    // O
    // P
    RegisterOptKernelCreator<pmx::ParallelEmbeddingOp>("pmx", "ParallelEmbedding", 1, 1);
    // Q
    // R
    RegisterOptKernelCreator<pmx::RMSNormOp>("pmx", "RMSNorm", 1, 1);
    RegisterOptKernelCreator<pmx::RotaryPositionEmbeddingOp>("pmx", "RotaryPositionEmbedding", 1, 1);
    RegisterOptKernelCreator<pmx::Rotary2DPositionEmbeddingOp>("pmx", "Rotary2DPositionEmbedding", 1, 1);
    RegisterOptKernelCreator<pmx::RowParallelLinearOp>("pmx", "RowParallelLinear", 1, 1);
    // S
    RegisterOptKernelCreator<pmx::SiLUOp>("pmx", "SiLU", 1, 1);
    RegisterOptKernelCreator<pmx::SwiGLUOp>("pmx", "SwiGLU", 1, 1);
    RegisterOptKernelCreator<pmx::SwishOp>("pmx", "Swish", 1, 1);
    // T
    // U
    // V
    RegisterOptKernelCreator<pmx::VisionEmbeddingOp>("pmx", "VisionEmbedding", 1, 1);
    // W
    // X
    // Y
    // Z

    /*                                                                        */
    /*                        PMX.DYNAMIC_BATCHING                            */
    /*                                                                        */
    // A
    // B
    // C
    // D
    // E
    // F
    // G
    // H
    // I
    // J
    // K
    RegisterOptKernelCreator<pmx::DynamicBatchingKeyValueCacheOp>("pmx.dynamic_batching", "KeyValueCache", 1, 1);
    // L
    // M
    // N
    RegisterOptKernelCreator<pmx::DynamicBatchingMultiHeadAttentionOp>("pmx.dynamic_batching", "MultiHeadAttention", 1, 1);
    RegisterOptKernelCreator<pmx::DynamicBatchingMultiHeadCacheAttentionOp>("pmx.dynamic_batching", "MultiHeadCacheAttention", 1, 1);
    // O
    // P
    RegisterOptKernelCreator<pmx::DynamicBatchingPositionIndexOp>("pmx.dynamic_batching", "PositionIndex", 1, 1);
    // Q
    // R
    RegisterOptKernelCreator<pmx::DynamicBatchingRotaryPositionEmbeddingOp>("pmx.dynamic_batching", "RotaryPositionEmbedding", 1, 1);
    RegisterOptKernelCreator<pmx::DynamicBatchingRotary2DPositionEmbeddingOp>("pmx.dynamic_batching", "Rotary2DPositionEmbedding", 1, 1);

    // S
    // T
    // U
    // V
    // W
    // X
    // Y
    // Z

    /*                                                                        */
    /*                        PMX.I8I8                                        */
    /*                                                                        */
    // A
    // B
    // C
    RegisterOptKernelCreator<pmx::I8I8ColumnParallelLinearOp>("pmx.i8i8", "ColumnParallelLinear", 1, 1);
    // D
    // E
    // F
    // G
    // H
    // I
    // J
    // K
    // L
    // M
    // N
    // O
    RegisterOptKernelCreator<pmx::I8I8OnlineDequantizeOp>("pmx.i8i8", "OnlineDequantize", 1, 1);
    RegisterOptKernelCreator<pmx::I8I8OnlineQuantizeOp>("pmx.i8i8", "OnlineQuantize", 1, 1);
    RegisterOptKernelCreator<pmx::I8I8OnlineQuantizeRMSNormOp>("pmx.i8i8", "OnlineQuantizeRMSNorm", 1, 1);
    // P
    // Q
    // R
    RegisterOptKernelCreator<pmx::I8I8RowParallelLinearOp>("pmx.i8i8", "RowParallelLinear", 1, 1);
    // S
    // T
    // U
    // V
    // W
    // X
    // Y
    // Z
}

}}}} // namespace ppl::nn::llm::cuda
