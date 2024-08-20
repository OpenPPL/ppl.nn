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

#include "ppl/nn/engines/llm_cuda/ops/opmx/column_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/geglu_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/gelu_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/key_value_cache_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/layer_norm_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/moe_column_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/moe_reduce_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/moe_row_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/moe_select_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/multi_head_attention_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/parallel_embedding_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/pixel_unshuffle_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/rms_norm_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/rotary_position_embedding_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/row_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/silu_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/swiglu_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/swish_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/tensor_parallel_rms_norm_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/vision_embedding_op.h"

#include "ppl/nn/engines/llm_cuda/ops/opmx/dynamic_batching/key_value_cache_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/dynamic_batching/multi_head_attention_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/dynamic_batching/multi_head_cache_attention_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/dynamic_batching/position_index_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/dynamic_batching/rotary_position_embedding_op.h"

#include "ppl/nn/engines/llm_cuda/ops/opmx/i8i8/column_parallel_linear_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/i8i8/online_dequantize_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/i8i8/online_quantize_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/i8i8/online_quantize_rms_norm_op.h"
#include "ppl/nn/engines/llm_cuda/ops/opmx/i8i8/row_parallel_linear_op.h"

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
    /*                                 OPMX                                    */
    /*                                                                        */
    // A
    // B
    // C
    RegisterOptKernelCreator<opmx::ColumnParallelLinearOp>("opmx", "ColumnParallelLinear", 1, 1);
    // D
    // E
    // F
    // G
    RegisterOptKernelCreator<opmx::GeGLUOp>("opmx", "GeGLU", 1, 1);
    RegisterOptKernelCreator<opmx::GELUOp>("opmx", "GELU", 1, 1);
    // H
    // I
    // J
    // K
    RegisterOptKernelCreator<opmx::KeyValueCacheOp>("opmx", "KeyValueCache", 1, 1);
    // L
    RegisterOptKernelCreator<opmx::LayerNormOp>("opmx", "LayerNorm", 1, 1);
    RegisterOptKernelCreator<opmx::LinearOp>("opmx", "Linear", 1, 1);

    // M
    RegisterOptKernelCreator<opmx::MoeColumnParallelLinearOp>("opmx", "MoeColumnParallelLinear", 1, 1);
    RegisterOptKernelCreator<opmx::MoeReduceOp>("opmx", "MoeReduce", 1, 1);
    RegisterOptKernelCreator<opmx::MoeRowParallelLinearOp>("opmx", "MoeRowParallelLinear", 1, 1);
    RegisterOptKernelCreator<opmx::MoeSelectOp>("opmx", "MoeSelect", 1, 1);
    // N
    RegisterOptKernelCreator<opmx::MultiHeadAttentionOp>("opmx", "MultiHeadAttention", 1, 1);

    // N
    // O
    // P
    RegisterOptKernelCreator<opmx::ParallelEmbeddingOp>("opmx", "ParallelEmbedding", 1, 1);
    RegisterOptKernelCreator<opmx::PixelUnshuffleOp>("opmx", "PixelUnshuffle", 1, 1);
    // Q
    // R
    RegisterOptKernelCreator<opmx::RMSNormOp>("opmx", "RMSNorm", 1, 1);
    RegisterOptKernelCreator<opmx::RotaryPositionEmbeddingOp>("opmx", "RotaryPositionEmbedding", 1, 1);
    RegisterOptKernelCreator<opmx::RowParallelLinearOp>("opmx", "RowParallelLinear", 1, 1);
    // S
    RegisterOptKernelCreator<opmx::SiLUOp>("opmx", "SiLU", 1, 1);
    RegisterOptKernelCreator<opmx::SwiGLUOp>("opmx", "SwiGLU", 1, 1);
    RegisterOptKernelCreator<opmx::SwishOp>("opmx", "Swish", 1, 1);
    // T
    RegisterOptKernelCreator<opmx::TensorParallelRMSNormOp>("opmx", "TensorParallelRMSNorm", 1, 1);
    // U
    // V
    RegisterOptKernelCreator<opmx::VisionEmbeddingOp>("opmx", "VisionEmbedding", 1, 1);
    // W
    // X
    // Y
    // Z

    /*                                                                        */
    /*                        OPMX.DYNAMIC_BATCHING                            */
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
    RegisterOptKernelCreator<opmx::DynamicBatchingKeyValueCacheOp>("opmx.dynamic_batching", "KeyValueCache", 1, 1);
    // L
    // M
    // N
    RegisterOptKernelCreator<opmx::DynamicBatchingMultiHeadAttentionOp>("opmx.dynamic_batching", "MultiHeadAttention", 1, 1);
    RegisterOptKernelCreator<opmx::DynamicBatchingMultiHeadCacheAttentionOp>("opmx.dynamic_batching", "MultiHeadCacheAttention", 1, 1);
    // O
    // P
    RegisterOptKernelCreator<opmx::DynamicBatchingPositionIndexOp>("opmx.dynamic_batching", "PositionIndex", 1, 1);
    // Q
    // R
    RegisterOptKernelCreator<opmx::DynamicBatchingRotaryPositionEmbeddingOp>("opmx.dynamic_batching", "RotaryPositionEmbedding", 1, 1);

    // S
    // T
    // U
    // V
    // W
    // X
    // Y
    // Z

    /*                                                                        */
    /*                        OPMX.I8I8                                        */
    /*                                                                        */
    // A
    // B
    // C
    RegisterOptKernelCreator<opmx::I8I8ColumnParallelLinearOp>("opmx.i8i8", "ColumnParallelLinear", 1, 1);
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
    RegisterOptKernelCreator<opmx::I8I8OnlineDequantizeOp>("opmx.i8i8", "OnlineDequantize", 1, 1);
    RegisterOptKernelCreator<opmx::I8I8OnlineQuantizeOp>("opmx.i8i8", "OnlineQuantize", 1, 1);
    RegisterOptKernelCreator<opmx::I8I8OnlineQuantizeRMSNormOp>("opmx.i8i8", "OnlineQuantizeRMSNorm", 1, 1);
    // P
    // Q
    // R
    RegisterOptKernelCreator<opmx::I8I8RowParallelLinearOp>("opmx.i8i8", "RowParallelLinear", 1, 1);
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
