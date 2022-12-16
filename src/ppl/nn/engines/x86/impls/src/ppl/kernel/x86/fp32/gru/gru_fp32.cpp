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

#include <math.h>
#include <string.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gru.h"
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/kernel/x86/common/memory.h"

namespace ppl { namespace kernel { namespace x86 {

static inline float sigmoidf(const float x)
{
    return 1.0f / (1.0f + expf(-x));
}

uint64_t gru_fp32_get_buffer_bytes(
    const ppl::common::TensorShape *X_shape,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    const bool has_sequence_lens,
    const bool has_Y,
    const bool has_Y_h)
{
    if (!has_Y && !has_Y_h)
        return 64u;

    const int64_t seq_len       = X_shape->GetDim(0);
    const int64_t batch         = X_shape->GetDim(1);
    const int64_t input_size    = X_shape->GetDim(2);
    const int64_t num_direction = direction == rnn_direction::BIDIRECTIONAL ? 2 : 1;
    const bool    has_reverse   = direction == rnn_direction::BIDIRECTIONAL || direction == rnn_direction::REVERSE;

    const uint64_t gate_buff_size  = seq_len * batch * rnn_num_gate::GRU * hidden_size;
    const uint64_t extra_gate_size = batch * hidden_size; // (rt (.) Ht-1)*(Rh^T)  (rt (.) (Ht-1*(Rh^T) + Rbh))
    const uint64_t yh_size         = has_Y_h ? num_direction * batch * hidden_size : 0;
    const uint64_t rev_seq_size    = has_sequence_lens && has_reverse ? seq_len * batch * input_size : 0;

    return (gate_buff_size + extra_gate_size + yh_size + rev_seq_size) * sizeof(float);
}

ppl::common::RetCode gru_fp32(
    const ppl::common::isa_t isa,
    const ppl::common::TensorShape *X_shape,
    const float *X,
    const float **W,
    const float **Rzr,
    const float **Rh,
    const float *bias,
    const int32_t *sequence_lens,
    const float *initial_h,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    bool packed_W,
    bool packed_Rzr,
    bool packed_Rh,
    void *temp_buffer,
    float *Y,
    float *Y_h)
{
#ifdef PPL_USE_X86_AVX512
    if (isa & ppl::common::ISA_X86_AVX512) {
        return kernel::x86::gru_fp32_avx512(X_shape, X, W, Rzr, Rh, bias, sequence_lens, initial_h, direction, hidden_size, packed_W, packed_Rzr, packed_Rh, temp_buffer, Y, Y_h);
    }
#endif
    if (isa & ppl::common::ISA_X86_FMA) {
        return kernel::x86::gru_fp32_fma(X_shape, X, W, Rzr, Rh, bias, sequence_lens, initial_h, direction, hidden_size, packed_W, packed_Rzr, packed_Rh, temp_buffer, Y, Y_h);
    }
    return kernel::x86::gru_fp32_sse(X_shape, X, W, Rzr, Rh, bias, sequence_lens, initial_h, direction, hidden_size, packed_W, packed_Rzr, packed_Rh, temp_buffer, Y, Y_h);
}

}}}; // namespace ppl::kernel::x86
