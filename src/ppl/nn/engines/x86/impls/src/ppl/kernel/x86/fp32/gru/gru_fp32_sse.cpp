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
#include <nmmintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/gru.h"
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/kernel/x86/common/sse_tools.h"
#include "ppl/kernel/x86/common/math_sse.h"

namespace ppl { namespace kernel { namespace x86 {

static inline float sigmoidf(const float x)
{
    return 1.0f / (1.0f + expf(-x));
}

ppl::common::RetCode gru_fp32_sse(
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
    if (!Y && !Y_h) {
        return ppl::common::RC_SUCCESS;
    }

    const int64_t simd_w = 4;

    const int64_t num_direction = direction == rnn_direction::BIDIRECTIONAL ? 2 : 1;
    const bool    has_reverse   = direction == rnn_direction::BIDIRECTIONAL || direction == rnn_direction::REVERSE;
    const int64_t seq_len       = X_shape->GetDim(0);
    const int64_t batch         = X_shape->GetDim(1);
    const int64_t input_size    = X_shape->GetDim(2);

    float *Yh_buf = Y_h;
    float *rX = nullptr;

    // set temp buffer
    float *temp_buffer_fp32 = reinterpret_cast<float *>(temp_buffer);
    if (!Yh_buf) {
        Yh_buf = temp_buffer_fp32;
        temp_buffer_fp32 += num_direction * batch * hidden_size;
    }
    if (sequence_lens && has_reverse) {
        // need reverse X if seq_len of each batch is different
        rX = temp_buffer_fp32;
        temp_buffer_fp32 += seq_len * batch * input_size;
    }

    float *gate_buf   = temp_buffer_fp32;
    float *extra_gate = gate_buf + seq_len * batch * rnn_num_gate::GRU * hidden_size;

    for (int64_t nd = 0; nd < num_direction; ++nd) {
        const bool is_reverse = nd || (direction == rnn_direction::REVERSE);

        float *nd_Yh = Yh_buf + nd * batch * hidden_size;
        float *nd_Y  = Y + nd * batch * hidden_size;

        const float *nd_W   = W[nd];
        const float *nd_Rzr = Rzr[nd];
        const float *nd_Rh  = Rh[nd];

        const float *nd_Wb   = bias ? bias + nd * 2 * rnn_num_gate::GRU * hidden_size : nullptr;
        const float *nd_Rbzr = bias ? nd_Wb + rnn_num_gate::GRU * hidden_size : nullptr;
        const float *nd_Rbh  = bias ? nd_Rbzr + (rnn_num_gate::GRU - 1) * hidden_size : nullptr;

        const float *nd_init_h = initial_h ? initial_h + nd * batch * hidden_size : nullptr;

        const float *nd_X = X;
        if (sequence_lens && is_reverse) {
            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t work = 0; work < seq_len * batch; ++work) {
                const int64_t seq_idx = work / batch;
                const int64_t b       = work % batch;
                const int64_t seq_end = sequence_lens[b];
                auto src = X + ((seq_idx < seq_end) ? (seq_end - seq_idx - 1) : seq_idx) * batch * input_size + b * input_size;
                auto dst = rX + seq_idx * batch * input_size + b * input_size;
                memcpy32_sse(dst, src, input_size);
            }
            nd_X = rX;
        }

        gemm_fp32_sse( // Xt*(Wz^T) + Wbz ; Xt*(Wr^T) + Wbr ; Xt*(Wh^T) + Wbh
            nd_X, nd_W, nd_Wb, nullptr,
            gemm_m_type::NOTRANS, packed_W ? gemm_m_type::PACKED : gemm_m_type::TRANS,
            nd_Wb ? gemm_v_type::ROW_VEC : gemm_v_type::EMPTY, gemm_m_type::EMPTY,
            seq_len * batch, rnn_num_gate::GRU * hidden_size, input_size,
            input_size, input_size, rnn_num_gate::GRU * hidden_size, 0,
            1.0f, 0.0f, 1.0f, 0.0f, gemm_post::NONE, gate_buf);

        for (int64_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            auto seq_gate = gate_buf + ((!sequence_lens && is_reverse) ? (seq_len - seq_idx - 1) : seq_idx) * batch * rnn_num_gate::GRU * hidden_size;
            auto Y_h_prev = seq_idx == 0 ? nd_init_h : nd_Yh;

            gemm_fp32_sse( // Ht-1*(Rz^T) + Rbz; Ht-1*(Rr^T) + Rbr
                Y_h_prev, nd_Rzr, nd_Rbzr, nullptr,
                gemm_m_type::NOTRANS, packed_Rzr ? gemm_m_type::PACKED : gemm_m_type::TRANS,
                nd_Rbzr ? gemm_v_type::ROW_VEC : gemm_v_type::EMPTY, gemm_m_type::EMPTY,
                batch, (rnn_num_gate::GRU - 1) * hidden_size, hidden_size,
                hidden_size, hidden_size, rnn_num_gate::GRU * hidden_size, 0,
                !Y_h_prev ? 0.0f : 1.0f, 1.0f, 1.0f, 0.0f, gemm_post::NONE, seq_gate);

            gemm_fp32_sse( //  (Ht-1*(Rh^T) + Rbh)
                Y_h_prev, nd_Rh, nd_Rbh, nullptr,
                gemm_m_type::NOTRANS, packed_Rh ? gemm_m_type::PACKED : gemm_m_type::TRANS,
                nd_Rbh ? gemm_v_type::ROW_VEC : gemm_v_type::EMPTY, gemm_m_type::EMPTY,
                batch, hidden_size, hidden_size,
                hidden_size, hidden_size, hidden_size, 0,
                !Y_h_prev ? 0.0f : 1.0f, 0.0f, 1.0f, 0.0f, gemm_post::NONE, extra_gate);

            bool reset_Ht = false;
            if (seq_idx == 0 && !Y_h_prev) {
                Y_h_prev = nd_Yh; // preprocess Y_h_prev
                reset_Ht = true;
            }

            PRAGMA_OMP_PARALLEL_FOR()
            for (int64_t b = 0; b < batch; ++b) {
                const float *Hprev = Y_h_prev + b * hidden_size;
                float *Ht          = nd_Yh + b * hidden_size;

                if (reset_Ht) memset32_sse(Ht, 0, hidden_size);

                const int64_t seq_end = sequence_lens ? sequence_lens[b] : seq_len;
                if (seq_idx < seq_end) {
                    const float *gZ = seq_gate + b * rnn_num_gate::GRU * hidden_size;
                    const float *gR = gZ + hidden_size;
                    const float *gH = gR + hidden_size;
                    const float *gE = extra_gate + b * hidden_size;
                    int64_t h       = 0;
                    for (; h <= hidden_size - simd_w; h += simd_w) {
                        auto zt = _sse_sigmoid_ps(_mm_loadu_ps(gZ + h));
                        auto rt = _sse_sigmoid_ps(_mm_loadu_ps(gR + h));
                        auto ht = _sse_tanh_ps(rt * _mm_loadu_ps(gE + h) + _mm_loadu_ps(gH + h));
                        _mm_storeu_ps(Ht + h, ht - zt * ht + zt * _mm_loadu_ps(Hprev + h));
                    }
                    for (; h < hidden_size; ++h) {
                        const float zt = sigmoidf(gZ[h]);
                        const float rt = sigmoidf(gR[h]);
                        const float ht = ::tanhf(rt * gE[h] + gH[h]);
                        Ht[h]          = ht - zt * ht + zt * Hprev[h];
                    }
                    if (Y) {
                        float *Yt = nd_Y + (is_reverse ? (seq_end - seq_idx - 1) : seq_idx) * num_direction * batch * hidden_size + b * hidden_size;
                        memcpy32_sse(Yt, Ht, hidden_size);
                    }
                } else { // pass through the initial_h, initial_c
                    if (Hprev != Ht) memcpy32_sse(Ht, Hprev, hidden_size);
                    if (Y) {
                        float *Yt = nd_Y + seq_idx * num_direction * batch * hidden_size + b * hidden_size;
                        memset32_sse(Yt, 0, hidden_size);
                    }
                }
            }
        }
    }
    return ppl::common::RC_SUCCESS;
}

}}}; // namespace ppl::kernel::x86
