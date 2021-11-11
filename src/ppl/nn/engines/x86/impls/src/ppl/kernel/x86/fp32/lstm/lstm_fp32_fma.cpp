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
#include <immintrin.h>

#include "ppl/kernel/x86/common/internal_include.h"
#include "ppl/kernel/x86/fp32/lstm.h"
#include "ppl/kernel/x86/fp32/gemm.h"
#include "ppl/kernel/x86/common/avx_tools.h"
#include "ppl/kernel/x86/common/math_fma.h"

namespace ppl { namespace kernel { namespace x86 {

static inline float sigmoidf(const float x) {
    return 1.0f / (1.0f + expf(-x));
}

uint64_t lstm_fp32_fma_get_buffer_bytes(
    const ppl::nn::TensorShape *X_shape,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    const bool has_Y,
    const bool has_Y_h,
    const bool has_Y_c)
{
    if (!has_Y && !has_Y_h && !has_Y_c)
        return 64u;

    const int64_t batch = X_shape->GetDim(1);
    const int64_t num_direction = direction == rnn_direction::BIDIRECTIONAL ? 2 : 1;

    const uint64_t gate_buff_size = batch * rnn_num_gate::LSTM * hidden_size;
    const uint64_t yh_size = has_Y_h ? num_direction * batch * hidden_size : 0;
    const uint64_t yc_size = has_Y_c ? num_direction * batch * hidden_size : 0;

    return (gate_buff_size + yh_size + yc_size) * sizeof(float);
}

ppl::common::RetCode lstm_fp32_fma(
    const ppl::nn::TensorShape *X_shape,
    const float *X,
    const float *X_weight,
    const float *R_weight,
    const float *P_weight,
    const float *bias,
    const int32_t *sequence_lens,
    const float *initial_h,
    const float *initial_c,
    const rnn_direction_t direction,
    const int64_t hidden_size,
    void *temp_buffer,
    float *Y,
    float *Y_h,
    float *Y_c)
{
    if (!Y && !Y_h && !Y_c) {
        return ppl::common::RC_SUCCESS;
    }

    const int64_t simd_w = 8;

    const int64_t num_direction = direction == rnn_direction::BIDIRECTIONAL ? 2 : 1;
    const int64_t seq_len = X_shape->GetDim(0);
    const int64_t batch = X_shape->GetDim(1);
    const int64_t input_size = X_shape->GetDim(2);

    float *Yh_buf = Y_h;
    float *Yc_buf = Y_c;

    // set temp buffer
    float *temp_buffer_fp32 = reinterpret_cast<float*>(temp_buffer);
    if (!Yh_buf) {
        Yh_buf = temp_buffer_fp32;
        temp_buffer_fp32 += num_direction * batch * hidden_size;
    }
    if (!Yc_buf) {
        Yc_buf = temp_buffer_fp32;
        temp_buffer_fp32 += num_direction * batch * hidden_size;
    }
    float *gate_buf = temp_buffer_fp32;

    for (int64_t nd = 0; nd < num_direction; ++nd) {
        const bool is_reverse = nd || (direction == rnn_direction::REVERSE);

        float *nd_Yh = Yh_buf + nd * batch * hidden_size;
        float *nd_Yc = Yc_buf + nd * batch * hidden_size;
        float *nd_Y = Y + nd * batch * hidden_size;

        const float *nd_W = X_weight + nd * rnn_num_gate::LSTM * hidden_size * input_size;
        const float *nd_R = R_weight + nd * rnn_num_gate::LSTM * hidden_size * hidden_size;
        const float *nd_P = P_weight ? P_weight + nd * (rnn_num_gate::LSTM - 1) * hidden_size : nullptr;
        const float *nd_Wb = bias ? bias + nd * 2 * rnn_num_gate::LSTM * hidden_size : nullptr;
        const float *nd_Rb = bias ? nd_Wb + rnn_num_gate::LSTM * hidden_size : nullptr;

        const float *nd_init_h = initial_h ? initial_h + nd * batch * hidden_size : nullptr;
        const float *nd_init_c = initial_c ? initial_c + nd * batch * hidden_size : nullptr;

        for (int64_t seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
            const int64_t mapped_seq_index = is_reverse ? (seq_len - seq_idx - 1) : seq_idx;
            const bool is_first_seq = seq_idx == 0;

            // X (seq_len, batch, input_size)
            // h_0 (num_direction, batch, hidden_size)
            // c_0 (num_direction, batch, hidden_size)
            // Y (seq_len, num_direction, batch, hidden_size)
            // h_n (num_direction, batch, hidden_size)
            // c_n (num_direction, batch, hidden_size)

            const float *sX = X + mapped_seq_index * batch * input_size;
            const float *Y_h_prev = is_first_seq ? nd_init_h : nd_Yh;
            const float *Y_c_prev = is_first_seq ? nd_init_c : nd_Yc;
            float *sY = nd_Y + mapped_seq_index * num_direction * batch * hidden_size;

            gemm_fp32_fma( // X[s]*W[nd]_{iofc}^T+Wb_{iofc}
                sX, nd_W, nd_Wb, nullptr,
                gemm_m_type::NOTRANS, gemm_m_type::TRANS,
                gemm_v_type::ROW_VEC, gemm_m_type::EMPTY,
                batch, rnn_num_gate::LSTM * hidden_size, input_size,
                input_size, input_size, rnn_num_gate::LSTM * hidden_size, 0,
                1.0f, 1.0f, gemm_post::NONE, gate_buf);

            const float alpha = !Y_h_prev ? 0.0f : 1.0f; // some hack, gemm will skip aAxB if alpha is 0
            gemm_fp32_fma( // h_0[nd]*R[nd]_{iofc}^T+Rb_{iofc}
                Y_h_prev, nd_R, nd_Rb, gate_buf,
                gemm_m_type::NOTRANS, gemm_m_type::TRANS,
                gemm_v_type::ROW_VEC, gemm_m_type::NOTRANS,
                batch, rnn_num_gate::LSTM * hidden_size, hidden_size,
                hidden_size, hidden_size, rnn_num_gate::LSTM * hidden_size, rnn_num_gate::LSTM * hidden_size,
                alpha, 1.0f, gemm_post::NONE, gate_buf);

            if (is_first_seq && !Y_h_prev) {
                Y_h_prev = nd_Yh; // preprocess Y_h_prev
                memset32_avx(nd_Yh, 0, batch * hidden_size);
            }
            if (is_first_seq && !Y_c_prev) {
                Y_c_prev = nd_Yc; // preprocess Y_c_prev
                memset32_avx(nd_Yc, 0, batch * hidden_size);
            }

            if (!P_weight) {
PRAGMA_OMP_PARALLEL_FOR()
                for (int64_t b = 0; b < batch; ++b) {
                    if (!sequence_lens || seq_idx < sequence_lens[b]) {
                        const float *gI = gate_buf + b * rnn_num_gate::LSTM * hidden_size;
                        const float *gO = gI + hidden_size;
                        const float *gF = gO + hidden_size;
                        const float *gC = gF + hidden_size;
                        const float *Cprev = Y_c_prev + b * hidden_size;
                        float *Ct = nd_Yc + b * hidden_size;
                        float *Ht = nd_Yh + b * hidden_size;
                        float *Yt = sY + b * hidden_size;
                        int64_t h = 0;
                        for (; h <= hidden_size - simd_w; h += simd_w) {
                            const __m256 it = _fma_sigmoid_ps(_mm256_loadu_ps(gI + h));
                            const __m256 ft = _fma_sigmoid_ps(_mm256_loadu_ps(gF + h));
                            const __m256 ct = _fma_tanh_ps(_mm256_loadu_ps(gC + h));
                            const __m256 cn = ft * _mm256_loadu_ps(Cprev + h) + it * ct;
                            const __m256 ot = _fma_sigmoid_ps(_mm256_loadu_ps(gO + h));
                            const __m256 hn = ot *_fma_tanh_ps(cn);
                            _mm256_storeu_ps(Ct + h, cn);
                            _mm256_storeu_ps(Ht + h, hn);
                        }
                        for (; h < hidden_size; ++h) {
                            const float it = sigmoidf(gI[h]);
                            const float ft = sigmoidf(gF[h]);
                            const float ct = ::tanhf(gC[h]);
                            Ct[h] = ft * Cprev[h] + it * ct;
                            const float ot = sigmoidf(gO[h]);
                            Ht[h] = ot * ::tanhf(Ct[h]);
                        }
                        if (Y) {
                            memcpy32_avx(Yt, Ht, hidden_size);
                        }
                    } else { // pass through the initial_h, initial_c
                        const float *Hprev = Y_h_prev + b * hidden_size;
                        const float *Cprev = Y_c_prev + b * hidden_size;
                        float *Ct = nd_Yc + b * hidden_size;
                        float *Ht = nd_Yh + b * hidden_size;
                        memcpy32_avx(Ct, Cprev, hidden_size);
                        memcpy32_avx(Ht, Hprev, hidden_size);
                    }
                }
            } else {
                const float *pI = nd_P;
                const float *pO = pI + hidden_size;
                const float *pF = pO + hidden_size;
PRAGMA_OMP_PARALLEL_FOR()
                for (int64_t b = 0; b < batch; ++b) {
                    if (!sequence_lens || seq_idx < sequence_lens[b]) {
                        const float *gI = gate_buf + b * rnn_num_gate::LSTM * hidden_size;
                        const float *gO = gI + hidden_size;
                        const float *gF = gO + hidden_size;
                        const float *gC = gF + hidden_size;
                        const float *Cprev = Y_c_prev + b * hidden_size;
                        float *Ct = nd_Yc + b * hidden_size;
                        float *Ht = nd_Yh + b * hidden_size;
                        float *Yt = sY + b * hidden_size;
                        int64_t h = 0;
                        for (; h <= hidden_size - simd_w; h += simd_w) {
                            const __m256 cp = _mm256_loadu_ps(Cprev + h);
                            const __m256 it = _fma_sigmoid_ps(_mm256_loadu_ps(gI + h) + cp * _mm256_loadu_ps(pI + h));
                            const __m256 ft = _fma_sigmoid_ps(_mm256_loadu_ps(gF + h) + cp * _mm256_loadu_ps(pF + h));
                            const __m256 ct = _fma_tanh_ps(_mm256_loadu_ps(gC + h));
                            const __m256 cn = ft * cp + it * ct;
                            const __m256 ot = _fma_sigmoid_ps(_mm256_loadu_ps(gO + h) + cn * _mm256_loadu_ps(pO + h));
                            const __m256 hn = ot *_fma_tanh_ps(cn);
                            _mm256_storeu_ps(Ct + h, cn);
                            _mm256_storeu_ps(Ht + h, hn);
                        }
                        for (; h < hidden_size; ++h) {
                            const float it = sigmoidf(gI[h] + pI[h] * Cprev[h]);
                            const float ft = sigmoidf(gF[h] + pF[h] * Cprev[h]);
                            const float ct = ::tanhf(gC[h]);
                            Ct[h] = ft * Cprev[h] + it * ct;
                            const float ot = sigmoidf(gO[h] + pO[h] * Ct[h]);
                            Ht[h] = ot * ::tanhf(Ct[h]);
                        }
                        if (Y) {
                            memcpy32_avx(Yt, Ht, hidden_size * sizeof(float));
                        }
                    } else { // pass through the initial_h, initial_c
                        const float *Hprev = Y_h_prev + b * hidden_size;
                        const float *Cprev = Y_c_prev + b * hidden_size;
                        float *Ct = nd_Yc + b * hidden_size;
                        float *Ht = nd_Yh + b * hidden_size;
                        memcpy32_avx(Ct, Cprev, hidden_size);
                        memcpy32_avx(Ht, Hprev, hidden_size);
                    }
                }
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}


}}}; // namespace ppl::kernel::x86
