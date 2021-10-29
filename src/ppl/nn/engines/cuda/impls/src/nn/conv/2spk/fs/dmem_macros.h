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

////////////////////////////////////////
// load dB macros
////////////////////////////////////////

#define LOAD_dBv4_SIZE_16TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid)         \
    {                                                                                     \
        if (tid < (CTA_SIZE_IN_THD / 16))                                                 \
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[_dBv4_off[0]] : ZEROv4; \
                                                                                          \
        _dBv4_off[0] += TILE_K_V8_PER_CTA;                                                \
    }

#define LOAD_dBv4_SIZE_8TH(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid)          \
    {                                                                                     \
        if (tid < (CTA_SIZE_IN_THD / 8))                                                  \
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[_dBv4_off[0]] : ZEROv4; \
                                                                                          \
        _dBv4_off[0] += TILE_K_V8_PER_CTA;                                                \
    }

#define LOAD_dBv4_SIZE_QTR(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid)          \
    {                                                                                     \
        if (tid < (CTA_SIZE_IN_THD / 4))                                                  \
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[_dBv4_off[0]] : ZEROv4; \
                                                                                          \
        _dBv4_off[0] += TILE_K_V8_PER_CTA;                                                \
    }

#define LOAD_dBv4_SIZE_HALF(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid)         \
    {                                                                                     \
        if (tid < (CTA_SIZE_IN_THD / 2))                                                  \
            _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[_dBv4_off[0]] : ZEROv4; \
                                                                                          \
        _dBv4_off[0] += TILE_K_V8_PER_CTA;                                                \
    }

#define LOAD_dBv4_SIZE1(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid)         \
    {                                                                                 \
        _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[_dBv4_off[0]] : ZEROv4; \
                                                                                      \
        _dBv4_off[0] += TILE_K_V8_PER_CTA;                                            \
    }

#define LOAD_dBv4_SIZE2(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid)         \
    {                                                                                 \
        _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[_dBv4_off[0]] : ZEROv4; \
        _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[_dBv4_off[1]] : ZEROv4; \
                                                                                      \
        _dBv4_off[0] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[1] += TILE_K_V8_PER_CTA;                                            \
    }

#define LOAD_dBv4_SIZE4(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid)         \
    {                                                                                 \
        _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[_dBv4_off[0]] : ZEROv4; \
        _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[_dBv4_off[1]] : ZEROv4; \
        _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[_dBv4_off[2]] : ZEROv4; \
        _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[_dBv4_off[3]] : ZEROv4; \
                                                                                      \
        _dBv4_off[0] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[1] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[2] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[3] += TILE_K_V8_PER_CTA;                                            \
    }

#define LOAD_dBv4_SIZE8(_regB, _dB, _dBv4_off, _flt_c_v8_valid, _flt_n_valid)         \
    {                                                                                 \
        _regB[0] = (_flt_n_valid[0] && _flt_c_v8_valid) ? _dB[_dBv4_off[0]] : ZEROv4; \
        _regB[1] = (_flt_n_valid[1] && _flt_c_v8_valid) ? _dB[_dBv4_off[1]] : ZEROv4; \
        _regB[2] = (_flt_n_valid[2] && _flt_c_v8_valid) ? _dB[_dBv4_off[2]] : ZEROv4; \
        _regB[3] = (_flt_n_valid[3] && _flt_c_v8_valid) ? _dB[_dBv4_off[3]] : ZEROv4; \
        _regB[4] = (_flt_n_valid[4] && _flt_c_v8_valid) ? _dB[_dBv4_off[4]] : ZEROv4; \
        _regB[5] = (_flt_n_valid[5] && _flt_c_v8_valid) ? _dB[_dBv4_off[5]] : ZEROv4; \
        _regB[6] = (_flt_n_valid[6] && _flt_c_v8_valid) ? _dB[_dBv4_off[6]] : ZEROv4; \
        _regB[7] = (_flt_n_valid[7] && _flt_c_v8_valid) ? _dB[_dBv4_off[7]] : ZEROv4; \
                                                                                      \
        _dBv4_off[0] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[1] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[2] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[3] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[4] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[5] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[6] += TILE_K_V8_PER_CTA;                                            \
        _dBv4_off[7] += TILE_K_V8_PER_CTA;                                            \
    }

#define SET_dBv4_BOUND(_step_id, _dBv4_off, _flt_n_valid)                            \
    {                                                                                \
        int _flt_n_id = cta_idx * TILE_N_PER_CTA +                                   \
                        _step_id * (TILE_N_PER_CTA / READ_dBv4_STEPS) +              \
                        ldg_idy;                                                     \
                                                                                     \
        _flt_n_valid = _flt_n_id < num_flt_per_grp_pad;                              \
                                                                                     \
        _dBv4_off = grp_id * num_chl_per_grp_pad_v8 * flt_hw * num_flt_per_grp_pad + \
                    _flt_n_id * num_chl_per_grp_pad_v8 * flt_hw +                    \
                    spf_id * num_chl_per_grp_pad_v8 +                                \
                    flt_c_v8_id;                                                     \
    }

////////////////////////////////////////
// load dA macros
////////////////////////////////////////

#define SET_dAv4_BOUND(_step_id, _dAv4_off, _in_hw_valid)                                                    \
    {                                                                                                        \
        int _out_nhw_id = cta_idy * TILE_M_PER_CTA +                                                         \
                          _step_id * (TILE_M_PER_CTA / READ_dAv4_STEPS) +                                    \
                          ldg_idy;                                                                           \
                                                                                                             \
        int _out_w_id = (_out_nhw_id % out_width);                                                           \
        int _out_h_id = (_out_nhw_id / out_width) % out_height;                                              \
                                                                                                             \
        int _in_n_id = _out_nhw_id / out_hw;                                                                 \
        int _in_h_id = _out_h_id * stride_height;                                                            \
        int _in_w_id = _out_w_id * stride_width;                                                             \
                                                                                                             \
        int _flt_h_id = spf_id / flt_width;                                                                  \
        int _flt_w_id = spf_id % flt_width;                                                                  \
                                                                                                             \
        _in_h_id = _in_h_id + _flt_h_id * hole_height - pad_height;                                          \
        _in_w_id = _in_w_id + _flt_w_id * hole_width - pad_width;                                            \
                                                                                                             \
        _dAv4_off = (_in_n_id * in_hw + _in_h_id * in_width + _in_w_id) * num_chl_per_grp_pad_v8 * num_grp + \
                    grp_id * num_chl_per_grp_pad_v8 +                                                        \
                    flt_c_v8_id;                                                                             \
                                                                                                             \
        SET_BOUND_FLT1(_in_hw_valid, _in_n_id, _in_h_id, _in_w_id);                                          \
    }

#define LOAD_dAv4_SIZE_16TH(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid)         \
    {                                                                                    \
        if (tid < (CTA_SIZE_IN_THD / 16))                                                \
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[_dAv4_off[0]] : ZEROv4; \
                                                                                         \
        _dAv4_off[0] += TILE_K_V8_PER_CTA;                                               \
    }

#define LOAD_dAv4_SIZE_8TH(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid)          \
    {                                                                                    \
        if (tid < (CTA_SIZE_IN_THD / 8))                                                 \
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[_dAv4_off[0]] : ZEROv4; \
                                                                                         \
        _dAv4_off[0] += TILE_K_V8_PER_CTA;                                               \
    }

#define LOAD_dAv4_SIZE_QTR(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid)          \
    {                                                                                    \
        if (tid < (CTA_SIZE_IN_THD / 4))                                                 \
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[_dAv4_off[0]] : ZEROv4; \
                                                                                         \
        _dAv4_off[0] += TILE_K_V8_PER_CTA;                                               \
    }

#define LOAD_dAv4_SIZE_HALF(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid)         \
    {                                                                                    \
        if (tid < (CTA_SIZE_IN_THD / 2))                                                 \
            _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[_dAv4_off[0]] : ZEROv4; \
                                                                                         \
        _dAv4_off[0] += TILE_K_V8_PER_CTA;                                               \
    }

#define LOAD_dAv4_SIZE1(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid)         \
    {                                                                                \
        _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[_dAv4_off[0]] : ZEROv4; \
                                                                                     \
        _dAv4_off[0] += TILE_K_V8_PER_CTA;                                           \
    }

#define LOAD_dAv4_SIZE2(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid)         \
    {                                                                                \
        _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[_dAv4_off[0]] : ZEROv4; \
        _regA[1] = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[_dAv4_off[1]] : ZEROv4; \
                                                                                     \
        _dAv4_off[0] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[1] += TILE_K_V8_PER_CTA;                                           \
    }

#define LOAD_dAv4_SIZE4(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid)         \
    {                                                                                \
        _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[_dAv4_off[0]] : ZEROv4; \
        _regA[1] = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[_dAv4_off[1]] : ZEROv4; \
        _regA[2] = (_in_hw_valid[2] && _in_c_v8_valid) ? _dA[_dAv4_off[2]] : ZEROv4; \
        _regA[3] = (_in_hw_valid[3] && _in_c_v8_valid) ? _dA[_dAv4_off[3]] : ZEROv4; \
                                                                                     \
        _dAv4_off[0] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[1] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[2] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[3] += TILE_K_V8_PER_CTA;                                           \
    }

#define LOAD_dAv4_SIZE8(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid)         \
    {                                                                                \
        _regA[0] = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[_dAv4_off[0]] : ZEROv4; \
        _regA[1] = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[_dAv4_off[1]] : ZEROv4; \
        _regA[2] = (_in_hw_valid[2] && _in_c_v8_valid) ? _dA[_dAv4_off[2]] : ZEROv4; \
        _regA[3] = (_in_hw_valid[3] && _in_c_v8_valid) ? _dA[_dAv4_off[3]] : ZEROv4; \
        _regA[4] = (_in_hw_valid[4] && _in_c_v8_valid) ? _dA[_dAv4_off[4]] : ZEROv4; \
        _regA[5] = (_in_hw_valid[5] && _in_c_v8_valid) ? _dA[_dAv4_off[5]] : ZEROv4; \
        _regA[6] = (_in_hw_valid[6] && _in_c_v8_valid) ? _dA[_dAv4_off[6]] : ZEROv4; \
        _regA[7] = (_in_hw_valid[7] && _in_c_v8_valid) ? _dA[_dAv4_off[7]] : ZEROv4; \
                                                                                     \
        _dAv4_off[0] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[1] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[2] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[3] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[4] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[5] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[6] += TILE_K_V8_PER_CTA;                                           \
        _dAv4_off[7] += TILE_K_V8_PER_CTA;                                           \
    }

#define LOAD_dAv4_SIZE16(_regA, _dA, _dAv4_off, _in_c_v8_valid, _in_hw_valid)           \
    {                                                                                   \
        _regA[0]  = (_in_hw_valid[0] && _in_c_v8_valid) ? _dA[_dAv4_off[0]] : ZEROv4;   \
        _regA[1]  = (_in_hw_valid[1] && _in_c_v8_valid) ? _dA[_dAv4_off[1]] : ZEROv4;   \
        _regA[2]  = (_in_hw_valid[2] && _in_c_v8_valid) ? _dA[_dAv4_off[2]] : ZEROv4;   \
        _regA[3]  = (_in_hw_valid[3] && _in_c_v8_valid) ? _dA[_dAv4_off[3]] : ZEROv4;   \
        _regA[4]  = (_in_hw_valid[4] && _in_c_v8_valid) ? _dA[_dAv4_off[4]] : ZEROv4;   \
        _regA[5]  = (_in_hw_valid[5] && _in_c_v8_valid) ? _dA[_dAv4_off[5]] : ZEROv4;   \
        _regA[6]  = (_in_hw_valid[6] && _in_c_v8_valid) ? _dA[_dAv4_off[6]] : ZEROv4;   \
        _regA[7]  = (_in_hw_valid[7] && _in_c_v8_valid) ? _dA[_dAv4_off[7]] : ZEROv4;   \
        _regA[8]  = (_in_hw_valid[8] && _in_c_v8_valid) ? _dA[_dAv4_off[8]] : ZEROv4;   \
        _regA[9]  = (_in_hw_valid[9] && _in_c_v8_valid) ? _dA[_dAv4_off[9]] : ZEROv4;   \
        _regA[10] = (_in_hw_valid[10] && _in_c_v8_valid) ? _dA[_dAv4_off[10]] : ZEROv4; \
        _regA[11] = (_in_hw_valid[11] && _in_c_v8_valid) ? _dA[_dAv4_off[11]] : ZEROv4; \
        _regA[12] = (_in_hw_valid[12] && _in_c_v8_valid) ? _dA[_dAv4_off[12]] : ZEROv4; \
        _regA[13] = (_in_hw_valid[13] && _in_c_v8_valid) ? _dA[_dAv4_off[13]] : ZEROv4; \
        _regA[14] = (_in_hw_valid[14] && _in_c_v8_valid) ? _dA[_dAv4_off[14]] : ZEROv4; \
        _regA[15] = (_in_hw_valid[15] && _in_c_v8_valid) ? _dA[_dAv4_off[15]] : ZEROv4; \
                                                                                        \
        _dAv4_off[0] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[1] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[2] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[3] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[4] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[5] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[6] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[7] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[8] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[9] += TILE_K_V8_PER_CTA;                                              \
        _dAv4_off[10] += TILE_K_V8_PER_CTA;                                             \
        _dAv4_off[11] += TILE_K_V8_PER_CTA;                                             \
        _dAv4_off[12] += TILE_K_V8_PER_CTA;                                             \
        _dAv4_off[13] += TILE_K_V8_PER_CTA;                                             \
        _dAv4_off[14] += TILE_K_V8_PER_CTA;                                             \
        _dAv4_off[15] += TILE_K_V8_PER_CTA;                                             \
    }
