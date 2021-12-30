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

#define LOAD_dBv4_SIZE_16TH(_regB, _dB, _dBv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dBv4_off[0] += flt_lut.idx[lut_id]; \
            \
            if(tid < (CTA_SIZE_IN_THD / 16))  \
                _regB[0] = (_flt_n_valid[0] && _flt_c_v16_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE_8TH(_regB, _dB, _dBv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dBv4_off[0] += flt_lut.idx[lut_id]; \
            \
            if(tid < (CTA_SIZE_IN_THD / 8))  \
                _regB[0] = (_flt_n_valid[0] && _flt_c_v16_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE_QTR(_regB, _dB, _dBv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dBv4_off[0] += flt_lut.idx[lut_id]; \
            \
            if(tid < (CTA_SIZE_IN_THD / 4))  \
                _regB[0] = (_flt_n_valid[0] && _flt_c_v16_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE_HALF(_regB, _dB, _dBv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dBv4_off[0] += flt_lut.idx[lut_id]; \
            \
            if(tid < (CTA_SIZE_IN_THD / 2))  \
                _regB[0] = (_flt_n_valid[0] && _flt_c_v16_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE1(_regB, _dB, _dBv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dBv4_off[0] += flt_lut.idx[lut_id]; \
            \
            _regB[0] = (_flt_n_valid[0] && _flt_c_v16_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE2(_regB, _dB, _dBv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dBv4_off[0] += flt_lut.idx[lut_id]; \
            _dBv4_off[1] += flt_lut.idx[lut_id]; \
            \
            _regB[0] = (_flt_n_valid[0] && _flt_c_v16_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v16_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE4(_regB, _dB, _dBv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dBv4_off[0] += flt_lut.idx[lut_id]; \
            _dBv4_off[1] += flt_lut.idx[lut_id]; \
            _dBv4_off[2] += flt_lut.idx[lut_id]; \
            _dBv4_off[3] += flt_lut.idx[lut_id]; \
            \
            _regB[0] = (_flt_n_valid[0] && _flt_c_v16_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v16_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v16_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v16_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\
        }

#define LOAD_dBv4_SIZE8(_regB, _dB, _dBv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dBv4_off[0] += flt_lut.idx[lut_id]; \
            _dBv4_off[1] += flt_lut.idx[lut_id]; \
            _dBv4_off[2] += flt_lut.idx[lut_id]; \
            _dBv4_off[3] += flt_lut.idx[lut_id]; \
            _dBv4_off[4] += flt_lut.idx[lut_id]; \
            _dBv4_off[5] += flt_lut.idx[lut_id]; \
            _dBv4_off[6] += flt_lut.idx[lut_id]; \
            _dBv4_off[7] += flt_lut.idx[lut_id]; \
            \
            _regB[0] = (_flt_n_valid[0] && _flt_c_v16_valid) ? _dB[ _dBv4_off[0] ] : ZEROv4;\
            _regB[1] = (_flt_n_valid[1] && _flt_c_v16_valid) ? _dB[ _dBv4_off[1] ] : ZEROv4;\
            _regB[2] = (_flt_n_valid[2] && _flt_c_v16_valid) ? _dB[ _dBv4_off[2] ] : ZEROv4;\
            _regB[3] = (_flt_n_valid[3] && _flt_c_v16_valid) ? _dB[ _dBv4_off[3] ] : ZEROv4;\
            _regB[4] = (_flt_n_valid[4] && _flt_c_v16_valid) ? _dB[ _dBv4_off[4] ] : ZEROv4;\
            _regB[5] = (_flt_n_valid[5] && _flt_c_v16_valid) ? _dB[ _dBv4_off[5] ] : ZEROv4;\
            _regB[6] = (_flt_n_valid[6] && _flt_c_v16_valid) ? _dB[ _dBv4_off[6] ] : ZEROv4;\
            _regB[7] = (_flt_n_valid[7] && _flt_c_v16_valid) ? _dB[ _dBv4_off[7] ] : ZEROv4;\
        }

#define SET_dBv4_BOUND(_step_id, _dBv4_off, _flt_n_valid) \
        { \
            int _flt_n_id  =  cta_idx *  TILE_N_PER_CTA + \
                             _step_id * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \
                             ldg_idy; \
            \
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp; \
            \
            _dBv4_off  =   grp_id   * flt_hw * num_chl_per_grp_pad_v16 * num_flt_per_grp + \
                          _flt_n_id * flt_hw * num_chl_per_grp_pad_v16 + \
                           flt_c_v16_id; \
        }

////////////////////////////////////////
// load dA macros
////////////////////////////////////////

#define SET_dAv4_BOUND(_step_id, _dAv4_off, _in_n_id, _in_h_start, _in_w_start) \
        { \
            int _out_nhw_id    = cta_idy  *  TILE_M_PER_CTA + \
                                 _step_id * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \
                                 ldg_idy; \
            \
            int _out_w_id =  (_out_nhw_id % out_width); \
            int _out_h_id =  (_out_nhw_id / out_width) % out_height; \
            int _in_h_id  =     _out_h_id * stride_height; \
            int _in_w_id  =     _out_w_id * stride_width; \
            \
            _in_n_id      =  _out_nhw_id / out_hw; \
            _in_h_start   =  _in_h_id - pad_height; \
            _in_w_start   =  _in_w_id - pad_width;  \
            \
            _dAv4_off  =  (_in_n_id  * in_hw + _in_h_id  * in_width + _in_w_id) * num_chl_per_grp_pad_v16 * num_grp + \
                           grp_id   * num_chl_per_grp_pad_v16 + \
                           flt_c_v16_id; \
        }

#define LOAD_dAv4_SIZE_16TH(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _dAv4_off[0] += in_lut.idx[lut_id]; \
            \
            if(tid < (CTA_SIZE_IN_THD / 16))  \
                _regA[0] = (flt_c_v16_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE_8TH(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _dAv4_off[0] += in_lut.idx[lut_id]; \
            \
            if(tid < (CTA_SIZE_IN_THD / 8))  \
                _regA[0] = (flt_c_v16_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE_QTR(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _dAv4_off[0] += in_lut.idx[lut_id]; \
            \
            if(tid < (CTA_SIZE_IN_THD / 4))  \
                _regA[0] = (flt_c_v16_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE_HALF(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _dAv4_off[0] += in_lut.idx[lut_id]; \
            \
            if(tid < (CTA_SIZE_IN_THD / 2))  \
                _regA[0] = (flt_c_v16_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE1(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _dAv4_off[0] += in_lut.idx[lut_id]; \
            \
            _regA[0] = (flt_c_v16_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE2(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _dAv4_off[0] += in_lut.idx[lut_id]; \
            _dAv4_off[1] += in_lut.idx[lut_id]; \
            \
            _regA[0] = (flt_c_v16_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = (flt_c_v16_valid && HeightInRange(_in_h_id[1]) && WidthInRange(_in_w_id[1]) && (_in_n_id[1] < in_num)) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE4(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _dAv4_off[0] += in_lut.idx[lut_id]; \
            _dAv4_off[1] += in_lut.idx[lut_id]; \
            _dAv4_off[2] += in_lut.idx[lut_id]; \
            _dAv4_off[3] += in_lut.idx[lut_id]; \
            \
            _regA[0] = (flt_c_v16_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = (flt_c_v16_valid && HeightInRange(_in_h_id[1]) && WidthInRange(_in_w_id[1]) && (_in_n_id[1] < in_num)) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
            _regA[2] = (flt_c_v16_valid && HeightInRange(_in_h_id[2]) && WidthInRange(_in_w_id[2]) && (_in_n_id[2] < in_num)) ? _dA[ _dAv4_off[2] ] : ZEROv4;\
            _regA[3] = (flt_c_v16_valid && HeightInRange(_in_h_id[3]) && WidthInRange(_in_w_id[3]) && (_in_n_id[3] < in_num)) ? _dA[ _dAv4_off[3] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE8(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _dAv4_off[0] += in_lut.idx[lut_id]; \
            _dAv4_off[1] += in_lut.idx[lut_id]; \
            _dAv4_off[2] += in_lut.idx[lut_id]; \
            _dAv4_off[3] += in_lut.idx[lut_id]; \
            _dAv4_off[4] += in_lut.idx[lut_id]; \
            _dAv4_off[5] += in_lut.idx[lut_id]; \
            _dAv4_off[6] += in_lut.idx[lut_id]; \
            _dAv4_off[7] += in_lut.idx[lut_id]; \
            \
            _regA[0] = (flt_c_v16_valid && HeightInRange(_in_h_id[0]) && WidthInRange(_in_w_id[0]) && (_in_n_id[0] < in_num)) ? _dA[ _dAv4_off[0] ] : ZEROv4;\
            _regA[1] = (flt_c_v16_valid && HeightInRange(_in_h_id[1]) && WidthInRange(_in_w_id[1]) && (_in_n_id[1] < in_num)) ? _dA[ _dAv4_off[1] ] : ZEROv4;\
            _regA[2] = (flt_c_v16_valid && HeightInRange(_in_h_id[2]) && WidthInRange(_in_w_id[2]) && (_in_n_id[2] < in_num)) ? _dA[ _dAv4_off[2] ] : ZEROv4;\
            _regA[3] = (flt_c_v16_valid && HeightInRange(_in_h_id[3]) && WidthInRange(_in_w_id[3]) && (_in_n_id[3] < in_num)) ? _dA[ _dAv4_off[3] ] : ZEROv4;\
            _regA[4] = (flt_c_v16_valid && HeightInRange(_in_h_id[4]) && WidthInRange(_in_w_id[4]) && (_in_n_id[4] < in_num)) ? _dA[ _dAv4_off[4] ] : ZEROv4;\
            _regA[5] = (flt_c_v16_valid && HeightInRange(_in_h_id[5]) && WidthInRange(_in_w_id[5]) && (_in_n_id[5] < in_num)) ? _dA[ _dAv4_off[5] ] : ZEROv4;\
            _regA[6] = (flt_c_v16_valid && HeightInRange(_in_h_id[6]) && WidthInRange(_in_w_id[6]) && (_in_n_id[6] < in_num)) ? _dA[ _dAv4_off[6] ] : ZEROv4;\
            _regA[7] = (flt_c_v16_valid && HeightInRange(_in_h_id[7]) && WidthInRange(_in_w_id[7]) && (_in_n_id[7] < in_num)) ? _dA[ _dAv4_off[7] ] : ZEROv4;\
        }

#define LOAD_dAv4_SIZE16(_regA, _dA, _dAv4_off, _in_n_id, _in_h_id, _in_w_id) \
        { \
            _dAv4_off[0]  += in_lut.idx[lut_id]; \
            _dAv4_off[1]  += in_lut.idx[lut_id]; \
            _dAv4_off[2]  += in_lut.idx[lut_id]; \
            _dAv4_off[3]  += in_lut.idx[lut_id]; \
            _dAv4_off[4]  += in_lut.idx[lut_id]; \
            _dAv4_off[5]  += in_lut.idx[lut_id]; \
            _dAv4_off[6]  += in_lut.idx[lut_id]; \
            _dAv4_off[7]  += in_lut.idx[lut_id]; \
            _dAv4_off[8]  += in_lut.idx[lut_id]; \
            _dAv4_off[9]  += in_lut.idx[lut_id]; \
            _dAv4_off[10] += in_lut.idx[lut_id]; \
            _dAv4_off[11] += in_lut.idx[lut_id]; \
            _dAv4_off[12] += in_lut.idx[lut_id]; \
            _dAv4_off[13] += in_lut.idx[lut_id]; \
            _dAv4_off[14] += in_lut.idx[lut_id]; \
            _dAv4_off[15] += in_lut.idx[lut_id]; \
            \
            _regA[0]  = (flt_c_v16_valid && HeightInRange(_in_h_id[0])  && WidthInRange(_in_w_id[0])  && (_in_n_id[0]  < in_num)) ? _dA[ _dAv4_off[0] ]  : ZEROv4;\
            _regA[1]  = (flt_c_v16_valid && HeightInRange(_in_h_id[1])  && WidthInRange(_in_w_id[1])  && (_in_n_id[1]  < in_num)) ? _dA[ _dAv4_off[1] ]  : ZEROv4;\
            _regA[2]  = (flt_c_v16_valid && HeightInRange(_in_h_id[2])  && WidthInRange(_in_w_id[2])  && (_in_n_id[2]  < in_num)) ? _dA[ _dAv4_off[2] ]  : ZEROv4;\
            _regA[3]  = (flt_c_v16_valid && HeightInRange(_in_h_id[3])  && WidthInRange(_in_w_id[3])  && (_in_n_id[3]  < in_num)) ? _dA[ _dAv4_off[3] ]  : ZEROv4;\
            _regA[4]  = (flt_c_v16_valid && HeightInRange(_in_h_id[4])  && WidthInRange(_in_w_id[4])  && (_in_n_id[4]  < in_num)) ? _dA[ _dAv4_off[4] ]  : ZEROv4;\
            _regA[5]  = (flt_c_v16_valid && HeightInRange(_in_h_id[5])  && WidthInRange(_in_w_id[5])  && (_in_n_id[5]  < in_num)) ? _dA[ _dAv4_off[5] ]  : ZEROv4;\
            _regA[6]  = (flt_c_v16_valid && HeightInRange(_in_h_id[6])  && WidthInRange(_in_w_id[6])  && (_in_n_id[6]  < in_num)) ? _dA[ _dAv4_off[6] ]  : ZEROv4;\
            _regA[7]  = (flt_c_v16_valid && HeightInRange(_in_h_id[7])  && WidthInRange(_in_w_id[7])  && (_in_n_id[7]  < in_num)) ? _dA[ _dAv4_off[7] ]  : ZEROv4;\
            _regA[8]  = (flt_c_v16_valid && HeightInRange(_in_h_id[8])  && WidthInRange(_in_w_id[8])  && (_in_n_id[8]  < in_num)) ? _dA[ _dAv4_off[8] ]  : ZEROv4;\
            _regA[9]  = (flt_c_v16_valid && HeightInRange(_in_h_id[9])  && WidthInRange(_in_w_id[9])  && (_in_n_id[9]  < in_num)) ? _dA[ _dAv4_off[9] ]  : ZEROv4;\
            _regA[10] = (flt_c_v16_valid && HeightInRange(_in_h_id[10]) && WidthInRange(_in_w_id[10]) && (_in_n_id[10] < in_num)) ? _dA[ _dAv4_off[10] ] : ZEROv4;\
            _regA[11] = (flt_c_v16_valid && HeightInRange(_in_h_id[11]) && WidthInRange(_in_w_id[11]) && (_in_n_id[11] < in_num)) ? _dA[ _dAv4_off[11] ] : ZEROv4;\
            _regA[12] = (flt_c_v16_valid && HeightInRange(_in_h_id[12]) && WidthInRange(_in_w_id[12]) && (_in_n_id[12] < in_num)) ? _dA[ _dAv4_off[12] ] : ZEROv4;\
            _regA[13] = (flt_c_v16_valid && HeightInRange(_in_h_id[13]) && WidthInRange(_in_w_id[13]) && (_in_n_id[13] < in_num)) ? _dA[ _dAv4_off[13] ] : ZEROv4;\
            _regA[14] = (flt_c_v16_valid && HeightInRange(_in_h_id[14]) && WidthInRange(_in_w_id[14]) && (_in_n_id[14] < in_num)) ? _dA[ _dAv4_off[14] ] : ZEROv4;\
            _regA[15] = (flt_c_v16_valid && HeightInRange(_in_h_id[15]) && WidthInRange(_in_w_id[15]) && (_in_n_id[15] < in_num)) ? _dA[ _dAv4_off[15] ] : ZEROv4;\
        }

