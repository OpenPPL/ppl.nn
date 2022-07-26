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

/////////////////////////////////////////////////////
// common load global memory macros
/////////////////////////////////////////////////////

////////////////////////////////////////
// load dA macros
////////////////////////////////////////

#define LOAD_dAv4_SIZE_16TH(_sAv4, _sAv4_off, _dA, _dAv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dAv4_off[0] += flt_lut.idx[lut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                CP_ASYNC( (_flt_n_valid[0] && _flt_c_v16_valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE_8TH(_sAv4, _sAv4_off, _dA, _dAv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dAv4_off[0] += flt_lut.idx[lut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                CP_ASYNC( (_flt_n_valid[0] && _flt_c_v16_valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE_QTR(_sAv4, _sAv4_off, _dA, _dAv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dAv4_off[0] += flt_lut.idx[lut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                CP_ASYNC( (_flt_n_valid[0] && _flt_c_v16_valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE_HALF(_sAv4, _sAv4_off, _dA, _dAv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dAv4_off[0] += flt_lut.idx[lut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                CP_ASYNC( (_flt_n_valid[0] && _flt_c_v16_valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE1(_sAv4, _sAv4_off, _dA, _dAv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dAv4_off[0] += flt_lut.idx[lut_id]; \
            \
            CP_ASYNC( (_flt_n_valid[0] && _flt_c_v16_valid), _sAv4, _sAv4_off, _dA, _dAv4_off[0]); \
        }

#define LOAD_dAv4_SIZE2(_sAv4, _sAv4_off, _dA, _dAv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dAv4_off[0] += flt_lut.idx[lut_id]; \
            _dAv4_off[1] += flt_lut.idx[lut_id]; \
            \
            CP_ASYNC( (_flt_n_valid[0] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( (_flt_n_valid[1] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
        }

#define LOAD_dAv4_SIZE4(_sAv4, _sAv4_off, _dA, _dAv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dAv4_off[0] += flt_lut.idx[lut_id]; \
            _dAv4_off[1] += flt_lut.idx[lut_id]; \
            _dAv4_off[2] += flt_lut.idx[lut_id]; \
            _dAv4_off[3] += flt_lut.idx[lut_id]; \
            \
            CP_ASYNC( (_flt_n_valid[0] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( (_flt_n_valid[1] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
            CP_ASYNC( (_flt_n_valid[2] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2, _dA, _dAv4_off[2]); \
            CP_ASYNC( (_flt_n_valid[3] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3, _dA, _dAv4_off[3]); \
        }

#define LOAD_dAv4_SIZE8(_sAv4, _sAv4_off, _dA, _dAv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dAv4_off[0] += flt_lut.idx[lut_id]; \
            _dAv4_off[1] += flt_lut.idx[lut_id]; \
            _dAv4_off[2] += flt_lut.idx[lut_id]; \
            _dAv4_off[3] += flt_lut.idx[lut_id]; \
            _dAv4_off[4] += flt_lut.idx[lut_id]; \
            _dAv4_off[5] += flt_lut.idx[lut_id]; \
            _dAv4_off[6] += flt_lut.idx[lut_id]; \
            _dAv4_off[7] += flt_lut.idx[lut_id]; \
            \
            CP_ASYNC( (_flt_n_valid[0] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0, _dA, _dAv4_off[0]); \
            CP_ASYNC( (_flt_n_valid[1] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1, _dA, _dAv4_off[1]); \
            CP_ASYNC( (_flt_n_valid[2] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2, _dA, _dAv4_off[2]); \
            CP_ASYNC( (_flt_n_valid[3] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3, _dA, _dAv4_off[3]); \
            CP_ASYNC( (_flt_n_valid[4] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4, _dA, _dAv4_off[4]); \
            CP_ASYNC( (_flt_n_valid[5] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5, _dA, _dAv4_off[5]); \
            CP_ASYNC( (_flt_n_valid[6] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6, _dA, _dAv4_off[6]); \
            CP_ASYNC( (_flt_n_valid[7] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7, _dA, _dAv4_off[7]); \
        }

#define LOAD_dAv4_SIZE16(_sAv4, _sAv4_off, _dA, _dAv4_off, _flt_c_v16_valid, _flt_n_valid) \
        { \
            _dAv4_off[0]  += flt_lut.idx[lut_id]; \
            _dAv4_off[1]  += flt_lut.idx[lut_id]; \
            _dAv4_off[2]  += flt_lut.idx[lut_id]; \
            _dAv4_off[3]  += flt_lut.idx[lut_id]; \
            _dAv4_off[4]  += flt_lut.idx[lut_id]; \
            _dAv4_off[5]  += flt_lut.idx[lut_id]; \
            _dAv4_off[6]  += flt_lut.idx[lut_id]; \
            _dAv4_off[7]  += flt_lut.idx[lut_id]; \
            _dAv4_off[8]  += flt_lut.idx[lut_id]; \
            _dAv4_off[9]  += flt_lut.idx[lut_id]; \
            _dAv4_off[10] += flt_lut.idx[lut_id]; \
            _dAv4_off[11] += flt_lut.idx[lut_id]; \
            _dAv4_off[12] += flt_lut.idx[lut_id]; \
            _dAv4_off[13] += flt_lut.idx[lut_id]; \
            _dAv4_off[14] += flt_lut.idx[lut_id]; \
            _dAv4_off[15] += flt_lut.idx[lut_id]; \
            \
            CP_ASYNC( (_flt_n_valid[0]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 0,  _dA, _dAv4_off[0]);  \
            CP_ASYNC( (_flt_n_valid[1]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 1,  _dA, _dAv4_off[1]);  \
            CP_ASYNC( (_flt_n_valid[2]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 2,  _dA, _dAv4_off[2]);  \
            CP_ASYNC( (_flt_n_valid[3]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 3,  _dA, _dAv4_off[3]);  \
            CP_ASYNC( (_flt_n_valid[4]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 4,  _dA, _dAv4_off[4]);  \
            CP_ASYNC( (_flt_n_valid[5]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 5,  _dA, _dAv4_off[5]);  \
            CP_ASYNC( (_flt_n_valid[6]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 6,  _dA, _dAv4_off[6]);  \
            CP_ASYNC( (_flt_n_valid[7]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 7,  _dA, _dAv4_off[7]);  \
            CP_ASYNC( (_flt_n_valid[8]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 8,  _dA, _dAv4_off[8]);  \
            CP_ASYNC( (_flt_n_valid[9]  && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 9,  _dA, _dAv4_off[9]);  \
            CP_ASYNC( (_flt_n_valid[10] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 10, _dA, _dAv4_off[10]); \
            CP_ASYNC( (_flt_n_valid[11] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 11, _dA, _dAv4_off[11]); \
            CP_ASYNC( (_flt_n_valid[12] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 12, _dA, _dAv4_off[12]); \
            CP_ASYNC( (_flt_n_valid[13] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 13, _dA, _dAv4_off[13]); \
            CP_ASYNC( (_flt_n_valid[14] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 14, _dA, _dAv4_off[14]); \
            CP_ASYNC( (_flt_n_valid[15] && _flt_c_v16_valid), _sAv4, _sAv4_off + CTA_SIZE_IN_THD * 15, _dA, _dAv4_off[15]); \
        }

#define SET_dAv4_BOUND(_step_id, _dAv4_off, _flt_n_valid) \
        { \
            int _flt_n_id  =  cta_idy  *  TILE_M_PER_CTA + \
                            _step_id  * (TILE_M_PER_CTA / READ_dAv4_STEPS) + \
                             ldg_idy; \
            \
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp; \
            \
            _dAv4_off  =   grp_id   * flt_hw * num_chl_per_grp_pad_v16 * num_flt_per_grp + \
                          _flt_n_id  * flt_hw * num_chl_per_grp_pad_v16 + \
                           flt_c_v16_id; \
        }

////////////////////////////////////////
// load dB macros
////////////////////////////////////////

#define SET_dBv4_BOUND(_step_id, _dBv4_off, _in_hw_mask) \
        { \
            int _out_nhw_id    =  cta_idx  *  TILE_N_PER_CTA + \
                                _step_id  * (TILE_N_PER_CTA / READ_dBv4_STEPS) + \
                                 ldg_idy; \
            \
            int _out_w_id =  (_out_nhw_id % out_width); \
            int _out_h_id =  (_out_nhw_id / out_width) % out_height; \
            \
            int _in_n_id  =   _out_nhw_id / out_hw; \
            int _in_h_id  =     _out_h_id * stride_height; \
            int _in_w_id  =     _out_w_id * stride_width; \
            \
            _dBv4_off  =  (_in_n_id  * in_hw + _in_h_id  * in_width + _in_w_id) * num_chl_per_grp_pad_v16 * num_grp + \
                           grp_id   * num_chl_per_grp_pad_v16 + \
                           flt_c_v16_id; \
            \
            _in_h_id =  _in_h_id - pad_height; \
            _in_w_id =  _in_w_id - pad_width;  \
            \
            SET_BOUND_FLT3(_in_hw_mask, _in_n_id, _in_h_id, _in_w_id); \
        }

#define LOAD_dBv4_SIZE_16TH(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0] += in_lut.idx[lut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v16_valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE_8TH(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0] += in_lut.idx[lut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v16_valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE_QTR(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0] += in_lut.idx[lut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v16_valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE_HALF(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0] += in_lut.idx[lut_id]; \
            \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v16_valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE1(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0] += in_lut.idx[lut_id]; \
            \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v16_valid), _sBv4, _sBv4_off, _dB, _dBv4_off[0]); \
        }

#define LOAD_dBv4_SIZE2(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0] += in_lut.idx[lut_id]; \
            _dBv4_off[1] += in_lut.idx[lut_id]; \
            \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[1]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
        }

#define LOAD_dBv4_SIZE4(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0] += in_lut.idx[lut_id]; \
            _dBv4_off[1] += in_lut.idx[lut_id]; \
            _dBv4_off[2] += in_lut.idx[lut_id]; \
            _dBv4_off[3] += in_lut.idx[lut_id]; \
            \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[1]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[2]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2, _dB, _dBv4_off[2]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[3]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3, _dB, _dBv4_off[3]); \
        }

#define LOAD_dBv4_SIZE8(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0] += in_lut.idx[lut_id]; \
            _dBv4_off[1] += in_lut.idx[lut_id]; \
            _dBv4_off[2] += in_lut.idx[lut_id]; \
            _dBv4_off[3] += in_lut.idx[lut_id]; \
            _dBv4_off[4] += in_lut.idx[lut_id]; \
            _dBv4_off[5] += in_lut.idx[lut_id]; \
            _dBv4_off[6] += in_lut.idx[lut_id]; \
            _dBv4_off[7] += in_lut.idx[lut_id]; \
            \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0, _dB, _dBv4_off[0]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[1]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1, _dB, _dBv4_off[1]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[2]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2, _dB, _dBv4_off[2]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[3]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3, _dB, _dBv4_off[3]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[4]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 4, _dB, _dBv4_off[4]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[5]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 5, _dB, _dBv4_off[5]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[6]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 6, _dB, _dBv4_off[6]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[7]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 7, _dB, _dBv4_off[7]); \
        }

#define LOAD_dBv4_SIZE16(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0]  += in_lut.idx[lut_id]; \
            _dBv4_off[1]  += in_lut.idx[lut_id]; \
            _dBv4_off[2]  += in_lut.idx[lut_id]; \
            _dBv4_off[3]  += in_lut.idx[lut_id]; \
            _dBv4_off[4]  += in_lut.idx[lut_id]; \
            _dBv4_off[5]  += in_lut.idx[lut_id]; \
            _dBv4_off[6]  += in_lut.idx[lut_id]; \
            _dBv4_off[7]  += in_lut.idx[lut_id]; \
            _dBv4_off[8]  += in_lut.idx[lut_id]; \
            _dBv4_off[9]  += in_lut.idx[lut_id]; \
            _dBv4_off[10] += in_lut.idx[lut_id]; \
            _dBv4_off[11] += in_lut.idx[lut_id]; \
            _dBv4_off[12] += in_lut.idx[lut_id]; \
            _dBv4_off[13] += in_lut.idx[lut_id]; \
            _dBv4_off[14] += in_lut.idx[lut_id]; \
            _dBv4_off[15] += in_lut.idx[lut_id]; \
            \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0,  _dB, _dBv4_off[0]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[1])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1,  _dB, _dBv4_off[1]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[2])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2,  _dB, _dBv4_off[2]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[3])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3,  _dB, _dBv4_off[3]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[4])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 4,  _dB, _dBv4_off[4]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[5])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 5,  _dB, _dBv4_off[5]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[6])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 6,  _dB, _dBv4_off[6]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[7])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 7,  _dB, _dBv4_off[7]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[8])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 8,  _dB, _dBv4_off[8]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[9])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 9,  _dB, _dBv4_off[9]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[10]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 10, _dB, _dBv4_off[10]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[11]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 11, _dB, _dBv4_off[11]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[12]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 12, _dB, _dBv4_off[12]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[13]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 13, _dB, _dBv4_off[13]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[14]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 14, _dB, _dBv4_off[14]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[15]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 15, _dB, _dBv4_off[15]); \
        }

#define LOAD_dBv4_SIZE32(_sBv4, _sBv4_off, _dB, _dBv4_off, _in_c_v16_valid, _flt_hw_bid) \
        { \
            _dBv4_off[0]  += in_lut.idx[lut_id]; \
            _dBv4_off[1]  += in_lut.idx[lut_id]; \
            _dBv4_off[2]  += in_lut.idx[lut_id]; \
            _dBv4_off[3]  += in_lut.idx[lut_id]; \
            _dBv4_off[4]  += in_lut.idx[lut_id]; \
            _dBv4_off[5]  += in_lut.idx[lut_id]; \
            _dBv4_off[6]  += in_lut.idx[lut_id]; \
            _dBv4_off[7]  += in_lut.idx[lut_id]; \
            _dBv4_off[8]  += in_lut.idx[lut_id]; \
            _dBv4_off[9]  += in_lut.idx[lut_id]; \
            _dBv4_off[10] += in_lut.idx[lut_id]; \
            _dBv4_off[11] += in_lut.idx[lut_id]; \
            _dBv4_off[12] += in_lut.idx[lut_id]; \
            _dBv4_off[13] += in_lut.idx[lut_id]; \
            _dBv4_off[14] += in_lut.idx[lut_id]; \
            _dBv4_off[15] += in_lut.idx[lut_id]; \
            _dBv4_off[16] += in_lut.idx[lut_id]; \
            _dBv4_off[17] += in_lut.idx[lut_id]; \
            _dBv4_off[18] += in_lut.idx[lut_id]; \
            _dBv4_off[19] += in_lut.idx[lut_id]; \
            _dBv4_off[20] += in_lut.idx[lut_id]; \
            _dBv4_off[21] += in_lut.idx[lut_id]; \
            _dBv4_off[22] += in_lut.idx[lut_id]; \
            _dBv4_off[23] += in_lut.idx[lut_id]; \
            _dBv4_off[24] += in_lut.idx[lut_id]; \
            _dBv4_off[25] += in_lut.idx[lut_id]; \
            _dBv4_off[26] += in_lut.idx[lut_id]; \
            _dBv4_off[27] += in_lut.idx[lut_id]; \
            _dBv4_off[28] += in_lut.idx[lut_id]; \
            _dBv4_off[29] += in_lut.idx[lut_id]; \
            _dBv4_off[30] += in_lut.idx[lut_id]; \
            _dBv4_off[31] += in_lut.idx[lut_id]; \
            \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[0])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 0,  _dB, _dBv4_off[0]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[1])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 1,  _dB, _dBv4_off[1]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[2])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 2,  _dB, _dBv4_off[2]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[3])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 3,  _dB, _dBv4_off[3]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[4])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 4,  _dB, _dBv4_off[4]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[5])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 5,  _dB, _dBv4_off[5]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[6])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 6,  _dB, _dBv4_off[6]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[7])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 7,  _dB, _dBv4_off[7]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[8])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 8,  _dB, _dBv4_off[8]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[9])  && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 9,  _dB, _dBv4_off[9]);  \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[10]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 10, _dB, _dBv4_off[10]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[11]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 11, _dB, _dBv4_off[11]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[12]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 12, _dB, _dBv4_off[12]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[13]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 13, _dB, _dBv4_off[13]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[14]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 14, _dB, _dBv4_off[14]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[15]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 15, _dB, _dBv4_off[15]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[16]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 16, _dB, _dBv4_off[16]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[17]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 17, _dB, _dBv4_off[17]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[18]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 18, _dB, _dBv4_off[18]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[19]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 19, _dB, _dBv4_off[19]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[20]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 20, _dB, _dBv4_off[20]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[21]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 21, _dB, _dBv4_off[21]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[22]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 22, _dB, _dBv4_off[22]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[23]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 23, _dB, _dBv4_off[23]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[24]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 24, _dB, _dBv4_off[24]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[25]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 25, _dB, _dBv4_off[25]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[26]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 26, _dB, _dBv4_off[26]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[27]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 27, _dB, _dBv4_off[27]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[28]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 28, _dB, _dBv4_off[28]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[29]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 29, _dB, _dBv4_off[29]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[30]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 30, _dB, _dBv4_off[30]); \
            CP_ASYNC( ((_flt_hw_bid & in_hw_mask[31]) && _in_c_v16_valid), _sBv4, _sBv4_off + CTA_SIZE_IN_THD * 31, _dB, _dBv4_off[31]); \
        }
