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
// load dB macros
////////////////////////////////////////

#define LOAD_dBv1_SIZE1(_regB, _dBv1, _dBv1_off) \
        { \
            _regB[0] = ( flt_n_valid[0] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[0] * _INT4_TO_4INT_] : ZEROv1;\
            \
            _dBv1_off[0] += TILE_K_V4_PER_STEP; \
            \
            flt_hwc_v4_acc  += TILE_K_V4_PER_STEP; \
        }

#define LOAD_dBv1_SIZE2(_regB, _dBv1, _dBv1_off) \
        { \
            _regB[0] = ( flt_n_valid[0] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[0] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[1] = ( flt_n_valid[1] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[1] * _INT4_TO_4INT_] : ZEROv1;\
            \
            _dBv1_off[0] += TILE_K_V4_PER_STEP; \
            _dBv1_off[1] += TILE_K_V4_PER_STEP; \
            \
            flt_hwc_v4_acc  += TILE_K_V4_PER_STEP; \
        }

#define LOAD_dBv1_SIZE4(_regB, _dBv1, _dBv1_off) \
        { \
            _regB[0] = ( flt_n_valid[0] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[0] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[1] = ( flt_n_valid[1] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[1] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[2] = ( flt_n_valid[2] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[2] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[3] = ( flt_n_valid[3] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[3] * _INT4_TO_4INT_] : ZEROv1;\
            \
            _dBv1_off[0] += TILE_K_V4_PER_STEP; \
            _dBv1_off[1] += TILE_K_V4_PER_STEP; \
            _dBv1_off[2] += TILE_K_V4_PER_STEP; \
            _dBv1_off[3] += TILE_K_V4_PER_STEP; \
            \
            flt_hwc_v4_acc  += TILE_K_V4_PER_STEP; \
        }

#define LOAD_dBv1_SIZE8(_regB, _dBv1, _dBv1_off) \
        { \
            _regB[0] = ( flt_n_valid[0] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[0] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[1] = ( flt_n_valid[1] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[1] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[2] = ( flt_n_valid[2] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[2] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[3] = ( flt_n_valid[3] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[3] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[4] = ( flt_n_valid[4] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[4] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[5] = ( flt_n_valid[5] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[5] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[6] = ( flt_n_valid[6] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[6] * _INT4_TO_4INT_] : ZEROv1;\
            _regB[7] = ( flt_n_valid[7] && (flt_hwc_v4_acc < flt_hwc_v4) ) ? _dBv1[ _dBv1_off[7] * _INT4_TO_4INT_] : ZEROv1;\
            \
            _dBv1_off[0] += TILE_K_V4_PER_STEP; \
            _dBv1_off[1] += TILE_K_V4_PER_STEP; \
            _dBv1_off[2] += TILE_K_V4_PER_STEP; \
            _dBv1_off[3] += TILE_K_V4_PER_STEP; \
            _dBv1_off[4] += TILE_K_V4_PER_STEP; \
            _dBv1_off[5] += TILE_K_V4_PER_STEP; \
            _dBv1_off[6] += TILE_K_V4_PER_STEP; \
            _dBv1_off[7] += TILE_K_V4_PER_STEP; \
            \
            flt_hwc_v4_acc  += TILE_K_V4_PER_STEP; \
        }

#define SET_dBv1_BOUND(_step_id, _dBv1_off, _flt_n_valid) \
        { \
            int _flt_n_id  =  cta_idx  *  TILE_N_PER_CTA  + \
                            _step_id  *  TILE_N_PER_STEP + \
                            warp_idx  *  TILE_N_PER_MMA  + \
                             tid_y; \
            \
            _flt_n_valid  =  _flt_n_id < num_flt_per_grp; \
            \
            _dBv1_off  =   grp_id   * flt_hwc_v4 * num_flt_per_grp + \
                          _flt_n_id  * flt_hwc_v4 + \
                           tid_x; \
        }

////////////////////////////////////////
// load dA macros
////////////////////////////////////////

#define SET_IN_Mv1_ID(_tid, _sm_base_v4) \
        { \
            int _out_nhw_id =  cta_idy    * TILE_M_PER_CTA + _tid; \
            \
            int _out_w_id   = (_out_nhw_id % out_width); \
            int _out_h_id   = (_out_nhw_id / out_width)      % out_height; \
            \
            int4 _in_id; \
            \
            _in_id.y = _out_w_id   * stride_width    - pad_width; \
            _in_id.z = _out_h_id   * stride_height   - pad_height; \
            _in_id.w = _out_nhw_id / out_hw; \
            \
            _in_id.x = (_in_id.w * in_hw + _in_id.z * in_width + _in_id.y) * img_chl_per_grp_pad_v16 * num_grp + \
                         grp_id  * img_chl_per_grp_pad_v16; \
            \
            _sm_base_v4[_tid] = _in_id; \
        }

#define SET_IN_Kv16_OFF(_tid, _sm_base_v4) \
        { \
            int _inNHWC_id =  _tid; \
            \
            int4 _in_off; \
            \
            _in_off.y = ((_inNHWC_id /  img_chl_per_grp_pad_v16) % fltWidth              ) * hole_width; \
            _in_off.z = ((_inNHWC_id / (img_chl_per_grp_pad_v16  * fltWidth)) % fltHeight) * hole_height; \
            _in_off.w =   _inNHWC_id / (img_chl_per_grp_pad_v16  * fltWidth   * fltHeight); \
            \
            _in_off.x = (_in_off.w  * in_hw + _in_off.z * in_width + _in_off.y) * img_chl_per_grp_pad_v16 * num_grp + \
                        (_inNHWC_id %  img_chl_per_grp_pad_v16); \
            \
            _sm_base_v4[SM_IN_ID_SIZE + _tid] = _in_off; \
         }

#define LOAD_dAv1_SIZE1(_regA, _dAv1, _inId, _inOff) \
        { \
            int4 _in; \
            \
            _in.x = (_inId[0].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[0].y + _inOff.y; \
            _in.z =  _inId[0].z + _inOff.z; \
            _in.w =  _inId[0].w + _inOff.w; \
            _regA[0] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
        }

#define LOAD_dAv1_SIZE2(_regA, _dAv1, _inId, _inOff) \
        { \
            int4 _in; \
            \
            _in.x = (_inId[0].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[0].y + _inOff.y; \
            _in.z =  _inId[0].z + _inOff.z; \
            _in.w =  _inId[0].w + _inOff.w; \
            _regA[0] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[1].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[1].y + _inOff.y; \
            _in.z =  _inId[1].z + _inOff.z; \
            _in.w =  _inId[1].w + _inOff.w; \
            _regA[1] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
        }

#define LOAD_dAv1_SIZE4(_regA, _dAv1, _inId, _inOff) \
        { \
            int4 _in; \
            \
            _in.x = (_inId[0].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[0].y + _inOff.y; \
            _in.z =  _inId[0].z + _inOff.z; \
            _in.w =  _inId[0].w + _inOff.w; \
            _regA[0] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[1].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[1].y + _inOff.y; \
            _in.z =  _inId[1].z + _inOff.z; \
            _in.w =  _inId[1].w + _inOff.w; \
            _regA[1] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[2].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[2].y + _inOff.y; \
            _in.z =  _inId[2].z + _inOff.z; \
            _in.w =  _inId[2].w + _inOff.w; \
            _regA[2] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[3].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[3].y + _inOff.y; \
            _in.z =  _inId[3].z + _inOff.z; \
            _in.w =  _inId[3].w + _inOff.w; \
            _regA[3] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
        }

#define LOAD_dAv1_SIZE8(_regA, _dAv1, _inId, _inOff) \
        { \
            int4 _in; \
            \
            _in.x = (_inId[0].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[0].y + _inOff.y; \
            _in.z =  _inId[0].z + _inOff.z; \
            _in.w =  _inId[0].w + _inOff.w; \
            _regA[0] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[1].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[1].y + _inOff.y; \
            _in.z =  _inId[1].z + _inOff.z; \
            _in.w =  _inId[1].w + _inOff.w; \
            _regA[1] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[2].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[2].y + _inOff.y; \
            _in.z =  _inId[2].z + _inOff.z; \
            _in.w =  _inId[2].w + _inOff.w; \
            _regA[2] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[3].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[3].y + _inOff.y; \
            _in.z =  _inId[3].z + _inOff.z; \
            _in.w =  _inId[3].w + _inOff.w; \
            _regA[3] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[4].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[4].y + _inOff.y; \
            _in.z =  _inId[4].z + _inOff.z; \
            _in.w =  _inId[4].w + _inOff.w; \
            _regA[4] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[5].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[5].y + _inOff.y; \
            _in.z =  _inId[5].z + _inOff.z; \
            _in.w =  _inId[5].w + _inOff.w; \
            _regA[5] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[6].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[6].y + _inOff.y; \
            _in.z =  _inId[6].z + _inOff.z; \
            _in.w =  _inId[6].w + _inOff.w; \
            _regA[6] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[7].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[7].y + _inOff.y; \
            _in.z =  _inId[7].z + _inOff.z; \
            _in.w =  _inId[7].w + _inOff.w; \
            _regA[7] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
        }

#define LOAD_dAv1_SIZE16(_regA, _dAv1, _inId, _inOff) \
        { \
            int4 _in; \
            \
            _in.x = (_inId[0].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[0].y + _inOff.y; \
            _in.z =  _inId[0].z + _inOff.z; \
            _in.w =  _inId[0].w + _inOff.w; \
            _regA[0] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[1].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[1].y + _inOff.y; \
            _in.z =  _inId[1].z + _inOff.z; \
            _in.w =  _inId[1].w + _inOff.w; \
            _regA[1] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[2].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[2].y + _inOff.y; \
            _in.z =  _inId[2].z + _inOff.z; \
            _in.w =  _inId[2].w + _inOff.w; \
            _regA[2] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[3].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[3].y + _inOff.y; \
            _in.z =  _inId[3].z + _inOff.z; \
            _in.w =  _inId[3].w + _inOff.w; \
            _regA[3] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[4].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[4].y + _inOff.y; \
            _in.z =  _inId[4].z + _inOff.z; \
            _in.w =  _inId[4].w + _inOff.w; \
            _regA[4] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[5].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[5].y + _inOff.y; \
            _in.z =  _inId[5].z + _inOff.z; \
            _in.w =  _inId[5].w + _inOff.w; \
            _regA[5] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[6].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[6].y + _inOff.y; \
            _in.z =  _inId[6].z + _inOff.z; \
            _in.w =  _inId[6].w + _inOff.w; \
            _regA[6] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[7].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[7].y + _inOff.y; \
            _in.z =  _inId[7].z + _inOff.z; \
            _in.w =  _inId[7].w + _inOff.w; \
            _regA[7] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[8].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[8].y + _inOff.y; \
            _in.z =  _inId[8].z + _inOff.z; \
            _in.w =  _inId[8].w + _inOff.w; \
            _regA[8] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[9].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[9].y + _inOff.y; \
            _in.z =  _inId[9].z + _inOff.z; \
            _in.w =  _inId[9].w + _inOff.w; \
            _regA[9] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[10].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[10].y + _inOff.y; \
            _in.z =  _inId[10].z + _inOff.z; \
            _in.w =  _inId[10].w + _inOff.w; \
            _regA[10] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[11].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[11].y + _inOff.y; \
            _in.z =  _inId[11].z + _inOff.z; \
            _in.w =  _inId[11].w + _inOff.w; \
            _regA[11] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[12].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[12].y + _inOff.y; \
            _in.z =  _inId[12].z + _inOff.z; \
            _in.w =  _inId[12].w + _inOff.w; \
            _regA[12] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[13].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[13].y + _inOff.y; \
            _in.z =  _inId[13].z + _inOff.z; \
            _in.w =  _inId[13].w + _inOff.w; \
            _regA[13] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[14].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[14].y + _inOff.y; \
            _in.z =  _inId[14].z + _inOff.z; \
            _in.w =  _inId[14].w + _inOff.w; \
            _regA[14] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
            \
            _in.x = (_inId[15].x + _inOff.x) * _INT4_TO_4INT_; \
            _in.y =  _inId[15].y + _inOff.y; \
            _in.z =  _inId[15].z + _inOff.z; \
            _in.w =  _inId[15].w + _inOff.w; \
            _regA[15] = ( BatchInRange(_in.w) && WidthInRange(_in.y) && HeightInRange(_in.z) ) ? _dAv1[_in.x] : ZEROv1;\
        }
