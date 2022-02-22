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

//////////////////////////////////////////////////////
// half output interface
//////////////////////////////////////////////////////

#if defined(ENABLE_FUSE)

#define OUTPUT_BY_INT1() \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv1_y_valid[0] && dCv1_x_valid[i] ) dCv1[concat_v1_off0 + dCv1_idx[i]] = C[Cv1_off + i]; \
                \
                if( dCv1_y_valid[1] && dCv1_x_valid[i] ) dCv1[concat_v1_off1 + dCv1_idx[i]] = C[Cv1_off + i + NUM_N_STEPS]; \
            } \
            \
            dCv1_idy[0]  += TILE_M_PER_STEP; \
            dCv1_idy[1]  += TILE_M_PER_STEP; \
            dCv1_y_valid[0] = (dCv1_idy[0] < out_nhw); \
            dCv1_y_valid[1] = (dCv1_idy[1] < out_nhw); \
        }

#else

#define OUTPUT_BY_INT1() \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv1_y_valid[0] && dCv1_x_valid[i] ) dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[i]] = C[Cv1_off + i]; \
                \
                if( dCv1_y_valid[1] && dCv1_x_valid[i] ) dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[i]] = C[Cv1_off + i + NUM_N_STEPS]; \
            } \
            \
            dCv1_idy[0]  += TILE_M_PER_STEP; \
            dCv1_idy[1]  += TILE_M_PER_STEP; \
            dCv1_y_valid[0] = (dCv1_idy[0] < out_nhw); \
            dCv1_y_valid[1] = (dCv1_idy[1] < out_nhw); \
        }

#endif

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define ADD_BIAS_V1(_has_bias, _bias) \
        { \
            if( _has_bias ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HADD2_INST(C[Cv1_off + i],               C[Cv1_off + i],               ((int *) _bias) [dCv1_idx[i]]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HADD2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], ((int *) _bias) [dCv1_idx[i]]); \
                } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_V1(_has_relu) \
        { \
	        if( _has_relu == 1) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i],               C[Cv1_off + i],               0, C[Cv1_off + i]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], 0, C[Cv1_off + i + NUM_N_STEPS]); \
	            } \
	        } \
        }

#if 0
#define FUSE_RELU_V1(_has_relu) \
        { \
	        if( _has_relu == 1) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i],               C[Cv1_off + i],               0, C[Cv1_off + i]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], 0, C[Cv1_off + i + NUM_N_STEPS]); \
	            } \
	        } \
            else if( _has_relu == 2) \
            { \
                __half2 h2ONE((__half)1.f, (__half)1.f); \
                __half2 *h2C = (__half2 *)C; \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) \
                        h2C[Cv1_off + i]               = __h2div(h2exp(h2C[Cv1_off + i]),               __hadd2(h2ONE, h2exp(h2C[Cv1_off + i]))); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) \
                        h2C[Cv1_off + i + NUM_N_STEPS] = __h2div(h2exp(h2C[Cv1_off + i + NUM_N_STEPS]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + i + NUM_N_STEPS]))); \
	            } \
	        } \
        }
#endif

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_V1(_has_clip, _clip_max, _clip_min) \
        { \
	        if( _has_clip ) \
            { \
                int * _r_clip_max = (int *) &_clip_max; \
                int * _r_clip_min = (int *) &_clip_min; \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMIN2_INST(C[Cv1_off + i],               C[Cv1_off + i],               _r_clip_max[0], C[Cv1_off + i]); \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i],               C[Cv1_off + i],               _r_clip_min[0], C[Cv1_off + i]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMIN2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], _r_clip_max[0], C[Cv1_off + i + NUM_N_STEPS]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], _r_clip_min[0], C[Cv1_off + i + NUM_N_STEPS]); \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_V1(_has_prelu, _prelu, _leaky) \
        { \
	        if( _has_prelu == 1) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], 0) ) \
                        hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], _leaky); \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], 0) ) \
                        hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], _leaky); \
                    \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], 0) ) \
                        hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], _leaky); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], 0) ) \
                        hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], _leaky); \
	            } \
	        } \
            \
	        if( _has_prelu == 2) \
            { \
                int _scale_v1[NUM_N_STEPS]; \
                __half * _hscale[NUM_N_STEPS]; \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    _scale_v1[i] = dCv1_x_valid[i] ? ((int *)_prelu)[dCv1_idx[i]] : 0; ; \
                    _hscale[i] = (__half *) &_scale_v1[i]; \
                } \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], 0) ) \
                        hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], _hscale[i][0]); \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], 0) ) \
                        hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], _hscale[i][1]); \
                    \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], 0) ) \
                        hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], _hscale[i][0]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], 0) ) \
                        hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], _hscale[i][1]); \
	            } \
	        } \
	        if( _has_prelu == 3) \
            { \
                int _scale_v1[BLK_M_PER_MMA * NUM_N_STEPS]; \
                __half * _hscale[BLK_M_PER_MMA * NUM_N_STEPS]; \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    _scale_v1[i * BLK_M_PER_MMA + 0] = (dCv1_y_valid[0] && dCv1_x_valid[i]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[i]] : 0; \
                    _scale_v1[i * BLK_M_PER_MMA + 1] = (dCv1_y_valid[1] && dCv1_x_valid[i]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[i]] : 0; \
                    \
                    _hscale[i * BLK_M_PER_MMA + 0] = (__half *)&_scale_v1[i * BLK_M_PER_MMA + 0]; \
                    _hscale[i * BLK_M_PER_MMA + 1] = (__half *)&_scale_v1[i * BLK_M_PER_MMA + 1]; \
                } \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], 0) ) \
                        hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], _hscale[i * BLK_M_PER_MMA + 0][0]); \
                    if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], 0) ) \
                        hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], _hscale[i * BLK_M_PER_MMA + 0][1]); \
                    \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], 0) ) \
                        hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], _hscale[i * BLK_M_PER_MMA + 1][0]); \
                    if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], 0) ) \
                        hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], _hscale[i * BLK_M_PER_MMA + 1][1]); \
	            } \
	        } \
    }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_V1(_has_elt, _pre_data) \
    { \
	    if( _has_elt ) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[i] ) \
                    HADD2_INST(C[Cv1_off + i],               C[Cv1_off + i],               ((int *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[i]]); \
                if(dCv1_y_valid[1] && dCv1_x_valid[i] ) \
                    HADD2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], ((int *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[i]]); \
	        } \
	    } \
    }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

#define SET_CONCAT_OFF_V1(_has_concat, _concat_v1_off0, _concat_v1_off1)                                                   \
    {                                                                                                                      \
        _concat_v1_off0 = dCv1_idy[0] * num_flt_v2;                                                                        \
        _concat_v1_off1 = dCv1_idy[1] * num_flt_v2;                                                                        \
        if (_has_concat) {                                                                                                 \
            if (dCv1_y_valid[0])                                                                                           \
                _concat_v1_off0 = concat_offset_v8 * _INT4_TO_4HALF2_ + dCv1_idy[0] * concat_stride_v8 * _INT4_TO_4HALF2_; \
            if (dCv1_y_valid[1])                                                                                           \
                _concat_v1_off1 = concat_offset_v8 * _INT4_TO_4HALF2_ + dCv1_idy[1] * concat_stride_v8 * _INT4_TO_4HALF2_; \
        }                                                                                                                  \
    }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define JIT_FUSE_RELU_V1() \
    { \
        _Pragma("unroll") \
        for(int i = 0; i < NUM_N_STEPS; i++) \
        { \
            if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i],               C[Cv1_off + i],               0, C[Cv1_off + i]); \
            if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], 0, C[Cv1_off + i + NUM_N_STEPS]); \
	    } \
    }

#define JIT_FUSE_SIGMOID_V1() \
    { \
        __half2 h2ONE((__half)1.f, (__half)1.f); \
        __half2 *h2C = (__half2 *)C; \
        \
        _Pragma("unroll") \
        for(int i = 0; i < NUM_N_STEPS; i++) \
        { \
            if( dCv1_y_valid[0] && dCv1_x_valid[i] ) \
                h2C[Cv1_off + i]               = __h2div(h2exp(h2C[Cv1_off + i]),               __hadd2(h2ONE, h2exp(h2C[Cv1_off + i]))); \
            if( dCv1_y_valid[1] && dCv1_x_valid[i] ) \
                h2C[Cv1_off + i + NUM_N_STEPS] = __h2div(h2exp(h2C[Cv1_off + i + NUM_N_STEPS]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + i + NUM_N_STEPS]))); \
	    } \
    }


//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define JIT_FUSE_CLIP_V1(_clip_max, _clip_min) \
    { \
        int * _r_clip_max = (int *) &_clip_max; \
        int * _r_clip_min = (int *) &_clip_min; \
        \
        _Pragma("unroll") \
        for(int i = 0; i < NUM_N_STEPS; i++) \
        { \
            if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMIN2_INST(C[Cv1_off + i],               C[Cv1_off + i],               _r_clip_max[0], C[Cv1_off + i]); \
            if( dCv1_y_valid[0] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i],               C[Cv1_off + i],               _r_clip_min[0], C[Cv1_off + i]); \
            if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMIN2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], _r_clip_max[0], C[Cv1_off + i + NUM_N_STEPS]); \
            if( dCv1_y_valid[1] && dCv1_x_valid[i] ) HMAX2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], _r_clip_min[0], C[Cv1_off + i + NUM_N_STEPS]); \
	    } \
    }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define JIT_FUSE_LEAKY_V1(_leaky) \
   { \
       _Pragma("unroll") \
       for(int i = 0; i < NUM_N_STEPS; i++) \
       { \
           if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], 0) ) \
               hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], _leaky); \
           if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], 0) ) \
               hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], _leaky); \
           \
           if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], 0) ) \
               hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], _leaky); \
           if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], 0) ) \
               hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], _leaky); \
       } \
    }

#define JIT_FUSE_PRELU_V1(_has_prelu, _prelu) \
    { \
	    if( _has_prelu == 2) \
        { \
            int _scale_v1[NUM_N_STEPS]; \
            __half * _hscale[NUM_N_STEPS]; \
            \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                _scale_v1[i] = dCv1_x_valid[i] ? ((int *)_prelu)[dCv1_idx[i]] : 0; ; \
                _hscale[i] = (__half *) &_scale_v1[i]; \
            } \
            \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], 0) ) \
                    hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], _hscale[i][0]); \
                if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], 0) ) \
                    hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], _hscale[i][1]); \
                \
                if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], 0) ) \
                    hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], _hscale[i][0]); \
                if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], 0) ) \
                    hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], _hscale[i][1]); \
            } \
        } \
        \
	    if( _has_prelu == 3) \
        { \
            int _scale_v1[BLK_M_PER_MMA * NUM_N_STEPS]; \
            __half * _hscale[BLK_M_PER_MMA * NUM_N_STEPS]; \
            \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                _scale_v1[i * BLK_M_PER_MMA + 0] = (dCv1_y_valid[0] && dCv1_x_valid[i]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[i]] : 0; \
                _scale_v1[i * BLK_M_PER_MMA + 1] = (dCv1_y_valid[1] && dCv1_x_valid[i]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[i]] : 0; \
                \
                _hscale[i * BLK_M_PER_MMA + 0] = (__half *)&_scale_v1[i * BLK_M_PER_MMA + 0]; \
                _hscale[i * BLK_M_PER_MMA + 1] = (__half *)&_scale_v1[i * BLK_M_PER_MMA + 1]; \
            } \
            \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], 0) ) \
                    hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 0], _hscale[i * BLK_M_PER_MMA + 0][0]); \
                if( dCv1_y_valid[0] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], 0) ) \
                    hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i) * _INT_TO_2HALF_ + 1], _hscale[i * BLK_M_PER_MMA + 0][1]); \
                \
                if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], 0) ) \
                    hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 0], _hscale[i * BLK_M_PER_MMA + 1][0]); \
                if( dCv1_y_valid[1] && dCv1_x_valid[i] && __hlt(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], 0) ) \
                    hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1] = __hmul(hC[(Cv1_off + i + NUM_N_STEPS) * _INT_TO_2HALF_ + 1], _hscale[i * BLK_M_PER_MMA + 1][1]); \
	        } \
	    } \
    }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define JIT_FUSE_ELT_V1(_pre_data) \
    { \
        _Pragma("unroll") \
        for(int i = 0; i < NUM_N_STEPS; i++) \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[i] ) \
                HADD2_INST(C[Cv1_off + i],               C[Cv1_off + i],               ((int *) _pre_data) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[i]]); \
            if(dCv1_y_valid[1] && dCv1_x_valid[i] ) \
                HADD2_INST(C[Cv1_off + i + NUM_N_STEPS], C[Cv1_off + i + NUM_N_STEPS], ((int *) _pre_data) [dCv1_idy[1] * num_flt_v2 + dCv1_idx[i]]); \
	    } \
    }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

#define JIT_SET_CONCAT_OFF_V1(_concat_v1_off0, _concat_v1_off1)                                                        \
    {                                                                                                                  \
        if (dCv1_y_valid[0])                                                                                           \
            _concat_v1_off0 = concat_offset_v8 * _INT4_TO_4HALF2_ + dCv1_idy[0] * concat_stride_v8 * _INT4_TO_4HALF2_; \
        if (dCv1_y_valid[1])                                                                                           \
            _concat_v1_off1 = concat_offset_v8 * _INT4_TO_4HALF2_ + dCv1_idy[1] * concat_stride_v8 * _INT4_TO_4HALF2_; \
    }
