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

#define PACK_V2(_C, _Cv1_off) \
        { \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(_C[_Cv1_off + 0]) : "r"(_C[_Cv1_off + 1]), "r"(_C[_Cv1_off + 0]) ); \
        }

#define PACK_V4(_C, _Cv1_off) \
        { \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(_C[_Cv1_off + 2]) : "r"(_C[_Cv1_off + 3]), "r"(_C[_Cv1_off + 2]) ); \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;\n": "=r"(_C[_Cv1_off + 0]) : "r"(_C[_Cv1_off + 1]), "r"(_C[_Cv1_off + 0]), "r"(_C[_Cv1_off + 2])); \
        }

#define OUTPUT_BY_HALF_X1() \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv2_y_valid[0] && dCv2_x_valid[i] ) \
                { \
                    PACK_V2(C, (Cv2_off + i) * _INT2_TO_2INT_); \
                    dCvHalf[concat_v2_off0 + dCv2_idx[i]] = CvHalf[(Cv2_off + i) * _INT2_TO_4HALF_]; \
                } \
            } \
            \
            dCv2_idy[0]  += TILE_M_PER_STEP; \
            dCv2_y_valid[0] = (dCv2_idy[0] < out_nhw); \
        }

//////////////////////////////////////////////////////
// quant interface
//////////////////////////////////////////////////////

#define GET_DEQUANTSCALE_V2(_de_scale_v2, _d_flt_scale, _in_scale) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv2_y_valid[0] && dCv2_x_valid[i] ) { \
                    _de_scale_v2[i] = ((float2 *)_d_flt_scale) [dCv2_idx[i]]; \
                    _de_scale_v2[i].x *= _in_scale; \
                    _de_scale_v2[i].y *= _in_scale; \
                } \
	        } \
        }

#define DEQUANT_V2(_fCv2, _Cv2, _de_scale_v2) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv2_y_valid[0] && dCv2_x_valid[i] ) { \
                    _fCv2[Cv2_off + i].x = _Cv2[Cv2_off + i].x * _de_scale_v2[i].x; \
                    _fCv2[Cv2_off + i].y = _Cv2[Cv2_off + i].y * _de_scale_v2[i].y; \
                } \
	        } \
        }

#define QUANT_V2(_Cv2, _fCv2, _quant_scale) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < NUM_N_STEPS; i++) \
            { \
                if( dCv2_y_valid[0] && dCv2_x_valid[i] ) { \
                    _Cv2[Cv2_off + i].x = __float2int_rn(_fCv2[Cv2_off + i].x * _quant_scale); \
                    _Cv2[Cv2_off + i].y = __float2int_rn(_fCv2[Cv2_off + i].y * _quant_scale); \
                } \
	        } \
        }

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define ADD_BIAS_V2(_has_bias, _bias) \
        { \
            if( _has_bias ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] ) { \
                        float2 f2_bias = ((float2 *)_bias) [dCv2_idx[i]]; \
                        fCv2[Cv2_off + i].x += f2_bias.x; \
                        fCv2[Cv2_off + i].y += f2_bias.y; \
                    } \
                } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_V2(_has_relu) \
        { \
	        if( _has_relu == 1) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] ) { \
                        fCv2[Cv2_off + i].x = Max(fCv2[Cv2_off + i].x, _FLOAT_ZERO_); \
                        fCv2[Cv2_off + i].y = Max(fCv2[Cv2_off + i].y, _FLOAT_ZERO_); \
                    } \
	            } \
	        } \
        }

#if 0
#define FUSE_RELU_V2(_has_relu) \
        { \
	        if( _has_relu == 1) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] ) { \
                        fCv2[Cv2_off + i].x = Max(fCv2[Cv2_off + i].x, _FLOAT_ZERO_); \
                        fCv2[Cv2_off + i].y = Max(fCv2[Cv2_off + i].y, _FLOAT_ZERO_); \
                    } \
	            } \
	        } \
            else if( _has_relu == 2) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] ) { \
                        fCv2[Cv2_off + i].x = __expf(fCv2[Cv2_off + i].x) / (_FLOAT_ONE_ + __expf(fCv2[Cv2_off + i].x)); \
                        fCv2[Cv2_off + i].y = __expf(fCv2[Cv2_off + i].y) / (_FLOAT_ONE_ + __expf(fCv2[Cv2_off + i].y)); \
                    } \
	            } \
	        } \
        }
#endif

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_V2(_has_clip, _clip_max, _clip_min) \
        { \
	        if( _has_clip ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] ) { \
                        fCv2[Cv2_off + i].x = Min(fCv2[Cv2_off + i].x, _clip_max); \
                        fCv2[Cv2_off + i].y = Min(fCv2[Cv2_off + i].y, _clip_max); \
                        fCv2[Cv2_off + i].x = Max(fCv2[Cv2_off + i].x, _clip_min); \
                        fCv2[Cv2_off + i].y = Max(fCv2[Cv2_off + i].y, _clip_min); \
                    } \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_V2(_has_prelu, _prelu, _leaky) \
        { \
	        if( _has_prelu == 1) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] && fCv2[Cv2_off + i].x < _FLOAT_ZERO_) \
                        fCv2[Cv2_off + i].x *= _leaky; \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] && fCv2[Cv2_off + i].y < _FLOAT_ZERO_) \
                        fCv2[Cv2_off + i].y *= _leaky; \
	            } \
	        } \
            \
	        if( _has_prelu == 2) \
            { \
                int2 _scaleV2[NUM_N_STEPS]; \
                float * _f_scale = (float *) _scaleV2;\
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                    _scaleV2[i] = dCv2_x_valid[i] ? ((int2 *)_prelu)[dCv2_idx[i]] : {0, 0}; \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] && fCv2[Cv2_off + i].x < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i].x *= _f_scale[i * _INT2_TO_2INT_ + 0]; \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] && fCv2[Cv2_off + i].y < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i].y *= _f_scale[i * _INT2_TO_2INT_ + 1]; \
	            } \
	        } \
	        if( _has_prelu == 3) \
            { \
                int2 _scaleV2[NUM_N_STEPS]; \
                float * _f_scale = (float *) _scaleV2;\
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    _scaleV2[i] = (dCv2_y_valid[0] && dCv2_x_valid[i]) ? ((int2 *)_prelu)[dCv2_idy[0] * num_flt_v2 + dCv2_idx[i]] : {0, 0}; \
                } \
                \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] && fCv2[Cv2_off + i].x < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i].x *= _f_scale[i * _INT2_TO_2INT_ + 0]; \
                    if( dCv2_y_valid[0] && dCv2_x_valid[i] && fCv2[Cv2_off + i].y < _FLOAT_ZERO_ ) \
                        fCv2[Cv2_off + i].y *= _f_scale[i * _INT2_TO_2INT_ + 1]; \
	            } \
	        } \
    }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_V2(_has_elt, _pre_data) \
        { \
	        if( _has_elt ) \
            { \
                _Pragma("unroll") \
                for(int i = 0; i < NUM_N_STEPS; i++) \
                { \
                    if(dCv2_y_valid[0] && dCv2_x_valid[i] ) { \
                        int16_t  elt_v2 = ((int16_t*) _pre_data) [dCv2_idy[0] * num_flt_v2 + dCv2_idx[i]]; \
                        int8_t * elt_v1 = (int8_t *) &elt_v2; \
                        \
                        fCv2[Cv2_off + i].x += (int)elt_v1[0] * pre_scale; \
                        fCv2[Cv2_off + i].y += (int)elt_v1[1] * pre_scale; \
                    } \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

#define SET_CONCAT_OFF_V2(_has_concat, _concat_v2_off0) \
        { \
            _concat_v2_off0 = dCv2_idy[0] * num_flt_v2; \
            if (_has_concat) { \
                if (dCv2_y_valid[0]) \
                    _concat_v2_off0 = concat_offset_v4 * _INT4_TO_8HALF_ + dCv2_idy[0] * concat_stride_v4 * _INT4_TO_8HALF_; \
            } \
        }

