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
// output interface
//////////////////////////////////////////////////////

#if defined(ENABLE_FUSE)

#define PACK_V4(_R, _R_off) \
        { \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, 0;\n" : "=r"(_R[_R_off + 2]) : "r"(_R[_R_off + 3]), "r"(_R[_R_off + 2]) ); \
            asm volatile("cvt.pack.sat.s8.s32.b32 %0, %1, %2, %3;\n": "=r"(_R[_R_off + 0]) : "r"(_R[_R_off + 1]), "r"(_R[_R_off + 0]), "r"(_R[_R_off + 2])); \
        }

#define OUTPUT_BY_INT8_V4(_R) \
        { \
            if( dCv4_x_valid && dCv4_y_valid ) \
                ((int*) dC)[concat_v4_off + dCv4_off] = _R[0]; \
        }

#elif defined(ENABLE_SPLITK) || defined(ENABLE_SPLITF)

#define OUTPUT_BY_INT4_V1(_Rv4) \
        { \
            if( dCv4_x_valid && dCv4_y_valid ) \
            { \
                ((int4 *)dC)[ dCv4_off ] = _Rv4[0]; \
            } \
        }

#endif

//////////////////////////////////////////////////////
// quant interface
//////////////////////////////////////////////////////

#define GET_DEQUANTSCALE(_de_scale_v4, _de_scale, _d_flt_scale, _in_scale) \
        { \
        	if(dCv4_x_valid && dCv4_y_valid) \
            { \
                _de_scale_v4 = ((float4 *) _d_flt_scale)[grp_id * num_flt_per_grp_pad_v4 + dCv4_idx]; \
                \
                _de_scale[0] *= _in_scale; \
                _de_scale[1] *= _in_scale; \
                _de_scale[2] *= _in_scale; \
                _de_scale[3] *= _in_scale; \
            } \
        }

#define DEQUANT_V4(_fR, _R, _de_scale) \
        { \
        	_fR[0] = _R[0] * _de_scale[0]; \
        	_fR[1] = _R[1] * _de_scale[1]; \
        	_fR[2] = _R[2] * _de_scale[2]; \
        	_fR[3] = _R[3] * _de_scale[3]; \
        }

#define QUANT_V4(_R, _fR, _quant_scale) \
        { \
           _R[0] = __float2int_rn(_fR[0] * _quant_scale); \
           _R[1] = __float2int_rn(_fR[1] * _quant_scale); \
           _R[2] = __float2int_rn(_fR[2] * _quant_scale); \
           _R[3] = __float2int_rn(_fR[3] * _quant_scale); \
        }

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define ADD_BIAS_V4(_has_bias, _bias) \
        { \
            if(_has_bias && dCv4_x_valid && dCv4_y_valid) \
            { \
                int4 _bias_v4 = ((int4 *)_bias)[grp_id * num_flt_per_grp_pad_v4 + dCv4_idx]; \
	            float* _f_bias = (float *) &_bias_v4; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
	                fR[i] = fR[i] + _f_bias[i]; \
	            } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_V4(_has_relu) \
        { \
	        if(_has_relu && dCv4_x_valid && dCv4_y_valid) \
            { \
                if (_has_relu == 1) \
                {  \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	                    fR[i] = Max(fR[i], 0); \
	                } \
                } \
		    } \
        }

#if 0
#define FUSE_RELU_V4(_has_relu) \
        { \
	        if(_has_relu && dCv4_x_valid && dCv4_y_valid) \
            { \
                if (_has_relu == 1) {  \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	                    fR[i] = Max(fR[i], 0); \
	                } \
                } \
                else if( _has_relu == 2) { \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
			            fR[i] = __expf(fR[i]) / (1.f + __expf(fR[i])); \
                    } \
	            } \
		    } \
        }
#endif

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_V4(_has_clip, _clip_max, _clip_min) \
        { \
	        if(_has_clip && dCv4_x_valid && dCv4_y_valid) \
            { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
			        fR[i] = Min(fR[i], _clip_max); \
			        fR[i] = Max(fR[i], _clip_min); \
	            } \
		    } \
        }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_V4(_has_prelu, _prelu, _leaky) \
        { \
            if (_has_prelu && dCv4_x_valid && dCv4_y_valid) \
            { \
                if (_has_prelu == 1) \
                { \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	    		        if(fR[i] < _FLOAT_ZERO_) \
                            fR[i] *= _leaky; \
                    } \
                } \
                \
                else if (_has_prelu == 2) \
                { \
                    int4 _scale_v4  = ((int4 *)_prelu)[grp_id * num_flt_per_grp_pad_v4 + dCv4_idx]; \
                    float * _f_scale = (float *) &_scale_v4; \
                    \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	    		        if(fR[i] < _FLOAT_ZERO_) \
                            fR[i] *= _f_scale[i]; \
                    } \
                } \
                \
                else if (_has_prelu == 3) \
                { \
                    int4 _scale_v4  = ((int4 *)_prelu)[dCv4_off]; \
                    float * _f_scale = (float *) &_scale_v4; \
                    \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	    		        if(fR[i] < _FLOAT_ZERO_) \
                            fR[i] *= _f_scale[i]; \
                    } \
                } \
            } \
        }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_V4(_has_elt, _pre_data) \
        { \
	        if( _has_elt && dCv4_x_valid && dCv4_y_valid) \
            { \
	            int  elt_v4 = ((int *) _pre_data)[dCv4_off]; \
	            int8_t *elt_v1 = (int8_t *) &elt_v4; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
			        fR[i] += (int)elt_v1[i] * pre_scale; \
	            } \
	        } \
        }

#define SET_CONCAT_OFF_V4(_has_concat, _concat_v4_off) \
        { \
            if (_has_concat && dCv4_x_valid && dCv4_y_valid) \
            { \
                dCv4_off = concat_offset_v4 + dCv4_idy * concat_stride_v4 + dCv4_base + dCv4_idx; \
            } \
        }
        
//////////////////////////////////////////////////////
// jit macros
//////////////////////////////////////////////////////

#define JIT_FUSE_RELU_V4() \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < _INT4_TO_4INT_; i++) \
            { \
                fR[i] = Max(fR[i], 0); \
            } \
        }

#define JIT_FUSE_SIGMOID_V4() \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < _INT4_TO_4INT_; i++) \
            { \
              fR[i] = __expf(fR[i]) / (1.f + __expf(fR[i])); \
            } \
        }

#define JIT_FUSE_CLIP_V4(_clip_max, _clip_min) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < _INT4_TO_4INT_; i++) \
            { \
                fR[i] = Min(fR[i], _clip_max); \
                fR[i] = Max(fR[i], _clip_min); \
            } \
        }

#define JIT_FUSE_PRELU_V4(_has_prelu, _prelu) \
        { \
            if(_has_prelu == 2) \
            { \
                int4 _scale_v4  = ((int4 *)_prelu)[grp_id * num_flt_per_grp_pad_v4 + dCv4_idx]; \
	            float *_f_scale  = (float *) &_scale_v4; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
                    if(fR[i] < _FLOAT_ZERO_) \
                        fR[i] *= _f_scale[i]; \
	            } \
            } \
            \
	        else if(_has_prelu == 3) \
            { \
                int4 _scale_v4 = ((int4  *) _prelu) [dCv4_off]; \
	            float* _f_scale  = (float *) &_scale_v4; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
                    if(fR[i] < _FLOAT_ZERO_) \
                        fR[i] *= _f_scale[i]; \
	            } \
	        } \
        }

#define JIT_FUSE_LEAKY_V4(_leaky) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < _INT4_TO_4INT_; i++) \
            { \
                if(fR[i] < _FLOAT_ZERO_) \
                    fR[i] *= _leaky; \
            } \
        }

#define JIT_FUSE_ELT_V4(_pre_data) \
        { \
            int  _elt_v4 = ((int *)   _pre_data) [dCv4_off]; \
            int8_t *_elt_v1 = (int8_t *) &_elt_v4; \
            \
            _Pragma("unroll") \
            for(int i = 0; i < _INT4_TO_4INT_; i++) \
            { \
                fR[i] += (int)_elt_v1[i] * pre_scale; \
            } \
        }

#define JIT_SET_CONCAT_OFF_V4(_concat_v4_off) \
        { \
            dCv4_off = concat_offset_v4 + dCv4_idy * concat_stride_v4 + dCv4_base + dCv4_idx; \
        }
