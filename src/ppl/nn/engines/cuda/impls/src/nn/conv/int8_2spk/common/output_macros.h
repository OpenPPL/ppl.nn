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

#if defined(ENABLE_FUSE)

#define OUTPUT_PRC_INT8_V4(_Rv1) \
        { \
            if( dCv4_x_valid && dCv4_y_valid ) \
            { \
                ((unsigned int*)dC)[ concatV4_off + dCv4_off ] = _Rv1; \
            } \
        }

#else

#define OUTPUT_PRC_FLOAT_V4(_Rv4) \
        { \
            if( dCv4_x_valid && dCv4_y_valid ) \
            { \
                ((int4 *)dC)[ dCv4_off ] = _Rv4; \
            } \
        }
#endif

#define ADD_BIAS_V4(_fR, _has_bias, _bias) \
        { \
            if( _has_bias && dCv4_x_valid && dCv4_y_valid ) \
            { \
	            int4  _biasV4 = ((int4 *) _bias) [grp_id * num_flt_per_grp_pad_v4 + dCv4_idx]; \
	            float* _fBias = (float *) &_biasV4; \
                \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                { \
	            _fR[i] = _fR[i] + _fBias[i]; \
		} \
            } \
        }

#define FUSE_RELU_V4(_fR, _has_relu) \
        { \
	        if( _has_relu && dCv4_x_valid  && dCv4_y_valid ) \
            { \
		        if(_has_relu == 1) \
                { \
                    \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	                    _fR[i] = MAX(_fR[i], 0); \
	                } \
	            } \
                else if(_has_relu == 2) \
                { \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
			    _fR[i] = __expf(_fR[i]) / (1.f + __expf(_fR[i])); \
	            } \
	        } \
            } \
        }

#define FUSE_CLIP_V4(_fR, _has_clip, _clip_max, _clip_min) \
        { \
	        if( _has_clip && dCv4_x_valid  && dCv4_y_valid ) { \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
			_fR[i] = MIN(_fR[i], _clip_max); \
			_fR[i] = MAX(_fR[i], _clip_min); \
		    } \
	        } \
        }

#define FUSE_PRELU_V4(_fR, _has_prelu, _prelu, _leaky) \
        { \
	        if( _has_prelu && dCv4_x_valid  && dCv4_y_valid ) { \
                \
       	        if(_has_prelu == 1) \
                { \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
			    if(_fR[i] < 0.f)    _fR[i] *= _leaky; \
	                } \
	            } \
                \
	        else if(_has_prelu == 2) \
                { \
	                int4 _scale_v4 = ( (int4 *) _prelu) [grp_id * num_flt_per_grp_pad_v4 + dCv4_idx]; \
	                float *_scale  = (float *) &_scale_v4; \
                    \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	            	    if( _fR[i] < 0 )   _fR[i] *= _scale[i]; \
	                } \
	            } \
                \
	        else if(_has_prelu == 3) \
                { \
                    int4 _scale_v4 = ((int4  *) _prelu) [dCv4_off]; \
	            float* _scale  = (float *) &_scale_v4; \
                    \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	            	    if( _fR[i] < 0 )   _fR[i] *= _scale[i]; \
	            } \
	        } \
		\
	        } \
        }

#define FUSE_ELT_V4(_fR, _has_elt, _pre_data) \
        { \
	        if( _has_elt && dCv4_x_valid && dCv4_y_valid ) \
            { \
	            int  _elt_v4 = ((int *)   _pre_data) [dCv4_off]; \
	            int8_t *_elt = (int8_t *) &_elt_v4; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++){ \
			_fR[i] += (int)_elt[i] * pre_scale; \
	            } \
	        } \
        }

#define SET_CONCAT_OFF_V4(_has_concat, _concatV4_off) \
        { \
	        if( _has_concat && dCv4_x_valid && dCv4_y_valid ) \
            { \
	            dCv4_off = concat_offset_v16 + dCv4_idy * concat_stride_v16 + dCv4_base + dCv4_idx; \
	        } \
        }

#define JIT_FUSE_RELU_V4(_fR) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < _INT4_TO_4INT_; i++) \
            { \
                _fR[i] = MAX(_fR[i], 0); \
            } \
        }

#define JIT_FUSE_SIGMOID_V4(_fR) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < _INT4_TO_4INT_; i++) \
            { \
              _fR[i] = __expf(_fR[i]) / (1.f + __expf(_fR[i])); \
            } \
        }

#define JIT_FUSE_CLIP_V4(_fR, _clip_max, _clip_min) \
        { \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
			_fR[i] = MIN(_fR[i], _clip_max); \
			_fR[i] = MAX(_fR[i], _clip_min); \
		    } \
        }

#define JIT_FUSE_PRELU_V4(_fR, _has_prelu, _prelu) \
        { \
            if(_has_prelu == 2) \
                { \
	                int4 _scale_v4 = ( (int4 *) _prelu) [grp_id * num_flt_per_grp_pad_v4 + dCv4_idx]; \
	                float *_scale  = (float *) &_scale_v4; \
                    \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	            	    if( _fR[i] < 0 )   _fR[i] *= _scale[i]; \
	                } \
	            } \
                \
	        else if(_has_prelu == 3) \
                { \
                    int4 _scale_v4 = ((int4  *) _prelu) [dCv4_off]; \
	            float* _scale  = (float *) &_scale_v4; \
                    \
                    _Pragma("unroll") \
	                for(int i = 0; i < _INT4_TO_4INT_; i++) \
                    { \
	            	    if( _fR[i] < 0 )   _fR[i] *= _scale[i]; \
	            } \
	        } \
        }

#define JIT_FUSE_LEAKY_V4(_fR, _leaky) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < _INT4_TO_4INT_; i++) \
            { \
			    if(_fR[i] < 0.f)    _fR[i] *= _leaky; \
            } \
        }

#define JIT_FUSE_ELT_V4(_fR, _pre_data) \
        { \
	            int  _elt_v4 = ((int *)   _pre_data) [dCv4_off]; \
	            int8_t *_elt = (int8_t *) &_elt_v4; \
                \
                _Pragma("unroll") \
	            for(int i = 0; i < _INT4_TO_4INT_; i++){ \
			_fR[i] += (int)_elt[i] * pre_scale; \
	            } \
        }

#define JIT_SET_CONCAT_OFF_V4(_concatV4_off) \
        { \
            dCv4_off = concat_offset_v16 + dCv4_idy * concat_stride_v16 + dCv4_base + dCv4_idx; \
        }
