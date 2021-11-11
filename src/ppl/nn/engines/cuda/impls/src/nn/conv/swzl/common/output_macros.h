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

#define OUTPUT_BY_INT4(_Rv4) \
        { \
            _Pragma("unroll") for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
                if( dCv4_y_valid && dCv4_x_valid[i] ) \
                    dC[dCv4_base + concat_v4_off[i]] = _Rv4[i]; \
                \
                dCv4_idx[i]   +=  OUTPUT_SIZE_X_IN_THD * OUTPUT_BLKS_PER_STEP; \
                dCv4_x_valid[i]  = (dCv4_idx[i] / out_hw) < in_num; \
            } \
        }

#else

#define OUTPUT_BY_INT4(_Rv4) \
        { \
            _Pragma("unroll") for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
                if( dCv4_y_valid && dCv4_x_valid[i] ) \
                    dC[dCv4_base + dCv4_idx[i] * num_flt_per_grp_pad_v8 * num_grp] = _Rv4[i]; \
                \
                dCv4_idx[i]   +=  OUTPUT_SIZE_X_IN_THD * OUTPUT_BLKS_PER_STEP; \
                dCv4_x_valid[i]  = (dCv4_idx[i] / out_hw) < in_num; \
            } \
        }

#endif

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define ADD_BIAS_V4(_has_bias, _bias) \
        { \
            if(_has_bias && dCv4_y_valid) \
            { \
                int4 _bias_v4[OUTPUT_BLKS_PER_STEP]; \
                __half2 *_h2_bias = (__half2 *)&_bias_v4; \
                \
                _Pragma("unroll") for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    _bias_v4[i] = ((int4 *)_bias)[grp_id * num_flt_per_grp_pad_v8 + dCv4_idy]; \
                    \
                    _Pragma("unroll") for(int j = 0; j < _INT4_TO_4HALF2_; j++) \
                    { \
                        h2R[i * _INT4_TO_4HALF2_ + j] = __hadd2(_h2_bias[i * _INT4_TO_4HALF2_ + j], h2R[i * _INT4_TO_4HALF2_ + j]);   \
	                } \
                } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_V4(_has_relu)                                                                           \
        {                                                                                                 \
	        if(_has_relu & dCv4_y_valid)                                                                  \
            {                                                                                             \
                _Pragma("unroll") for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++)                           \
                {                                                                                         \
                    int *Rv1 = (int *)Rv4;                                                                \
                                                                                                          \
                    if (_has_relu == 1) {                                                                 \
                        _Pragma("unroll") for (int j = 0; j < _INT4_TO_4HALF2_; j++) {                    \
                            Rv1[i * _INT4_TO_4HALF2_ + j] = __vmaxs2(Rv1[i * _INT4_TO_4HALF2_ + j], 0);   \
                        }                                                                                 \
                    } else if (_has_relu == 2) {                                                          \
                        __half2 h2ONE((__half)1.f, (__half)1.f);                                          \
                                                                                                          \
                        _Pragma("unroll") for (int j = 0; j < _INT4_TO_4HALF2_; j++) {                    \
                            h2R[i * _INT4_TO_4HALF2_ + j] = __h2div(h2exp(h2R[i * _INT4_TO_4HALF2_ + j]), \
                                    __hadd2(h2ONE, h2exp(h2R[i * _INT4_TO_4HALF2_ + j])));                \
                        }                                                                                 \
                    }                                                                                     \
		        }                                                                                         \
		    }                                                                                             \
        }

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_V4(_has_clip, _clip_max, _clip_min) \
        { \
	        if(_has_clip & dCv4_y_valid) \
            { \
                _Pragma("unroll") for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    _Pragma("unroll") for(int j = 0; j < _INT4_TO_4INT_; j++) \
                    { \
                        h2R[i * _INT4_TO_4INT_ + j].x = __hgt(h2R[i * _INT4_TO_4INT_ + j].x, _clip_max.x) ? _clip_max.x : h2R[i * _INT4_TO_4INT_ + j].x; \
                        h2R[i * _INT4_TO_4INT_ + j].y = __hgt(h2R[i * _INT4_TO_4INT_ + j].y, _clip_max.x) ? _clip_max.y : h2R[i * _INT4_TO_4INT_ + j].y; \
                        h2R[i * _INT4_TO_4INT_ + j].x = __hlt(h2R[i * _INT4_TO_4INT_ + j].x, _clip_min.x) ? _clip_min.x : h2R[i * _INT4_TO_4INT_ + j].x; \
                        h2R[i * _INT4_TO_4INT_ + j].y = __hlt(h2R[i * _INT4_TO_4INT_ + j].y, _clip_min.x) ? _clip_min.y : h2R[i * _INT4_TO_4INT_ + j].y; \
	                } \
		        } \
		    } \
        }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_V4(_has_prelu, _prelu, _leaky) \
    { \
        if (_has_prelu && dCv4_y_valid) {                                                                                        \
            int4 _scale_v4[OUTPUT_BLKS_PER_STEP];                                                                                \
            __half *_hscale = (__half *)&_scale_v4;                                                                              \
                                                                                                                                 \
            _Pragma("unroll") for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++)                                                      \
            {                                                                                                                    \
                if (_has_prelu == 1) {                                                                                           \
                    _Pragma("unroll") for (int j = 0; j < _INT4_TO_8HALF_; j++)                                                  \
                    {                                                                                                            \
                        if (__hlt(hR[i * _INT4_TO_8HALF_ + j], 0))                                                               \
                            hR[i * _INT4_TO_8HALF_ + j] = __hmul(hR[i * _INT4_TO_8HALF_ + j], _leaky);                           \
                    }                                                                                                            \
                }                                                                                                                \
                                                                                                                                 \
                if (_has_prelu == 2) {                                                                                           \
                    _scale_v4[i] = ((int4 *)_prelu)[grp_id * num_flt_per_grp_pad_v8 + dCv4_idy];                                 \
                                                                                                                                 \
                    _Pragma("unroll") for (int j = 0; j < _INT4_TO_8HALF_; j++)                                                  \
                    {                                                                                                            \
                        if (__hlt(hR[i * _INT4_TO_8HALF_ + j], 0))                                                               \
                            hR[i * _INT4_TO_8HALF_ + j] = __hmul(hR[i * _INT4_TO_8HALF_ + j], _hscale[i * _INT4_TO_8HALF_ + j]); \
                    }                                                                                                            \
                }                                                                                                                \
                                                                                                                                 \
                if (_has_prelu == 3) {                                                                                           \
                    if(dCv4_x_valid[i])                                                                                          \
                        _scale_v4[i] = ((int4 *)_prelu)[dCv4_base + dCv4_idx[i] * num_flt_per_grp_pad_v8 * num_grp];             \
                                                                                                                                 \
                    _Pragma("unroll") for (int j = 0; j < _INT4_TO_8HALF_; j++)                                                  \
                    {                                                                                                            \
                        if (__hlt(hR[i * _INT4_TO_8HALF_ + j], 0))                                                               \
                            hR[i * _INT4_TO_8HALF_ + j] = __hmul(hR[i * _INT4_TO_8HALF_ + j], _hscale[i * _INT4_TO_8HALF_ + j]); \
                    }                                                                                                            \
                } \
            } \
        } \
    }


//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_V4(_has_elt, _pre_data) \
        { \
       	    if(_has_elt && dCv4_y_valid) \
            { \
                int4 _elt_v4[OUTPUT_BLKS_PER_STEP]; \
                __half2 *_h2_elt = (__half2 *)&_elt_v4; \
                \
                _Pragma("unroll") for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
                { \
                    if(dCv4_x_valid[i]) \
                        _elt_v4[i] = ((int4 *) _pre_data) [dCv4_base + dCv4_idx[i] * num_flt_per_grp_pad_v8 * num_grp]; \
                    \
                    _Pragma("unroll") for (int j = 0; j < _INT4_TO_4HALF2_; j++) \
                    { \
                        h2R[i * _INT4_TO_4HALF2_ + j] = __hadd2(h2R[i * _INT4_TO_4HALF2_ + j], _h2_elt[i * _INT4_TO_4HALF2_ + j]); \
	                } \
	            } \
	        } \
        }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

#define SET_CONCAT_OFF_V4(_has_concat, _concat_v4_off) \
        { \
            _Pragma("unroll") \
            for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) \
            { \
	            _concat_v4_off[i] = (_has_concat) ? dCv4_idx[i] * concat_stride_v8 + concat_offset_v8 : dCv4_idx[i] * num_flt_per_grp_pad_v8 * num_grp; \
	        } \
        }
        
