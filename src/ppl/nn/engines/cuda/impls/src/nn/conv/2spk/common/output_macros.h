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

#define OUTPUT_PRC_HALF(_Rv4)                      \
    {                                              \
        if (dCv4_x_valid && dCv4_y_valid) {        \
            dC[concatV4_off + dCv4_off] = _Rv4[0]; \
        }                                          \
    }

#else

#define OUTPUT_PRC_HALF(_Rv4)               \
    {                                       \
        if (dCv4_x_valid && dCv4_y_valid) { \
            dC[dCv4_off] = _Rv4[0];         \
        }                                   \
    }
#endif

#define ADD_BIAS_V4(_has_bias, _bias)                                                       \
    {                                                                                       \
        if (_has_bias && dCv4_x_valid && dCv4_y_valid) {                                    \
            int4 _biasV4     = ((int4 *)_bias)[grp_id * num_flt_per_grp_pad_v8 + dCv4_idx]; \
            __half2 *_h2Bias = (__half2 *)&_biasV4;                                         \
                                                                                            \
            _Pragma("unroll") for (int i = 0; i < _INT4_TO_4HALF2_; i++)                    \
            {                                                                               \
                h2R[i] = __hadd2(h2R[i], _h2Bias[i]);                                       \
            }                                                                               \
        }                                                                                   \
    }

#define FUSE_RELU_V4(_has_relu)                                                     \
    {                                                                               \
        if (_has_relu && dCv4_x_valid && dCv4_y_valid) {                            \
            if (_has_relu == 1) {                                                   \
                int *Rv1 = (int *)Rv4;                                              \
                                                                                    \
                _Pragma("unroll") for (int i = 0; i < _INT4_TO_4HALF2_; i++)        \
                {                                                                   \
                    Rv1[i] = __vmaxs2(Rv1[i], 0);                                   \
                }                                                                   \
            } else if (_has_relu == 2) {                                            \
                __half *hR = (__half*)Rv4;                                          \
                _Pragma("unroll") for (int i = 0; i < _INT4_TO_8HALF_; i++)         \
                {                                                                   \
                    hR[i] = __expf((float)hR[i]) / (1.f + __expf((float)hR[i]));    \
                }                                                                   \
            }                                                                       \
        }                                                                           \
    }

#define FUSE_CLIP_V4(_has_clip, _clip_max, _clip_min)                             \
    {                                                                             \
        if (_has_clip && dCv4_x_valid && dCv4_y_valid) {                          \
            _Pragma("unroll") for (int i = 0; i < _INT4_TO_4HALF2_; i++)          \
            {                                                                     \
                h2R[i].x = __hgt(h2R[i].x, _clip_max.x) ? _clip_max.x : h2R[i].x; \
                h2R[i].y = __hgt(h2R[i].y, _clip_max.x) ? _clip_max.y : h2R[i].y; \
                h2R[i].x = __hlt(h2R[i].x, _clip_min.x) ? _clip_min.x : h2R[i].x; \
                h2R[i].y = __hlt(h2R[i].y, _clip_min.x) ? _clip_min.y : h2R[i].y; \
            }                                                                     \
        }                                                                         \
    }

#define FUSE_PRELU_V4(_has_prelu, _prelu, _leaky)                                               \
    {                                                                                           \
        if (_has_prelu && dCv4_x_valid && dCv4_y_valid) {                                       \
            if (_has_prelu == 1) {                                                              \
                _Pragma("unroll") for (int i = 0; i < _INT4_TO_8HALF_; i++)                     \
                {                                                                               \
                    if (__hlt(hR[i], 0))                                                        \
                        hR[i] = __hmul(hR[i], _leaky);                                          \
                }                                                                               \
            }                                                                                   \
                                                                                                \
            if (_has_prelu == 2) {                                                              \
                int4 _scale_v4  = ((int4 *)_prelu)[grp_id * num_flt_per_grp_pad_v8 + dCv4_idx]; \
                __half *_hscale = (__half *)&_scale_v4;                                         \
                                                                                                \
                _Pragma("unroll") for (int i = 0; i < _INT4_TO_8HALF_; i++)                     \
                {                                                                               \
                    if (__hlt(hR[i], 0))                                                        \
                        hR[i] = __hmul(hR[i], _hscale[i]);                                      \
                }                                                                               \
            }                                                                                   \
                                                                                                \
            if (_has_prelu == 3) {                                                              \
                int4 _scale_v4  = ((int4 *)_prelu)[dCv4_off];                                   \
                __half *_hscale = (__half *)&_scale_v4;                                         \
                                                                                                \
                _Pragma("unroll") for (int i = 0; i < _INT4_TO_8HALF_; i++)                     \
                {                                                                               \
                    if (__hlt(hR[i], 0))                                                        \
                        hR[i] = __hmul(hR[i], _hscale[i]);                                      \
                }                                                                               \
            }                                                                                   \
        }                                                                                       \
    }

#define FUSE_ELT_V4(_has_elt, _pre_data)                                 \
    {                                                                    \
        if (_has_elt && dCv4_x_valid && dCv4_y_valid) {                  \
            int4 _elt_v4     = ((int4 *)_pre_data)[dCv4_off];            \
            __half2 *_h2_elt = (__half2 *)&_elt_v4;                      \
                                                                         \
            _Pragma("unroll") for (int i = 0; i < _INT4_TO_4HALF2_; i++) \
            {                                                            \
                h2R[i] = __hadd2(h2R[i], _h2_elt[i]);                    \
            }                                                            \
        }                                                                \
    }

#define SET_CONCAT_OFF_V4(_has_concat, _concatV4_off)                                         \
    {                                                                                         \
        if (_has_concat && dCv4_x_valid && dCv4_y_valid) {                                    \
            dCv4_off = concat_offset_v8 + dCv4_idy * concat_stride_v8 + dCv4_base + dCv4_idx; \
        }                                                                                     \
    }

#define JIT_FUSE_RELU_V4()                                           \
    {                                                                \
        int *Rv1 = (int *)Rv4;                                       \
                                                                     \
        _Pragma("unroll") for (int i = 0; i < _INT4_TO_4HALF2_; i++) \
        {                                                            \
            Rv1[i] = __vmaxs2(Rv1[i], 0);                            \
        }                                                            \
    }

#define JIT_FUSE_SIGMOID_V4()                                               \
    {                                                                       \
        __half *hR = (__half*)Rv4;                                          \
        _Pragma("unroll") for (int i = 0; i < _INT4_TO_8HALF_; i++)         \
        {                                                                   \
            hR[i] = __expf((float)hR[i]) / (1.f + __expf((float)hR[i]));    \
        }                                                                   \
    }

#define JIT_FUSE_CLIP_V4(_clip_max, _clip_min)                                \
    {                                                                         \
        _Pragma("unroll") for (int i = 0; i < _INT4_TO_4HALF2_; i++)          \
        {                                                                     \
            h2R[i].x = __hgt(h2R[i].x, _clip_max.x) ? _clip_max.x : h2R[i].x; \
            h2R[i].y = __hgt(h2R[i].y, _clip_max.x) ? _clip_max.y : h2R[i].y; \
            h2R[i].x = __hlt(h2R[i].x, _clip_min.x) ? _clip_min.x : h2R[i].x; \
            h2R[i].y = __hlt(h2R[i].y, _clip_min.x) ? _clip_min.y : h2R[i].y; \
        }                                                                     \
    }

#define JIT_FUSE_LEAKY_V4(_leaky)                                   \
    {                                                               \
        _Pragma("unroll") for (int i = 0; i < _INT4_TO_8HALF_; i++) \
        {                                                           \
            if (__hlt(hR[i], 0))                                    \
                hR[i] = __hmul(hR[i], _leaky);                      \
        }                                                           \
    }

#define JIT_FUSE_PRELU_V4(_has_prelu, _prelu)                                               \
    {                                                                                       \
        if (_has_prelu == 2) {                                                              \
            int4 _scale_v4  = ((int4 *)_prelu)[grp_id * num_flt_per_grp_pad_v8 + dCv4_idx]; \
            __half *_hscale = (__half *)&_scale_v4;                                         \
                                                                                            \
            _Pragma("unroll") for (int i = 0; i < _INT4_TO_8HALF_; i++)                     \
            {                                                                               \
                if (__hlt(hR[i], 0))                                                        \
                    hR[i] = __hmul(hR[i], _hscale[i]);                                      \
            }                                                                               \
        }                                                                                   \
                                                                                            \
        if (_has_prelu == 3) {                                                              \
            int4 _scale_v4  = ((int4 *)_prelu)[dCv4_off];                                   \
            __half *_hscale = (__half *)&_scale_v4;                                         \
                                                                                            \
            _Pragma("unroll") for (int i = 0; i < _INT4_TO_8HALF_; i++)                     \
            {                                                                               \
                if (__hlt(hR[i], 0))                                                        \
                    hR[i] = __hmul(hR[i], _hscale[i]);                                      \
            }                                                                               \
        }                                                                                   \
    }

#define JIT_FUSE_ELT_V4(_pre_data)                                   \
    {                                                                \
        int4 _elt_v4     = ((int4 *)_pre_data)[dCv4_off];            \
        __half2 *_h2_elt = (__half2 *)&_elt_v4;                      \
                                                                     \
        _Pragma("unroll") for (int i = 0; i < _INT4_TO_4HALF2_; i++) \
        {                                                            \
            h2R[i] = __hadd2(h2R[i], _h2_elt[i]);                    \
        }                                                            \
    }

#define JIT_SET_CONCAT_OFF_V4(concatV4_off)                                               \
    {                                                                                     \
        dCv4_off = concat_offset_v8 + dCv4_idy * concat_stride_v8 + dCv4_base + dCv4_idx; \
    }
