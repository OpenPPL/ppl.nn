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

#define OUTPUT_2x1_BY_INT1()                                     \
    {                                                            \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                  \
            dCv1[dCv1_idx[0] + concat_v1_off0] = C[Cv1_off + 0]; \
                                                                 \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                  \
            dCv1[dCv1_idx[0] + concat_v1_off1] = C[Cv1_off + 1]; \
    }

#define OUTPUT_2x2_BY_INT1()                                     \
    {                                                            \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                  \
            dCv1[dCv1_idx[0] + concat_v1_off0] = C[Cv1_off + 0]; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                  \
            dCv1[dCv1_idx[1] + concat_v1_off0] = C[Cv1_off + 1]; \
                                                                 \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                  \
            dCv1[dCv1_idx[0] + concat_v1_off1] = C[Cv1_off + 2]; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                  \
            dCv1[dCv1_idx[1] + concat_v1_off1] = C[Cv1_off + 3]; \
    }

#define OUTPUT_2x4_BY_INT1()                                     \
    {                                                            \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                  \
            dCv1[dCv1_idx[0] + concat_v1_off0] = C[Cv1_off + 0]; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                  \
            dCv1[dCv1_idx[1] + concat_v1_off0] = C[Cv1_off + 1]; \
        if (dCv1_y_valid[0] && dCv1_x_valid[2])                  \
            dCv1[dCv1_idx[2] + concat_v1_off0] = C[Cv1_off + 2]; \
        if (dCv1_y_valid[0] && dCv1_x_valid[3])                  \
            dCv1[dCv1_idx[3] + concat_v1_off0] = C[Cv1_off + 3]; \
                                                                 \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                  \
            dCv1[dCv1_idx[0] + concat_v1_off1] = C[Cv1_off + 4]; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                  \
            dCv1[dCv1_idx[1] + concat_v1_off1] = C[Cv1_off + 5]; \
        if (dCv1_y_valid[1] && dCv1_x_valid[2])                  \
            dCv1[dCv1_idx[2] + concat_v1_off1] = C[Cv1_off + 6]; \
        if (dCv1_y_valid[1] && dCv1_x_valid[3])                  \
            dCv1[dCv1_idx[3] + concat_v1_off1] = C[Cv1_off + 7]; \
    }

#else

#define OUTPUT_2x1_BY_INT1()                                               \
    {                                                                      \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                            \
            dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 0]; \
                                                                           \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                            \
            dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 1]; \
    }

#define OUTPUT_2x2_BY_INT1()                                               \
    {                                                                      \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                            \
            dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 0]; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                            \
            dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] = C[Cv1_off + 1]; \
                                                                           \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                            \
            dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 2]; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                            \
            dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] = C[Cv1_off + 3]; \
    }

#define OUTPUT_2x4_BY_INT1()                                               \
    {                                                                      \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                            \
            dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 0]; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                            \
            dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] = C[Cv1_off + 1]; \
        if (dCv1_y_valid[0] && dCv1_x_valid[2])                            \
            dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] = C[Cv1_off + 2]; \
        if (dCv1_y_valid[0] && dCv1_x_valid[3])                            \
            dCv1[dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] = C[Cv1_off + 3]; \
                                                                           \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                            \
            dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] = C[Cv1_off + 4]; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                            \
            dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] = C[Cv1_off + 5]; \
        if (dCv1_y_valid[1] && dCv1_x_valid[2])                            \
            dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]] = C[Cv1_off + 6]; \
        if (dCv1_y_valid[1] && dCv1_x_valid[3])                            \
            dCv1[dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]] = C[Cv1_off + 7]; \
    }

#endif

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define ADD_BIAS_2x1_V1(_has_bias, _bias, _step)                                               \
    {                                                                                          \
        if (_has_bias) {                                                                       \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                            \
                h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *)_bias)[dCv1_idx[0]]); \
                                                                                               \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                            \
                h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *)_bias)[dCv1_idx[0]]); \
        }                                                                                      \
    }

#define ADD_BIAS_2x2_V1(_has_bias, _bias, _step)                                               \
    {                                                                                          \
        if (_has_bias) {                                                                       \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                            \
                h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *)_bias)[dCv1_idx[0]]); \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                            \
                h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *)_bias)[dCv1_idx[1]]); \
                                                                                               \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                            \
                h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *)_bias)[dCv1_idx[0]]); \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                            \
                h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *)_bias)[dCv1_idx[1]]); \
        }                                                                                      \
    }

#define ADD_BIAS_2x4_V1(_has_bias, _bias, _step)                                               \
    {                                                                                          \
        if (_has_bias) {                                                                       \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                            \
                h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *)_bias)[dCv1_idx[0]]); \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                            \
                h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *)_bias)[dCv1_idx[1]]); \
            if (dCv1_y_valid[0] && dCv1_x_valid[2])                                            \
                h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *)_bias)[dCv1_idx[2]]); \
            if (dCv1_y_valid[0] && dCv1_x_valid[3])                                            \
                h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *)_bias)[dCv1_idx[3]]); \
                                                                                               \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                            \
                h2C[Cv1_off + 4] = __hadd2(h2C[Cv1_off + 4], ((__half2 *)_bias)[dCv1_idx[0]]); \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                            \
                h2C[Cv1_off + 5] = __hadd2(h2C[Cv1_off + 5], ((__half2 *)_bias)[dCv1_idx[1]]); \
            if (dCv1_y_valid[1] && dCv1_x_valid[2])                                            \
                h2C[Cv1_off + 6] = __hadd2(h2C[Cv1_off + 6], ((__half2 *)_bias)[dCv1_idx[2]]); \
            if (dCv1_y_valid[1] && dCv1_x_valid[3])                                            \
                h2C[Cv1_off + 7] = __hadd2(h2C[Cv1_off + 7], ((__half2 *)_bias)[dCv1_idx[3]]); \
        }                                                                                      \
    }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_2x1_V1(_has_relu)                                                                           \
    {                                                                                                         \
        if (_has_relu == 1) {                                                                                 \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                           \
                C[Cv1_off + 0] = __vmaxs2(C[Cv1_off + 0], 0);                                                 \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                           \
                C[Cv1_off + 1] = __vmaxs2(C[Cv1_off + 1], 0);                                                 \
        } else if (_has_relu == 2) {                                                                          \
            __half2 h2ONE((__half)1.f, (__half)1.f);                                                          \
                                                                                                              \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                           \
                h2C[Cv1_off + 0] = __h2div(h2exp(h2C[Cv1_off + 0]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 0]))); \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                           \
                h2C[Cv1_off + 1] = __h2div(h2exp(h2C[Cv1_off + 1]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 1]))); \
        }                                                                                                     \
    }

#define FUSE_RELU_2x2_V1(_has_relu)                                                                           \
    {                                                                                                         \
        if (_has_relu == 1) {                                                                                 \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                           \
                C[Cv1_off + 0] = __vmaxs2(C[Cv1_off + 0], 0);                                                 \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                           \
                C[Cv1_off + 1] = __vmaxs2(C[Cv1_off + 1], 0);                                                 \
                                                                                                              \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                           \
                C[Cv1_off + 2] = __vmaxs2(C[Cv1_off + 2], 0);                                                 \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                           \
                C[Cv1_off + 3] = __vmaxs2(C[Cv1_off + 3], 0);                                                 \
        } else if (_has_relu == 2) {                                                                          \
            __half2 h2ONE((__half)1.f, (__half)1.f);                                                          \
                                                                                                              \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                           \
                h2C[Cv1_off + 0] = __h2div(h2exp(h2C[Cv1_off + 0]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 0]))); \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                           \
                h2C[Cv1_off + 1] = __h2div(h2exp(h2C[Cv1_off + 1]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 1]))); \
                                                                                                              \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                           \
                h2C[Cv1_off + 2] = __h2div(h2exp(h2C[Cv1_off + 2]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 2]))); \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                           \
                h2C[Cv1_off + 3] = __h2div(h2exp(h2C[Cv1_off + 3]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 3]))); \
        }                                                                                                     \
    }

#define FUSE_RELU_2x4_V1(_has_relu)                                                                           \
    {                                                                                                         \
        if (_has_relu == 1) {                                                                                 \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                           \
                C[Cv1_off + 0] = __vmaxs2(C[Cv1_off + 0], 0);                                                 \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                           \
                C[Cv1_off + 1] = __vmaxs2(C[Cv1_off + 1], 0);                                                 \
            if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                           \
                C[Cv1_off + 2] = __vmaxs2(C[Cv1_off + 2], 0);                                                 \
            if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                           \
                C[Cv1_off + 3] = __vmaxs2(C[Cv1_off + 3], 0);                                                 \
                                                                                                              \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                           \
                C[Cv1_off + 4] = __vmaxs2(C[Cv1_off + 4], 0);                                                 \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                           \
                C[Cv1_off + 5] = __vmaxs2(C[Cv1_off + 5], 0);                                                 \
            if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                           \
                C[Cv1_off + 6] = __vmaxs2(C[Cv1_off + 6], 0);                                                 \
            if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                           \
                C[Cv1_off + 7] = __vmaxs2(C[Cv1_off + 7], 0);                                                 \
        } else if (_has_relu == 2) {                                                                          \
            __half2 h2ONE((__half)1.f, (__half)1.f);                                                          \
                                                                                                              \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                           \
                h2C[Cv1_off + 0] = __h2div(h2exp(h2C[Cv1_off + 0]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 0]))); \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                           \
                h2C[Cv1_off + 1] = __h2div(h2exp(h2C[Cv1_off + 1]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 1]))); \
            if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                           \
                h2C[Cv1_off + 2] = __h2div(h2exp(h2C[Cv1_off + 2]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 2]))); \
            if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                           \
                h2C[Cv1_off + 3] = __h2div(h2exp(h2C[Cv1_off + 3]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 3]))); \
                                                                                                              \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                           \
                h2C[Cv1_off + 4] = __h2div(h2exp(h2C[Cv1_off + 4]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 4]))); \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                           \
                h2C[Cv1_off + 5] = __h2div(h2exp(h2C[Cv1_off + 5]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 5]))); \
            if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                           \
                h2C[Cv1_off + 6] = __h2div(h2exp(h2C[Cv1_off + 6]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 6]))); \
            if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                           \
                h2C[Cv1_off + 7] = __h2div(h2exp(h2C[Cv1_off + 7]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 7]))); \
        }                                                                                                     \
    }

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_2x1_V1(_has_clip, _clip_max, _clip_min)                                                       \
    {                                                                                                           \
        if (_has_clip) {                                                                                        \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].x = __hgt(h2C[Cv1_off + 0].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 0].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].y = __hgt(h2C[Cv1_off + 0].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 0].y; \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].x = __hlt(h2C[Cv1_off + 0].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 0].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].y = __hlt(h2C[Cv1_off + 0].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 0].y; \
                                                                                                                \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 1].x = __hgt(h2C[Cv1_off + 1].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 1].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 1].y = __hgt(h2C[Cv1_off + 1].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 1].y; \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 1].x = __hlt(h2C[Cv1_off + 1].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 1].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 1].y = __hlt(h2C[Cv1_off + 1].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 1].y; \
        }                                                                                                       \
    }

#define FUSE_CLIP_2x2_V1(_has_clip, _clip_max, _clip_min)                                                       \
    {                                                                                                           \
        if (_has_clip) {                                                                                        \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].x = __hgt(h2C[Cv1_off + 0].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 0].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].y = __hgt(h2C[Cv1_off + 0].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 0].y; \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].x = __hlt(h2C[Cv1_off + 0].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 0].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].y = __hlt(h2C[Cv1_off + 0].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 0].y; \
                                                                                                                \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 1].x = __hgt(h2C[Cv1_off + 1].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 1].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 1].y = __hgt(h2C[Cv1_off + 1].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 1].y; \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 1].x = __hlt(h2C[Cv1_off + 1].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 1].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 1].y = __hlt(h2C[Cv1_off + 1].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 1].y; \
                                                                                                                \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 2].x = __hgt(h2C[Cv1_off + 2].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 2].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 2].y = __hgt(h2C[Cv1_off + 2].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 2].y; \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 2].x = __hlt(h2C[Cv1_off + 2].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 2].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 2].y = __hlt(h2C[Cv1_off + 2].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 2].y; \
                                                                                                                \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 3].x = __hgt(h2C[Cv1_off + 3].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 3].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 3].y = __hgt(h2C[Cv1_off + 3].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 3].y; \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 3].x = __hlt(h2C[Cv1_off + 3].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 3].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 3].y = __hlt(h2C[Cv1_off + 3].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 3].y; \
        }                                                                                                       \
    }

#define FUSE_CLIP_2x4_V1(_has_clip, _clip_max, _clip_min)                                                       \
    {                                                                                                           \
        if (_has_clip) {                                                                                        \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].x = __hgt(h2C[Cv1_off + 0].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 0].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].y = __hgt(h2C[Cv1_off + 0].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 0].y; \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].x = __hlt(h2C[Cv1_off + 0].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 0].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 0].y = __hlt(h2C[Cv1_off + 0].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 0].y; \
                                                                                                                \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 1].x = __hgt(h2C[Cv1_off + 1].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 1].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 1].y = __hgt(h2C[Cv1_off + 1].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 1].y; \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 1].x = __hlt(h2C[Cv1_off + 1].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 1].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 1].y = __hlt(h2C[Cv1_off + 1].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 1].y; \
                                                                                                                \
            if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                             \
                h2C[Cv1_off + 2].x = __hgt(h2C[Cv1_off + 2].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 2].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                             \
                h2C[Cv1_off + 2].y = __hgt(h2C[Cv1_off + 2].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 2].y; \
            if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                             \
                h2C[Cv1_off + 2].x = __hlt(h2C[Cv1_off + 2].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 2].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                             \
                h2C[Cv1_off + 2].y = __hlt(h2C[Cv1_off + 2].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 2].y; \
                                                                                                                \
            if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                             \
                h2C[Cv1_off + 3].x = __hgt(h2C[Cv1_off + 3].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 3].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                             \
                h2C[Cv1_off + 3].y = __hgt(h2C[Cv1_off + 3].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 3].y; \
            if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                             \
                h2C[Cv1_off + 3].x = __hlt(h2C[Cv1_off + 3].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 3].x; \
            if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                             \
                h2C[Cv1_off + 3].y = __hlt(h2C[Cv1_off + 3].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 3].y; \
                                                                                                                \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 4].x = __hgt(h2C[Cv1_off + 4].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 4].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 4].y = __hgt(h2C[Cv1_off + 4].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 4].y; \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 4].x = __hlt(h2C[Cv1_off + 4].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 4].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
                h2C[Cv1_off + 4].y = __hlt(h2C[Cv1_off + 4].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 4].y; \
                                                                                                                \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 5].x = __hgt(h2C[Cv1_off + 5].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 5].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 5].y = __hgt(h2C[Cv1_off + 5].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 5].y; \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 5].x = __hlt(h2C[Cv1_off + 5].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 5].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
                h2C[Cv1_off + 5].y = __hlt(h2C[Cv1_off + 5].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 5].y; \
                                                                                                                \
            if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                             \
                h2C[Cv1_off + 6].x = __hgt(h2C[Cv1_off + 6].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 6].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                             \
                h2C[Cv1_off + 6].y = __hgt(h2C[Cv1_off + 6].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 6].y; \
            if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                             \
                h2C[Cv1_off + 6].x = __hlt(h2C[Cv1_off + 6].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 6].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                             \
                h2C[Cv1_off + 6].y = __hlt(h2C[Cv1_off + 6].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 6].y; \
                                                                                                                \
            if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                             \
                h2C[Cv1_off + 7].x = __hgt(h2C[Cv1_off + 7].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 7].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                             \
                h2C[Cv1_off + 7].y = __hgt(h2C[Cv1_off + 7].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 7].y; \
            if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                             \
                h2C[Cv1_off + 7].x = __hlt(h2C[Cv1_off + 7].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 7].x; \
            if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                             \
                h2C[Cv1_off + 7].y = __hlt(h2C[Cv1_off + 7].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 7].y; \
        }                                                                                                       \
    }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_2x1_V1(_has_prelu, _prelu, _leaky)                                                                     \
    {                                                                                                                     \
        if (_has_prelu == 1 && dCv1_x_valid[0]) {                                                                         \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                    \
            {                                                                                                             \
                if (dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky);      \
                                                                                                                          \
                if (dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky);      \
            }                                                                                                             \
        }                                                                                                                 \
                                                                                                                          \
        if (_has_prelu == 2 && dCv1_x_valid[0]) {                                                                         \
            int _scale0_v1   = ((int *)_prelu)[dCv1_idx[0]];                                                              \
            __half *_hscale0 = (__half *)&_scale0_v1;                                                                     \
                                                                                                                          \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                    \
            {                                                                                                             \
                if (dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                                                                                                                          \
                if (dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale0[i]); \
            }                                                                                                             \
        }                                                                                                                 \
                                                                                                                          \
        if (_has_prelu == 3 && dCv1_x_valid[0]) {                                                                         \
            int _scale0_v1 = dCv1_y_valid[0] ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0;               \
            int _scale1_v1 = dCv1_y_valid[1] ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0;               \
                                                                                                                          \
            __half *_hscale0 = (__half *)&_scale0_v1;                                                                     \
            __half *_hscale1 = (__half *)&_scale1_v1;                                                                     \
                                                                                                                          \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                    \
            {                                                                                                             \
                if (dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                                                                                                                          \
                if (dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]); \
            }                                                                                                             \
        }                                                                                                                 \
    }

#define FUSE_PRELU_2x2_V1(_has_prelu, _prelu, _leaky)                                                                             \
    {                                                                                                                             \
        if (_has_prelu == 1) {                                                                                                    \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky);              \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky);              \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _leaky);              \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _leaky);              \
            }                                                                                                                     \
        }                                                                                                                         \
                                                                                                                                  \
        if (_has_prelu == 2) {                                                                                                    \
            int _scale0_v1   = dCv1_x_valid[0] ? ((int *)_prelu)[dCv1_idx[0]] : 0;                                                \
            int _scale1_v1   = dCv1_x_valid[1] ? ((int *)_prelu)[dCv1_idx[1]] : 0;                                                \
            __half *_hscale0 = (__half *)&_scale0_v1;                                                                             \
            __half *_hscale1 = (__half *)&_scale1_v1;                                                                             \
                                                                                                                                  \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]);         \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]);         \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale0[i]);         \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale1[i]);         \
            }                                                                                                                     \
        }                                                                                                                         \
                                                                                                                                  \
        if (_has_prelu == 3) {                                                                                                    \
            int _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
            int _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
                                                                                                                                  \
            int _scale10_v1 = (dCv1_y_valid[1] && dCv1_x_valid[0]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \
            int _scale11_v1 = (dCv1_y_valid[1] && dCv1_x_valid[1]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] : 0; \
                                                                                                                                  \
            __half *_hscale00 = (__half *)&_scale00_v1;                                                                           \
            __half *_hscale01 = (__half *)&_scale01_v1;                                                                           \
                                                                                                                                  \
            __half *_hscale10 = (__half *)&_scale10_v1;                                                                           \
            __half *_hscale11 = (__half *)&_scale11_v1;                                                                           \
                                                                                                                                  \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale00[i]);        \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale01[i]);        \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale10[i]);        \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale11[i]);        \
            }                                                                                                                     \
        }                                                                                                                         \
    }

#define FUSE_PRELU_2x4_V1(_has_prelu, _prelu, _leaky)                                                                             \
    {                                                                                                                             \
        if (_has_prelu == 1) {                                                                                                    \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky);              \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky);              \
                if (dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _leaky);              \
                if (dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _leaky);              \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _leaky);              \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _leaky);              \
                if (dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _leaky);              \
                if (dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _leaky);              \
            }                                                                                                                     \
        }                                                                                                                         \
                                                                                                                                  \
        if (_has_prelu == 2) {                                                                                                    \
            int _scale0_v1   = dCv1_x_valid[0] ? ((int *)_prelu)[dCv1_idx[0]] : 0;                                                \
            int _scale1_v1   = dCv1_x_valid[1] ? ((int *)_prelu)[dCv1_idx[1]] : 0;                                                \
            int _scale2_v1   = dCv1_x_valid[2] ? ((int *)_prelu)[dCv1_idx[2]] : 0;                                                \
            int _scale3_v1   = dCv1_x_valid[3] ? ((int *)_prelu)[dCv1_idx[3]] : 0;                                                \
            __half *_hscale0 = (__half *)&_scale0_v1;                                                                             \
            __half *_hscale1 = (__half *)&_scale1_v1;                                                                             \
            __half *_hscale2 = (__half *)&_scale2_v1;                                                                             \
            __half *_hscale3 = (__half *)&_scale3_v1;                                                                             \
                                                                                                                                  \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]);         \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]);         \
                if (dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale2[i]);         \
                if (dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale3[i]);         \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _hscale0[i]);         \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _hscale1[i]);         \
                if (dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _hscale2[i]);         \
                if (dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _hscale3[i]);         \
            }                                                                                                                     \
        }                                                                                                                         \
                                                                                                                                  \
        if (_has_prelu == 3) {                                                                                                    \
            int _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
            int _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
            int _scale02_v1 = (dCv1_y_valid[0] && dCv1_x_valid[2]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] : 0; \
            int _scale03_v1 = (dCv1_y_valid[0] && dCv1_x_valid[3]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] : 0; \
                                                                                                                                  \
            int _scale10_v1 = (dCv1_y_valid[1] && dCv1_x_valid[0]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \
            int _scale11_v1 = (dCv1_y_valid[1] && dCv1_x_valid[1]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] : 0; \
            int _scale12_v1 = (dCv1_y_valid[1] && dCv1_x_valid[2]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]] : 0; \
            int _scale13_v1 = (dCv1_y_valid[1] && dCv1_x_valid[3]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]] : 0; \
                                                                                                                                  \
            __half *_hscale00 = (__half *)&_scale00_v1;                                                                           \
            __half *_hscale01 = (__half *)&_scale01_v1;                                                                           \
            __half *_hscale02 = (__half *)&_scale02_v1;                                                                           \
            __half *_hscale03 = (__half *)&_scale03_v1;                                                                           \
                                                                                                                                  \
            __half *_hscale10 = (__half *)&_scale10_v1;                                                                           \
            __half *_hscale11 = (__half *)&_scale11_v1;                                                                           \
            __half *_hscale12 = (__half *)&_scale12_v1;                                                                           \
            __half *_hscale13 = (__half *)&_scale13_v1;                                                                           \
                                                                                                                                  \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale00[i]);        \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale01[i]);        \
                if (dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale02[i]);        \
                if (dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale03[i]);        \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _hscale10[i]);        \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _hscale11[i]);        \
                if (dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _hscale12[i]);        \
                if (dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _hscale13[i]);        \
            }                                                                                                                     \
        }                                                                                                                         \
    }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_2x1_V1(_has_elt, _pre_data)                                                                                  \
    {                                                                                                                         \
        if (_has_elt) {                                                                                                       \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                                           \
                h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \
                                                                                                                              \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                                           \
                h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \
        }                                                                                                                     \
    }

#define FUSE_ELT_2x2_V1(_has_elt, _pre_data)                                                                                  \
    {                                                                                                                         \
        if (_has_elt) {                                                                                                       \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                                           \
                h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                                           \
                h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]]); \
                                                                                                                              \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                                           \
                h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                                           \
                h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]]); \
        }                                                                                                                     \
    }

#define FUSE_ELT_2x4_V1(_has_elt, _pre_data)                                                                                  \
    {                                                                                                                         \
        if (_has_elt) {                                                                                                       \
            if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                                           \
                h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \
            if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                                           \
                h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]]); \
            if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                                           \
                h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]]); \
            if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                                           \
                h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]]); \
                                                                                                                              \
            if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                                           \
                h2C[Cv1_off + 4] = __hadd2(h2C[Cv1_off + 4], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \
            if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                                           \
                h2C[Cv1_off + 5] = __hadd2(h2C[Cv1_off + 5], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]]); \
            if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                                           \
                h2C[Cv1_off + 6] = __hadd2(h2C[Cv1_off + 6], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]]); \
            if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                                           \
                h2C[Cv1_off + 7] = __hadd2(h2C[Cv1_off + 7], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]]); \
        }                                                                                                                     \
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

#define JIT_FUSE_RELU_2x1_V1()                            \
    {                                                     \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])           \
            C[Cv1_off + 0] = __vmaxs2(C[Cv1_off + 0], 0); \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])           \
            C[Cv1_off + 1] = __vmaxs2(C[Cv1_off + 1], 0); \
    }

#define JIT_FUSE_SIGMOID_2x1_V1()                                                                         \
    \                                                                                                     \
    {                                                                                                     \
        __half2 h2ONE((__half)1.f, (__half)1.f);                                                          \
                                                                                                          \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                           \
            h2C[Cv1_off + 0] = __h2div(h2exp(h2C[Cv1_off + 0]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 0]))); \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                           \
            h2C[Cv1_off + 1] = __h2div(h2exp(h2C[Cv1_off + 1]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 1]))); \
    }                                                                                                     \
    }

#define JIT_FUSE_RELU_2x2_V1()                            \
    {                                                     \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])           \
            C[Cv1_off + 0] = __vmaxs2(C[Cv1_off + 0], 0); \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])           \
            C[Cv1_off + 1] = __vmaxs2(C[Cv1_off + 1], 0); \
                                                          \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])           \
            C[Cv1_off + 2] = __vmaxs2(C[Cv1_off + 2], 0); \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])           \
            C[Cv1_off + 3] = __vmaxs2(C[Cv1_off + 3], 0); \
    }

#define JIT_FUSE_SIGMOID_2x2_V1()                                                                         \
    {                                                                                                     \
        __half2 h2ONE((__half)1.f, (__half)1.f);                                                          \
                                                                                                          \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                           \
            h2C[Cv1_off + 0] = __h2div(h2exp(h2C[Cv1_off + 0]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 0]))); \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                           \
            h2C[Cv1_off + 1] = __h2div(h2exp(h2C[Cv1_off + 1]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 1]))); \
                                                                                                          \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                           \
            h2C[Cv1_off + 2] = __h2div(h2exp(h2C[Cv1_off + 2]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 2]))); \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                           \
            h2C[Cv1_off + 3] = __h2div(h2exp(h2C[Cv1_off + 3]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 3]))); \
    }

#define JIT_FUSE_RELU_2x4_V1()                            \
    {                                                     \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])           \
            C[Cv1_off + 0] = __vmaxs2(C[Cv1_off + 0], 0); \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])           \
            C[Cv1_off + 1] = __vmaxs2(C[Cv1_off + 1], 0); \
        if (dCv1_y_valid[0] && dCv1_x_valid[2])           \
            C[Cv1_off + 2] = __vmaxs2(C[Cv1_off + 2], 0); \
        if (dCv1_y_valid[0] && dCv1_x_valid[3])           \
            C[Cv1_off + 3] = __vmaxs2(C[Cv1_off + 3], 0); \
                                                          \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])           \
            C[Cv1_off + 4] = __vmaxs2(C[Cv1_off + 4], 0); \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])           \
            C[Cv1_off + 5] = __vmaxs2(C[Cv1_off + 5], 0); \
        if (dCv1_y_valid[1] && dCv1_x_valid[2])           \
            C[Cv1_off + 6] = __vmaxs2(C[Cv1_off + 6], 0); \
        if (dCv1_y_valid[1] && dCv1_x_valid[3])           \
            C[Cv1_off + 7] = __vmaxs2(C[Cv1_off + 7], 0); \
    }

#define JIT_FUSE_SIGMOID_2x4_V1()                                                                         \
    {                                                                                                     \
        __half2 h2ONE((__half)1.f, (__half)1.f);                                                          \
                                                                                                          \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                           \
            h2C[Cv1_off + 0] = __h2div(h2exp(h2C[Cv1_off + 0]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 0]))); \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                           \
            h2C[Cv1_off + 1] = __h2div(h2exp(h2C[Cv1_off + 1]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 1]))); \
        if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                           \
            h2C[Cv1_off + 2] = __h2div(h2exp(h2C[Cv1_off + 2]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 2]))); \
        if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                           \
            h2C[Cv1_off + 3] = __h2div(h2exp(h2C[Cv1_off + 3]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 3]))); \
                                                                                                          \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                           \
            h2C[Cv1_off + 4] = __h2div(h2exp(h2C[Cv1_off + 4]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 4]))); \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                           \
            h2C[Cv1_off + 5] = __h2div(h2exp(h2C[Cv1_off + 5]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 5]))); \
        if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                           \
            h2C[Cv1_off + 6] = __h2div(h2exp(h2C[Cv1_off + 6]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 6]))); \
        if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                           \
            h2C[Cv1_off + 7] = __h2div(h2exp(h2C[Cv1_off + 7]), __hadd2(h2ONE, h2exp(h2C[Cv1_off + 7]))); \
    }

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define JIT_FUSE_CLIP_2x1_V1(_clip_max, _clip_min)                                                          \
    {                                                                                                       \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].x = __hgt(h2C[Cv1_off + 0].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 0].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].y = __hgt(h2C[Cv1_off + 0].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 0].y; \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].x = __hlt(h2C[Cv1_off + 0].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 0].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].y = __hlt(h2C[Cv1_off + 0].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 0].y; \
                                                                                                            \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 1].x = __hgt(h2C[Cv1_off + 1].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 1].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 1].y = __hgt(h2C[Cv1_off + 1].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 1].y; \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 1].x = __hlt(h2C[Cv1_off + 1].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 1].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 1].y = __hlt(h2C[Cv1_off + 1].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 1].y; \
    }

#define JIT_FUSE_CLIP_2x2_V1(_clip_max, _clip_min)                                                          \
    {                                                                                                       \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].x = __hgt(h2C[Cv1_off + 0].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 0].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].y = __hgt(h2C[Cv1_off + 0].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 0].y; \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].x = __hlt(h2C[Cv1_off + 0].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 0].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].y = __hlt(h2C[Cv1_off + 0].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 0].y; \
                                                                                                            \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 1].x = __hgt(h2C[Cv1_off + 1].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 1].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 1].y = __hgt(h2C[Cv1_off + 1].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 1].y; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 1].x = __hlt(h2C[Cv1_off + 1].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 1].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 1].y = __hlt(h2C[Cv1_off + 1].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 1].y; \
                                                                                                            \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 2].x = __hgt(h2C[Cv1_off + 2].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 2].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 2].y = __hgt(h2C[Cv1_off + 2].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 2].y; \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 2].x = __hlt(h2C[Cv1_off + 2].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 2].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 2].y = __hlt(h2C[Cv1_off + 2].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 2].y; \
                                                                                                            \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 3].x = __hgt(h2C[Cv1_off + 3].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 3].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 3].y = __hgt(h2C[Cv1_off + 3].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 3].y; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 3].x = __hlt(h2C[Cv1_off + 3].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 3].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 3].y = __hlt(h2C[Cv1_off + 3].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 3].y; \
    }

#define JIT_FUSE_CLIP_2x4_V1(_clip_max, _clip_min)                                                          \
    {                                                                                                       \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].x = __hgt(h2C[Cv1_off + 0].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 0].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].y = __hgt(h2C[Cv1_off + 0].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 0].y; \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].x = __hlt(h2C[Cv1_off + 0].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 0].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 0].y = __hlt(h2C[Cv1_off + 0].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 0].y; \
                                                                                                            \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 1].x = __hgt(h2C[Cv1_off + 1].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 1].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 1].y = __hgt(h2C[Cv1_off + 1].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 1].y; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 1].x = __hlt(h2C[Cv1_off + 1].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 1].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 1].y = __hlt(h2C[Cv1_off + 1].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 1].y; \
                                                                                                            \
        if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                             \
            h2C[Cv1_off + 2].x = __hgt(h2C[Cv1_off + 2].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 2].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                             \
            h2C[Cv1_off + 2].y = __hgt(h2C[Cv1_off + 2].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 2].y; \
        if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                             \
            h2C[Cv1_off + 2].x = __hlt(h2C[Cv1_off + 2].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 2].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                             \
            h2C[Cv1_off + 2].y = __hlt(h2C[Cv1_off + 2].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 2].y; \
                                                                                                            \
        if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                             \
            h2C[Cv1_off + 3].x = __hgt(h2C[Cv1_off + 3].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 3].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                             \
            h2C[Cv1_off + 3].y = __hgt(h2C[Cv1_off + 3].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 3].y; \
        if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                             \
            h2C[Cv1_off + 3].x = __hlt(h2C[Cv1_off + 3].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 3].x; \
        if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                             \
            h2C[Cv1_off + 3].y = __hlt(h2C[Cv1_off + 3].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 3].y; \
                                                                                                            \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 4].x = __hgt(h2C[Cv1_off + 4].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 4].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 4].y = __hgt(h2C[Cv1_off + 4].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 4].y; \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 4].x = __hlt(h2C[Cv1_off + 4].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 4].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                             \
            h2C[Cv1_off + 4].y = __hlt(h2C[Cv1_off + 4].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 4].y; \
                                                                                                            \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 5].x = __hgt(h2C[Cv1_off + 5].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 5].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 5].y = __hgt(h2C[Cv1_off + 5].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 5].y; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 5].x = __hlt(h2C[Cv1_off + 5].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 5].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                             \
            h2C[Cv1_off + 5].y = __hlt(h2C[Cv1_off + 5].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 5].y; \
                                                                                                            \
        if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                             \
            h2C[Cv1_off + 6].x = __hgt(h2C[Cv1_off + 6].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 6].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                             \
            h2C[Cv1_off + 6].y = __hgt(h2C[Cv1_off + 6].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 6].y; \
        if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                             \
            h2C[Cv1_off + 6].x = __hlt(h2C[Cv1_off + 6].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 6].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                             \
            h2C[Cv1_off + 6].y = __hlt(h2C[Cv1_off + 6].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 6].y; \
                                                                                                            \
        if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                             \
            h2C[Cv1_off + 7].x = __hgt(h2C[Cv1_off + 7].x, _clip_max.x) ? _clip_max.x : h2C[Cv1_off + 7].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                             \
            h2C[Cv1_off + 7].y = __hgt(h2C[Cv1_off + 7].y, _clip_max.x) ? _clip_max.y : h2C[Cv1_off + 7].y; \
        if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                             \
            h2C[Cv1_off + 7].x = __hlt(h2C[Cv1_off + 7].x, _clip_min.x) ? _clip_min.x : h2C[Cv1_off + 7].x; \
        if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                             \
            h2C[Cv1_off + 7].y = __hlt(h2C[Cv1_off + 7].y, _clip_min.x) ? _clip_min.y : h2C[Cv1_off + 7].y; \
    }

//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define JIT_FUSE_LEAKY_2x1_V1(_leaky)                                                                                \
    {                                                                                                                \
        if (dCv1_x_valid[0])                                                                                         \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                               \
            {                                                                                                        \
                if (dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                             \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky); \
                                                                                                                     \
                if (dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                             \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky); \
            }                                                                                                        \
    }                                                                                                                \
    }
#define JIT_FUSE_PRELU_2x1_V1(_has_prelu, _prelu)                                                                         \
    {                                                                                                                     \
        if (_has_prelu == 2 && dCv1_x_valid[0]) {                                                                         \
            int _scale0_v1   = ((int *)_prelu)[dCv1_idx[0]];                                                              \
            __half *_hscale0 = (__half *)&_scale0_v1;                                                                     \
                                                                                                                          \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                    \
            {                                                                                                             \
                if (dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                                                                                                                          \
                if (dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale0[i]); \
            }                                                                                                             \
        }                                                                                                                 \
                                                                                                                          \
        if (_has_prelu == 3 && dCv1_x_valid[0]) {                                                                         \
            int _scale0_v1 = dCv1_y_valid[0] ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0;               \
            int _scale1_v1 = dCv1_y_valid[1] ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0;               \
                                                                                                                          \
            __half *_hscale0 = (__half *)&_scale0_v1;                                                                     \
            __half *_hscale1 = (__half *)&_scale1_v1;                                                                     \
                                                                                                                          \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                    \
            {                                                                                                             \
                if (dCv1_y_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]); \
                                                                                                                          \
                if (dCv1_y_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                                  \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]); \
            }                                                                                                             \
        }                                                                                                                 \
    }

#define JIT_FUSE_LEAKY_2x2_V1(_leaky)                                                                            \
    {                                                                                                            \
        _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                               \
        {                                                                                                        \
            if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky); \
            if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky); \
                                                                                                                 \
            if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _leaky); \
            if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _leaky); \
        }                                                                                                        \
    }

#define JIT_FUSE_PRELU_2x2_V1(_has_prelu, _prelu)                                                                                 \
    {                                                                                                                             \
        if (_has_prelu == 2) {                                                                                                    \
            int _scale0_v1   = dCv1_x_valid[0] ? ((int *)_prelu)[dCv1_idx[0]] : 0;                                                \
            int _scale1_v1   = dCv1_x_valid[1] ? ((int *)_prelu)[dCv1_idx[1]] : 0;                                                \
            __half *_hscale0 = (__half *)&_scale0_v1;                                                                             \
            __half *_hscale1 = (__half *)&_scale1_v1;                                                                             \
                                                                                                                                  \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]);         \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]);         \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale0[i]);         \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale1[i]);         \
            }                                                                                                                     \
        }                                                                                                                         \
                                                                                                                                  \
        if (_has_prelu == 3) {                                                                                                    \
            int _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
            int _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
                                                                                                                                  \
            int _scale10_v1 = (dCv1_y_valid[1] && dCv1_x_valid[0]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \
            int _scale11_v1 = (dCv1_y_valid[1] && dCv1_x_valid[1]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] : 0; \
                                                                                                                                  \
            __half *_hscale00 = (__half *)&_scale00_v1;                                                                           \
            __half *_hscale01 = (__half *)&_scale01_v1;                                                                           \
                                                                                                                                  \
            __half *_hscale10 = (__half *)&_scale10_v1;                                                                           \
            __half *_hscale11 = (__half *)&_scale11_v1;                                                                           \
                                                                                                                                  \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale00[i]);        \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale01[i]);        \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale10[i]);        \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale11[i]);        \
            }                                                                                                                     \
        }                                                                                                                         \
    }

#define JIT_FUSE_LEAKY_2x4_V1(_leaky)                                                                            \
    {                                                                                                            \
        _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                               \
        {                                                                                                        \
            if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _leaky); \
            if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _leaky); \
            if (dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _leaky); \
            if (dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _leaky); \
                                                                                                                 \
            if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _leaky); \
            if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _leaky); \
            if (dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _leaky); \
            if (dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0))          \
                hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _leaky); \
        }                                                                                                        \
    }

#define JIT_FUSE_PRELU_2x4_V1(_has_prelu, _prelu)                                                                                 \
    {                                                                                                                             \
        if (_has_prelu == 2) {                                                                                                    \
            int _scale0_v1   = dCv1_x_valid[0] ? ((int *)_prelu)[dCv1_idx[0]] : 0;                                                \
            int _scale1_v1   = dCv1_x_valid[1] ? ((int *)_prelu)[dCv1_idx[1]] : 0;                                                \
            int _scale2_v1   = dCv1_x_valid[2] ? ((int *)_prelu)[dCv1_idx[2]] : 0;                                                \
            int _scale3_v1   = dCv1_x_valid[3] ? ((int *)_prelu)[dCv1_idx[3]] : 0;                                                \
            __half *_hscale0 = (__half *)&_scale0_v1;                                                                             \
            __half *_hscale1 = (__half *)&_scale1_v1;                                                                             \
            __half *_hscale2 = (__half *)&_scale2_v1;                                                                             \
            __half *_hscale3 = (__half *)&_scale3_v1;                                                                             \
                                                                                                                                  \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale0[i]);         \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale1[i]);         \
                if (dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale2[i]);         \
                if (dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale3[i]);         \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _hscale0[i]);         \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _hscale1[i]);         \
                if (dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _hscale2[i]);         \
                if (dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _hscale3[i]);         \
            }                                                                                                                     \
        }                                                                                                                         \
                                                                                                                                  \
        if (_has_prelu == 3) {                                                                                                    \
            int _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
            int _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
            int _scale02_v1 = (dCv1_y_valid[0] && dCv1_x_valid[2]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] : 0; \
            int _scale03_v1 = (dCv1_y_valid[0] && dCv1_x_valid[3]) ? ((int *)_prelu)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] : 0; \
                                                                                                                                  \
            int _scale10_v1 = (dCv1_y_valid[1] && dCv1_x_valid[0]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]] : 0; \
            int _scale11_v1 = (dCv1_y_valid[1] && dCv1_x_valid[1]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]] : 0; \
            int _scale12_v1 = (dCv1_y_valid[1] && dCv1_x_valid[2]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]] : 0; \
            int _scale13_v1 = (dCv1_y_valid[1] && dCv1_x_valid[3]) ? ((int *)_prelu)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]] : 0; \
                                                                                                                                  \
            __half *_hscale00 = (__half *)&_scale00_v1;                                                                           \
            __half *_hscale01 = (__half *)&_scale01_v1;                                                                           \
            __half *_hscale02 = (__half *)&_scale02_v1;                                                                           \
            __half *_hscale03 = (__half *)&_scale03_v1;                                                                           \
                                                                                                                                  \
            __half *_hscale10 = (__half *)&_scale10_v1;                                                                           \
            __half *_hscale11 = (__half *)&_scale11_v1;                                                                           \
            __half *_hscale12 = (__half *)&_scale12_v1;                                                                           \
            __half *_hscale13 = (__half *)&_scale13_v1;                                                                           \
                                                                                                                                  \
            _Pragma("unroll") for (int i = 0; i < _INT_TO_2HALF_; i++)                                                            \
            {                                                                                                                     \
                if (dCv1_y_valid[0] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 0) * _INT_TO_2HALF_ + i], _hscale00[i]);        \
                if (dCv1_y_valid[0] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 1) * _INT_TO_2HALF_ + i], _hscale01[i]);        \
                if (dCv1_y_valid[0] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 2) * _INT_TO_2HALF_ + i], _hscale02[i]);        \
                if (dCv1_y_valid[0] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 3) * _INT_TO_2HALF_ + i], _hscale03[i]);        \
                                                                                                                                  \
                if (dCv1_y_valid[1] && dCv1_x_valid[0] && __hlt(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 4) * _INT_TO_2HALF_ + i], _hscale10[i]);        \
                if (dCv1_y_valid[1] && dCv1_x_valid[1] && __hlt(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 5) * _INT_TO_2HALF_ + i], _hscale11[i]);        \
                if (dCv1_y_valid[1] && dCv1_x_valid[2] && __hlt(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 6) * _INT_TO_2HALF_ + i], _hscale12[i]);        \
                if (dCv1_y_valid[1] && dCv1_x_valid[3] && __hlt(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], 0))                       \
                    hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i] = __hmul(hC[(Cv1_off + 7) * _INT_TO_2HALF_ + i], _hscale13[i]);        \
            }                                                                                                                     \
        }                                                                                                                         \
    }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define JIT_FUSE_ELT_2x1_V1(_pre_data)                                                                                    \
    {                                                                                                                     \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                                           \
            h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \
                                                                                                                          \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                                           \
            h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \
    }

#define JIT_FUSE_ELT_2x2_V1(_pre_data)                                                                                    \
    {                                                                                                                     \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                                           \
            h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                                           \
            h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]]); \
                                                                                                                          \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                                           \
            h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                                           \
            h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]]); \
    }

#define JIT_FUSE_ELT_2x4_V1(_pre_data)                                                                                    \
    {                                                                                                                     \
        if (dCv1_y_valid[0] && dCv1_x_valid[0])                                                                           \
            h2C[Cv1_off + 0] = __hadd2(h2C[Cv1_off + 0], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]]); \
        if (dCv1_y_valid[0] && dCv1_x_valid[1])                                                                           \
            h2C[Cv1_off + 1] = __hadd2(h2C[Cv1_off + 1], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]]); \
        if (dCv1_y_valid[0] && dCv1_x_valid[2])                                                                           \
            h2C[Cv1_off + 2] = __hadd2(h2C[Cv1_off + 2], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]]); \
        if (dCv1_y_valid[0] && dCv1_x_valid[3])                                                                           \
            h2C[Cv1_off + 3] = __hadd2(h2C[Cv1_off + 3], ((__half2 *)_pre_data)[dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]]); \
                                                                                                                          \
        if (dCv1_y_valid[1] && dCv1_x_valid[0])                                                                           \
            h2C[Cv1_off + 4] = __hadd2(h2C[Cv1_off + 4], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[0]]); \
        if (dCv1_y_valid[1] && dCv1_x_valid[1])                                                                           \
            h2C[Cv1_off + 5] = __hadd2(h2C[Cv1_off + 5], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[1]]); \
        if (dCv1_y_valid[1] && dCv1_x_valid[2])                                                                           \
            h2C[Cv1_off + 6] = __hadd2(h2C[Cv1_off + 6], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[2]]); \
        if (dCv1_y_valid[1] && dCv1_x_valid[3])                                                                           \
            h2C[Cv1_off + 7] = __hadd2(h2C[Cv1_off + 7], ((__half2 *)_pre_data)[dCv1_idy[1] * num_flt_v2 + dCv1_idx[3]]); \
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
