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

#define OUTPUT_1x1_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idx[0] + concat_v1_off0] = outData[0]; \
        }

#define OUTPUT_1x2_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idx[0] + concat_v1_off0] = outData[0]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv2[dCv1_idx[1] + concat_v1_off0] = outData[1]; \
        }

#define OUTPUT_1x4_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idx[0] + concat_v1_off0] = outData[0]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv2[dCv1_idx[1] + concat_v1_off0] = outData[1]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[2]) dCv2[dCv1_idx[2] + concat_v1_off0] = outData[2]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[3]) dCv2[dCv1_idx[3] + concat_v1_off0] = outData[3]; \
        }

#else

#define OUTPUT_1x1_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = outData[0]; \
        }

#define OUTPUT_1x2_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = outData[0]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] = outData[1]; \
        }

#define OUTPUT_1x4_BY_INT1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] = outData[0]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] = outData[1]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[2]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] = outData[2]; \
            if(dCv1_y_valid[0] && dCv1_x_valid[3]) dCv2[dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] = outData[3]; \
        }

#endif

//////////////////////////////////////////////////////
// bias macros
//////////////////////////////////////////////////////

#define ADD_BIAS_1x1_V1(_has_bias, _bias, _step) \
        { \
            if(_has_bias) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[0]]; \
		    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + f2Bias.x; \
		    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + f2Bias.y; \
	        } \
            } \
        }

#define ADD_BIAS_1x2_V1(_has_bias, _bias, _step) \
        { \
            if(_has_bias) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[0]]; \
		    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + f2Bias.x; \
		    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + f2Bias.y; \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[1]]; \
		    fCv2[Cv1_off + 1].x = fCv2[Cv1_off + 1].x + f2Bias.x; \
		    fCv2[Cv1_off + 1].y = fCv2[Cv1_off + 1].y + f2Bias.y; \
	        } \
            } \
        }

#define ADD_BIAS_1x4_V1(_has_bias, _bias, _step) \
        { \
            if(_has_bias) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[0]]; \
		    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + f2Bias.x; \
		    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + f2Bias.y; \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[1]]; \
		    fCv2[Cv1_off + 1].x = fCv2[Cv1_off + 1].x + f2Bias.x; \
		    fCv2[Cv1_off + 1].y = fCv2[Cv1_off + 1].y + f2Bias.y; \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[2]]; \
		    fCv2[Cv1_off + 2].x = fCv2[Cv1_off + 2].x + f2Bias.x; \
		    fCv2[Cv1_off + 2].y = fCv2[Cv1_off + 2].y + f2Bias.y; \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[3]){ \
	            float2 f2Bias = ((float2*)_bias)[dCv1_idx[3]]; \
		    fCv2[Cv1_off + 3].x = fCv2[Cv1_off + 3].x + f2Bias.x; \
		    fCv2[Cv1_off + 3].y = fCv2[Cv1_off + 3].y + f2Bias.y; \
	        } \
            } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////

#define FUSE_RELU_1x1_V1(_has_relu) \
        { \
	        if(_has_relu == 1) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, 0); \
		    fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, 0); \
	        } \
	    } \
            else if(_has_relu == 2) \
            { \
	        float ONE = 1.f; \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = __expf(fCv2[Cv1_off + 0].x) / (ONE + __expf(fCv2[Cv1_off + 0].x)); \
		    fCv2[Cv1_off + 0].y = __expf(fCv2[Cv1_off + 0].y) / (ONE + __expf(fCv2[Cv1_off + 0].y)); \
		} \
	    } \
        }

#define FUSE_RELU_1x2_V1(_has_relu) \
        { \
	        if(_has_relu == 1) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, 0); \
		    fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
		    fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, 0); \
		    fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, 0); \
	        } \
	    } \
            else if(_has_relu == 2) \
            { \
	        float ONE = 1.f; \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = __expf(fCv2[Cv1_off + 0].x) / (ONE + __expf(fCv2[Cv1_off + 0].x)); \
		    fCv2[Cv1_off + 0].y = __expf(fCv2[Cv1_off + 0].y) / (ONE + __expf(fCv2[Cv1_off + 0].y)); \
		} \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
		    fCv2[Cv1_off + 1].x = __expf(fCv2[Cv1_off + 1].x) / (ONE + __expf(fCv2[Cv1_off + 1].x)); \
		    fCv2[Cv1_off + 1].y = __expf(fCv2[Cv1_off + 1].y) / (ONE + __expf(fCv2[Cv1_off + 1].y)); \
		} \
	    } \
        } 
#define FUSE_RELU_1x4_V1(_has_relu) \
        { \
	        if(_has_relu == 1) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, 0); \
		    fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
		    fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, 0); \
		    fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]){ \
		    fCv2[Cv1_off + 2].x = Max(fCv2[Cv1_off + 2].x, 0); \
		    fCv2[Cv1_off + 2].y = Max(fCv2[Cv1_off + 2].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[3]){ \
		    fCv2[Cv1_off + 3].x = Max(fCv2[Cv1_off + 3].x, 0); \
		    fCv2[Cv1_off + 3].y = Max(fCv2[Cv1_off + 3].y, 0); \
	        } \
	    } \
            else if(_has_relu == 2) \
            { \
	        float ONE = 1.f; \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = __expf(fCv2[Cv1_off + 0].x) / (ONE + __expf(fCv2[Cv1_off + 0].x)); \
		    fCv2[Cv1_off + 0].y = __expf(fCv2[Cv1_off + 0].y) / (ONE + __expf(fCv2[Cv1_off + 0].y)); \
		} \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
		    fCv2[Cv1_off + 1].x = __expf(fCv2[Cv1_off + 1].x) / (ONE + __expf(fCv2[Cv1_off + 1].x)); \
		    fCv2[Cv1_off + 1].y = __expf(fCv2[Cv1_off + 1].y) / (ONE + __expf(fCv2[Cv1_off + 1].y)); \
		} \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]){ \
		    fCv2[Cv1_off + 2].x = __expf(fCv2[Cv1_off + 2].x) / (ONE + __expf(fCv2[Cv1_off + 2].x)); \
		    fCv2[Cv1_off + 2].y = __expf(fCv2[Cv1_off + 2].y) / (ONE + __expf(fCv2[Cv1_off + 2].y)); \
		} \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[3]){ \
		    fCv2[Cv1_off + 3].x = __expf(fCv2[Cv1_off + 3].x) / (ONE + __expf(fCv2[Cv1_off + 3].x)); \
		    fCv2[Cv1_off + 3].y = __expf(fCv2[Cv1_off + 3].y) / (ONE + __expf(fCv2[Cv1_off + 3].y)); \
		} \
	    } \
        }

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define FUSE_CLIP_1x1_V1(_has_clip, _clip_max, _clip_min) \
        { \
	    if(_has_clip) \
            { \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Min(fCv2[Cv1_off + 0].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Min(fCv2[Cv1_off + 0].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, _clip_min); \
	    } \
        }

#define FUSE_CLIP_1x2_V1(_has_clip, _clip_max, _clip_min) \
        { \
	        if(_has_clip) \
            { \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Min(fCv2[Cv1_off + 0].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Min(fCv2[Cv1_off + 0].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Min(fCv2[Cv1_off + 1].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Min(fCv2[Cv1_off + 1].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, _clip_min); \
	    } \
        }

#define FUSE_CLIP_1x4_V1(_has_clip, _clip_max, _clip_min) \
        { \
	        if(_has_clip) \
            { \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Min(fCv2[Cv1_off + 0].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Min(fCv2[Cv1_off + 0].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Min(fCv2[Cv1_off + 1].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Min(fCv2[Cv1_off + 1].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].x = Min(fCv2[Cv1_off + 2].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].y = Min(fCv2[Cv1_off + 2].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].x = Max(fCv2[Cv1_off + 2].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].y = Max(fCv2[Cv1_off + 2].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].x = Min(fCv2[Cv1_off + 3].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].y = Min(fCv2[Cv1_off + 3].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].x = Max(fCv2[Cv1_off + 3].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].y = Max(fCv2[Cv1_off + 3].y, _clip_min); \
	    } \
        } 
//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define FUSE_PRELU_1x1_V1(_has_prelu, _prelu, _leaky) \
        { \
       	    if(_has_prelu == 1 && dCv1_x_valid[0]) \
            { \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _leaky; \
            } \
            \
       	    if(_has_prelu == 2 && dCv1_x_valid[0]) \
            { \
	            int  _scale0_v1 = ((int  *) _prelu) [dCv1_idx[0]]; \
	            float* _hscale0 = (float *) &_scale0_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
	    } \
            \
       	    if(_has_prelu == 3 && dCv1_x_valid[0]) \
            { \
                int   _scale0_v1 = dCv1_y_valid[0] ? ((int *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
                float *_hscale0  = (float *) &_scale0_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
	    } \
        }

#define FUSE_PRELU_1x2_V1(_has_prelu, _prelu, _leaky) \
        { \
       	    if(_has_prelu == 1) \
            { \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _leaky; \
	    } \
            \
       	    if(_has_prelu == 2) \
            { \
	            int     _scale0_v1 = dCv1_x_valid[0] ? ((int  *) _prelu) [dCv1_idx[0]] : 0; \
	            int     _scale1_v1 = dCv1_x_valid[1] ? ((int  *) _prelu) [dCv1_idx[1]] : 0; \
	            float * _hscale0  = (float *) &_scale0_v1; \
	            float * _hscale1  = (float *) &_scale1_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _hscale1[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _hscale1[1]; \
	    } \
            \
       	    if(_has_prelu == 3) \
            { \
                int      _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
                int      _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
                \
	            float * _hscale0  = (float *) &_scale00_v1; \
	            float * _hscale1  = (float *) &_scale01_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _hscale1[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _hscale1[1]; \
	    } \
        }

#define FUSE_PRELU_1x4_V1(_has_prelu, _prelu, _leaky) \
        { \
       	    if(_has_prelu == 1) \
            { \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 2].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 2].y * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 3].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 3].y * _leaky; \
	    } \
            \
       	    if(_has_prelu == 2) \
            { \
	            int      _scale0_v1 = dCv1_x_valid[0] ? ((int  *) _prelu) [dCv1_idx[0]] : 0; \
	            int      _scale1_v1 = dCv1_x_valid[1] ? ((int  *) _prelu) [dCv1_idx[1]] : 0; \
	            int      _scale2_v1 = dCv1_x_valid[2] ? ((int  *) _prelu) [dCv1_idx[2]] : 0; \
	            int      _scale3_v1 = dCv1_x_valid[3] ? ((int  *) _prelu) [dCv1_idx[3]] : 0; \
	            float * _hscale0  = (float *) &_scale0_v1; \
	            float * _hscale1  = (float *) &_scale1_v1; \
	            float * _hscale2  = (float *) &_scale2_v1; \
	            float * _hscale3  = (float *) &_scale3_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _hscale1[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _hscale1[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 2].x * _hscale2[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 2].y * _hscale2[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 3].x * _hscale3[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 3].y * _hscale3[1]; \
	    } \
            \
       	    if(_has_prelu == 3) \
            { \
                int      _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
                int      _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
                int      _scale02_v1 = (dCv1_y_valid[0] && dCv1_x_valid[2]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] : 0; \
                int      _scale03_v1 = (dCv1_y_valid[0] && dCv1_x_valid[3]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] : 0; \
                \
	            float * _hscale0  = (float *) &_scale00_v1; \
	            float * _hscale1  = (float *) &_scale01_v1; \
	            float * _hscale2  = (float *) &_scale02_v1; \
	            float * _hscale3  = (float *) &_scale03_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _hscale1[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _hscale1[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 2].x * _hscale2[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 2].y * _hscale2[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 3].x * _hscale3[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 3].y * _hscale3[1]; \
	    } \
        }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define FUSE_ELT_1x1_V1(_has_elt, _pre_data) \
        { \
	        if(_has_elt) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) { \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
	    } \
        }

#define FUSE_ELT_1x2_V1(_has_elt, _pre_data) \
        { \
	        if(_has_elt) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) { \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) { \
                    fCv2[Cv1_off + 1].x = fCv2[Cv1_off + 1].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[1])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 1].y = fCv2[Cv1_off + 1].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[1])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
	    } \
        }

#define FUSE_ELT_1x4_V1(_has_elt, _pre_data) \
        { \
	        if(_has_elt) \
            { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) { \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) { \
                    fCv2[Cv1_off + 1].x = fCv2[Cv1_off + 1].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[1])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 1].y = fCv2[Cv1_off + 1].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[1])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]) { \
                    fCv2[Cv1_off + 2].x = fCv2[Cv1_off + 2].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[2])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 2].y = fCv2[Cv1_off + 2].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[2])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) { \
                    fCv2[Cv1_off + 3].x = fCv2[Cv1_off + 3].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[3])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 3].y = fCv2[Cv1_off + 3].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[3])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
	    } \
        }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

//FIXME _INT4_TO_4HALF2_
#define SET_CONCAT_OFF_V1(_has_concat, _concat_v1_off0) \
        { \
                _concat_v1_off0 = dCv1_idy[0] * num_flt_v2; \
	        if(_has_concat) \
            { \
                if(dCv1_y_valid[0]) _concat_v1_off0 = concat_offset_v16 * _INT4_TO_8HALF_ + dCv1_idy[0] * concat_stride_v16 * _INT4_TO_8HALF_; \
	    } \
        }

//////////////////////////////////////////////////////
// relu macros
//////////////////////////////////////////////////////


#define JIT_FUSE_RELU_1x1_V1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
                fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, 0); \
                fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, 0); \
            } \
        }

#define JIT_FUSE_SIGMOID_1x1_V1() \
        { \
            float ONE = 1.f; \
                \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
                fCv2[Cv1_off + 0].x = __expf(fCv2[Cv1_off + 0].x) / (ONE + __expf(fCv2[Cv1_off + 0].x)); \
                fCv2[Cv1_off + 0].y = __expf(fCv2[Cv1_off + 0].y) / (ONE + __expf(fCv2[Cv1_off + 0].y)); \
            } \
        }

#define JIT_FUSE_RELU_1x2_V1() \
        { \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
                fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, 0); \
                fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, 0); \
            } \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
                fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, 0); \
                fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, 0); \
	        } \
        } 


#define JIT_FUSE_SIGMOID_1x2_V1() \
        { \
	        float ONE = 1.f; \
                \
            if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
                fCv2[Cv1_off + 0].x = __expf(fCv2[Cv1_off + 0].x) / (ONE + __expf(fCv2[Cv1_off + 0].x)); \
                fCv2[Cv1_off + 0].y = __expf(fCv2[Cv1_off + 0].y) / (ONE + __expf(fCv2[Cv1_off + 0].y)); \
            } \
            if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
                fCv2[Cv1_off + 1].x = __expf(fCv2[Cv1_off + 1].x) / (ONE + __expf(fCv2[Cv1_off + 1].x)); \
                fCv2[Cv1_off + 1].y = __expf(fCv2[Cv1_off + 1].y) / (ONE + __expf(fCv2[Cv1_off + 1].y)); \
            } \
        } 


#define JIT_FUSE_RELU_1x4_V1() \
        { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
		    fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, 0); \
		    fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
		    fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, 0); \
		    fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]){ \
		    fCv2[Cv1_off + 2].x = Max(fCv2[Cv1_off + 2].x, 0); \
		    fCv2[Cv1_off + 2].y = Max(fCv2[Cv1_off + 2].y, 0); \
	        } \
                if(dCv1_y_valid[0] && dCv1_x_valid[3]){ \
		    fCv2[Cv1_off + 3].x = Max(fCv2[Cv1_off + 3].x, 0); \
		    fCv2[Cv1_off + 3].y = Max(fCv2[Cv1_off + 3].y, 0); \
	        } \
        }


#define JIT_FUSE_SIGMOID_1x4_V1() \
        { \
	        float ONE = 1.f; \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]){ \
                fCv2[Cv1_off + 0].x = __expf(fCv2[Cv1_off + 0].x) / (ONE + __expf(fCv2[Cv1_off + 0].x)); \
                fCv2[Cv1_off + 0].y = __expf(fCv2[Cv1_off + 0].y) / (ONE + __expf(fCv2[Cv1_off + 0].y)); \
            } \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]){ \
                fCv2[Cv1_off + 1].x = __expf(fCv2[Cv1_off + 1].x) / (ONE + __expf(fCv2[Cv1_off + 1].x)); \
                fCv2[Cv1_off + 1].y = __expf(fCv2[Cv1_off + 1].y) / (ONE + __expf(fCv2[Cv1_off + 1].y)); \
            } \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]){ \
                fCv2[Cv1_off + 2].x = __expf(fCv2[Cv1_off + 2].x) / (ONE + __expf(fCv2[Cv1_off + 2].x)); \
                fCv2[Cv1_off + 2].y = __expf(fCv2[Cv1_off + 2].y) / (ONE + __expf(fCv2[Cv1_off + 2].y)); \
            } \
                \
                if(dCv1_y_valid[0] && dCv1_x_valid[3]){ \
                fCv2[Cv1_off + 3].x = __expf(fCv2[Cv1_off + 3].x) / (ONE + __expf(fCv2[Cv1_off + 3].x)); \
                fCv2[Cv1_off + 3].y = __expf(fCv2[Cv1_off + 3].y) / (ONE + __expf(fCv2[Cv1_off + 3].y)); \
            } \
        }

//////////////////////////////////////////////////////
// clip macros
//////////////////////////////////////////////////////

#define JIT_FUSE_CLIP_1x1_V1(_clip_max, _clip_min) \
        { \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Min(fCv2[Cv1_off + 0].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Min(fCv2[Cv1_off + 0].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, _clip_min); \
        }

#define JIT_FUSE_CLIP_1x2_V1(_clip_max, _clip_min) \
        { \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Min(fCv2[Cv1_off + 0].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Min(fCv2[Cv1_off + 0].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Min(fCv2[Cv1_off + 1].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Min(fCv2[Cv1_off + 1].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, _clip_min); \
        }

#define JIT_FUSE_CLIP_1x4_V1(_clip_max, _clip_min) \
        { \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Min(fCv2[Cv1_off + 0].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Min(fCv2[Cv1_off + 0].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].x = Max(fCv2[Cv1_off + 0].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[0]) fCv2[Cv1_off + 0].y = Max(fCv2[Cv1_off + 0].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Min(fCv2[Cv1_off + 1].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Min(fCv2[Cv1_off + 1].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].x = Max(fCv2[Cv1_off + 1].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[1]) fCv2[Cv1_off + 1].y = Max(fCv2[Cv1_off + 1].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].x = Min(fCv2[Cv1_off + 2].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].y = Min(fCv2[Cv1_off + 2].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].x = Max(fCv2[Cv1_off + 2].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[2]) fCv2[Cv1_off + 2].y = Max(fCv2[Cv1_off + 2].y, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].x = Min(fCv2[Cv1_off + 3].x, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].y = Min(fCv2[Cv1_off + 3].y, _clip_max); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].x = Max(fCv2[Cv1_off + 3].x, _clip_min); \
	        if(dCv1_y_valid[0] && dCv1_x_valid[3]) fCv2[Cv1_off + 3].y = Max(fCv2[Cv1_off + 3].y, _clip_min); \
        } 
//////////////////////////////////////////////////////
// prelu macros
//////////////////////////////////////////////////////

#define JIT_FUSE_LEAKY_1x1_V1(_leaky) \
        { \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _leaky; \
        }

#define JIT_FUSE_PRELU_1x1_V1(_has_prelu, _prelu) \
        { \
       	    if(_has_prelu == 2 && dCv1_x_valid[0]) \
            { \
	            int  _scale0_v1 = ((int  *) _prelu) [dCv1_idx[0]]; \
	            float* _hscale0 = (float *) &_scale0_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
	    } \
            \
       	    if(_has_prelu == 3 && dCv1_x_valid[0]) \
            { \
                int   _scale0_v1 = dCv1_y_valid[0] ? ((int *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
                float *_hscale0  = (float *) &_scale0_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
	    } \
        }


#define JIT_FUSE_LEAKY_1x2_V1(_leaky) \
        { \
            if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _leaky; \
            if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _leaky; \
            if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _leaky; \
            if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _leaky; \
        }

#define JIT_FUSE_PRELU_1x2_V1(_has_prelu, _prelu) \
        { \
       	    if(_has_prelu == 2) \
            { \
	            int     _scale0_v1 = dCv1_x_valid[0] ? ((int  *) _prelu) [dCv1_idx[0]] : 0; \
	            int     _scale1_v1 = dCv1_x_valid[1] ? ((int  *) _prelu) [dCv1_idx[1]] : 0; \
	            float * _hscale0  = (float *) &_scale0_v1; \
	            float * _hscale1  = (float *) &_scale1_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _hscale1[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _hscale1[1]; \
	    } \
            \
       	    if(_has_prelu == 3) \
            { \
                int      _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
                int      _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
                \
	            float * _hscale0  = (float *) &_scale00_v1; \
	            float * _hscale1  = (float *) &_scale01_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _hscale1[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _hscale1[1]; \
	    } \
        }


#define JIT_FUSE_LEAKY_1x4_V1(_leaky) \
        { \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 2].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 2].y * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 3].x * _leaky; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 3].y * _leaky; \
        }

#define JIT_FUSE_PRELU_1x4_V1(_has_prelu, _prelu) \
        {\
       	    if(_has_prelu == 2) \
            { \
	            int      _scale0_v1 = dCv1_x_valid[0] ? ((int  *) _prelu) [dCv1_idx[0]] : 0; \
	            int      _scale1_v1 = dCv1_x_valid[1] ? ((int  *) _prelu) [dCv1_idx[1]] : 0; \
	            int      _scale2_v1 = dCv1_x_valid[2] ? ((int  *) _prelu) [dCv1_idx[2]] : 0; \
	            int      _scale3_v1 = dCv1_x_valid[3] ? ((int  *) _prelu) [dCv1_idx[3]] : 0; \
	            float * _hscale0  = (float *) &_scale0_v1; \
	            float * _hscale1  = (float *) &_scale1_v1; \
	            float * _hscale2  = (float *) &_scale2_v1; \
	            float * _hscale3  = (float *) &_scale3_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _hscale1[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _hscale1[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 2].x * _hscale2[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 2].y * _hscale2[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 3].x * _hscale3[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 3].y * _hscale3[1]; \
	        } \
            \
       	    if(_has_prelu == 3) \
            { \
                int      _scale00_v1 = (dCv1_y_valid[0] && dCv1_x_valid[0]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[0]] : 0; \
                int      _scale01_v1 = (dCv1_y_valid[0] && dCv1_x_valid[1]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[1]] : 0; \
                int      _scale02_v1 = (dCv1_y_valid[0] && dCv1_x_valid[2]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[2]] : 0; \
                int      _scale03_v1 = (dCv1_y_valid[0] && dCv1_x_valid[3]) ? ((int   *) _prelu) [dCv1_idy[0] * num_flt_v2 + dCv1_idx[3]] : 0; \
                \
	            float * _hscale0  = (float *) &_scale00_v1; \
	            float * _hscale1  = (float *) &_scale01_v1; \
	            float * _hscale2  = (float *) &_scale02_v1; \
	            float * _hscale3  = (float *) &_scale03_v1; \
                \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x * _hscale0[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 0].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y * _hscale0[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 1].x * _hscale1[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 1].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 1].y * _hscale1[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 2].x * _hscale2[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 2].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 2].y * _hscale2[1]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].x < 0) \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 3].x * _hscale3[0]; \
                if(dCv1_y_valid[0] && fCv2[Cv1_off + 3].y < 0) \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 3].y * _hscale3[1]; \
	        } \
        }

//////////////////////////////////////////////////////
// eltwise macros
//////////////////////////////////////////////////////

#define JIT_FUSE_ELT_1x1_V1( _pre_data) \
        { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) { \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
        }

#define JIT_FUSE_ELT_1x2_V1(_pre_data) \
        { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) { \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) { \
                    fCv2[Cv1_off + 1].x = fCv2[Cv1_off + 1].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[1])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 1].y = fCv2[Cv1_off + 1].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[1])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
        }

#define JIT_FUSE_ELT_1x4_V1(_pre_data) \
        { \
                if(dCv1_y_valid[0] && dCv1_x_valid[0]) { \
                    fCv2[Cv1_off + 0].x = fCv2[Cv1_off + 0].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 0].y = fCv2[Cv1_off + 0].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[0])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) { \
                    fCv2[Cv1_off + 1].x = fCv2[Cv1_off + 1].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[1])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 1].y = fCv2[Cv1_off + 1].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[1])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
                if(dCv1_y_valid[0] && dCv1_x_valid[2]) { \
                    fCv2[Cv1_off + 2].x = fCv2[Cv1_off + 2].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[2])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 2].y = fCv2[Cv1_off + 2].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[2])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
                if(dCv1_y_valid[0] && dCv1_x_valid[1]) { \
                    fCv2[Cv1_off + 3].x = fCv2[Cv1_off + 3].x + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[3])*_INT16_TO_INT8_ + 0] * pre_scale; \
                    fCv2[Cv1_off + 3].y = fCv2[Cv1_off + 3].y + (int)((int8_t*) _pre_data) [(dCv1_idy[0] * num_flt_v2 + dCv1_idx[3])*_INT16_TO_INT8_ + 1] * pre_scale; \
                } \
        }

//////////////////////////////////////////////////////
// concat macros
//////////////////////////////////////////////////////

//FIXME _INT4_TO_4HALF2_
#define JIT_SET_CONCAT_OFF_V1(_concat_v1_off0) \
        { \
            if(dCv1_y_valid[0]) _concat_v1_off0 = concat_offset_v16 * _INT4_TO_8HALF_ + dCv1_idy[0] * concat_stride_v16 * _INT4_TO_8HALF_; \
        }
