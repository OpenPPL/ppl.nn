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

#if defined(_WIN64) || defined(_WIN32)
    #define uint unsigned int

#ifdef _MSC_VER
    #if _MSC_VER >= 1600
    #include <cstdint>
    #else
    typedef __int8 int8_t;
    typedef __int16 int16_t;
    typedef __int32 int32_t;
    typedef __int64 int64_t;
    typedef unsigned __int8 uint8_t;
    typedef unsigned __int16 uint16_t;
    typedef unsigned __int32 uint32_t;
    typedef unsigned __int64 uint64_t;
    #endif
    #endif
#endif

#ifndef PPLNN_ENABLE_CUDA_JIT
#if (defined(_WIN64) || defined(_WIN32))
    #define _Pragma __pragma
#endif
#endif

////////////////////////////////////////
// kernel list macros
////////////////////////////////////////

#define TOTAL_KPARAM_LIST                                \
    int4 *dA,                                            \
        int4 *dB,                                        \
        int4 *dC,                                        \
        int kloop_num, int koff_num_pad,                 \
        int in_hw, int out_hw,                           \
        int flt_hw, int out_nhw,                         \
        int in_height, int in_width,                     \
        int in_num, int num_grp,                         \
        int num_chl, int num_chl_per_grp,                \
        int in_chl_per_grp_pad, int flt_chl_per_grp_pad, \
        int flt_height, int flt_width,                   \
        int num_flt_per_grp, int num_flt_per_grp_pad,    \
        int out_height, int out_width,                   \
        int stride_height, int stride_width,             \
        int pad_height, int pad_width,                   \
        int hole_height, int hole_width,                 \
        int has_bias, const int4 *bias,                  \
        int has_relu, const __half2 clip_min,            \
        bool has_clip, const __half2 clip_max,           \
        int has_prelu, const void *prelu,                \
        bool has_elt, const int4 *pre_data,              \
        int has_elt_relu, const __half2 elt_clip_min,    \
        bool has_elt_clip, const __half2 elt_clip_max,   \
        int has_elt_prelu, const void *elt_prelu,        \
        const __half leaky, const __half elt_leaky,      \
        bool has_concat, int concat_offset_v8,           \
        int concat_stride_v8

////////////////////////////////////////
// align functions
////////////////////////////////////////

#define Align(x, y) (((x) + (y)-1) / (y) * (y))
#define DivUp(x, y) (((x) + (y)-1) / (y))

#define Min(x, y) (((x) < (y)) ? (x) : (y))
#define Max(x, y) (((x) > (y)) ? (x) : (y))

////////////////////////////////////////
// boundary check
////////////////////////////////////////

#define WidthInRange(_w)  ((_w < in_width) && (_w >= 0))
#define HeightInRange(_h) ((_h < in_height) && (_h >= 0))
#define BatchInRange(_b)  ((_b < in_num))

////////////////////////////////////////
// constant cta size macros
////////////////////////////////////////

#define _4CHAR_TO_INT_ 4
#define _4INT_TO_INT4_ 4
#define _2INT_TO_INT2_ 2

#define _2HALF_TO_INT_  2
#define _2INT2_TO_INT4_ 2

#define _C1_  1
#define _C2_  2
#define _C4_  4
#define _C8_  8
#define _C16_ 16
#define _C32_ 32

#define _1INT_ 1
#define _2INT_ 2
#define _4INT_ 4
#define _8INT_ 8

#define _1INT4_ 1
#define _2INT4_ 2
#define _4INT4_ 4
#define _8INT4_ 8

#define _1INT8_ 1
#define _2INT8_ 2
#define _4INT8_ 4
#define _8INT8_ 8

#define _1HALF_ 1
#define _2HALF_ 2
#define _4HALF_ 4
#define _8HALF_ 8

#define _1HALF2_ 1
#define _2HALF2_ 2
#define _4HALF2_ 4
#define _8HALF2_ 8

#define _1MMA_ 1
#define _2MMA_ 2
#define _4MMA_ 4
#define _8MMA_ 8

#define _HALF_ZERO_ 0.0

#define _1INT_X1_ (_1INT_ * 1)
#define _1INT_X2_ (_1INT_ * 2)
#define _1INT_X4_ (_1INT_ * 4)

#define _2INT_X1_ (_2INT_ * 1)
#define _2INT_X2_ (_2INT_ * 2)
#define _2INT_X4_ (_2INT_ * 4)

#define _4INT_X1_ (_4INT_ * 1)
#define _4INT_X2_ (_4INT_ * 2)
#define _4INT_X4_ (_4INT_ * 4)

#define _INT_TO_BYTE_    4
#define _INT_TO_2HALF_   2
#define _INT2_TO_2HALF2_ 2
#define _INT2_TO_2INT_   2

#define _INT4_TO_INT4_   1
#define _INT4_TO_2INT2_  2
#define _INT4_TO_4INT_   4
#define _INT4_TO_4HALF2_ 4
#define _INT4_TO_8HALF_  8

////////////////////////////////////////
// mma size macros
////////////////////////////////////////

#define TILE_M_PER_MMA      16
#define TILE_K_PER_MMA      8
#define TILE_N_PER_MMA      8
#define TILE_M_PER_MMA_HALF ((TILE_M_PER_MMA) / 2)

#define MMA_SIZE_X_IN_THD 4
#define MMA_SIZE_Y_IN_THD 8

#define BLK_M_PER_MMA 2
#define BLK_N_PER_MMA 1

////////////////////////////////////////
// thread / warp / cta size macros
////////////////////////////////////////

#define WARP_SIZE_IN_THD  32
#define WARP_SIZE_IN_BITS 5

#define WARP_SIZE_X_IN_THD 4
#define WARP_SIZE_Y_IN_THD 8

#define CTA_SIZE_X_IN_WARP ((TILE_N_PER_CTA) / (TILE_N_PER_WARP))
#define CTA_SIZE_Y_IN_WARP ((TILE_M_PER_CTA) / (TILE_M_PER_WARP))

#define CTA_SIZE_IN_WARP ((CTA_SIZE_X_IN_WARP) * (CTA_SIZE_Y_IN_WARP))
#define CTA_SIZE_IN_THD  ((CTA_SIZE_IN_WARP) * (WARP_SIZE_IN_THD))

#define WARP_SIZE_IN_THD_HALF (WARP_SIZE_IN_THD / 2)
#define WARP_SIZE_IN_THD_QTR  (WARP_SIZE_IN_THD / 4)

#define NUM_M_STEPS (TILE_M_PER_WARP / TILE_M_PER_MMA)
#define NUM_N_STEPS (TILE_N_PER_WARP / TILE_N_PER_MMA)

////////////////////////////////////////
// tiling size macros
////////////////////////////////////////

#define TILE_M_PER_STEP ((TILE_M_PER_MMA) * (CTA_SIZE_Y_IN_WARP))
#define TILE_N_PER_STEP ((TILE_N_PER_MMA) * (CTA_SIZE_X_IN_WARP))

#define TILE_M_PER_THD ((TILE_M_PER_WARP) / (WARP_SIZE_Y_IN_THD))
#define TILE_N_PER_THD ((TILE_N_PER_WARP) / (WARP_SIZE_X_IN_THD))

/////////////////////
// tile m

#define TILE_M_V1_PER_CTA ((TILE_M_PER_CTA) / 1)
#define TILE_M_V2_PER_CTA ((TILE_M_PER_CTA) / 2)
#define TILE_M_V4_PER_CTA ((TILE_M_PER_CTA) / 4)
#define TILE_M_V8_PER_CTA ((TILE_M_PER_CTA) / 8)

#define TILE_M_V1_PER_WARP ((TILE_M_PER_WARP) / 1)
#define TILE_M_V2_PER_WARP ((TILE_M_PER_WARP) / 2)
#define TILE_M_V4_PER_WARP ((TILE_M_PER_WARP) / 4)
#define TILE_M_V8_PER_WARP ((TILE_M_PER_WARP) / 8)

#define TILE_M_V1_PER_THD ((TILE_M_PER_THD) / 1)
#define TILE_M_V2_PER_THD ((TILE_M_PER_THD) / 2)
#define TILE_M_V4_PER_THD ((TILE_M_PER_THD) / 4)
#define TILE_M_V8_PER_THD ((TILE_M_PER_THD) / 8)

#define TILE_M_V1_PER_MMA      ((TILE_M_PER_MMA) / 1)
#define TILE_M_V2_PER_MMA      ((TILE_M_PER_MMA) / 2)
#define TILE_M_V4_PER_MMA      ((TILE_M_PER_MMA) / 4)
#define TILE_M_V8_PER_MMA      ((TILE_M_PER_MMA) / 8)
#define TILE_M_V1_PER_MMA_HALF ((TILE_M_PER_MMA) / 2)

/////////////////////
// tile k

#define TILE_K_V1_PER_CTA ((TILE_K_PER_CTA) / 1)
#define TILE_K_V2_PER_CTA ((TILE_K_PER_CTA) / 2)
#define TILE_K_V4_PER_CTA ((TILE_K_PER_CTA) / 4)
#define TILE_K_V8_PER_CTA ((TILE_K_PER_CTA) / 8)

#define TILE_K_V1_PER_STEP ((TILE_K_PER_STEP) / 1)
#define TILE_K_V2_PER_STEP ((TILE_K_PER_STEP) / 2)
#define TILE_K_V4_PER_STEP ((TILE_K_PER_STEP) / 4)
#define TILE_K_V8_PER_STEP ((TILE_K_PER_STEP) / 8)

#define TILE_K_V1_PER_MMA ((TILE_K_PER_MMA) / 1)
#define TILE_K_V2_PER_MMA ((TILE_K_PER_MMA) / 2)
#define TILE_K_V4_PER_MMA ((TILE_K_PER_MMA) / 4)
#define TILE_K_V8_PER_MMA ((TILE_K_PER_MMA) / 8)

/////////////////////
// tile n

#define TILE_N_V1_PER_CTA ((TILE_N_PER_CTA) / 1)
#define TILE_N_V2_PER_CTA ((TILE_N_PER_CTA) / 2)
#define TILE_N_V4_PER_CTA ((TILE_N_PER_CTA) / 4)
#define TILE_N_V8_PER_CTA ((TILE_N_PER_CTA) / 8)

#define TILE_N_V1_PER_WARP ((TILE_N_PER_WARP) / 1)
#define TILE_N_V2_PER_WARP ((TILE_N_PER_WARP) / 2)
#define TILE_N_V4_PER_WARP ((TILE_N_PER_WARP) / 4)
#define TILE_N_V8_PER_WARP ((TILE_N_PER_WARP) / 8)

#define TILE_N_V1_PER_THD ((TILE_N_PER_THD) / 1)
#define TILE_N_V2_PER_THD ((TILE_N_PER_THD) / 2)
#define TILE_N_V4_PER_THD ((TILE_N_PER_THD) / 4)
#define TILE_N_V8_PER_THD ((TILE_N_PER_THD) / 8)

#define TILE_N_V1_PER_MMA ((TILE_N_PER_MMA) / 1)
#define TILE_N_V2_PER_MMA ((TILE_N_PER_MMA) / 2)
#define TILE_N_V4_PER_MMA ((TILE_N_PER_MMA) / 4)
#define TILE_N_V8_PER_MMA ((TILE_N_PER_MMA) / 8)

#define TILE_N_V1_PER_STEP ((TILE_N_PER_STEP) / 1)
#define TILE_N_V2_PER_STEP ((TILE_N_PER_STEP) / 2)
#define TILE_N_V4_PER_STEP ((TILE_N_PER_STEP) / 4)
#define TILE_N_V8_PER_STEP ((TILE_N_PER_STEP) / 8)

////////////////////////////////////////
// main loop macros
////////////////////////////////////////

#define C_ITEMS_PER_THD   ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (CTA_SIZE_IN_THD * _INT_TO_2HALF_))
#define HC_ITEMS_PER_THD  ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (CTA_SIZE_IN_THD))
#define Cv4_ITEMS_PER_THD ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (CTA_SIZE_IN_THD * _INT_TO_2HALF_ * _4INT_TO_INT4_))

////////////////////////////////////////
// load A and B from device memory macros
////////////////////////////////////////

#define REG_dAv1_SIZE (NUM_M_STEPS * BLK_M_PER_MMA)
#define REG_dBv1_SIZE (NUM_N_STEPS * BLK_N_PER_MMA)

#define REG_dAv2_SIZE (NUM_M_STEPS * BLK_M_PER_MMA)
#define REG_dBv2_SIZE (NUM_N_STEPS * BLK_N_PER_MMA)

#define REG_dAv4_SIZE (NUM_M_STEPS * BLK_M_PER_MMA)
#define REG_dBv4_SIZE (NUM_N_STEPS * BLK_N_PER_MMA)

#define READ_dAv1_STEPS (REG_dAv1_SIZE)
#define READ_dBv1_STEPS (REG_dBv1_SIZE)

#define READ_dAv2_STEPS (REG_dAv2_SIZE)
#define READ_dBv2_STEPS (REG_dBv2_SIZE)

#define READ_dAv4_STEPS (REG_dAv4_SIZE)
#define READ_dBv4_STEPS (REG_dBv4_SIZE)

////////////////////////////////////////
// shared memory size macros
////////////////////////////////////////

#define SM_IN_ID_SIZE  (TILE_M_PER_CTA)
#define SM_IN_OFF_SIZE (CTA_SIZE_IN_THD)

////////////////////////////////////////
// bit size macros
////////////////////////////////////////

#if MMA_SIZE_X_IN_THD == 1
#define MMA_SIZE_X_IN_BITS 0
#elif MMA_SIZE_X_IN_THD == 2
#define MMA_SIZE_X_IN_BITS 1
#elif MMA_SIZE_X_IN_THD == 4
#define MMA_SIZE_X_IN_BITS 2
#elif MMA_SIZE_X_IN_THD == 8
#define MMA_SIZE_X_IN_BITS 3
#endif

#if CTA_SIZE_X_IN_WARP == 1
#define CTA_SIZE_X_IN_BITS 0
#elif CTA_SIZE_X_IN_WARP == 2
#define CTA_SIZE_X_IN_BITS 1
#elif CTA_SIZE_X_IN_WARP == 4
#define CTA_SIZE_X_IN_BITS 2
#endif

////////////////////////////////////////
// fuse macros
////////////////////////////////////////

#define HADD2_INST(_d, _a, _b) \
        asm volatile("add.ftz.f16x2 %0, %1, %2;\n":   "=r"(_d): "r"(_a), "r"(_b));

#define HMAX2_INST(_d, _a, _b, _c) \
        asm volatile("vmax2.s32.s32.s32 %0, %1, %2, %3;\n":   "=r"(_d): "r"(_a), "r"(_b), "r"(_c));

#define HMIN2_INST(_d, _a, _b, _c) \
        asm volatile("vmin2.s32.s32.s32 %0, %1, %2, %3;\n":   "=r"(_d): "r"(_a), "r"(_b), "r"(_c));

