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

#define SPK_KPARAM_LIST                                   \
    int4 *dA,                                             \
        int4 *dB,                                         \
        int4 *dC,                                         \
        int kloop_num,                                    \
        struct lut_t in_lut, int in_lut_size,             \
        struct lut_t flt_lut, int flt_lut_size,           \
        int num_chl_per_spk_head,                         \
        int num_chl_per_spk_tail,                         \
        int in_hw, int out_hw,                            \
        int flt_hw, int splitk,                           \
        int in_height, int in_width,                      \
        int in_num, int num_grp,                          \
        int num_chl_per_grp, int num_chl_per_grp_pad,     \
        int flt_height, int flt_width,                    \
        int num_flt_per_grp, int num_flt_per_grp_pad,     \
        int out_height, int out_width,                    \
        int stride_height, int stride_width,              \
        int pad_height, int pad_width,                    \
        int hole_height, int hole_width,                  \
        int has_bias, int *bias

#define TOTAL_KPARAM_LIST                              \
    int4 *dA,                                          \
        int4 *dB,                                      \
        int4 *dC,                                      \
        int kloop_num,                                 \
        struct lut_t in_lut, int in_lut_size,          \
        struct lut_t flt_lut, int flt_lut_size,        \
        int in_hw, int out_hw,                         \
        int flt_hw, int splitk,                        \
        int in_height, int in_width,                   \
        int in_num, int num_grp,                       \
        int num_chl_per_grp, int num_chl_per_grp_pad,  \
        int flt_height, int flt_width,                 \
        int num_flt_per_grp, int num_flt_per_grp_pad,  \
        int out_height, int out_width,                 \
        int stride_height, int stride_width,           \
        int pad_height, int pad_width,                 \
        int hole_height, int hole_width,               \
        int has_bias, const int4 *bias,                \
        int has_relu, const __half2 clip_min,          \
        bool has_clip, const __half2 clip_max,         \
        int has_prelu, const void *prelu,              \
        bool has_elt, const int4 *pre_data,            \
        int has_elt_relu, const __half2 elt_clip_min,  \
        bool has_elt_clip, const __half2 elt_clip_max, \
        int has_elt_prelu, const void *elt_prelu,      \
        const __half leaky, const __half elt_leaky,    \
        bool has_concat, int concat_offset_v8,         \
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

#define _0BYTE_  0
#define _1BYTE_  1
#define _2BYTE_  2
#define _4BYTE_  4
#define _8BYTE_  8
#define _16BYTE_ 16

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

#define _INT_TO_BYTE_    4
#define _INT_TO_2HALF_   2
#define _INT2_TO_2HALF2_ 2
#define _INT2_TO_2INT_   2
#define _INT2_TO_4HALF_  4

#define _INT8_TO_2INT4_  2
#define _INT4_TO_INT4_   1
#define _INT4_TO_2INT2_  2
#define _INT4_TO_4INT_   4
#define _INT4_TO_4HALF2_ 4
#define _INT4_TO_8HALF_  8
#define _INT4_TO_16BYTE_ 16

#define SMEM_ROW_V4_SIZE   8
#define SMEM_ROW_V2_SIZE   16
#define SMEM_ROW_V1_SIZE   32
#define SMEM_ROW_BYTE_SIZE 128
#define SMEM_ROW_BIT_SIZE  1024

#define _K16_TO_2K8_ 2
#define _K32_TO_4K8_ 4

////////////////////////////////////////
// mma size macros
////////////////////////////////////////

#define TILE_M_PER_MMA      8
#define TILE_K_PER_MMA      8
#define TILE_N_PER_MMA      16
#define TILE_N_PER_MMA_HALF ((TILE_N_PER_MMA) / 2)

#define BLK_M_PER_MMA       1
#define BLK_N_PER_MMA       2

#define MMA_SIZE_X_IN_THD   8
#define MMA_SIZE_Y_IN_THD   4

////////////////////////////////////////
// thread / warp / cta size macros
////////////////////////////////////////

#define WARP_SIZE_IN_THD        32
#define WARP_SIZE_IN_BITS       5

#define WARP_SIZE_X_IN_THD      8
#define WARP_SIZE_Y_IN_THD      4

#define CTA_SIZE_X_IN_WARP      ((TILE_N_PER_CTA) / (TILE_N_PER_WARP))
#define CTA_SIZE_Y_IN_WARP      ((TILE_M_PER_CTA) / (TILE_M_PER_WARP))

#define CTA_SIZE_IN_WARP        ((CTA_SIZE_X_IN_WARP) * (CTA_SIZE_Y_IN_WARP))
#define CTA_SIZE_IN_THD         ((CTA_SIZE_IN_WARP)   * (WARP_SIZE_IN_THD))

#define WARP_SIZE_IN_THD_HALF   (WARP_SIZE_IN_THD / 2)
#define WARP_SIZE_IN_THD_QTR    (WARP_SIZE_IN_THD / 4)

#define NUM_M_STEPS             (TILE_M_PER_WARP / TILE_M_PER_MMA)
#define NUM_N_STEPS             (TILE_N_PER_WARP / TILE_N_PER_MMA)

////////////////////////////////////////
// tiling size macros
////////////////////////////////////////

#define TILE_M_PER_THD          ((TILE_M_PER_WARP) / (WARP_SIZE_Y_IN_THD))
#define TILE_N_PER_THD          ((TILE_N_PER_WARP) / (WARP_SIZE_X_IN_THD))

/////////////////////
// tile m

#define TILE_M_V1_PER_CTA       ((TILE_M_PER_CTA)  / 1)
#define TILE_M_V2_PER_CTA       ((TILE_M_PER_CTA)  / 2)
#define TILE_M_V4_PER_CTA       ((TILE_M_PER_CTA)  / 4)
#define TILE_M_V8_PER_CTA       ((TILE_M_PER_CTA)  / 8)

#define TILE_M_V1_PER_WARP      ((TILE_M_PER_WARP) / 1)
#define TILE_M_V2_PER_WARP      ((TILE_M_PER_WARP) / 2)
#define TILE_M_V4_PER_WARP      ((TILE_M_PER_WARP) / 4)
#define TILE_M_V8_PER_WARP      ((TILE_M_PER_WARP) / 8)

#define TILE_M_V1_PER_THD       ((TILE_M_PER_THD)  / 1)
#define TILE_M_V2_PER_THD       ((TILE_M_PER_THD)  / 2)
#define TILE_M_V4_PER_THD       ((TILE_M_PER_THD)  / 4)
#define TILE_M_V8_PER_THD       ((TILE_M_PER_THD)  / 8)

#define TILE_M_V1_PER_MMA       ((TILE_M_PER_MMA)  / 1)
#define TILE_M_V2_PER_MMA       ((TILE_M_PER_MMA)  / 2)
#define TILE_M_V4_PER_MMA       ((TILE_M_PER_MMA)  / 4)
#define TILE_M_V8_PER_MMA       ((TILE_M_PER_MMA)  / 8)

/////////////////////
// tile k

#define TILE_K_V1_PER_CTA       ((TILE_K_PER_CTA)  / 1)
#define TILE_K_V2_PER_CTA       ((TILE_K_PER_CTA)  / 2)
#define TILE_K_V4_PER_CTA       ((TILE_K_PER_CTA)  / 4)
#define TILE_K_V8_PER_CTA       ((TILE_K_PER_CTA)  / 8)

#define TILE_K_V1_PER_WARP      ((TILE_K_PER_WARP) / 1)
#define TILE_K_V2_PER_WARP      ((TILE_K_PER_WARP) / 2)
#define TILE_K_V4_PER_WARP      ((TILE_K_PER_WARP) / 4)
#define TILE_K_V8_PER_WARP      ((TILE_K_PER_WARP) / 8)

#define TILE_K_V1_PER_MMA       ((TILE_K_PER_MMA) / 1)
#define TILE_K_V2_PER_MMA       ((TILE_K_PER_MMA) / 2)
#define TILE_K_V4_PER_MMA       ((TILE_K_PER_MMA) / 4)
#define TILE_K_V8_PER_MMA       ((TILE_K_PER_MMA) / 8)

/////////////////////
// tile n

#define TILE_N_V1_PER_CTA       ((TILE_N_PER_CTA)  / 1)
#define TILE_N_V2_PER_CTA       ((TILE_N_PER_CTA)  / 2)
#define TILE_N_V4_PER_CTA       ((TILE_N_PER_CTA)  / 4)
#define TILE_N_V8_PER_CTA       ((TILE_N_PER_CTA)  / 8)

#define TILE_N_V1_PER_WARP      ((TILE_N_PER_WARP) / 1)
#define TILE_N_V2_PER_WARP      ((TILE_N_PER_WARP) / 2)
#define TILE_N_V4_PER_WARP      ((TILE_N_PER_WARP) / 4)
#define TILE_N_V8_PER_WARP      ((TILE_N_PER_WARP) / 8)

#define TILE_N_V1_PER_THD       ((TILE_N_PER_THD)  / 1)
#define TILE_N_V2_PER_THD       ((TILE_N_PER_THD)  / 2)
#define TILE_N_V4_PER_THD       ((TILE_N_PER_THD)  / 4)
#define TILE_N_V8_PER_THD       ((TILE_N_PER_THD)  / 8)

#define TILE_N_V1_PER_MMA       ((TILE_N_PER_MMA)  / 1)
#define TILE_N_V2_PER_MMA       ((TILE_N_PER_MMA)  / 2)
#define TILE_N_V4_PER_MMA       ((TILE_N_PER_MMA)  / 4)
#define TILE_N_V8_PER_MMA       ((TILE_N_PER_MMA)  / 8)

////////////////////////////////////////
// shared memory size macros
////////////////////////////////////////

#define OUTPUT_STEPS            ((TILE_N_V1_PER_WARP) / (TILE_N_V1_PER_MMA))

#if TILE_M_PER_WARP == 8 || TILE_M_PER_WARP == 16
#define OUTPUT_BLKS_PER_STEP    1
#elif TILE_M_PER_WARP == 32
#define OUTPUT_BLKS_PER_STEP    2
#elif TILE_M_PER_WARP == 64
#define OUTPUT_BLKS_PER_STEP    4
#endif

#define M_ROWS_PER_SMEM_ROW     (SMEM_ROW_V4_SIZE / TILE_M_V8_PER_CTA)
#define K_ROWS_PER_SMEM_ROW     (SMEM_ROW_V4_SIZE / TILE_K_V8_PER_CTA)

#if M_ROWS_PER_SMEM_ROW < 1
#undef  M_ROWS_PER_SMEM_ROW
#define M_ROWS_PER_SMEM_ROW     1
#endif

#if K_ROWS_PER_SMEM_ROW < 1
#undef  K_ROWS_PER_SMEM_ROW
#define K_ROWS_PER_SMEM_ROW     1
#endif

#define OUTPUT_SIZE_Y_IN_THD    (TILE_M_V8_PER_WARP)
#define OUTPUT_SIZE_X_IN_THD    ((WARP_SIZE_IN_THD) / (OUTPUT_SIZE_Y_IN_THD))

////////////////////////////////////////
// k group macros
////////////////////////////////////////

#if BUF_NUM == 2

#define FWD_BUF(_addr0, _size0, _base0, _addr1, _size1, _base1) \
        { \
            _addr0 = ((_addr0 - _base0) ^ _size0) + _base0; \
            _addr1 = ((_addr1 - _base1) ^ _size1) + _base1; \
        }

#elif BUF_NUM > 2

#define INFLIGHT_BUF_NUM        2

#define FWD_BUF(_buf, _addr0, _size0, _addr1, _size1) \
        { \
            _buf   = (_buf == BUF_NUM - 1) ? 0 : _buf + 1; \
            \
            _addr0 = (_buf == 0) ? _addr0 - _size0 * (BUF_NUM - 1) : _addr0 + _size0; \
            _addr1 = (_buf == 0) ? _addr1 - _size1 * (BUF_NUM - 1) : _addr1 + _size1; \
        }

#endif

#if defined(USE_HMMA1688)

#define FWD_KGROUP_GAP1(_sUv1_read) \
        { \
            _sUv1_read = _sUv1_read ^ 0x4; \
        }

#define FWD_KGROUP_GAP2(_sUv1_read) \
        { \
            _sUv1_read = _sUv1_read ^ 0xc; \
        }

#define FWD_KGROUP_GAP3(_sUv1_read) \
        { \
            _sUv1_read = _sUv1_read ^ 0x1c; \
        }

#define FWD_KGROUP_STEP1(_sUv1_read)     FWD_KGROUP_GAP1(_sUv1_read)
#define FWD_KGROUP_STEP3(_sUv1_read)     FWD_KGROUP_GAP1(_sUv1_read)
#define FWD_KGROUP_STEP5(_sUv1_read)     FWD_KGROUP_GAP1(_sUv1_read)
#define FWD_KGROUP_STEP7(_sUv1_read)     FWD_KGROUP_GAP1(_sUv1_read)

#if TILE_K_PER_CTA == 16
#define FWD_KGROUP_STEP2(_sUv1_read)     FWD_KGROUP_GAP1(_sUv1_read)
#elif TILE_K_PER_CTA == 32
#define FWD_KGROUP_STEP2(_sUv1_read)     FWD_KGROUP_GAP2(_sUv1_read)
#define FWD_KGROUP_STEP4(_sUv1_read)     FWD_KGROUP_GAP2(_sUv1_read)
#elif TILE_K_PER_CTA == 64
#define FWD_KGROUP_STEP2(_sUv1_read)     FWD_KGROUP_GAP2(_sUv1_read)
#define FWD_KGROUP_STEP4(_sUv1_read)     FWD_KGROUP_GAP3(_sUv1_read)
#define FWD_KGROUP_STEP6(_sUv1_read)     FWD_KGROUP_GAP2(_sUv1_read)
#define FWD_KGROUP_STEP8(_sUv1_read)     FWD_KGROUP_GAP3(_sUv1_read)
#endif

#elif defined(USE_HMMA16816)

#define FWD_KGROUP_GAP1(_sUv1_read) \
        { \
            _sUv1_read = _sUv1_read ^ 0x8; \
        }

// 0x18 means 0x6 << 2
#define FWD_KGROUP_GAP2(_sUv1_read) \
        { \
            _sUv1_read = _sUv1_read ^ 0x18; \
        }

#define FWD_KGROUP_STEP1(_sUv1_read)     FWD_KGROUP_GAP1(_sUv1_read)
#define FWD_KGROUP_STEP3(_sUv1_read)     FWD_KGROUP_GAP1(_sUv1_read)

#if TILE_K_PER_CTA == 32
#define FWD_KGROUP_STEP2(_sUv1_read)     FWD_KGROUP_GAP1(_sUv1_read)
#elif TILE_K_PER_CTA == 64
#define FWD_KGROUP_STEP2(_sUv1_read)     FWD_KGROUP_GAP2(_sUv1_read)
#define FWD_KGROUP_STEP4(_sUv1_read)     FWD_KGROUP_GAP2(_sUv1_read)
#endif

#endif

////////////////////////////////////////
// main loop macros
////////////////////////////////////////

#define   C_ITEMS_PER_THD       ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (CTA_SIZE_IN_THD * _INT_TO_2HALF_))
#define  HC_ITEMS_PER_THD       ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (CTA_SIZE_IN_THD))
#define Cv4_ITEMS_PER_THD       ((TILE_M_PER_CTA) * (TILE_N_PER_CTA) / (CTA_SIZE_IN_THD * _INT_TO_2HALF_ * _4INT_TO_INT4_))

#if Cv4_ITEMS_PER_THD < 1
#undef Cv4_ITEMS_PER_THD
#define Cv4_ITEMS_PER_THD 1
#endif

////////////////////////////////////////
// load A and B from device memory macros
////////////////////////////////////////

#define REG_dAv4_SIZE           ( ((TILE_M_PER_CTA) * (TILE_K_PER_CTA)) / ((_2HALF_TO_INT_) * (_4INT_TO_INT4_) * (CTA_SIZE_IN_THD)) )
#define REG_dBv4_SIZE           ( ((TILE_N_PER_CTA) * (TILE_K_PER_CTA)) / ((_2HALF_TO_INT_) * (_4INT_TO_INT4_) * (CTA_SIZE_IN_THD)) )

#if REG_dAv4_SIZE < 1
#undef  REG_dAv4_SIZE
#define REG_dAv4_SIZE 1
#endif

#if REG_dBv4_SIZE < 1
#undef  REG_dBv4_SIZE
#define REG_dBv4_SIZE 1
#endif

#define READ_dAv4_STEPS         (REG_dAv4_SIZE)
#define READ_dBv4_STEPS         (REG_dBv4_SIZE)

////////////////////////////////////////
// shared memory size macros
////////////////////////////////////////

#define SM_A_SIZE               ((TILE_M_PER_CTA) * (TILE_K_PER_CTA) / (_2HALF_TO_INT_))
#define SM_B_SIZE               ((TILE_K_PER_CTA) * (TILE_N_PER_CTA) / (_2HALF_TO_INT_))

#if TILE_M_PER_WARP == 8
#define SM_C_SIZE               ((TILE_M_PER_CTA) * (TILE_N_PER_MMA) * (CTA_SIZE_X_IN_WARP) * (_2MMA_) / (_2HALF_TO_INT_))
#elif TILE_M_PER_WARP == 16 || TILE_M_PER_WARP == 32 || TILE_M_PER_WARP == 64
#define SM_C_SIZE               ((TILE_M_PER_CTA) * (TILE_N_PER_MMA) * (CTA_SIZE_X_IN_WARP) / (_2HALF_TO_INT_))
#endif

#define SM_A_1BUF               (SM_A_SIZE)
#define SM_B_1BUF               (SM_B_SIZE)
#define SM_C_1BUF               (SM_C_SIZE)

#define SM_A_2BUF               ((SM_A_SIZE) * 2)
#define SM_B_2BUF               ((SM_B_SIZE) * 2)
#define SM_C_2BUF               ((SM_C_SIZE) * 2)

#define SM_A_V1_1BUF            (SM_A_1BUF)
#define SM_B_V1_1BUF            (SM_B_1BUF)
#define SM_C_V1_1BUF            (SM_C_1BUF)

#define SM_A_V2_1BUF            ((SM_A_1BUF) / (_2INT_TO_INT2_))
#define SM_B_V2_1BUF            ((SM_B_1BUF) / (_2INT_TO_INT2_))
#define SM_C_V2_1BUF            ((SM_C_1BUF) / (_2INT_TO_INT2_))

#define SM_A_V4_1BUF            ((SM_A_1BUF) / (_4INT_TO_INT4_))
#define SM_B_V4_1BUF            ((SM_B_1BUF) / (_4INT_TO_INT4_))
#define SM_C_V4_1BUF            ((SM_C_1BUF) / (_4INT_TO_INT4_))

#define SM_A_V1_2BUF            ((SM_A_V1_1BUF) * 2)
#define SM_B_V1_2BUF            ((SM_B_V1_1BUF) * 2)
#define SM_C_V1_2BUF            ((SM_C_V1_1BUF) * 2)

#define SM_A_V2_2BUF            ((SM_A_V2_1BUF) * 2)
#define SM_B_V2_2BUF            ((SM_B_V2_1BUF) * 2)
#define SM_C_V2_2BUF            ((SM_C_V2_1BUF) * 2)

#define SM_A_V4_2BUF            ((SM_A_V4_1BUF) * 2)
#define SM_B_V4_2BUF            ((SM_B_V4_1BUF) * 2)
#define SM_C_V4_2BUF            ((SM_C_V4_1BUF) * 2)

#define SM_BASE_V4_1BUF         Max((SM_A_V4_1BUF + SM_B_V4_1BUF), (SM_C_V4_1BUF))
#define SM_BASE_V4_2BUF         Max((SM_A_V4_2BUF + SM_B_V4_2BUF), (SM_C_V4_1BUF))

#if (defined(__CUDA_ARCH__) && __CUDACC_VER_MAJOR__ >= 11)

#define CVT_SM_PTR(_smp_base, _sm_base) \
    _smp_base = static_cast<unsigned>(__cvta_generic_to_shared(_sm_base));

#elif defined(__CUDA_ARCH__)

#define CVT_SM_PTR(smp_base_v1, sm_base_v1) \
    asm("{ .reg .u64 smp_base_v1; cvta.to.shared.u64 smp_base_v1, %1; cvt.u32.u64 %0, smp_base_v1; }\n" \
            : "=r"(smp_base_v1) : "l"(sm_base_v1));

#endif

#define FWD_LUT(_lut_id) \
        { \
            _lut_id = (_lut_id == flt_hw) ? 1 : _lut_id + 1; \
        }

////////////////////////////////////////
// bit size macros
////////////////////////////////////////

#if CTA_SIZE_X_IN_WARP == 1
#define CTA_SIZE_X_IN_BITS      0
#elif CTA_SIZE_X_IN_WARP == 2
#define CTA_SIZE_X_IN_BITS      1
#elif CTA_SIZE_X_IN_WARP == 4
#define CTA_SIZE_X_IN_BITS      2
#elif CTA_SIZE_X_IN_WARP == 8
#define CTA_SIZE_X_IN_BITS      3
#endif

#if MMA_SIZE_Y_IN_THD == 1
#define MMA_SIZE_Y_IN_BITS      0
#elif MMA_SIZE_Y_IN_THD == 2
#define MMA_SIZE_Y_IN_BITS      1
#elif MMA_SIZE_Y_IN_THD == 4
#define MMA_SIZE_Y_IN_BITS      2
#elif MMA_SIZE_Y_IN_THD == 8
#define MMA_SIZE_Y_IN_BITS      3
#endif

#if CTA_SIZE_IN_WARP == 1
#define CTA_SIZE_IN_BITS        5
#elif CTA_SIZE_IN_WARP == 2
#define CTA_SIZE_IN_BITS        6
#elif CTA_SIZE_IN_WARP == 4
#define CTA_SIZE_IN_BITS        7
#elif CTA_SIZE_IN_WARP == 8
#define CTA_SIZE_IN_BITS        8
#endif

////////////////////////////////////////
// fuse size macros
////////////////////////////////////////

#define REDUCE_V4_SIZE              (OUTPUT_BLKS_PER_STEP)
#define Rv4_SIZE                    Max((REG_dAv4_SIZE + REG_dBv4_SIZE), REDUCE_V4_SIZE)

////////////////////////////////////////
// fuse macros
////////////////////////////////////////

#define HADD2_INST(_d, _a, _b) \
        asm volatile("add.ftz.f16x2 %0, %1, %2;\n":   "=r"(_d): "r"(_a), "r"(_b));

#define HMAX2_INST(_d, _a, _b, _c) \
        asm volatile("vmax2.s32.s32.s32 %0, %1, %2, %3;\n":   "=r"(_d): "r"(_a), "r"(_b), "r"(_c));

#define HMIN2_INST(_d, _a, _b, _c) \
        asm volatile("vmin2.s32.s32.s32 %0, %1, %2, %3;\n":   "=r"(_d): "r"(_a), "r"(_b), "r"(_c));

