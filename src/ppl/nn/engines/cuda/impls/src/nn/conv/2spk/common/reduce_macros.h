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

/////////////////////////////////////////////////////
// reduce half2 macros
/////////////////////////////////////////////////////

#define REDUCE_HALF2_SIZE4(_h2R, _h2R_off)              \
    {                                                   \
        _h2R[0] = __hadd2(_h2R[0], _h2R[_h2R_off]);     \
        _h2R[1] = __hadd2(_h2R[1], _h2R[_h2R_off + 1]); \
        _h2R[2] = __hadd2(_h2R[2], _h2R[_h2R_off + 2]); \
        _h2R[3] = __hadd2(_h2R[3], _h2R[_h2R_off + 3]); \
    }

#define REDUCE_HALF2_1x4(_h2R)              \
    {                                       \
        REDUCE_HALF2_SIZE4(_h2R, _4HALF2_); \
    }

#define REDUCE_HALF2_3x4(_h2R)                  \
    {                                           \
        REDUCE_HALF2_SIZE4(_h2R, _4HALF2_);     \
        REDUCE_HALF2_SIZE4(_h2R, _4HALF2_ * 2); \
        REDUCE_HALF2_SIZE4(_h2R, _4HALF2_ * 3); \
    }

/////////////////////////////////////////////////////
// read sRv4 macros
/////////////////////////////////////////////////////

#define READ_sRv4_SIZE1(_Rv4, _sm_base_v4, _sRv4_read) \
    {                                                  \
        if (dCv4_x_valid) {                            \
            _Rv4[0] = _sm_base_v4[_sRv4_read];         \
        }                                              \
                                                       \
        _sRv4_read += CTA_SIZE_IN_THD;                 \
    }

#define READ_sRv4_SIZE2(_Rv4, _sm_base_v4, _sRv4_read)                                     \
    {                                                                                      \
        if (dCv4_x_valid) {                                                                \
            _Rv4[0] = _sm_base_v4[_sRv4_read];                                             \
            _Rv4[1] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V8_PER_CTA * 1]; \
        }                                                                                  \
                                                                                           \
        _sRv4_read += CTA_SIZE_IN_THD;                                                     \
    }

#define READ_sRv4_SIZE4(_Rv4, _sm_base_v4, _sRv4_read)                                     \
    {                                                                                      \
        if (dCv4_x_valid) {                                                                \
            _Rv4[0] = _sm_base_v4[_sRv4_read];                                             \
            _Rv4[1] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V8_PER_CTA * 1]; \
            _Rv4[2] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V8_PER_CTA * 2]; \
            _Rv4[3] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V8_PER_CTA * 3]; \
        }                                                                                  \
                                                                                           \
        _sRv4_read += CTA_SIZE_IN_THD;                                                     \
    }

/////////////////////////////////////////////////////
// write sRv1 macros
/////////////////////////////////////////////////////

#define WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write, _C, _C_off)                                      \
    {                                                                                               \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _C[_C_off + 0]; \
    }

#define WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write, _C, _C_off)                                      \
    {                                                                                               \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _C[_C_off + 0]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x1) * TILE_N_V2_PER_MMA] = _C[_C_off + 1]; \
    }

#define WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write, _C, _C_off)                                      \
    {                                                                                               \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _C[_C_off + 0]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x1) * TILE_N_V2_PER_MMA] = _C[_C_off + 1]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x2) * TILE_N_V2_PER_MMA] = _C[_C_off + 2]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x3) * TILE_N_V2_PER_MMA] = _C[_C_off + 3]; \
    }

#define WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write, _C, _C_off)                                      \
    {                                                                                               \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _C[_C_off + 0]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x1) * TILE_N_V2_PER_MMA] = _C[_C_off + 1]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x2) * TILE_N_V2_PER_MMA] = _C[_C_off + 2]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x3) * TILE_N_V2_PER_MMA] = _C[_C_off + 3]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x4) * TILE_N_V2_PER_MMA] = _C[_C_off + 4]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x5) * TILE_N_V2_PER_MMA] = _C[_C_off + 5]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x6) * TILE_N_V2_PER_MMA] = _C[_C_off + 6]; \
        _sm_base_v1[_sRv1_write + (smem_row_write_off ^ 0x7) * TILE_N_V2_PER_MMA] = _C[_C_off + 7]; \
    }

/////////////////////////
// tile_n_per_warp = 8
/////////////////////////

#define WRITE_sRv1_1x1(_sm_base_v1, _sRv1_write, _C)       \
    {                                                      \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write, _C, 0); \
    }

#define WRITE_sRv1_2x1(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _1MMA_ * 0); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _1MMA_ * 1); \
    }

#define WRITE_sRv1_4x1(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _1MMA_ * 0); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _1MMA_ * 1); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _1MMA_ * 2); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _1MMA_ * 3); \
    }

#define WRITE_sRv1_8x1(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _1MMA_ * 0); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _1MMA_ * 1); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _1MMA_ * 2); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _1MMA_ * 3); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _1MMA_ * 4); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _1MMA_ * 5); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _1MMA_ * 6); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _1MMA_ * 7); \
    }

#define WRITE_sRv1_16x1(_sm_base_v1, _sRv1_write, _C)                                                               \
    {                                                                                                               \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _1MMA_ * 0);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _1MMA_ * 1);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _1MMA_ * 2);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _1MMA_ * 3);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _1MMA_ * 4);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _1MMA_ * 5);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _1MMA_ * 6);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _1MMA_ * 7);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 8, _C, _1MMA_ * 8);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 9, _C, _1MMA_ * 9);   \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 10, _C, _1MMA_ * 10); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 11, _C, _1MMA_ * 11); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 12, _C, _1MMA_ * 12); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 13, _C, _1MMA_ * 13); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 14, _C, _1MMA_ * 14); \
        WRITE_sRv1_SIZE1(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 15, _C, _1MMA_ * 15); \
    }

/////////////////////////
// tile_n_per_warp = 16
/////////////////////////

#define WRITE_sRv1_1x2(_sm_base_v1, _sRv1_write, _C)       \
    {                                                      \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write, _C, 0); \
    }

#define WRITE_sRv1_2x2(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _2MMA_ * 0); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _2MMA_ * 1); \
    }

#define WRITE_sRv1_4x2(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _2MMA_ * 0); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _2MMA_ * 1); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _2MMA_ * 2); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _2MMA_ * 3); \
    }

#define WRITE_sRv1_8x2(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _2MMA_ * 0); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _2MMA_ * 1); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _2MMA_ * 2); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _2MMA_ * 3); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _2MMA_ * 4); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _2MMA_ * 5); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _2MMA_ * 6); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _2MMA_ * 7); \
    }

#define WRITE_sRv1_16x2(_sm_base_v1, _sRv1_write, _C)                                                               \
    {                                                                                                               \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _2MMA_ * 0);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _2MMA_ * 1);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _2MMA_ * 2);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _2MMA_ * 3);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _2MMA_ * 4);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _2MMA_ * 5);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _2MMA_ * 6);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _2MMA_ * 7);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 8, _C, _2MMA_ * 8);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 9, _C, _2MMA_ * 9);   \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 10, _C, _2MMA_ * 10); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 11, _C, _2MMA_ * 11); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 12, _C, _2MMA_ * 12); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 13, _C, _2MMA_ * 13); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 14, _C, _2MMA_ * 14); \
        WRITE_sRv1_SIZE2(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 15, _C, _2MMA_ * 15); \
    }

/////////////////////////
// tile_n_per_warp = 32
/////////////////////////

#define WRITE_sRv1_1x4(_sm_base_v1, _sRv1_write, _C)       \
    {                                                      \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write, _C, 0); \
    }

#define WRITE_sRv1_2x4(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _4MMA_ * 0); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _4MMA_ * 1); \
    }

#define WRITE_sRv1_4x4(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _4MMA_ * 0); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _4MMA_ * 1); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _4MMA_ * 2); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _4MMA_ * 3); \
    }

#define WRITE_sRv1_8x4(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _4MMA_ * 0); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _4MMA_ * 1); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _4MMA_ * 2); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _4MMA_ * 3); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _4MMA_ * 4); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _4MMA_ * 5); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _4MMA_ * 6); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _4MMA_ * 7); \
    }

#define WRITE_sRv1_16x4(_sm_base_v1, _sRv1_write, _C)                                                               \
    {                                                                                                               \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _4MMA_ * 0);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _4MMA_ * 1);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _4MMA_ * 2);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _4MMA_ * 3);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _4MMA_ * 4);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _4MMA_ * 5);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _4MMA_ * 6);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _4MMA_ * 7);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 8, _C, _4MMA_ * 8);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 9, _C, _4MMA_ * 9);   \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 10, _C, _4MMA_ * 10); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 11, _C, _4MMA_ * 11); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 12, _C, _4MMA_ * 12); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 13, _C, _4MMA_ * 13); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 14, _C, _4MMA_ * 14); \
        WRITE_sRv1_SIZE4(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 15, _C, _4MMA_ * 15); \
    }

/////////////////////////
// tile_n_per_warp = 64
/////////////////////////

#define WRITE_sRv1_1x8(_sm_base_v1, _sRv1_write, _C)       \
    {                                                      \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write, _C, 0); \
    }

#define WRITE_sRv1_2x8(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _8MMA_ * 0); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _8MMA_ * 1); \
    }

#define WRITE_sRv1_4x8(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _8MMA_ * 0); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _8MMA_ * 1); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _8MMA_ * 2); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _8MMA_ * 3); \
    }

#define WRITE_sRv1_8x8(_sm_base_v1, _sRv1_write, _C)                                                              \
    {                                                                                                             \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 0, _C, _8MMA_ * 0); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 1, _C, _8MMA_ * 1); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 2, _C, _8MMA_ * 2); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 3, _C, _8MMA_ * 3); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 4, _C, _8MMA_ * 4); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 5, _C, _8MMA_ * 5); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 6, _C, _8MMA_ * 6); \
        WRITE_sRv1_SIZE8(_sm_base_v1, _sRv1_write + TILE_M_PER_MMA_HALF * TILE_N_V2_PER_CTA * 7, _C, _8MMA_ * 7); \
    }
