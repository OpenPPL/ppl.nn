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

#define REDUCE_INT_SIZE4(_R, _R_off) \
        { \
            _R[0] = _R[0] + _R[_R_off]; \
            _R[1] = _R[1] + _R[_R_off + 1]; \
            _R[2] = _R[2] + _R[_R_off + 2]; \
            _R[3] = _R[3] + _R[_R_off + 3]; \
        }

#define REDUCE_INT_1x4(_R) \
        { \
            REDUCE_INT_SIZE4(_R, _4INT_); \
        }

#define REDUCE_INT_3x4(_R) \
        { \
            REDUCE_INT_SIZE4(_R, _4INT_); \
            REDUCE_INT_SIZE4(_R, _4INT_ * 2); \
            REDUCE_INT_SIZE4(_R, _4INT_ * 3); \
        }

/////////////////////////////////////////////////////
// read sRv4 macros
/////////////////////////////////////////////////////

#define READ_sRv4_SIZE1(_Rv4, _sm_base_v4, _sRv4_read) \
        { \
            if(dCv4_x_valid) \
            { \
                _Rv4[0] = _sm_base_v4[_sRv4_read]; \
            } \
            \
            _sRv4_read += CTA_SIZE_IN_THD * SWIZZLE_GROUP; \
        }

#define READ_sRv4_SIZE2(_Rv4, _sm_base_v4, _sRv4_read) \
        { \
            if(dCv4_x_valid) \
            { \
                _Rv4[0] = _sm_base_v4[_sRv4_read]; \
                _Rv4[1] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V4_PER_CTA * 1]; \
            } \
            \
            _sRv4_read += CTA_SIZE_IN_THD * SWIZZLE_GROUP; \
        }

#define READ_sRv4_SIZE4(_Rv4, _sm_base_v4, _sRv4_read) \
        { \
            if(dCv4_x_valid) \
            { \
                _Rv4[0] = _sm_base_v4[_sRv4_read]; \
                _Rv4[1] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V4_PER_CTA * 1]; \
                _Rv4[2] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V4_PER_CTA * 2]; \
                _Rv4[3] = _sm_base_v4[_sRv4_read + TILE_M_V1_PER_CTA * TILE_N_V4_PER_CTA * 3]; \
            } \
            \
            _sRv4_read += CTA_SIZE_IN_THD * SWIZZLE_GROUP; \
        }


/////////////////////////////////////////////////////
// write sRv2 macros
/////////////////////////////////////////////////////

// actually is THD_IN_N_PER_MMA
#define WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write, _i2C, _C_off) \
        { \
	    /*if(blockIdx.x+blockIdx.y+blockIdx.z==0)    printf("tid:%d\tv2_off:%d\tv2write:%d\trow_off:%d\n", tid, _sRv2_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA, _sRv2_write, smem_row_write_off); */\
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 0]; \
        }

#define WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write, _i2C, _C_off) \
        { \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 0]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x1) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 1]; \
        }

#define WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write, _i2C, _C_off) \
        { \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 0]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x1) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 1]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x2) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 2]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x3) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 3]; \
        }

#define WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write, _i2C, _C_off) \
        { \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x0) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 0]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x1) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 1]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x2) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 2]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x3) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 3]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x4) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 4]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x5) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 5]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x6) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 6]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x7) * TILE_N_V2_PER_MMA] = _i2C[_C_off + 7]; \
        }

/////////////////////////
// tile_n_per_warp = 8
/////////////////////////

#define WRITE_sRv2_1x1(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write, _i2C, 0); \
        }

#define WRITE_sRv2_2x1(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _1MMA_ * 0); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _1MMA_ * 1); \
        }

#define WRITE_sRv2_4x1(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _1MMA_ * 0); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _1MMA_ * 1); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2, _i2C, _1MMA_ * 2); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3, _i2C, _1MMA_ * 3); \
        }

#define WRITE_sRv2_8x1(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _1MMA_ * 0); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _1MMA_ * 1); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2, _i2C, _1MMA_ * 2); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3, _i2C, _1MMA_ * 3); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 4, _i2C, _1MMA_ * 4); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 5, _i2C, _1MMA_ * 5); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 6, _i2C, _1MMA_ * 6); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 7, _i2C, _1MMA_ * 7); \
        }

#define WRITE_sRv2_16x1(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0,  _i2C, _1MMA_ * 0); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1,  _i2C, _1MMA_ * 1); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2,  _i2C, _1MMA_ * 2); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3,  _i2C, _1MMA_ * 3); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 4,  _i2C, _1MMA_ * 4); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 5,  _i2C, _1MMA_ * 5); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 6,  _i2C, _1MMA_ * 6); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 7,  _i2C, _1MMA_ * 7); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 8,  _i2C, _1MMA_ * 8); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 9,  _i2C, _1MMA_ * 9); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 10, _i2C, _1MMA_ * 10); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 11, _i2C, _1MMA_ * 11); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 12, _i2C, _1MMA_ * 12); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 13, _i2C, _1MMA_ * 13); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 14, _i2C, _1MMA_ * 14); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 15, _i2C, _1MMA_ * 15); \
        }

/////////////////////////
// tile_n_per_warp = 16
/////////////////////////

#define WRITE_sRv2_1x2(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write, _i2C, 0); \
        }

#define WRITE_sRv2_2x2(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _2MMA_ * 0); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _2MMA_ * 1); \
        }

#define WRITE_sRv2_4x2(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _2MMA_ * 0); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _2MMA_ * 1); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2, _i2C, _2MMA_ * 2); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3, _i2C, _2MMA_ * 3); \
        }

#define WRITE_sRv2_8x2(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _2MMA_ * 0); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _2MMA_ * 1); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2, _i2C, _2MMA_ * 2); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3, _i2C, _2MMA_ * 3); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 4, _i2C, _2MMA_ * 4); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 5, _i2C, _2MMA_ * 5); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 6, _i2C, _2MMA_ * 6); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 7, _i2C, _2MMA_ * 7); \
        }

#define WRITE_sRv2_16x2(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0,  _i2C, _2MMA_ * 0); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1,  _i2C, _2MMA_ * 1); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2,  _i2C, _2MMA_ * 2); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3,  _i2C, _2MMA_ * 3); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 4,  _i2C, _2MMA_ * 4); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 5,  _i2C, _2MMA_ * 5); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 6,  _i2C, _2MMA_ * 6); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 7,  _i2C, _2MMA_ * 7); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 8,  _i2C, _2MMA_ * 8); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 9,  _i2C, _2MMA_ * 9); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 10, _i2C, _2MMA_ * 10); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 11, _i2C, _2MMA_ * 11); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 12, _i2C, _2MMA_ * 12); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 13, _i2C, _2MMA_ * 13); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 14, _i2C, _2MMA_ * 14); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 15, _i2C, _2MMA_ * 15); \
        }

/////////////////////////
// tile_n_per_warp = 32
/////////////////////////

#define WRITE_sRv2_1x4(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write, _i2C, 0); \
        }

#define WRITE_sRv2_2x4(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _4MMA_ * 0); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _4MMA_ * 1); \
        }

#define WRITE_sRv2_4x4(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _4MMA_ * 0); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _4MMA_ * 1); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2, _i2C, _4MMA_ * 2); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3, _i2C, _4MMA_ * 3); \
        }

#define WRITE_sRv2_8x4(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _4MMA_ * 0); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _4MMA_ * 1); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2, _i2C, _4MMA_ * 2); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3, _i2C, _4MMA_ * 3); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 4, _i2C, _4MMA_ * 4); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 5, _i2C, _4MMA_ * 5); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 6, _i2C, _4MMA_ * 6); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 7, _i2C, _4MMA_ * 7); \
        }

#define WRITE_sRv2_16x4(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0,  _i2C, _4MMA_ * 0); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1,  _i2C, _4MMA_ * 1); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2,  _i2C, _4MMA_ * 2); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3,  _i2C, _4MMA_ * 3); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 4,  _i2C, _4MMA_ * 4); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 5,  _i2C, _4MMA_ * 5); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 6,  _i2C, _4MMA_ * 6); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 7,  _i2C, _4MMA_ * 7); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 8,  _i2C, _4MMA_ * 8); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 9,  _i2C, _4MMA_ * 9); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 10, _i2C, _4MMA_ * 10); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 11, _i2C, _4MMA_ * 11); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 12, _i2C, _4MMA_ * 12); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 13, _i2C, _4MMA_ * 13); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 14, _i2C, _4MMA_ * 14); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 15, _i2C, _4MMA_ * 15); \
        }

/////////////////////////
// tile_n_per_warp = 64
/////////////////////////

#define WRITE_sRv2_1x8(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write, _i2C, 0); \
        }

#define WRITE_sRv2_2x8(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _8MMA_ * 0); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _8MMA_ * 1); \
        }

#define WRITE_sRv2_4x8(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _8MMA_ * 0); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _8MMA_ * 1); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2, _i2C, _8MMA_ * 2); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3, _i2C, _8MMA_ * 3); \
        }

#define WRITE_sRv2_8x8(_sm_base_v2, _sRv2_write, _i2C) \
        { \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 0, _i2C, _8MMA_ * 0); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 1, _i2C, _8MMA_ * 1); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 2, _i2C, _8MMA_ * 2); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 3, _i2C, _8MMA_ * 3); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 4, _i2C, _8MMA_ * 4); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 5, _i2C, _8MMA_ * 5); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 6, _i2C, _8MMA_ * 6); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_PER_SUB_MMA * TILE_N_V2_PER_CTA * 7, _i2C, _8MMA_ * 7); \
        }

