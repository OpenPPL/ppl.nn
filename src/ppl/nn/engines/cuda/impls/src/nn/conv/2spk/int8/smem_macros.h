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
// common write shared memory macros
/////////////////////////////////////////////////////

#define WRITE_sUv4_SIZE_16TH(_sm_base_v4, _sm_off, _reg) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 16 ))  \
                _sm_base_v4[_sm_off] = _reg[0]; \
        }

#define WRITE_sUv4_SIZE_8TH(_sm_base_v4, _sm_off, _reg) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 8 ))  \
                _sm_base_v4[_sm_off] = _reg[0]; \
        }

#define WRITE_sUv4_SIZE_QTR(_sm_base_v4, _sm_off, _reg) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 4 ))  \
                _sm_base_v4[_sm_off] = _reg[0]; \
        }

#define WRITE_sUv4_SIZE_HALF(_sm_base_v4, _sm_off, _reg) \
        { \
            if(tid < ( CTA_SIZE_IN_THD / 2 ))  \
                _sm_base_v4[_sm_off] = _reg[0]; \
        }

#define WRITE_sUv4_SIZE1(_sm_base_v4, _sm_off, _reg) \
        { \
            _sm_base_v4[_sm_off] = _reg[0]; \
        }

#define WRITE_sUv4_SIZE2(_sm_base_v4, _sm_off, _reg) \
        { \
            _sm_base_v4[_sm_off] = _reg[0]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 1] = _reg[1]; \
        }

#define WRITE_sUv4_SIZE4(_sm_base_v4, _sm_off, _reg) \
        { \
            _sm_base_v4[_sm_off] = _reg[0]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 1] = _reg[1]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 2] = _reg[2]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 3] = _reg[3]; \
        }

#define WRITE_sUv4_SIZE8(_sm_base_v4, _sm_off, _reg) \
        { \
            _sm_base_v4[_sm_off] = _reg[0]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 1] = _reg[1]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 2] = _reg[2]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 3] = _reg[3]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 4] = _reg[4]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 5] = _reg[5]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 6] = _reg[6]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 7] = _reg[7]; \
        }

#define WRITE_sUv4_SIZE16(_sm_base_v4, _sm_off, _reg) \
        { \
            _sm_base_v4[_sm_off] = _reg[0]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 1]  = _reg[1];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 2]  = _reg[2];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 3]  = _reg[3];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 4]  = _reg[4];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 5]  = _reg[5];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 6]  = _reg[6];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 7]  = _reg[7];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 8]  = _reg[8];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 9]  = _reg[9];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 10] = _reg[10]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 11] = _reg[11]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 12] = _reg[12]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 13] = _reg[13]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 14] = _reg[14]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 15] = _reg[15]; \
        }

#define WRITE_sUv4_SIZE32(_sm_base_v4, _sm_off, _reg) \
        { \
            _sm_base_v4[_sm_off] = _reg[0]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 1]  = _reg[1];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 2]  = _reg[2];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 3]  = _reg[3];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 4]  = _reg[4];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 5]  = _reg[5];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 6]  = _reg[6];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 7]  = _reg[7];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 8]  = _reg[8];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 9]  = _reg[9];  \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 10] = _reg[10]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 11] = _reg[11]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 12] = _reg[12]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 13] = _reg[13]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 14] = _reg[14]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 15] = _reg[15]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 16] = _reg[16]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 17] = _reg[17]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 18] = _reg[18]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 19] = _reg[19]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 20] = _reg[20]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 21] = _reg[21]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 22] = _reg[22]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 23] = _reg[23]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 24] = _reg[24]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 25] = _reg[25]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 26] = _reg[26]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 27] = _reg[27]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 28] = _reg[28]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 29] = _reg[29]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 30] = _reg[30]; \
            _sm_base_v4[_sm_off + CTA_SIZE_IN_THD * 31] = _reg[31]; \
        }

////////////////////////////////////////////////////
// read shared memory macros
////////////////////////////////////////////////////

#define READ_sUv1_SIZE1(_reg, _reg_off, _smp_base_v1, _sUv1_read) \
        { \
            LDSM_ROW_X1_INST(_reg[_reg_off], _smp_base_v1 + _INT_TO_BYTE_ * (_sUv1_read) ); \
        }

#define READ_sUv1_SIZE2(_reg, _reg_off, _smp_base_v1, _sUv1_read) \
        { \
            LDSM_ROW_X2_INST(_reg[_reg_off], _reg[_reg_off + 1], _smp_base_v1 + _INT_TO_BYTE_ * (_sUv1_read) ); \
        }

#define READ_sUv1_SIZE4(_reg, _reg_off, _smp_base_v1, _sUv1_read) \
        { \
            LDSM_ROW_X4_INST(_reg[_reg_off], _reg[_reg_off + 1], _reg[_reg_off + 2], _reg[_reg_off + 3], _smp_base_v1 + _INT_TO_BYTE_ * (_sUv1_read) ); \
        }

#if defined(USE_IMMA8816) || defined(USE_IMMA16816)

#define REG_sAv1_SIZE   (TILE_M_V1_PER_THD)
#define REG_sBv1_SIZE   (TILE_N_V2_PER_THD)

#define READ_sUv1_K1_1x1(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE1(_reg, 0, _smp_base_v1, _sUv1_read); \
        }

#define READ_sUv1_K1_1x2(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE2(_reg, 0, _smp_base_v1, _sUv1_read); \
        }

#define READ_sUv1_K1_1x4(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \
        }

#define READ_sUv1_K1_1x8(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \
            READ_sUv1_SIZE4(_reg, 4, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * WARP_SIZE_IN_THD); \
        }

#define READ_sUv1_K1_1x16(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0,  _smp_base_v1, _sUv1_read); \
            READ_sUv1_SIZE4(_reg, 4,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * WARP_SIZE_IN_THD * 1); \
            READ_sUv1_SIZE4(_reg, 8,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * WARP_SIZE_IN_THD * 2); \
            READ_sUv1_SIZE4(_reg, 12, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * WARP_SIZE_IN_THD * 3); \
        }

#define READ_sUv1_K1_2x1(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE2(_reg, 0, _smp_base_v1, _sUv1_read); \
        }

#define READ_sUv1_K1_2x2(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \
        }

#define READ_sUv1_K1_2x4(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \
            READ_sUv1_SIZE4(_reg, 4, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * WARP_SIZE_IN_THD); \
        }

#define READ_sUv1_K1_2x8(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0,  _smp_base_v1, _sUv1_read); \
            READ_sUv1_SIZE4(_reg, 4,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * WARP_SIZE_IN_THD * 1); \
            READ_sUv1_SIZE4(_reg, 8,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * WARP_SIZE_IN_THD * 2); \
            READ_sUv1_SIZE4(_reg, 12, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * WARP_SIZE_IN_THD * 3); \
        }

#elif defined(USE_IMMA16832)

#define REG_sAv1_SIZE   (TILE_M_V1_PER_THD * _K32_TO_2K16_)
#define REG_sBv1_SIZE   (TILE_N_V2_PER_THD * _K32_TO_2K16_)

#define READ_sUv1_K2_1x1(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE2(_reg, 0, _smp_base_v1, _sUv1_read); \
        }

#define READ_sUv1_K2_1x2(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \
        }

#define READ_sUv1_K2_1x4(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \
            READ_sUv1_SIZE4(_reg, 4, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2)); \
        }

#define READ_sUv1_K2_1x8(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0,  _smp_base_v1, _sUv1_read); \
            READ_sUv1_SIZE4(_reg, 4,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 1); \
            READ_sUv1_SIZE4(_reg, 8,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 2); \
            READ_sUv1_SIZE4(_reg, 12, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 3); \
        }

#define READ_sUv1_K2_2x1(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \
        }

#define READ_sUv1_K2_2x2(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0, _smp_base_v1, _sUv1_read); \
            READ_sUv1_SIZE4(_reg, 4, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2)); \
        }

#define READ_sUv1_K2_2x4(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0,  _smp_base_v1, _sUv1_read); \
            READ_sUv1_SIZE4(_reg, 4,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 1); \
            READ_sUv1_SIZE4(_reg, 8,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 2); \
            READ_sUv1_SIZE4(_reg, 12, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 3); \
        }

#define READ_sUv1_K2_2x8(_reg, _smp_base_v1, _sUv1_read) \
        { \
            READ_sUv1_SIZE4(_reg, 0,  _smp_base_v1, _sUv1_read); \
            READ_sUv1_SIZE4(_reg, 4,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 1); \
            READ_sUv1_SIZE4(_reg, 8,  _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 2); \
            READ_sUv1_SIZE4(_reg, 12, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 3); \
            READ_sUv1_SIZE4(_reg, 16, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 4); \
            READ_sUv1_SIZE4(_reg, 20, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 5); \
            READ_sUv1_SIZE4(_reg, 24, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 6); \
            READ_sUv1_SIZE4(_reg, 28, _smp_base_v1, _sUv1_read + TILE_K_V4_PER_CTA * (WARP_SIZE_IN_THD / 2) * 7); \
        }

#endif
