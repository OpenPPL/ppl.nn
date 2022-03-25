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
// read sRv4 macros
/////////////////////////////////////////////////////

#define READ_sRv4_SIZE1(_Rv4, _sm_base_v4, _sRv4_read) \
        { \
            if(dCv4_y_valid) \
            { \
                _Rv4[0] = _sm_base_v4[_sRv4_read]; \
            } \
        }

#define READ_sRv4_SIZE2(_Rv4, _sm_base_v4, _sRv4_read) \
        { \
            if(dCv4_y_valid) \
            { \
                _Rv4[0] = _sm_base_v4[_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 0]; \
                _Rv4[1] = _sm_base_v4[_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 1]; \
            } \
        }

#define READ_sRv4_SIZE4(_Rv4, _sm_base_v4, _sRv4_read) \
        { \
            if(dCv4_y_valid) \
            { \
                _Rv4[0] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 0)]; \
                _Rv4[1] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 1)]; \
                _Rv4[2] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 2)]; \
                _Rv4[3] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 3)]; \
            } \
        }

#define READ_sRv4_SIZE8(_Rv4, _sm_base_v4, _sRv4_read) \
        { \
            if(dCv4_y_valid) \
            { \
                _Rv4[0] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 0)]; \
                _Rv4[1] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 1) ^ 4]; \
                _Rv4[2] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 2)]; \
                _Rv4[3] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 3) ^ 4]; \
                _Rv4[4] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 4)]; \
                _Rv4[5] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 5) ^ 4]; \
                _Rv4[6] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 6)]; \
                _Rv4[7] = _sm_base_v4[(_sRv4_read + WARP_SIZE_IN_THD * CTA_SIZE_Y_IN_WARP * 7) ^ 4]; \
            } \
        }

/////////////////////////////////////////////////////
// write sRv2 macros
/////////////////////////////////////////////////////

#define WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x0) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 0]; \
        }

#define WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x0) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 0]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x1) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 1]; \
        }

#define WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x0) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 0]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x1) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 1]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x2) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 2]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x3) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 3]; \
        }

#define WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x0) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 0]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x1) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 1]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x2) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 2]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x3) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 3]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x4) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 4]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x5) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 5]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x6) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 6]; \
            _sm_base_v2[_sRv2_write + (smem_row_write_off ^ 0x7) * TILE_M_V2_PER_MMA] = _Cv2[_Cv2_off + 7]; \
        }

#if defined(USE_IMMA16816) || defined(USE_IMMA16832)

/////////////////////////
// tile_m_per_warp = 8
/////////////////////////

#define WRITE_sRv2_1x1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off); \
        }

#define WRITE_sRv2_1x2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 0, _Cv2, _Cv2_off + _1MMA_ * 0); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 1, _Cv2, _Cv2_off + _1MMA_ * 1); \
        }

#define WRITE_sRv2_1x4(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 0, _Cv2, _Cv2_off + _1MMA_ * 0); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 1, _Cv2, _Cv2_off + _1MMA_ * 1); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 2, _Cv2, _Cv2_off + _1MMA_ * 2); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 3, _Cv2, _Cv2_off + _1MMA_ * 3); \
        }

/////////////////////////
// tile_m_per_warp = 16
/////////////////////////

#define WRITE_sRv2_2x1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off); \
        }

#define WRITE_sRv2_2x2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 0, _Cv2, _Cv2_off + _2MMA_ * 0); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 1, _Cv2, _Cv2_off + _2MMA_ * 1); \
        }

/////////////////////////
// tile_m_per_warp = 32
/////////////////////////

#define WRITE_sRv2_4x1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off); \
        }

#define WRITE_sRv2_4x2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 0, _Cv2, _Cv2_off + _4MMA_ * 0); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 1, _Cv2, _Cv2_off + _4MMA_ * 1); \
        }

/////////////////////////
// tile_m_per_warp = 64
/////////////////////////

#define WRITE_sRv2_8x1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off); \
        }

#define WRITE_sRv2_8x2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 0, _Cv2, _Cv2_off + _8MMA_ * 0); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA_HALF * 1, _Cv2, _Cv2_off + _8MMA_ * 1); \
        }

#elif defined(USE_IMMA8816)

/////////////////////////
// tile_m_per_warp = 8
/////////////////////////

#define WRITE_sRv2_1x1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off); \
        }

#define WRITE_sRv2_1x2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 0, _Cv2, _Cv2_off + _1MMA_ * 0); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 1, _Cv2, _Cv2_off + _1MMA_ * 1); \
        }

#define WRITE_sRv2_1x4(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 0, _Cv2, _Cv2_off + _1MMA_ * 0); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 1, _Cv2, _Cv2_off + _1MMA_ * 1); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 2, _Cv2, _Cv2_off + _1MMA_ * 2); \
            WRITE_sRv2_SIZE1(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 3, _Cv2, _Cv2_off + _1MMA_ * 3); \
        }

/////////////////////////
// tile_m_per_warp = 16
/////////////////////////

#define WRITE_sRv2_2x1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off); \
        }

#define WRITE_sRv2_2x2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 0, _Cv2, _Cv2_off + _2MMA_ * 0); \
            WRITE_sRv2_SIZE2(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 1, _Cv2, _Cv2_off + _2MMA_ * 1); \
        }

/////////////////////////
// tile_m_per_warp = 32
/////////////////////////

#define WRITE_sRv2_4x1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off); \
        }

#define WRITE_sRv2_4x2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 0, _Cv2, _Cv2_off + _4MMA_ * 0); \
            WRITE_sRv2_SIZE4(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 1, _Cv2, _Cv2_off + _4MMA_ * 1); \
        }

/////////////////////////
// tile_m_per_warp = 64
/////////////////////////

#define WRITE_sRv2_8x1(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off); \
        }

#define WRITE_sRv2_8x2(_sm_base_v2, _sRv2_write, _Cv2, _Cv2_off) \
        { \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 0, _Cv2, _Cv2_off + _8MMA_ * 0); \
            WRITE_sRv2_SIZE8(_sm_base_v2, _sRv2_write + TILE_M_V2_PER_CTA * TILE_N_PER_MMA * 1, _Cv2, _Cv2_off + _8MMA_ * 1); \
        }

#endif
