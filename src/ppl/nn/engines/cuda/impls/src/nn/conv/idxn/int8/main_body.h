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
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(TOTAL_KPARAM_LIST)
#endif
{
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__  * 1000  + __CUDACC_VER_MINOR__ * 10  >= 10020)
    int C[C_ITEMS_PER_THD];

    int16_t *  CvHalf = (int16_t *)  C;
    float2 *   fCv2   = (float2 *) C;

#if TILE_K_PER_STEP == 16
    int  * dAv1 = (int  *) dA;
    int  * dBv1 = (int  *) dB;
#elif TILE_K_PER_STEP == 32
    int2 * dAv2 = (int2 *) dA;
    int2 * dBv2 = (int2 *) dB;
#elif TILE_K_PER_STEP == 64
    int4 * dAv4 = (int4 *) dA;
    int4 * dBv4 = (int4 *) dB;
#endif

    int16_t * dCvHalf = (int16_t *) dC;

#pragma unroll
    for (int i = 0; i < C_ITEMS_PER_THD; i++) { C[i] = _ZERO_; }

    uint tid       =  threadIdx.x;
    uint tid_x     =  tid &  0x3;
    uint tid_y     = (tid & 0x1f) >> 2;

    uint warp_idx  = (tid >>  WARP_SIZE_IN_BITS) & (CTA_SIZE_X_IN_WARP - 1);
    uint warp_idy  =  tid >> (WARP_SIZE_IN_BITS  +  CTA_SIZE_X_IN_BITS);

    uint cta_idx   = blockIdx.y;
    uint cta_idy   = blockIdx.x;

    uint grp_id    = blockIdx.z;

    uint img_chl_per_grp_pad_v16 = img_chl_per_grp_pad >> 4;
#if TILE_K_PER_STEP == 16
    uint flt_chl_per_grp_pad_v4  = flt_chl_per_grp_pad >> 2;
#elif TILE_K_PER_STEP == 32
    uint flt_chl_per_grp_pad_v8  = flt_chl_per_grp_pad >> 3;
#elif TILE_K_PER_STEP == 64
    uint flt_chl_per_grp_pad_v16 = flt_chl_per_grp_pad >> 4;
#endif
    uint num_flt_per_grp_pad_v2 = num_flt_per_grp_pad >> 1;

    uint num_flt_v2 = num_flt_per_grp_pad_v2 * num_grp;
#if TILE_K_PER_STEP == 16
    uint flt_hwc_v4  = flt_hw * flt_chl_per_grp_pad_v4;
#elif TILE_K_PER_STEP == 32
    uint flt_hwc_v8  = flt_hw * flt_chl_per_grp_pad_v8;
#elif TILE_K_PER_STEP == 64
    uint flt_hwc_v16 = flt_hw * flt_chl_per_grp_pad_v16;
#endif

    bool dCv2_y_valid[BLK_M_PER_MMA];
    uint   dCv2_idy[BLK_M_PER_MMA];

    dCv2_idy[0] =  cta_idy     * TILE_M_V1_PER_CTA  +
                   warp_idy    * TILE_M_V1_PER_MMA  +
                   tid_y;

#if BLK_M_PER_MMA == 2
    dCv2_idy[1] =  dCv2_idy[0] + TILE_M_V1_PER_MMA_HALF;
#endif

    dCv2_y_valid[0] = (dCv2_idy[0] < out_nhw);
#if BLK_M_PER_MMA == 2
    dCv2_y_valid[1] = (dCv2_idy[1] < out_nhw);
#endif

    bool dCv2_x_valid[NUM_N_STEPS];
    uint   dCv2_idx[NUM_N_STEPS];

    uint dCv2_idx_base  =  grp_id      * num_flt_per_grp_pad_v2  +
                           cta_idx     * TILE_N_V2_PER_CTA  +
                           warp_idx    * TILE_N_V2_PER_MMA  +
                           tid_x;
    uint dCv2_idx_bound = (grp_id + 1) * num_flt_per_grp_pad_v2;

#pragma unroll
    for(uint i = 0; i < NUM_N_STEPS; i++)
    {
        dCv2_idx[i]     =  dCv2_idx_base + i * TILE_N_V2_PER_STEP;
        dCv2_x_valid[i]   = (dCv2_idx[i] < dCv2_idx_bound);
    }

#if TILE_K_PER_STEP == 16
    const int ZEROv1 = 0;
#elif TILE_K_PER_STEP == 32
    const int2 ZEROv2 = {0, 0};
#elif TILE_K_PER_STEP == 64
    const int4 ZEROv4 = {0, 0, 0, 0};
#endif

    __shared__ int4 sm_base_v4[SM_IN_ID_SIZE + SM_IN_OFF_SIZE];

#if TILE_K_PER_STEP == 16
    int  reg_dAv1[REG_dAv1_SIZE];
    int  reg_dBv1[REG_dBv1_SIZE];
#elif TILE_K_PER_STEP == 32
    int2 reg_dAv2[REG_dAv2_SIZE];
    int2 reg_dBv2[REG_dBv2_SIZE];

    int * reg_dAv1 = (int *) reg_dAv2;
    int * reg_dBv1 = (int *) reg_dBv2;
#elif TILE_K_PER_STEP == 64
    int4 reg_dAv4[REG_dAv4_SIZE];
    int4 reg_dBv4[REG_dBv4_SIZE];

    int * reg_dAv1 = (int *) reg_dAv4;
    int * reg_dBv1 = (int *) reg_dBv4;
#endif

#if (TILE_M_PER_CTA < CTA_SIZE_IN_THD)
    if(tid < TILE_M_PER_CTA)
        SET_IN_Mv1_ID(tid, sm_base_v4);
#else
#pragma unroll
    for(int i = 0; i < (TILE_M_PER_CTA / CTA_SIZE_IN_THD); i++)
        SET_IN_Mv1_ID(tid + CTA_SIZE_IN_THD * i, sm_base_v4);
#endif

    if(tid < koff_num_pad)
        SET_IN_Kv16_OFF(tid, sm_base_v4);

#if TILE_K_PER_STEP == 16
    int   dBv1_off[READ_dBv1_STEPS];
    bool flt_n_valid[READ_dBv1_STEPS];
    int  flt_hwc_v4_acc = tid_x;

    for(int i = 0; i < READ_dBv1_STEPS; i++)
    {
        SET_dBv1_BOUND(i, dBv1_off[i], flt_n_valid[i]);
    }
#elif TILE_K_PER_STEP == 32
    int   dBv2_off[READ_dBv2_STEPS];
    bool flt_n_valid[READ_dBv2_STEPS];
    int  flt_hwc_v8_acc = tid_x;

    for(int i = 0; i < READ_dBv2_STEPS; i++)
    {
        SET_dBv2_BOUND(i, dBv2_off[i], flt_n_valid[i]);
    }
#elif TILE_K_PER_STEP == 64
    int   dBv4_off[READ_dBv4_STEPS];
    bool flt_n_valid[READ_dBv4_STEPS];
    int  flt_hwc_v16_acc = tid_x;

    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], flt_n_valid[i]);
    }
#endif
    
    uint sInId_read  =  warp_idy * TILE_M_PER_MMA + tid_y;

    uint sInOff_read =  tid_x + SM_IN_ID_SIZE;

#if TILE_K_PER_STEP == 16
    int4  inId[READ_dAv1_STEPS];
#elif TILE_K_PER_STEP == 32
    int4  inId[READ_dAv2_STEPS];
#elif TILE_K_PER_STEP == 64
    int4  inId[READ_dAv4_STEPS];
#endif
    int4 inOff;

    __syncthreads();

    for(int i = 0; i < NUM_M_STEPS; i++)
    {
        inId[i * BLK_M_PER_MMA]     = sm_base_v4[sInId_read + TILE_M_PER_STEP * i];
#if BLK_M_PER_MMA == 2
        inId[i * BLK_M_PER_MMA + 1] = sm_base_v4[sInId_read + TILE_M_PER_STEP * i + TILE_M_PER_MMA_HALF];
#endif
    }

    for(uint i = 0; i < kloop_num; i++)
    {
        inOff = sm_base_v4[sInOff_read];

#if TILE_K_PER_STEP == 16
        LOAD_dBv1(reg_dBv1, dBv1, dBv1_off);
        LOAD_dAv1(reg_dAv1, dAv1, inId, inOff);
#elif TILE_K_PER_STEP == 32
        LOAD_dBv2(reg_dBv2, dBv2, dBv2_off);
        LOAD_dAv2(reg_dAv2, dAv2, inId, inOff);
#elif TILE_K_PER_STEP == 64
        LOAD_dBv4(reg_dBv4, dBv4, dBv4_off);
        LOAD_dAv4(reg_dAv4, dAv4, inId, inOff);
#endif

        MMA_INSTS(C, reg_dAv1, reg_dBv1);

#if 2 * TILE_K_PER_STEP == TILE_K_PER_CTA

#if TILE_K_PER_STEP == 16
        inOff = sm_base_v4[sInOff_read + TILE_K_V4_PER_STEP];
#elif TILE_K_PER_STEP == 32
        inOff = sm_base_v4[sInOff_read + TILE_K_V8_PER_STEP];
#elif TILE_K_PER_STEP == 64
        inOff = sm_base_v4[sInOff_read + TILE_K_V16_PER_STEP];
#endif

        __syncthreads();

#if TILE_K_PER_STEP == 16
        LOAD_dBv1(reg_dBv1, dBv1, dBv1_off);
        LOAD_dAv1(reg_dAv1, dAv1, inId, inOff);
#elif TILE_K_PER_STEP == 32
        LOAD_dBv2(reg_dBv2, dBv2, dBv2_off);
        LOAD_dAv2(reg_dAv2, dAv2, inId, inOff);
#elif TILE_K_PER_STEP == 64
        LOAD_dBv4(reg_dBv4, dBv4, dBv4_off);
        LOAD_dAv4(reg_dAv4, dAv4, inId, inOff);
#endif

        MMA_INSTS(C, reg_dAv1, reg_dBv1);
#endif

#if TILE_K_PER_STEP == 16
        sInOff_read += TILE_K_V4_PER_CTA;
#elif TILE_K_PER_STEP == 32
        sInOff_read += TILE_K_V8_PER_CTA;
#elif TILE_K_PER_STEP == 64
        sInOff_read += TILE_K_V16_PER_CTA;
#endif
    }

#pragma unroll
    for(int step = 0; step < NUM_M_STEPS; step++)
    {
        uint Cv2_off  = step * TILE_N_V2_PER_THD * BLK_M_PER_MMA;

#if defined(ENABLE_FUSE)
        float2 de_scale_v2[NUM_N_STEPS];
        int2 * Cv2 = (int2 *) C;
        GET_DEQUANTSCALE_V2(de_scale_v2, d_flt_scale, in_scale);
        DEQUANT_V2(fCv2, Cv2, de_scale_v2);
#endif

        ADD_BIAS_V2(has_bias, bias);

#if defined(ENABLE_FUSE)
        uint concat_v2_off0 = 0;
#if BLK_M_PER_MMA == 2
        uint concat_v2_off1 = 0;
#endif

        FUSE_RELU_V2(has_relu);
        FUSE_CLIP_V2(has_clip, clip_max, clip_min);
        // FUSE_PRELU_V2(has_prelu, prelu, leaky);

        FUSE_ELT_V2(has_elt, pre_data);
        FUSE_RELU_V2(has_elt_relu);
        FUSE_CLIP_V2(has_elt_clip, elt_clip_max, elt_clip_min);
        // FUSE_PRELU_V2(has_elt_prelu, elt_prelu, elt_leaky);

#if BLK_M_PER_MMA == 1
        SET_CONCAT_OFF_V2(has_concat, concat_v2_off0);
#elif BLK_M_PER_MMA == 2
        SET_CONCAT_OFF_V2(has_concat, concat_v2_off0, concat_v2_off1);
#endif

        QUANT_V2(Cv2, fCv2, out_scale);
#endif

#if BLK_M_PER_MMA == 1
        OUTPUT_BY_HALF_X1();
#elif BLK_M_PER_MMA == 2
        OUTPUT_BY_HALF_X2();
#endif
    }

#endif
}
