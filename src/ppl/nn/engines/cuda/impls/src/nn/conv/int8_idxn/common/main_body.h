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

    int2 *Cv2 = (int2 *) C;

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
    int16_t * dCv2 = (int16_t *) dC;

#pragma unroll
    for (int i = 0; i < C_ITEMS_PER_THD; i++) { C[i] = 0; }

    uint tid       =  threadIdx.x;
    uint tid_x     =  tid &  0x3;
    uint tid_y     = (tid & 0x1f) >> 2;

    uint warp_idx  = (tid >>  WARP_SIZE_IN_BITS) & (CTA_SIZE_X_IN_WARP - 1);
    uint warp_idy  =  tid >> (WARP_SIZE_IN_BITS  +  CTA_SIZE_X_IN_BITS);

    uint cta_idx   = blockIdx.y;
    uint cta_idy   = blockIdx.x;

    uint grp_id    = blockIdx.z;

    uint in_chl_per_grp_pad_v16 = in_chl_per_grp_pad >> 4;
#if TILE_K_PER_STEP == 16
    uint flt_chl_per_grp_pad_v4 = flt_chl_per_grp_pad >> 2;
#elif TILE_K_PER_STEP == 32
    uint flt_chl_per_grp_pad_v8 = flt_chl_per_grp_pad >> 3;
#elif TILE_K_PER_STEP == 64
    uint flt_chl_per_grp_pad_v16 = flt_chl_per_grp_pad >> 4;
#endif
    uint num_flt_per_grp_pad_v2 = num_flt_per_grp_pad >> 1;

    uint num_flt_v2 = num_flt_per_grp_pad_v2 * num_grp;
#if TILE_K_PER_STEP == 16
    uint flt_hwc_v2 = flt_hw * flt_chl_per_grp_pad_v4;
#elif TILE_K_PER_STEP == 32
    uint flt_hwc_v4 = flt_hw * flt_chl_per_grp_pad_v8;
#elif TILE_K_PER_STEP == 64
    uint flt_hwc_v8 = flt_hw * flt_chl_per_grp_pad_v16;
#endif

    bool dCv1_y_valid[BLK_M_PER_MMA];//FIXME assume as 1
    uint   dCv1_idy[BLK_M_PER_MMA];

    dCv1_idy[0] =  cta_idy     * TILE_M_V1_PER_CTA  +
                   warp_idy    * TILE_M_V1_PER_MMA  +
                   tid_y;

#pragma unroll
    for(int b = 1; b < BLK_M_PER_MMA; b++){
        dCv1_idy[b] =  dCv1_idy[b] + b * TILE_M_V1_PER_SUB_MMA;
    }

    bool dCv1_x_valid[NUM_N_STEPS];
    uint   dCv1_idx[NUM_N_STEPS];

    uint dCv1_idx_base  =  grp_id      * num_flt_per_grp_pad_v2  +
                           cta_idx     * TILE_N_V2_PER_CTA  +
                           warp_idx    * TILE_N_V2_PER_MMA  +
                           tid_x;
    uint dCv1_idx_bound = (grp_id + 1) * num_flt_per_grp_pad_v2;

#pragma unroll
    for(uint i = 0; i < NUM_N_STEPS; i++)
    {
        dCv1_idx[i]     =  dCv1_idx_base + i * TILE_N_V2_PER_STEP;
        dCv1_x_valid[i]   = (dCv1_idx[i] < dCv1_idx_bound);
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

#if (TILE_M_PER_CTA > CTA_SIZE_IN_THD)
    SET_IN_Mv1_ID(tid, sm_base_v4);
    SET_IN_Mv1_ID(tid + CTA_SIZE_IN_THD, sm_base_v4);
#elif (TILE_M_PER_CTA == CTA_SIZE_IN_THD)
    SET_IN_Mv1_ID(tid, sm_base_v4);
#elif (TILE_M_PER_CTA < CTA_SIZE_IN_THD)
    if(tid < TILE_M_PER_CTA)
	// tid: (base, w, h, n)
        SET_IN_Mv1_ID(tid, sm_base_v4);
#endif

    if(tid < koff_num_pad)
	// tid: (off, w, h, n)
        SET_IN_Kv8_OFF(tid, sm_base_v4);

#if TILE_K_PER_STEP == 16
    int   dBv1_off[READ_dBv1_STEPS];
    bool flt_n_valid[READ_dBv1_STEPS];
    int  flt_hwc_v2_off = tid_x;

#pragma unroll
    for(int i = 0; i < READ_dBv1_STEPS; i++)
    {
        SET_dBv1_BOUND(i, dBv1_off[i], flt_n_valid[i]);
    }
#elif TILE_K_PER_STEP == 32
    int   dBv2_off[READ_dBv2_STEPS];
    bool flt_n_valid[READ_dBv2_STEPS];
    int  flt_hwc_v4_off = tid_x;

#pragma unroll
    for(int i = 0; i < READ_dBv2_STEPS; i++)
    {
        SET_dBv2_BOUND(i, dBv2_off[i], flt_n_valid[i]);
    }
#elif TILE_K_PER_STEP == 64
    int   dBv4_off[READ_dBv4_STEPS];
    bool flt_n_valid[READ_dBv4_STEPS];
    int  flt_hwc_v8_off = tid_x;

#pragma unroll
    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], flt_n_valid[i]);
    }
#endif
    // nhw id
    uint in_id_read  =  warp_idy * TILE_M_PER_MMA + tid_y;
    // cin off
    uint in_off_read =  tid_x + SM_IN_ID_SIZE;

#if TILE_K_PER_STEP == 16
    int4  in_id[READ_dAv1_STEPS];
#elif TILE_K_PER_STEP == 32
    int4  in_id[READ_dAv2_STEPS];
#elif TILE_K_PER_STEP == 64
    int4  in_id[READ_dAv4_STEPS];
#endif
    int4 in_off;

    __syncthreads();

#pragma unroll
    for(int i = 0; i < NUM_M_STEPS; i++)
    {
#pragma unroll
        for(int b = 0; b < BLK_M_PER_MMA; b++)
        {
            in_id[i * BLK_M_PER_MMA + b] = sm_base_v4[in_id_read + TILE_M_PER_STEP * i + b*TILE_M_PER_SUB_MMA];
        }
    }

    for(uint i = 0; i < kloop_num; i++)
    {
        in_off = sm_base_v4[in_off_read];

#if TILE_K_PER_STEP == 16
        LOAD_dBv1(reg_dBv1, dBv1, dBv1_off);
        LOAD_dAv1(reg_dAv1, dAv1, in_id, in_off);
#elif TILE_K_PER_STEP == 32
        LOAD_dBv2(reg_dBv2, dBv2, dBv2_off);
        LOAD_dAv2(reg_dAv2, dAv2, in_id, in_off);
#elif TILE_K_PER_STEP == 64
        LOAD_dBv4(reg_dBv4, dBv4, dBv4_off);
        LOAD_dAv4(reg_dAv4, dAv4, in_id, in_off);
#endif

        MMA_INSTS(C, reg_dAv1, reg_dBv1);

#if 2 * TILE_K_PER_STEP == TILE_K_PER_CTA

#if TILE_K_PER_STEP == 16
        in_off = sm_base_v4[in_off_read + TILE_K_V4_PER_STEP];
#elif TILE_K_PER_STEP == 32
        in_off = sm_base_v4[in_off_read + TILE_K_V8_PER_STEP];
#elif TILE_K_PER_STEP == 64
        in_off = sm_base_v4[in_off_read + TILE_K_V16_PER_STEP];
#endif

        __syncthreads();

#if TILE_K_PER_STEP == 16
        LOAD_dBv1(reg_dBv1, dBv1, dBv1_off);
        LOAD_dAv1(reg_dAv1, dAv1, in_id, in_off);
#elif TILE_K_PER_STEP == 32
        LOAD_dBv2(reg_dBv2, dBv2, dBv2_off);
        LOAD_dAv2(reg_dAv2, dAv2, in_id, in_off);
#elif TILE_K_PER_STEP == 64
        LOAD_dBv4(reg_dBv4, dBv4, dBv4_off);
        LOAD_dAv4(reg_dAv4, dAv4, in_id, in_off);
#endif

        MMA_INSTS(C, reg_dAv1, reg_dBv1);
#endif

	// advance MMA_X_IN_THD
#if TILE_K_PER_STEP == 16
        in_off_read += TILE_K_V4_PER_CTA;
#elif TILE_K_PER_STEP == 32
        in_off_read += TILE_K_V8_PER_CTA;
#elif TILE_K_PER_STEP == 64
        in_off_read += TILE_K_V16_PER_CTA;
#endif
    }
    float2 deScale[NUM_N_STEPS];
    float2 *fCv2 = (float2*) C;
    int16_t outData[NUM_N_STEPS];
#pragma unroll
    for(int step = 0; step < NUM_M_STEPS; step++)
    {
#pragma unroll
	for(int b = 0; b < BLK_M_PER_MMA; b++)
        {
            dCv1_y_valid[b] = (dCv1_idy[b] < out_nhw);
	}

        uint Cv1_off  = step * TILE_N_V2_PER_THD * BLK_M_PER_MMA;

#if TILE_N_PER_WARP == 8
        LOAD_SCALE_x1(deScale, d_flt_scale);

        deQuantData_x1(fCv2, Cv2, deScale);

        ADD_BIAS_1x1_V1(has_bias, bias, step);

#if defined(ENABLE_FUSE)
        uint concat_v1_off0 = 0;

        FUSE_RELU_1x1_V1(has_relu);
        FUSE_CLIP_1x1_V1(has_clip, clip_max, clip_min);
        FUSE_PRELU_1x1_V1(has_prelu, prelu, leaky);

        FUSE_ELT_1x1_V1(has_elt, pre_data);
        FUSE_RELU_1x1_V1(has_elt_relu);
        FUSE_CLIP_1x1_V1(has_elt_clip, elt_clip_max, elt_clip_min);
        FUSE_PRELU_1x1_V1(has_elt_prelu, elt_prelu, elt_leaky);

        SET_CONCAT_OFF_V1(has_concat, concat_v1_off0);
#endif

        quantOutData_x1(Cv2, fCv2, out_scale);
	packChar2_x1(outData, Cv2);

        OUTPUT_1x1_BY_INT1();
#elif TILE_N_PER_WARP == 16
        LOAD_SCALE_x2(deScale, d_flt_scale);

        deQuantData_x2(fCv2, Cv2, deScale);

        ADD_BIAS_1x2_V1(has_bias, bias, step);

#if defined(ENABLE_FUSE)
        uint concat_v1_off0 = 0;

        FUSE_RELU_1x2_V1(has_relu);
        FUSE_CLIP_1x2_V1(has_clip, clip_max, clip_min);
        FUSE_PRELU_1x2_V1(has_prelu, prelu, leaky);

        FUSE_ELT_1x2_V1(has_elt, pre_data);
        FUSE_RELU_1x2_V1(has_elt_relu);
        FUSE_CLIP_1x2_V1(has_elt_clip, elt_clip_max, elt_clip_min);
        FUSE_PRELU_1x2_V1(has_elt_prelu, elt_prelu, elt_leaky);

        SET_CONCAT_OFF_V1(has_concat, concat_v1_off0);
#endif

        quantOutData_x2(Cv2, fCv2, out_scale);
	packChar2_x2(outData, Cv2);

        OUTPUT_1x2_BY_INT1();
#elif TILE_N_PER_WARP == 32
        LOAD_SCALE_x4(deScale, d_flt_scale);

        deQuantData_x4(fCv2, Cv2, deScale);

        ADD_BIAS_1x4_V1(has_bias, bias, step);

#if defined(ENABLE_FUSE)
        uint concat_v1_off0 = 0;

        FUSE_RELU_1x4_V1(has_relu);
        FUSE_CLIP_1x4_V1(has_clip, clip_max, clip_min);
        FUSE_PRELU_1x4_V1(has_prelu, prelu, leaky);

        FUSE_ELT_1x4_V1(has_elt, pre_data);
        FUSE_RELU_1x4_V1(has_elt_relu);
        FUSE_CLIP_1x4_V1(has_elt_clip, elt_clip_max, elt_clip_min);
        FUSE_PRELU_1x4_V1(has_elt_prelu, elt_prelu, elt_leaky);

        SET_CONCAT_OFF_V1(has_concat, concat_v1_off0);
#endif

        quantOutData_x4(Cv2, fCv2, out_scale);
	packChar2_x4(outData, Cv2);

        OUTPUT_1x4_BY_INT1();
#endif

#pragma unroll
        for(int b = 0; b < BLK_M_PER_MMA; b++){
            dCv1_idy[b] += TILE_M_PER_STEP;
	}
    }

#endif // __CUDA_ARCH__
}
