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
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 10020)
    int C[C_ITEMS_PER_THD];

    __half *hC   = (__half *)C;

#if TILE_K_PER_STEP == 8
    int *dAv1 = (int *)dA;
    int *dBv1 = (int *)dB;
#elif TILE_K_PER_STEP == 16
    int2 *dAv2 = (int2 *) dA;
    int2 *dBv2 = (int2 *) dB;
#elif TILE_K_PER_STEP == 32
    int4 *dAv4 = (int4 *) dA;
    int4 *dBv4 = (int4 *) dB;
#endif
    int *dCv1 = (int *)dC;

#pragma unroll
    for (int i = 0; i < HC_ITEMS_PER_THD; i++) {
        hC[i] = _HALF_ZERO_;
    }

    uint tid   = threadIdx.x;
    uint tid_x = tid & 0x3;
    uint tid_y = (tid & 0x1f) >> 2;

    uint warp_idx = (tid >> WARP_SIZE_IN_BITS) & (CTA_SIZE_X_IN_WARP - 1);
    uint warp_idy = tid >> (WARP_SIZE_IN_BITS + CTA_SIZE_X_IN_BITS);

    uint cta_idx = blockIdx.y;
    uint cta_idy = blockIdx.x;

    uint grp_id = blockIdx.z;

    uint in_chl_per_grp_pad_v8 = in_chl_per_grp_pad >> 3;
#if TILE_K_PER_STEP == 8
    uint flt_chl_per_grp_pad_v2 = flt_chl_per_grp_pad >> 1;
#elif TILE_K_PER_STEP == 16
    uint flt_chl_per_grp_pad_v4 = flt_chl_per_grp_pad >> 2;
#elif TILE_K_PER_STEP == 32
    uint flt_chl_per_grp_pad_v8 = flt_chl_per_grp_pad >> 3;
#endif
    uint num_flt_per_grp_pad_v2 = num_flt_per_grp_pad >> 1;

    uint num_flt_v2 = num_flt_per_grp_pad_v2 * num_grp;
#if TILE_K_PER_STEP == 8
    uint flt_hwc_v2 = flt_hw * flt_chl_per_grp_pad_v2;
#elif TILE_K_PER_STEP == 16
    uint flt_hwc_v4 = flt_hw * flt_chl_per_grp_pad_v4;
#elif TILE_K_PER_STEP == 32
    uint flt_hwc_v8 = flt_hw * flt_chl_per_grp_pad_v8;
#endif

    bool dCv1_y_valid[BLK_M_PER_MMA];
    uint dCv1_idy[BLK_M_PER_MMA];

    dCv1_idy[0] = cta_idy * TILE_M_V1_PER_CTA +
                  warp_idy * TILE_M_V1_PER_MMA +
                  tid_y;

    dCv1_idy[1] = dCv1_idy[0] + TILE_M_V1_PER_MMA_HALF;

    dCv1_y_valid[0] = (dCv1_idy[0] < out_nhw);
    dCv1_y_valid[1] = (dCv1_idy[1] < out_nhw);

    bool dCv1_x_valid[NUM_N_STEPS];
    uint dCv1_idx[NUM_N_STEPS];

    uint dCv1_idx_base = grp_id * num_flt_per_grp_pad_v2 +
                         cta_idx * TILE_N_V2_PER_CTA +
                         warp_idx * TILE_N_V2_PER_MMA +
                         tid_x;
    uint dCv1_idx_bound = (grp_id + 1) * num_flt_per_grp_pad_v2;

#pragma unroll
    for (uint i = 0; i < NUM_N_STEPS; i++) {
        dCv1_idx[i]     = dCv1_idx_base + i * TILE_N_V2_PER_STEP;
        dCv1_x_valid[i] = (dCv1_idx[i] < dCv1_idx_bound);
    }

#if TILE_K_PER_STEP == 8
    const int ZEROv1 = 0;
#elif TILE_K_PER_STEP == 16
    const int2 ZEROv2 = {0, 0};
#elif TILE_K_PER_STEP == 32
    const int4 ZEROv4 = {0, 0, 0, 0};
#endif

    __shared__ int4 sm_base_v4[SM_IN_ID_SIZE + SM_IN_OFF_SIZE];

#if TILE_K_PER_STEP == 8
    int reg_dAv1[REG_dAv1_SIZE];
    int reg_dBv1[REG_dBv1_SIZE];
#elif TILE_K_PER_STEP == 16
    int2 reg_dAv2[REG_dAv2_SIZE];
    int2 reg_dBv2[REG_dBv2_SIZE];

    int *reg_dAv1 = (int *)reg_dAv2;
    int *reg_dBv1 = (int *)reg_dBv2;
#elif TILE_K_PER_STEP == 32
    int4 reg_dAv4[REG_dAv4_SIZE];
    int4 reg_dBv4[REG_dBv4_SIZE];

    int *reg_dAv1 = (int *)reg_dAv4;
    int *reg_dBv1 = (int *)reg_dBv4;
#endif

#if (TILE_M_PER_CTA == 4 * CTA_SIZE_IN_THD)
    SET_IN_Mv1_ID(tid, sm_base_v4);
    SET_IN_Mv1_ID(tid + CTA_SIZE_IN_THD * 1, sm_base_v4);
    SET_IN_Mv1_ID(tid + CTA_SIZE_IN_THD * 2, sm_base_v4);
    SET_IN_Mv1_ID(tid + CTA_SIZE_IN_THD * 3, sm_base_v4);
#elif (TILE_M_PER_CTA == 2 * CTA_SIZE_IN_THD)
    SET_IN_Mv1_ID(tid, sm_base_v4);
    SET_IN_Mv1_ID(tid + CTA_SIZE_IN_THD, sm_base_v4);
#elif (TILE_M_PER_CTA == CTA_SIZE_IN_THD)
    SET_IN_Mv1_ID(tid, sm_base_v4);
#elif (TILE_M_PER_CTA < CTA_SIZE_IN_THD)
    if (tid < TILE_M_PER_CTA)
        SET_IN_Mv1_ID(tid, sm_base_v4);
#endif

    if (tid < koff_num_pad)
        SET_IN_Kv8_OFF(tid, sm_base_v4);

#if TILE_K_PER_STEP == 8
    int dBv1_off[READ_dBv1_STEPS];
    bool flt_n_valid[READ_dBv1_STEPS];
    int flt_hwc_v2_off = tid_x;

#pragma unroll
    for (int i = 0; i < READ_dBv1_STEPS; i++) {
        SET_dBv1_BOUND(i, dBv1_off[i], flt_n_valid[i]);
    }
#elif TILE_K_PER_STEP == 16
    int dBv2_off[READ_dBv2_STEPS];
    bool flt_n_valid[READ_dBv2_STEPS];
    int flt_hwc_v4_off = tid_x;

#pragma unroll
    for (int i = 0; i < READ_dBv2_STEPS; i++) {
        SET_dBv2_BOUND(i, dBv2_off[i], flt_n_valid[i]);
    }
#elif TILE_K_PER_STEP == 32
    int dBv4_off[READ_dBv4_STEPS];
    bool flt_n_valid[READ_dBv4_STEPS];
    int flt_hwc_v8_off = tid_x;

#pragma unroll
    for (int i = 0; i < READ_dBv4_STEPS; i++) {
        SET_dBv4_BOUND(i, dBv4_off[i], flt_n_valid[i]);
    }
#endif

    uint in_id_read = warp_idy * TILE_M_PER_MMA + tid_y;

    uint in_off_read = tid_x + SM_IN_ID_SIZE;

#if TILE_K_PER_STEP == 8
    int4 in_id[READ_dAv1_STEPS];
#elif TILE_K_PER_STEP == 16
    int4 in_id[READ_dAv2_STEPS];
#elif TILE_K_PER_STEP == 32
    int4 in_id[READ_dAv4_STEPS];
#endif
    int4 in_off;

    __syncthreads();

#pragma unroll
    for (int i = 0; i < NUM_M_STEPS; i++) {
        in_id[i * BLK_M_PER_MMA]     = sm_base_v4[in_id_read + TILE_M_PER_STEP * i];
        in_id[i * BLK_M_PER_MMA + 1] = sm_base_v4[in_id_read + TILE_M_PER_STEP * i + TILE_M_PER_MMA_HALF];
    }

    for (uint i = 0; i < kloop_num; i++) {
        in_off = sm_base_v4[in_off_read];

#if TILE_K_PER_STEP == 8
        LOAD_dBv1(reg_dBv1, dBv1, dBv1_off);
        LOAD_dAv1(reg_dAv1, dAv1, in_id, in_off);
#elif TILE_K_PER_STEP == 16
        LOAD_dBv2(reg_dBv2, dBv2, dBv2_off);
        LOAD_dAv2(reg_dAv2, dAv2, in_id, in_off);
#elif TILE_K_PER_STEP == 32
        LOAD_dBv4(reg_dBv4, dBv4, dBv4_off);
        LOAD_dAv4(reg_dAv4, dAv4, in_id, in_off);
#endif

        MMA_INSTS(C, reg_dAv1, reg_dBv1);

#if 2 * TILE_K_PER_STEP == TILE_K_PER_CTA

#if TILE_K_PER_STEP == 8
        in_off = sm_base_v4[in_off_read + TILE_K_V2_PER_STEP];
#elif TILE_K_PER_STEP == 16
        in_off = sm_base_v4[in_off_read + TILE_K_V4_PER_STEP];
#elif TILE_K_PER_STEP == 32
        in_off = sm_base_v4[in_off_read + TILE_K_V8_PER_STEP];
#endif

        __syncthreads();

#if TILE_K_PER_STEP == 8
        LOAD_dBv1(reg_dBv1, dBv1, dBv1_off);
        LOAD_dAv1(reg_dAv1, dAv1, in_id, in_off);
#elif TILE_K_PER_STEP == 16
        LOAD_dBv2(reg_dBv2, dBv2, dBv2_off);
        LOAD_dAv2(reg_dAv2, dAv2, in_id, in_off);
#elif TILE_K_PER_STEP == 32
        LOAD_dBv4(reg_dBv4, dBv4, dBv4_off);
        LOAD_dAv4(reg_dAv4, dAv4, in_id, in_off);
#endif

        MMA_INSTS(C, reg_dAv1, reg_dBv1);
#endif

#if TILE_K_PER_STEP == 8
        in_off_read += TILE_K_V2_PER_CTA;
#elif TILE_K_PER_STEP == 16
        in_off_read += TILE_K_V4_PER_CTA;
#elif TILE_K_PER_STEP == 32
        in_off_read += TILE_K_V8_PER_CTA;
#endif
    }

#pragma unroll
    for(int step = 0; step < NUM_M_STEPS; step++)
    {
        uint Cv1_off  = step * TILE_N_V2_PER_THD * BLK_M_PER_MMA;

        ADD_BIAS_V1(has_bias, bias);

#if defined(ENABLE_FUSE)
        uint concat_v1_off0 = 0;
        uint concat_v1_off1 = 0;

        FUSE_RELU_V1(has_relu);
        FUSE_CLIP_V1(has_clip, clip_max, clip_min);
#ifdef PPLNN_ENABLE_CUDA_JIT
        // FUSE_PRELU_V1(has_prelu, prelu, leaky);
#endif


        FUSE_ELT_V1(has_elt, pre_data);
        FUSE_RELU_V1(has_elt_relu);
        FUSE_CLIP_V1(has_elt_clip, elt_clip_max, elt_clip_min);
#ifdef PPLNN_ENABLE_CUDA_JIT
        // FUSE_PRELU_V1(has_elt_prelu, elt_prelu, elt_leaky);
#endif

        SET_CONCAT_OFF_V1(has_concat, concat_v1_off0, concat_v1_off1);
#endif

        OUTPUT_BY_INT1();
    }

#endif // __CUDA_ARCH__
}
