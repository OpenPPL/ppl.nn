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

#if defined(ENABLE_SPLITK)
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(SPK_KPARAM_LIST)
#elif defined(ENABLE_FUSE)
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(TOTAL_KPARAM_LIST)
#endif
{
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__  * 1000  + __CUDACC_VER_MINOR__ * 10  >= 10020)
    int4 Cv4[Cv4_ITEMS_PER_THD];

    __half *hC = (__half *)Cv4;
    int *C     = (int *)Cv4;

#pragma unroll
    for (int i = 0; i < HC_ITEMS_PER_THD; i++) {
        hC[i] = _HALF_ZERO_;
    }

    int4 Rv4[OUTPUT_BLKS_PER_STEP];

#if defined(ENABLE_FUSE)
    __half2 *h2R = (__half2 *)Rv4;
    // __half *hR = (__half *)Rv4;
#endif

    uint tid       =  threadIdx.x;

    uint local_tid =  tid & 0x1f;

    uint warp_idx  = (tid >>  WARP_SIZE_IN_BITS) & (CTA_SIZE_X_IN_WARP - 1);
    uint warp_idy  =  tid >> (WARP_SIZE_IN_BITS  +  CTA_SIZE_X_IN_BITS);

    uint ldg_idx   =  tid % TILE_K_V8_PER_CTA;
    uint ldg_idy   =  tid / TILE_K_V8_PER_CTA;

#if TILE_K_PER_CTA == 8
    uint sts_idx   =   0;
    uint sts_idy   =   tid;
#elif TILE_K_PER_CTA == 16
    uint sts_idx   = ((tid & 0x1) ^ ( (tid & 0xf) >> 3));
    uint sts_idy   =   tid >> 1;
#elif TILE_K_PER_CTA == 32
    uint sts_idx   = ((tid & 0x3) ^ ( (tid & 0x1f) >> 3));
    uint sts_idy   =   tid >> 2;
#elif TILE_K_PER_CTA == 64
    uint sts_idx   = ((tid & 0x7) ^ ( (tid & 0x3f) >> 3));
    uint sts_idy   =   tid >> 3;
#endif

    uint out_tid   =  warp_idy * WARP_SIZE_IN_THD + local_tid;

    /////////////////////////
    //  cta layout
    uint cta_idx   = 0;
    uint cta_idy   = 0;

    uint lsb_y_mask = 0x7;
    uint lsb_y_bits = 3;

    while(1)
    {
        uint msb_cta_y  =  blockIdx.y & (~lsb_y_mask);
        uint lsb_cta_y  =  blockIdx.y &   lsb_y_mask;

        uint flip_cta_y =  blockIdx.y &  (lsb_y_mask + 0x1);
        uint tail_cta_y =  blockIdx.y |   lsb_y_mask;

        uint local_cta_id = lsb_cta_y * gridDim.x + blockIdx.x;

        if(tail_cta_y < gridDim.y)
        {
            cta_idy = msb_cta_y | (local_cta_id & lsb_y_mask);

            cta_idx = local_cta_id >> lsb_y_bits;
            if(flip_cta_y) cta_idx = gridDim.x + (~cta_idx);

            break;
        }
        else {
            lsb_y_mask = lsb_y_mask >> 1;
            lsb_y_bits = lsb_y_bits -  1;
        }
    }

#if defined(ENABLE_SPLITK)
    uint grp_id    = blockIdx.z % num_grp;
    uint spk_id    = blockIdx.z / num_grp;

    uint num_chl_per_spk = (spk_id != splitk - 1) ? num_chl_per_spk_head : num_chl_per_spk_tail;
#elif defined(ENABLE_FUSE)
    uint grp_id    = blockIdx.z;
#endif

    uint num_chl_per_grp_pad_v8 = num_chl_per_grp_pad >> 3;
    uint num_flt_per_grp_pad_v8 = num_flt_per_grp_pad >> 3;

    uint   dCv4_idx[OUTPUT_BLKS_PER_STEP];
    bool dCv4_x_valid[OUTPUT_BLKS_PER_STEP];

    uint dCv4_idy   =  cta_idy  * TILE_M_V8_PER_CTA  +
                       out_tid  % TILE_M_V8_PER_CTA;

    dCv4_idx[0]     =  cta_idx  * TILE_N_V1_PER_CTA  +
                       warp_idx * TILE_N_V1_PER_WARP +
                       out_tid  / TILE_M_V8_PER_CTA;

#if TILE_M_PER_WARP == 8
    bool dCv4_y_valid =  (dCv4_idy < num_flt_per_grp_pad_v8) & ((out_tid / TILE_M_V8_PER_CTA) < TILE_N_PER_MMA * _2MMA_);
#elif TILE_M_PER_WARP == 16 || TILE_M_PER_WARP == 32 || TILE_M_PER_WARP == 64
    bool dCv4_y_valid =  (dCv4_idy < num_flt_per_grp_pad_v8) & ((out_tid / TILE_M_V8_PER_CTA) < TILE_N_PER_MMA);
#endif

#pragma unroll
    for(int i = 0; i < OUTPUT_BLKS_PER_STEP; i++)
    {
        dCv4_idx[i]   =  dCv4_idx[0] + OUTPUT_SIZE_X_IN_THD * i;
        dCv4_x_valid[i] = (dCv4_idx[i] / out_hw) < in_num;
    }

#if defined(ENABLE_SPLITK)
    uint dCv4_base  =  spk_id * num_flt_per_grp_pad_v8 * num_grp * out_hw * in_num +
                       grp_id * num_flt_per_grp_pad_v8 + dCv4_idy;
#elif defined(ENABLE_FUSE)
    uint dCv4_base  =  grp_id * num_flt_per_grp_pad_v8 + dCv4_idy;
#endif

    uint mma_idy    =  local_tid %  MMA_SIZE_Y_IN_THD;
    uint mma_idx    =  local_tid >> MMA_SIZE_Y_IN_BITS;

    uint smem_row_write_id  =  (warp_idy * TILE_M_V8_PER_WARP) / SMEM_ROW_V4_SIZE;
    uint smem_row_write_off = ((warp_idy * TILE_M_V8_PER_WARP) ^ (mma_idx  / M_ROWS_PER_SMEM_ROW)
                       ) % SMEM_ROW_V4_SIZE;

#if TILE_M_PER_WARP == 8
    uint sRv1_write =  warp_idx   * TILE_M_V2_PER_CTA    * TILE_N_V1_PER_MMA    * _2MMA_    +
#elif TILE_M_PER_WARP == 16 || TILE_M_PER_WARP == 32 || TILE_M_PER_WARP == 64
    uint sRv1_write =  warp_idx   * TILE_M_V2_PER_CTA    * TILE_N_V1_PER_MMA    +
#endif
                       mma_idx    * TILE_M_V2_PER_CTA    +
                       smem_row_write_id  * SMEM_ROW_V1_SIZE     +
                       mma_idy;

    uint sMmaRd_idy =  out_tid    % TILE_M_V8_PER_CTA;
    uint sMmaRd_idx =  out_tid    / TILE_M_V8_PER_CTA; 

    uint smem_row_read_id  =  sMmaRd_idy / SMEM_ROW_V4_SIZE;
    uint smem_row_read_off =  sMmaRd_idy % SMEM_ROW_V4_SIZE;

#if TILE_M_PER_WARP == 8
    uint sRv4_read  =  warp_idx                          * TILE_M_V8_PER_CTA    * TILE_N_PER_MMA          * _2MMA_   +
#elif TILE_M_PER_WARP == 16 || TILE_M_PER_WARP == 32 || TILE_M_PER_WARP == 64
    uint sRv4_read  =  warp_idx                          * TILE_M_V8_PER_CTA    * TILE_N_PER_MMA          +
#endif
                      (sMmaRd_idx / TILE_N_PER_MMA_HALF) * TILE_M_V8_PER_CTA    * TILE_N_PER_MMA_HALF     +
                      (sMmaRd_idx % TILE_N_PER_MMA_HALF) * TILE_M_V8_PER_CTA    +
                       smem_row_read_id  * SMEM_ROW_V4_SIZE     +
                    (((sMmaRd_idx % TILE_N_PER_MMA_HALF) / M_ROWS_PER_SMEM_ROW) ^ smem_row_read_off);

    const int4 ZEROv4 = {0, 0, 0, 0};

#if defined(FLT_SIZE3)
    int flt_hw_id  = 0;
    int flt_hw_bid = 0x1;

    int lut_id     = 0;
#elif defined(FLT_SIZEN)
    int  flt_h_id  = 0;
    int  flt_w_id  = 0;

    int lut_id     = 0;
#endif

#if defined(ENABLE_SPLITK)
    int  flt_c_v8_end = (spk_id * num_chl_per_spk_head + num_chl_per_spk) >> 3;
    int  flt_c_v8_id  = ldg_idx + ((spk_id * num_chl_per_spk_head) >> 3);
#elif defined(ENABLE_FUSE)
    int  flt_c_v8_end = num_chl_per_grp_pad_v8;
    int  flt_c_v8_id  = ldg_idx;
#endif

    bool flt_c_v8_valid  = flt_c_v8_id < flt_c_v8_end;

    int4 reg_dAv4[REG_dAv4_SIZE];
    int4 reg_dBv4[REG_dBv4_SIZE];

    int   dAv4_off[READ_dAv4_STEPS];
    bool flt_n_valid[READ_dAv4_STEPS];

    for(int i = 0; i < READ_dAv4_STEPS; i++)
    {
        SET_dAv4_BOUND(i, dAv4_off[i], flt_n_valid[i]);
    }


#if defined(FLT_SIZE1)
    int   dBv4_off[READ_dBv4_STEPS];
    bool in_hw_valid[READ_dBv4_STEPS];

    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], in_hw_valid[i]);
    }
#elif defined(FLT_SIZE3)
    int dBv4_off[READ_dBv4_STEPS];
    int in_hw_mask[READ_dBv4_STEPS];

    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], in_hw_mask[i]);
    }
#elif defined(FLT_SIZEN)
    int dBv4_off[READ_dBv4_STEPS];
    int   in_n_id[READ_dBv4_STEPS];
    int   in_h_id[READ_dBv4_STEPS];
    int   in_w_id[READ_dBv4_STEPS];

    int in_h_start[READ_dBv4_STEPS];
    int in_w_start[READ_dBv4_STEPS];

    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], in_n_id[i], in_h_start[i], in_w_start[i]);
        in_h_id[i] = in_h_start[i];
        in_w_id[i] = in_w_start[i];
    }
#endif

#if defined(USE_1BUF)
    __shared__ int4 sm_base_v4[SM_BASE_V4_1BUF];
#elif defined(USE_2BUF)
    __shared__ int4 sm_base_v4[SM_BASE_V4_2BUF];
#endif
    int * sm_base_v1 = (int *) sm_base_v4;
    
    uint32_t smp_base_v1;

    CVT_SM_PTR(smp_base_v1, sm_base_v1);

    uint sAv4_write =  sts_idy  * TILE_K_V8_PER_CTA + sts_idx;

#if defined(USE_1BUF)
    uint sBv4_write =  sAv4_write + SM_A_V4_1BUF;
#elif defined(USE_2BUF)
    uint sBv4_write =  sAv4_write + SM_A_V4_2BUF;
#endif

    uint lds_idy =  local_tid;
#if TILE_K_PER_CTA == 8
    uint lds_idx =  0;
#elif TILE_K_PER_CTA == 16
    uint lds_idx = (local_tid / K_ROWS_PER_SMEM_ROW) & 0x1;
#elif TILE_K_PER_CTA == 32
    uint lds_idx = (local_tid / K_ROWS_PER_SMEM_ROW) & 0x3;
#elif TILE_K_PER_CTA == 64
    uint lds_idx = (local_tid / K_ROWS_PER_SMEM_ROW) & 0x7;
#endif

    uint sAv1_read  =  warp_idy   * TILE_M_PER_WARP        * TILE_K_V2_PER_CTA +
#if TILE_M_PER_WARP == 8
                      (lds_idy    % WARP_SIZE_IN_THD_QTR)  * TILE_K_V2_PER_CTA +
#elif TILE_M_PER_WARP == 16
                      (lds_idy    % WARP_SIZE_IN_THD_HALF) * TILE_K_V2_PER_CTA +
#elif TILE_M_PER_WARP == 32 || TILE_M_PER_WARP == 64
                       lds_idy    * TILE_K_V2_PER_CTA      +
#endif
                       lds_idx    * _INT4_TO_4INT_;

    uint sBv1_read  =  warp_idx   * TILE_N_PER_WARP        * TILE_K_V2_PER_CTA +
#if TILE_N_PER_WARP == 16
                      (lds_idy    % WARP_SIZE_IN_THD_HALF) * TILE_K_V2_PER_CTA +
#elif TILE_N_PER_WARP == 32 || TILE_N_PER_WARP == 64 || TILE_N_PER_WARP == 128
                       lds_idy    * TILE_K_V2_PER_CTA      +
#endif
                       lds_idx    * _INT4_TO_4INT_         +
#if defined(USE_1BUF)
                       SM_A_V1_1BUF;
#elif defined(USE_2BUF)
                       SM_A_V1_2BUF;
#endif

    int db0_sBv1[REG_sBv1_SIZE];
#if TILE_K_PER_CTA == 16 || TILE_K_PER_CTA == 32 || TILE_K_PER_CTA == 64
    int db1_sBv1[REG_sBv1_SIZE];
#endif

    int db0_sAv1[REG_sAv1_SIZE];
#if TILE_K_PER_CTA == 16 || TILE_K_PER_CTA == 32 || TILE_K_PER_CTA == 64
    int db1_sAv1[REG_sAv1_SIZE];
#endif

#if defined(FLT_SIZE1)
    LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_n_valid);
    LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, in_hw_valid);

    FWD_FLT(flt_c_v8_id, flt_c_v8_valid);
#elif defined(FLT_SIZE3)
    LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_n_valid);
    LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_hw_bid);

    FWD_FLT(flt_hw_id, flt_hw_bid, flt_c_v8_id, flt_c_v8_valid);
    FWD_LUT(lut_id);
#elif defined(FLT_SIZEN)
    LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_n_valid);
    LOAD_dBv4(reg_dBv4, dB, dBv4_off, in_n_id, in_h_id, in_w_id);

    FWD_FLT(flt_h_id, flt_w_id, flt_c_v8_id, flt_c_v8_valid);
    FWD_LUT(lut_id);
#endif

    WRITE_sAv4(sm_base_v4, sAv4_write, reg_dAv4);
    WRITE_sBv4(sm_base_v4, sBv4_write, reg_dBv4);

    __syncthreads();

#if defined(USE_2BUF)
    SWITCH_BUFFER(sAv4_write, SM_A_V4_1BUF, 0);
    SWITCH_BUFFER(sBv4_write, SM_B_V4_1BUF, SM_A_V4_2BUF);
#endif

    READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
    READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

#if TILE_K_PER_CTA == 16 || TILE_K_PER_CTA == 32 || TILE_K_PER_CTA == 64
    FWD_KGROUP_STEP1(sAv1_read);
    FWD_KGROUP_STEP1(sBv1_read);
#endif

#if defined(ENABLE_SPLITK)
    for (uint j = 0; j < flt_hw * DivUp(num_chl_per_spk, TILE_K_PER_CTA); j++)
#elif defined(ENABLE_FUSE)
    for (uint j = 0; j < kloop_num; j++)
#endif
    {
#if defined(FLT_SIZE1)
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_n_valid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, in_hw_valid);

        FWD_FLT(flt_c_v8_id, flt_c_v8_valid);
#elif defined(FLT_SIZE3)
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_n_valid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v8_valid, flt_hw_bid);

        FWD_FLT(flt_hw_id, flt_hw_bid, flt_c_v8_id, flt_c_v8_valid);
        FWD_LUT(lut_id);
#elif defined(FLT_SIZEN)
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v8_valid, flt_n_valid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, in_n_id, in_h_id, in_w_id);

        FWD_FLT(flt_h_id, flt_w_id, flt_c_v8_id, flt_c_v8_valid);
        FWD_LUT(lut_id);
#endif

#if TILE_K_PER_CTA == 16 || TILE_K_PER_CTA == 32 || TILE_K_PER_CTA == 64
        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP2(sAv1_read);
        FWD_KGROUP_STEP2(sBv1_read);
#endif

        MMA_INSTS(C, db0_sBv1, db0_sAv1);

#if TILE_K_PER_CTA == 32 || TILE_K_PER_CTA == 64
        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP3(sAv1_read);
        FWD_KGROUP_STEP3(sBv1_read);

        MMA_INSTS(C, db1_sBv1, db1_sAv1);

        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP4(sAv1_read);
        FWD_KGROUP_STEP4(sBv1_read);
#endif

#if TILE_K_PER_CTA == 64
        MMA_INSTS(C, db0_sBv1, db0_sAv1);

        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP5(sAv1_read);
        FWD_KGROUP_STEP5(sBv1_read);

        MMA_INSTS(C, db1_sBv1, db1_sAv1);

        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP6(sAv1_read);
        FWD_KGROUP_STEP6(sBv1_read);

        MMA_INSTS(C, db0_sBv1, db0_sAv1);

        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP7(sAv1_read);
        FWD_KGROUP_STEP7(sBv1_read);

        MMA_INSTS(C, db1_sBv1, db1_sAv1);

        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP8(sAv1_read);
        FWD_KGROUP_STEP8(sBv1_read);
#endif

#if defined(USE_1BUF)
        __syncthreads();
#endif

        WRITE_sAv4(sm_base_v4, sAv4_write, reg_dAv4);
        WRITE_sBv4(sm_base_v4, sBv4_write, reg_dBv4);

#if TILE_K_PER_CTA == 16
        MMA_INSTS(C, db1_sBv1, db1_sAv1);
#elif TILE_K_PER_CTA == 32 || TILE_K_PER_CTA == 64
        MMA_INSTS(C, db0_sBv1, db0_sAv1);
#endif

#if defined(USE_2BUF)
        SWITCH_BUFFER(sAv4_write, SM_A_V4_1BUF, 0);
        SWITCH_BUFFER(sBv4_write, SM_B_V4_1BUF, SM_A_V4_2BUF);

        SWITCH_BUFFER(sAv1_read,  SM_A_V1_1BUF, 0);
        SWITCH_BUFFER(sBv1_read,  SM_B_V1_1BUF, SM_A_V1_2BUF);
#endif

        __syncthreads();

        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

#if TILE_K_PER_CTA == 16 || TILE_K_PER_CTA == 32 || TILE_K_PER_CTA == 64
        FWD_KGROUP_STEP1(sAv1_read);
        FWD_KGROUP_STEP1(sBv1_read);
#endif

#if TILE_K_PER_CTA == 32 || TILE_K_PER_CTA == 64
        MMA_INSTS(C, db1_sBv1, db1_sAv1);
#endif
    }

    __syncthreads();

#if defined(ENABLE_FUSE)
    uint concat_v4_off[OUTPUT_BLKS_PER_STEP];

#pragma unroll
    for (int i = 0; i < OUTPUT_BLKS_PER_STEP; i++) { concat_v4_off[i] = 0; }
#endif

#if TILE_M_PER_WARP == 8
#pragma unroll
    for(int s = 0; s < OUTPUT_STEPS / _2MMA_; s++)
    {
        WRITE_sRv1(sm_base_v1, sRv1_write, C, s * BLK_N_PER_MMA * NUM_M_STEPS * _2MMA_);
#elif TILE_M_PER_WARP == 16 || TILE_M_PER_WARP == 32 || TILE_M_PER_WARP == 64
#pragma unroll
    for(int s = 0; s < OUTPUT_STEPS; s++)
    {
        WRITE_sRv1(sm_base_v1, sRv1_write, C, s * BLK_N_PER_MMA * NUM_M_STEPS);
#endif

        __syncthreads();

        READ_sRv4(Rv4, sm_base_v4, sRv4_read);

        __syncthreads();

#if defined(ENABLE_FUSE)
        ADD_BIAS_V4(has_bias, bias);
#endif

#if defined(ENABLE_FUSE)

        FUSE_RELU_V4(has_relu);
        FUSE_CLIP_V4(has_clip, clip_max, clip_min);
        // FUSE_PRELU_V4(has_prelu, prelu, leaky);

        FUSE_ELT_V4(has_elt, pre_data);
        FUSE_RELU_V4(has_elt_relu);
        FUSE_CLIP_V4(has_elt_clip, elt_clip_max, elt_clip_min);
        // FUSE_PRELU_V4(has_elt_prelu, elt_prelu, elt_leaky);

        SET_CONCAT_OFF_V4(has_concat, concat_v4_off);
#endif

        OUTPUT_BY_INT4(Rv4);

    }

#endif // __CUDA_ARCH__
}
