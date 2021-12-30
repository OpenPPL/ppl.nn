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
#elif defined(ENABLE_FUSE) || defined(ENABLE_SPLITF)
__global__ void __launch_bounds__(CTA_SIZE_IN_THD) KERNEL_NAME(TOTAL_KPARAM_LIST)
#endif
{
#if (__CUDA_ARCH__ >= 750) && (__CUDACC_VER_MAJOR__ * 1000 + __CUDACC_VER_MINOR__ * 10 >= 10020)

    int4 Cv4[Cv4_ITEMS_PER_THD];

    int2 * i2C = (int2 *) Cv4;
    int *    C = (int *)  Cv4;

#pragma unroll
    for (int i = 0; i < C_ITEMS_PER_THD; i++) { C[i] = _ZERO_; }

    int4  Rv4[INTER_SET_REDUCE_RATIO];


    uint tid       =  threadIdx.x;

    uint local_tid =  tid & 0x1f;

    uint set_tid   =  tid & (SET_SIZE_IN_THD - 1);

    uint set_id    = (tid >> SET_SIZE_IN_BITS) & 0x7;

    uint set_widx  = (set_tid >>  WARP_SIZE_IN_BITS) & (SET_SIZE_X_IN_WARP - 1);
    uint set_widy  =  set_tid >> (WARP_SIZE_IN_BITS  +  SET_SIZE_X_IN_BITS);
 
    uint ldg_idx   =  tid % TILE_K_V16_PER_CTA;
    uint ldg_idy   =  tid / TILE_K_V16_PER_CTA;

#if TILE_K_PER_CTA == 16
    uint sts_idx   =   0;
    uint sts_idy   =   tid;
#elif TILE_K_PER_CTA == 32
    uint sts_idx   = ((tid & 0x1) ^ ((tid & 0xf) >> 3));
    uint sts_idy   =   tid >> 1;
#elif TILE_K_PER_CTA == 64
    uint sts_idx   = ((tid & 0x3) ^ ((tid & 0x1f) >> 3));
    uint sts_idy   =   tid >> 2;
#elif TILE_K_PER_CTA == 128
    uint sts_idx   = ((tid & 0x7) ^ ((tid & 0x3f) >> 3));
    uint sts_idy   =   tid >> 3;
#elif TILE_K_PER_CTA == 256
    uint sts_idx   = ((tid & 0xf) ^ ((tid & 0x7f) >> 4));
    uint sts_idy   =   tid >> 4;
#endif

    uint cta_idx   = blockIdx.y;
    uint cta_idy   = blockIdx.x;

#if defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint spk_id    =  blockIdx.z %  splitk;
    uint spf_id    = (blockIdx.z % (splitk * flt_hw)) / splitk;
    uint grp_id    =  blockIdx.z / (splitk * flt_hw);

    uint num_chl_per_spk = (spk_id != splitk - 1) ? num_chl_per_spk_head: num_chl_per_spk_tail;
#elif defined(ENABLE_SPLITK) && !defined(ENABLE_SPLITF)
    uint spk_id    =  blockIdx.z %  splitk;
    uint grp_id    =  blockIdx.z /  splitk;

    uint num_chl_per_spk = (spk_id != splitk - 1) ? num_chl_per_spk_head: num_chl_per_spk_tail;
#elif !defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint spf_id    =  blockIdx.z %  flt_hw;
    uint grp_id    =  blockIdx.z /  flt_hw;
#elif defined(ENABLE_FUSE)
    uint grp_id    = blockIdx.z;
#endif

    uint num_chl_per_grp_pad_v16 = num_chl_per_grp_pad >> 4;
    uint num_flt_per_grp_pad_v4 = num_flt_per_grp_pad >> 2;

    uint dCv4_idy   =  cta_idy  * TILE_M_V1_PER_CTA  +
                       tid      / TILE_N_V4_PER_CTA;

    uint dCv4_idx   =  cta_idx  * TILE_N_V4_PER_CTA  +
                       tid      % TILE_N_V4_PER_CTA;

    bool dCv4_x_valid =  (dCv4_idx < num_flt_per_grp_pad_v4) & ((tid / TILE_N_V4_PER_CTA) < TILE_M_PER_CTA);

#if defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint dCv4_base  = (spf_id   * splitk + spk_id)  * num_flt_per_grp_pad_v4 * num_grp * out_hw * in_num +
                       grp_id   * num_flt_per_grp_pad_v4;
#elif defined(ENABLE_SPLITK) && !defined(ENABLE_SPLITF)
    uint dCv4_base  =  spk_id   * num_flt_per_grp_pad_v4 * num_grp * out_hw * in_num +
                       grp_id   * num_flt_per_grp_pad_v4;
#elif !defined(ENABLE_SPLITK) && defined(ENABLE_SPLITF)
    uint dCv4_base  =  spf_id   * num_flt_per_grp_pad_v4 * num_grp * out_hw * in_num +
                       grp_id   * num_flt_per_grp_pad_v4;
#elif defined(ENABLE_FUSE)
    uint dCv4_base  =  grp_id   * num_flt_per_grp_pad_v4;
#endif

    uint mma_idx    =  local_tid %  MMA_SIZE_X_IN_THD;
    uint mma_idy    =  local_tid >> MMA_SIZE_X_IN_BITS;

    //可以用宏变量，可扩展的在smem上进行swizzle的原因，是单个mma的一行的size是2的幂，和smem 32 bank size（128B）互为2的幂指的关系
    uint smem_row_write_id  =  (set_widx * TILE_N_V4_PER_WARP) / SMEM_ROW_V4_SIZE;
//FIXME
    // inter_mma_x
    uint smem_row_write_off = ((set_widx * TILE_N_IN_MMA_PER_WARP) ^ (mma_idy  / N_ROWS_PER_SMEM_ROW)
                       ) % MMAs_PER_REDUCE_ROW;//SMEM_ROW_V8_SIZE;//8 int32 才表示有可以放多少个mma的行

    // 2 int32 is v2
    uint sRv2_write =  set_id     * TILE_N_V2_PER_CTA    * TILE_M_V1_PER_CTA  +
                       set_widy   * TILE_N_V2_PER_CTA    * TILE_M_V1_PER_WARP +
                       mma_idy    * TILE_N_V2_PER_CTA    +
                       //smem_row_write_id  * SMEM_ROW_V1_SIZE     +
                       smem_row_write_id  * SMEM_ROW_V2_SIZE     +
                       mma_idx;

    uint red_read_idx =  tid % TILE_N_V4_PER_CTA;
    uint red_read_idy =  tid / TILE_N_V4_PER_CTA; 

    uint smem_row_read_id   =  red_read_idx / SMEM_ROW_V4_SIZE;
    uint smem_row_read_off  =  red_read_idx % SMEM_ROW_V4_SIZE;
    uint inner_mma_read_idx =  smem_row_read_off & (TILE_N_PER_MMA/_4INT_TO_INT4_-1);
    uint inter_mma_read_idx =  smem_row_read_off / (TILE_N_PER_MMA/_4INT_TO_INT4_);

//#define SWIZZLE_GROUP MAX( 1, 4/(CTA_SIZE_IN_THD / TILE_N_V4_PER_CTA) )
    uint sRv4_read[SWIZZLE_GROUP];
    sRv4_read[0]  =   red_read_idy * TILE_N_V4_PER_CTA    +
                      smem_row_read_id  * SMEM_ROW_V4_SIZE +
                    //(((red_read_idy % TILE_M_PER_MMA_HALF) / N_ROWS_PER_SMEM_ROW) ^ smem_row_read_off);//这里可以直接用smem_row_read_off的原因是刚好一个mma_seg是int4
                    //(((red_read_idy / N_ROWS_PER_SMEM_ROW) % (SMEM_ROW_BYTE_SIZE/TILE_N_PER_MMA/sizeof(int))) ^ inter_mma_read_idx) * (TILE_N_PER_MMA/_4INT_TO_INT4_) +
                    (((red_read_idy / N_ROWS_PER_SMEM_ROW) % MMAs_PER_REDUCE_ROW) ^ inter_mma_read_idx) * (TILE_N_PER_MMA/_4INT_TO_INT4_) +
		     inner_mma_read_idx;
#if SWIZZLE_GROUP == 2
    sRv4_read[1] = (red_read_idy+2) * TILE_N_V4_PER_CTA    +
                 smem_row_read_id  * SMEM_ROW_V4_SIZE +
                 ((((red_read_idy+2) / N_ROWS_PER_SMEM_ROW) % MMAs_PER_REDUCE_ROW) ^ inter_mma_read_idx) * (TILE_N_PER_MMA/_4INT_TO_INT4_) +
    	         inner_mma_read_idx;
#elif SWIZZLE_GROUP == 4
    sRv4_read[1] = red_read_idy * TILE_N_V4_PER_CTA    +
                 smem_row_read_id  * SMEM_ROW_V4_SIZE +
                 ((((red_read_idy + 1) / N_ROWS_PER_SMEM_ROW) % MMAs_PER_REDUCE_ROW) ^ inter_mma_read_idx) * (TILE_N_PER_MMA/_4INT_TO_INT4_) +
    	         inner_mma_read_idx;
    sRv4_read[2] = red_read_idy * TILE_N_V4_PER_CTA    +
                 smem_row_read_id  * SMEM_ROW_V4_SIZE +
                 ((((red_read_idy + 2) / N_ROWS_PER_SMEM_ROW) % MMAs_PER_REDUCE_ROW) ^ inter_mma_read_idx) * (TILE_N_PER_MMA/_4INT_TO_INT4_) +
    	         inner_mma_read_idx;
    sRv4_read[3] = red_read_idy * TILE_N_V4_PER_CTA    +
                 smem_row_read_id  * SMEM_ROW_V4_SIZE +
                 ((((red_read_idy + 3) / N_ROWS_PER_SMEM_ROW) % MMAs_PER_REDUCE_ROW) ^ inter_mma_read_idx) * (TILE_N_PER_MMA/_4INT_TO_INT4_) +
    	         inner_mma_read_idx;
#endif

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
    int  flt_c_v16_end = (spk_id * num_chl_per_spk_head + num_chl_per_spk) >> 4;
    int  flt_c_v16_id  = ldg_idx + ((spk_id * num_chl_per_spk_head) >> 4);
#elif defined(ENABLE_SPLITF) || defined(ENABLE_FUSE)
    int  flt_c_v16_end = num_chl_per_grp_pad_v16;
    int  flt_c_v16_id  = ldg_idx;
#endif

    bool flt_c_v16_valid  = flt_c_v16_id < flt_c_v16_end;

    int4 reg_dAv4[REG_dAv4_SIZE];
    int4 reg_dBv4[REG_dBv4_SIZE];

#if defined(FLT_SIZE1)
    int     dAv4_off[READ_dAv4_STEPS];
    bool in_hw_valid[READ_dAv4_STEPS];

#pragma unroll
    for(int i = 0; i < READ_dAv4_STEPS; i++)
    {
        SET_dAv4_BOUND(i, dAv4_off[i], in_hw_valid[i]);
    }
#elif defined(FLT_SIZE3)
    int dAv4_off[READ_dAv4_STEPS];
    int in_hw_mask[READ_dAv4_STEPS];

#pragma unroll
    for(int i = 0; i < READ_dAv4_STEPS; i++)
    {
        SET_dAv4_BOUND(i, dAv4_off[i], in_hw_mask[i]);
    }
#elif defined(FLT_SIZEN)
    int dAv4_off[READ_dAv4_STEPS];
    int  in_n_id[READ_dAv4_STEPS];
    int  in_h_id[READ_dAv4_STEPS];
    int  in_w_id[READ_dAv4_STEPS];

    int in_h_start[READ_dAv4_STEPS];
    int in_w_start[READ_dAv4_STEPS];

#pragma unroll
    for(int i = 0; i < READ_dAv4_STEPS; i++)
    {
        SET_dAv4_BOUND(i, dAv4_off[i], in_n_id[i], in_h_start[i], in_w_start[i]);
        in_h_id[i] = in_h_start[i];
        in_w_id[i] = in_w_start[i];
    }
#endif

    int     dBv4_off[READ_dBv4_STEPS];
    bool flt_n_valid[READ_dBv4_STEPS];

#pragma unroll
    for(int i = 0; i < READ_dBv4_STEPS; i++)
    {
        SET_dBv4_BOUND(i, dBv4_off[i], flt_n_valid[i]);
    }

#if defined(USE_1BUF)
    __shared__ int4 sm_base_v4[SM_BASE_V4_1BUF];
#elif defined(USE_2BUF)
    __shared__ int4 sm_base_v4[SM_BASE_V4_2BUF];
#endif
    int * sm_base_v1 = (int *) sm_base_v4;
    int2 *sm_base_v2 = (int2 *)sm_base_v4;
    
    uint32_t smp_base_v1;

    CVT_SM_PTR(smp_base_v1, sm_base_v1);

    uint sAv4_write =  sts_idy  * TILE_K_V16_PER_CTA + sts_idx;

#if defined(USE_1BUF)
    uint sBv4_write =  sAv4_write + SM_A_V4_1BUF;
#elif defined(USE_2BUF)
    uint sBv4_write =  sAv4_write + SM_A_V4_2BUF;
#endif

    uint lds_idy =  local_tid;
#if TILE_K_PER_CTA == 16
    uint lds_idx =  0;
#elif TILE_K_PER_CTA == 32
    uint lds_idx = ((set_id * TILE_K_V16_PER_SET) & 0x1) ^ ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x1);
#elif TILE_K_PER_CTA == 64
    uint lds_idx = ((set_id * TILE_K_V16_PER_SET) & 0x3) ^ ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x3);
#elif TILE_K_PER_CTA == 128
    uint lds_idx = ((set_id * TILE_K_V16_PER_SET) & 0x7) ^ ((local_tid / K_ROWS_PER_SMEM_ROW) & 0x7);
#elif TILE_K_PER_CTA == 256
    uint lds_idx = ((set_id * TILE_K_V16_PER_SET) & 0xf) ^ (local_tid & 0x7);
#endif

    uint sAv1_read  =  set_widy   * TILE_M_PER_WARP        * TILE_K_V4_PER_CTA +
#if TILE_M_PER_WARP == 8
                      (lds_idy    % WARP_SIZE_IN_THD_QTR)  * TILE_K_V4_PER_CTA +
#elif TILE_M_PER_WARP == 16
                      (lds_idy    % WARP_SIZE_IN_THD_HALF) * TILE_K_V4_PER_CTA +
#elif TILE_M_PER_WARP == 32
                       lds_idy    * TILE_K_V4_PER_CTA      +
#elif TILE_M_PER_WARP == 64 || TILE_M_PER_WARP == 128
                       lds_idy    * TILE_K_V4_PER_CTA      +
#endif
                       lds_idx    * _INT4_TO_4INT_;

    uint sBv1_read  =  set_widx   * TILE_N_PER_WARP        * TILE_K_V4_PER_CTA +
#if TILE_N_PER_WARP == 8
                      (lds_idy    % WARP_SIZE_IN_THD_QTR)  * TILE_K_V4_PER_CTA +
#elif TILE_N_PER_WARP == 16
                      (lds_idy    % WARP_SIZE_IN_THD_HALF) * TILE_K_V4_PER_CTA +
#elif TILE_N_PER_WARP == 32 || TILE_N_PER_WARP == 64
                       lds_idy    * TILE_K_V4_PER_CTA      +
#endif
                       lds_idx    * _INT4_TO_4INT_         +
#if defined(USE_1BUF)
                       SM_A_V1_1BUF;
#elif defined(USE_2BUF)
                       SM_A_V1_2BUF;
#endif

    int db0_sBv1[REG_sBv1_SIZE];
#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
    int db1_sBv1[REG_sBv1_SIZE];
#endif

    int db0_sAv1[REG_sAv1_SIZE];
#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
    int db1_sAv1[REG_sAv1_SIZE];
#endif

#if defined(FLT_SIZE1)
    LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v16_valid, in_hw_valid);
    LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v16_valid, flt_n_valid);

    FWD_FLT(flt_c_v16_id, flt_c_v16_valid);
#elif defined(FLT_SIZE3)
    LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v16_valid, flt_hw_bid);
    LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v16_valid, flt_n_valid);

    FWD_FLT(flt_hw_id, flt_hw_bid, flt_c_v16_id, flt_c_v16_valid);
    FWD_LUT(lut_id);
#elif defined(FLT_SIZEN)
    LOAD_dAv4(reg_dAv4, dA, dAv4_off, in_n_id, in_h_id, in_w_id);
    LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v16_valid, flt_n_valid);

    FWD_FLT(flt_h_id, flt_w_id, flt_c_v16_id, flt_c_v16_valid);
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

#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
    FWD_KGROUP_STEP1(sAv1_read);
    FWD_KGROUP_STEP1(sBv1_read);
#endif

#if defined(ENABLE_SPLITK)
    for (uint j = 0; j < flt_hw * DivUp(num_chl_per_spk, TILE_K_PER_SET); j++)
#elif defined(ENABLE_SPLITF) || defined(ENABLE_FUSE)
    for (uint j = 0; j < kloop_num; j++)
#endif
    {
#if defined(FLT_SIZE1)
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v16_valid, in_hw_valid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v16_valid, flt_n_valid);

        FWD_FLT(flt_c_v16_id, flt_c_v16_valid);
#elif defined(FLT_SIZE3)
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, flt_c_v16_valid, flt_hw_bid);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v16_valid, flt_n_valid);

        FWD_FLT(flt_hw_id, flt_hw_bid, flt_c_v16_id, flt_c_v16_valid);
        FWD_LUT(lut_id);
#elif defined(FLT_SIZEN)
        LOAD_dAv4(reg_dAv4, dA, dAv4_off, in_n_id, in_h_id, in_w_id);
        LOAD_dBv4(reg_dBv4, dB, dBv4_off, flt_c_v16_valid, flt_n_valid);

        FWD_FLT(flt_h_id, flt_w_id, flt_c_v16_id, flt_c_v16_valid);
        FWD_LUT(lut_id);
#endif

#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);
#endif

#if TILE_K_PER_SET == 32
        FWD_KGROUP_STEP1(sAv1_read);
        FWD_KGROUP_STEP1(sBv1_read);
#elif TILE_K_PER_SET == 64
        FWD_KGROUP_STEP2(sAv1_read);
        FWD_KGROUP_STEP2(sBv1_read);
#endif

        MMA_INSTS(C, db0_sAv1, db0_sBv1);

#if TILE_K_PER_SET == 64
        READ_sAv1(db0_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db0_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP3(sAv1_read);
        FWD_KGROUP_STEP3(sBv1_read);

        MMA_INSTS(C, db1_sAv1, db1_sBv1);

        READ_sAv1(db1_sAv1, smp_base_v1, sAv1_read);
        READ_sBv1(db1_sBv1, smp_base_v1, sBv1_read);

        FWD_KGROUP_STEP4(sAv1_read);
        FWD_KGROUP_STEP4(sBv1_read);
#endif

#if defined(USE_1BUF)
        __syncthreads();
#endif

        WRITE_sAv4(sm_base_v4, sAv4_write, reg_dAv4);
        WRITE_sBv4(sm_base_v4, sBv4_write, reg_dBv4);

#if TILE_K_PER_SET == 32
        MMA_INSTS(C, db1_sAv1, db1_sBv1);
#elif TILE_K_PER_SET == 64
        MMA_INSTS(C, db0_sAv1, db0_sBv1);
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

#if TILE_K_PER_SET == 32 || TILE_K_PER_SET == 64
        FWD_KGROUP_STEP1(sAv1_read);
        FWD_KGROUP_STEP1(sBv1_read);
#endif

#if TILE_K_PER_SET == 64
        MMA_INSTS(C, db1_sAv1, db1_sBv1);
#endif
    }

    __syncthreads();

    WRITE_sRv2(sm_base_v2, sRv2_write, i2C);

    __syncthreads();

    float4 deScale;
    float *fR = (float *)Rv4;
#if defined(ENABLE_FUSE)
    int outData;
#endif
#pragma unroll
    for(int s = 0; s < OUTPUT_STEPS; s++)
    {
	//fp16时不用每次swizzle idx，是因为一个cta的线程可以把sub_mma行(即8行)的数据一次加载完，下一个sub_mma块就可以沿用上一次读的索引不变，只需加上偏移即可。
        //READ_sRv4(Rv4, sm_base_v4, sRv4_read);
        READ_sRv4(Rv4, sm_base_v4, sRv4_read[s % SWIZZLE_GROUP]);

#if TILE_K_PER_CTA > TILE_K_PER_SET
        int * Rv1  = (int *) Rv4;
        REDUCE(Rv1);
#endif

        bool dCv4_y_valid = (dCv4_idy  / out_hw) < in_num;
        uint dCv4_off     =  dCv4_base +
                             dCv4_idy  * num_flt_per_grp_pad_v4 * num_grp +
                             dCv4_idx;
        LOAD_SCALE(deScale, d_flt_scale);

        deQuantData(fR, Rv4[0], deScale);
#if defined(ENABLE_FUSE)
        ADD_BIAS_V4(fR, has_bias, bias);
#endif

#if defined(ENABLE_FUSE)
        uint concatV4_off = 0;

        FUSE_RELU_V4(fR, has_relu);
        FUSE_CLIP_V4(fR, has_clip, clip_max, clip_min);
        FUSE_PRELU_V4(fR, has_prelu, prelu, leaky);

        FUSE_ELT_V4(fR, has_elt, pre_data);
        FUSE_RELU_V4(fR, has_elt_relu);
        FUSE_CLIP_V4(fR, has_elt_clip, elt_clip_max, elt_clip_min);
        FUSE_PRELU_V4(fR, has_elt_prelu, elt_prelu, elt_leaky);

        SET_CONCAT_OFF_V4(has_concat, concatV4_off);
#endif

#if defined(ENABLE_FUSE)
        quantOutData(Rv4[0], fR, out_scale);
        packchar4(outData, Rv4[0].x, Rv4[0].y, Rv4[0].z, Rv4[0].w);
#endif
	

#if defined(ENABLE_FUSE)
        OUTPUT_PRC_INT8_V4(outData);
#else
        OUTPUT_PRC_FLOAT_V4(Rv4[0]);
#endif

        dCv4_idy += OUTPUT_SIZE_Y_IN_THD;
    }

#endif // __CUDA_ARCH__
}
