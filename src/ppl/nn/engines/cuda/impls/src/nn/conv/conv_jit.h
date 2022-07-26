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

#ifndef __PPLCUDA_CONV_JIT_H__
#define __PPLCUDA_CONV_JIT_H__

#define CPI_SM75_HMMA1688       8
#define CPI_SM75_IMMA8816       4
#define CPI_SM75_LDG32_L1D      4 
#define CPI_SM75_LDG64_L1D      4 
#define CPI_SM75_LDG128_L1D     8
#define CPI_SM75_LDG32_L2       4
#define CPI_SM75_LDG64_L2       8
#define CPI_SM75_LDG128_L2      16
#define CPI_SM75_LDS32          2
#define CPI_SM75_LDS64          4
#define CPI_SM75_LDS128         8
#define CPI_SM75_STS32          4
#define CPI_SM75_STS64          6
#define CPI_SM75_STS128         10

#define CPI_SM80_HMMA1688       4
#define CPI_SM80_HMMA16816      8
#define CPI_SM80_IMMA8816       4
#define CPI_SM80_IMMA16816      4
#define CPI_SM80_IMMA16832      8
#define CPI_SM80_LDG32_L1D      4 
#define CPI_SM80_LDG64_L1D      4 
#define CPI_SM80_LDG128_L1D     8
#define CPI_SM80_LDG32_L2       4
#define CPI_SM80_LDG64_L2       8
#define CPI_SM80_LDG128_L2      16
#define CPI_SM80_LDS32          2
#define CPI_SM80_LDS64          4
#define CPI_SM80_LDS128         8
#define CPI_SM80_STS32          4
#define CPI_SM80_STS64          6
#define CPI_SM80_STS128         10

#define LATENCY_SM75_HMMA1688   14
#define LATENCY_SM75_IMMA8816   10
#define LATENCY_SM75_SMEM       19
#define LATENCY_SM75_L1D_CACHE  32
#define LATENCY_SM75_L2_CACHE   188
#define LATENCY_SM75_DRAM       296

#define LATENCY_SM80_HMMA1688   14
#define LATENCY_SM80_IMMA8816   14
#define LATENCY_SM80_HMMA16816  32
#define LATENCY_SM80_IMMA16816  14
#define LATENCY_SM80_IMMA16832  32
#define LATENCY_SM80_SMEM       22
#define LATENCY_SM80_L1D_CACHE  34
#define LATENCY_SM80_L2_CACHE   200 // near:200, far:350
#define LATENCY_SM80_DRAM       360

#define PB_NUM_PER_SM           4

#include "ppl/common/types.h"

bool SortByAscendScore(const std::pair<algo_param_t, float> &a, const std::pair<algo_param_t, float> &b);
bool SortByDescendScore(const std::pair<algo_param_t, float> &a, const std::pair<algo_param_t, float> &b);

void GetHardwareInfo(
    int device_arch,
    ppl::common::datatype_t type,
    int num_chl_per_grp,
    int &cpi_mma,
    int &latency_mma,
    int &cpi_ldg32_l1d,
    int &cpi_ldg64_l1d,
    int &cpi_ldg128_l1d,
    int &cpi_ldg32_l2,
    int &cpi_ldg64_l2,
    int &cpi_ldg128_l2,
    int &cpi_lds32,
    int &cpi_lds64,
    int &cpi_lds128,
    int &cpi_sts32,
    int &cpi_sts64,
    int &cpi_sts128,
    int &latency_l2_cache,
    int &latency_dram,
    int &max_dyn_smem_per_cta);

int GetEstimateCtaNumber(
    int m_conv,
    int n_conv,
    int num_grp);

void GetIdxnMmaInfo(
    int device_arch,
    ppl::common::datatype_t type,
    int num_chl_per_grp,
    std::string &mma_shape,
    int &m_mma,
    int &n_mma,
    int &k_mma,
    int &m_mma_max,
    int &n_mma_max,
    int &k_mma_max);

void Get2spkMmaInfo(
    int device_arch,
    ppl::common::datatype_t type,
    std::string &mma_shape,
    int &m_mma,
    int &n_mma,
    int &k_mma,
    int &m_mma_max,
    int &n_mma_max,
    int &k_mma_max,
    int &k_blk_mma,
    int &buf_num_max);

void GetSwzlMmaInfo(
    int device_arch,
    ppl::common::datatype_t type,
    std::string &mma_shape,
    int &m_mma,
    int &n_mma,
    int &k_mma,
    int &m_mma_max,
    int &n_mma_max,
    int &k_mma_max,
    int &k_blk_mma,
    int &buf_num_max);

int GetIdxnRegsPerThread(
    ppl::common::datatype_t type,
    int m_cta,
    int n_cta,
    int m_warp,
    int n_warp,
    int k_per_step,
    int m_mma,
    int n_mma,
    int k_mma,
    int cta_size_in_thd);

int Get2spkRegsPerThread(
    ppl::common::datatype_t type,
    int type_size,
    int m_cta,
    int n_cta,
    int k_cta,
    int m_warp,
    int n_warp,
    int k_per_set,
    int m_mma,
    int n_mma,
    int k_mma,
    int k_blk_mma,
    int buf_num,
    int cta_size_in_thd,
    int set_size_in_thd);

int GetSwzlRegsPerThread(
    ppl::common::datatype_t type,
    int type_size,
    int m_cta,
    int n_cta,
    int k_cta,
    int m_warp,
    int n_warp,
    int m_mma,
    int n_mma,
    int k_mma,
    int k_blk_mma,
    int buf_num,
    int cta_size_in_thd);

int GetIdxnSmemUsage(
    int m_cta,
    int cta_size_in_thd);

int Get2spkSmemUsage(
    ppl::common::datatype_t type,
    int type_size,
    int m_cta,
    int n_cta,
    int k_cta,
    int set_num,
    int buf_num);

int GetSwzlSmemUsage(
    ppl::common::datatype_t type,
    int type_size,
    int m_cta,
    int n_cta,
    int k_cta,
    int m_warp,
    int n_warp,
    int m_mma,
    int n_mma,
    int buf_num,
    int cta_size_in_warp);

int GetTileKSize(
    int num_chl_per_grp_pad,
    int kloop_num);

float GetWarpOccupySMScore(
    int warp_num_per_sm,
    int cta_num_per_sm 
    );

float GetEfficiencyScore(
    int m_cta,
    int n_cta,
    int k_cta,
    int kloop_total,
    int m_conv,
    int n_conv,
    int k_conv);

float GetOccupancyScore(
    int cta_size_in_thd,
    int cta_size_in_warp,
    int sm_num,
    int cta_num,
    int regs_per_cta,
    int smem_per_cta,
    int max_ctas_per_sm,
    int max_thds_per_sm,
    int max_regs_per_sm,
    int max_smem_per_sm,
    float& cta_launch_times);

float GetIdxnPipelineScore(
    int type_size,
    float cta_launch_times,
    int out_w,
    int cta_size_in_thd,
    int cta_size_in_warp,
    int m_cta,
    int n_cta,
    int k_cta,
    int m_warp,
    int n_warp,
    int k_per_step,
    int m_mma,
    int n_mma,
    int k_mma,
    int cpi_mma,
    int cpi_ldg32_l1d,
    int cpi_ldg64_l1d,
    int cpi_ldg128_l1d,
    int cpi_ldg32_l2,
    int cpi_ldg64_l2,
    int cpi_ldg128_l2,
    int latency_mma,
    int latency_l2_cache,
    int latency_dram);

float Get2spkPipelineScore(
    int type_size,
    float cta_launch_times,
    int m_conv,
    int n_conv,
    int k_conv,
    int kloop_num,
    int splitk,
    int splitf,
    int out_w,
    int cta_size_in_thd,
    int cta_size_in_warp,
    int sm_num,
    int m_cta,
    int n_cta,
    int k_cta,
    int m_warp,
    int n_warp,
    int k_per_set,
    int set_num,
    int buf_num,
    int m_mma,
    int n_mma,
    int k_mma,
    int k_mma_max,
    int cpi_mma,
    int cpi_ldg128_l1d,
    int cpi_ldg128_l2,
    int cpi_lds128,
    int cpi_sts32,
    int latency_mma,
    int latency_l2_cache,
    int latency_dram);

float GetSwzlPipelineScore(
    int type_size,
    float cta_launch_times,
    int m_conv,
    int n_conv,
    int k_conv,
    int kloop_num,
    int out_w,
    int cta_size_in_thd,
    int cta_size_in_warp,
    int sm_num,
    int m_cta,
    int n_cta,
    int k_cta,
    int m_warp,
    int n_warp,
    int buf_num,
    int m_mma,
    int n_mma,
    int k_mma,
    int k_mma_max,
    int cpi_mma,
    int cpi_ldg128_l1d,
    int cpi_ldg128_l2,
    int cpi_lds128,
    int cpi_sts32,
    int latency_mma,
    int latency_l2_cache,
    int latency_dram);

#endif
