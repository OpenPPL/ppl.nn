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

#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/common/common.h"

#include "conv_common.h"
#include "conv_jit.h"

#include <string>
#include <vector>
#include <cmath>

// sort by ascending order
bool SortByAscendScore(const std::pair<algo_param_t, float> &a, const std::pair<algo_param_t, float> &b)
{
    return (a.second < b.second);
}

// sort by descending order
bool SortByDescendScore(const std::pair<algo_param_t, float> &a, const std::pair<algo_param_t, float> &b)
{
    return (a.second > b.second);
}

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
        int &max_dyn_smem_per_cta)
{
    if (device_arch == 75) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            cpi_mma = CPI_SM75_HMMA1688;
            latency_mma = LATENCY_SM75_HMMA1688;

        } else if( type == ppl::common::DATATYPE_INT8 ) {
            cpi_mma = CPI_SM75_IMMA8816;
            latency_mma = LATENCY_SM75_IMMA8816;
        }

        cpi_ldg32_l1d  = CPI_SM75_LDG32_L1D;
        cpi_ldg64_l1d  = CPI_SM75_LDG64_L1D;
        cpi_ldg128_l1d = CPI_SM75_LDG128_L1D;

        cpi_ldg32_l2  = CPI_SM75_LDG32_L2;
        cpi_ldg64_l2  = CPI_SM75_LDG64_L2;
        cpi_ldg128_l2 = CPI_SM75_LDG128_L2;

        cpi_lds32  = CPI_SM75_LDS32;
        cpi_lds64  = CPI_SM75_LDS64;
        cpi_lds128 = CPI_SM75_LDS128;

        cpi_sts32  = CPI_SM75_STS32;
        cpi_sts64  = CPI_SM75_STS64;
        cpi_sts128 = CPI_SM75_STS128;

        latency_l2_cache = LATENCY_SM75_L2_CACHE;
        latency_dram = LATENCY_SM75_DRAM;

        max_dyn_smem_per_cta = SM75_MAX_DYN_SMEM_SIZE_PER_CTA;

    } else if (device_arch >= 80) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            if(num_chl_per_grp <= 2) {
                cpi_mma = CPI_SM80_HMMA1688;
                latency_mma = LATENCY_SM80_HMMA1688;
            } else {
                cpi_mma = CPI_SM80_HMMA16816;
                latency_mma = LATENCY_SM80_HMMA16816;
            }

        } else if( type == ppl::common::DATATYPE_INT8 ) {
            if(num_chl_per_grp <= 4) {
                cpi_mma = CPI_SM80_IMMA16816;
                latency_mma = LATENCY_SM80_IMMA16816;
            } else {
                cpi_mma = CPI_SM80_IMMA16832;
                latency_mma = LATENCY_SM80_IMMA16832;
            }
        }
        cpi_ldg32_l1d  = CPI_SM80_LDG32_L1D;
        cpi_ldg64_l1d  = CPI_SM80_LDG64_L1D;
        cpi_ldg128_l1d = CPI_SM80_LDG128_L1D;

        cpi_ldg32_l2  = CPI_SM80_LDG32_L2;
        cpi_ldg64_l2  = CPI_SM80_LDG64_L2;
        cpi_ldg128_l2 = CPI_SM80_LDG128_L2;

        cpi_lds32  = CPI_SM80_LDS32;
        cpi_lds64  = CPI_SM80_LDS64;
        cpi_lds128 = CPI_SM80_LDS128;

        cpi_sts32  = CPI_SM80_STS32;
        cpi_sts64  = CPI_SM80_STS64;
        cpi_sts128 = CPI_SM80_STS128;

        latency_l2_cache = LATENCY_SM80_L2_CACHE;
        latency_dram = LATENCY_SM80_DRAM;

        max_dyn_smem_per_cta = SM80_MAX_DYN_SMEM_SIZE_PER_CTA;
    }
}

int GetEstimateCtaNumber(
        int m_conv,
        int n_conv,
        int num_grp)
{
    int cta_num = 0;

    const int M_CTA = 128;
    const int N_CTA = 128;

    cta_num = DivUp((m_conv * n_conv), (M_CTA * N_CTA)) * num_grp;

    return cta_num;
}

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
        int &k_mma_max)
{
    if (device_arch == 75) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma1688";
            m_mma = 16;
            n_mma = 8;
            k_mma = 8;
            k_mma_max = k_mma * 4;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma8816";
            m_mma = 8;
            n_mma = 8;
            k_mma = 16;
            k_mma_max = k_mma * 4;
        }
    } else if (device_arch >= 80) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            if(num_chl_per_grp <= 2) {
                mma_shape = "hmma1688";
                m_mma = 16;
                n_mma = 8;
                k_mma = 8;
                k_mma_max = k_mma * 4;
            } else {
                mma_shape = "hmma16816";
                m_mma = 16;
                n_mma = 8;
                k_mma = 16;
                k_mma_max = k_mma * 2;
            }

        } else if( type == ppl::common::DATATYPE_INT8 ) {
            if(num_chl_per_grp <= 4) {
                mma_shape = "imma16816";
                m_mma = 16;
                n_mma = 8;
                k_mma = 16;
                k_mma_max = k_mma * 4;
            } else {
                mma_shape = "imma16832";
                m_mma = 16;
                n_mma = 8;
                k_mma = 32;
                k_mma_max = k_mma * 2;
            }
        }
    }

    m_mma_max = m_mma * 8;
    n_mma_max = n_mma * 8;
}

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
        int &buf_num_max)
{
    if (device_arch == 75) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma1688";
            m_mma = 16;
            n_mma = 8;
            k_mma = 8;
            k_blk_mma = 1;
            buf_num_max = 2;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma8816";
            m_mma = 8;
            n_mma = 8;
            k_mma = 16;
            k_blk_mma = 1;
            buf_num_max = 2;
        }
    } else if (device_arch >= 80) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma16816";
            m_mma = 16;
            n_mma = 8;
            k_mma = 16;
            k_blk_mma = 2;
            buf_num_max = 6;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma16832";
            m_mma = 16;
            n_mma = 8;
            k_mma = 32;
            k_blk_mma = 2;
            buf_num_max = 6;
        }
    }

    m_mma_max = m_mma * 8;
    n_mma_max = n_mma * 8;
    k_mma_max = k_mma * 4;
}

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
        int &buf_num_max)
{
    if (device_arch == 75) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma1688";
            m_mma = 8;
            n_mma = 16;
            k_mma = 8;
            buf_num_max = 2;
            k_blk_mma = 1;
            k_mma_max = k_mma * 8;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma8816";
            m_mma = 8;
            n_mma = 8;
            k_mma = 16;
            buf_num_max = 2;
            k_blk_mma = 1;
            k_mma_max = k_mma * 4;
        }
    } else if (device_arch >= 80) {
        if( type == ppl::common::DATATYPE_FLOAT16 ) {
            mma_shape = "hmma16816";
            m_mma = 8;
            n_mma = 16;
            k_mma = 16;
            buf_num_max = 6;
            k_blk_mma = 2;
            k_mma_max = k_mma * 4;
        } else if( type == ppl::common::DATATYPE_INT8 ) {
            mma_shape = "imma16832";
            m_mma = 8;
            n_mma = 16;
            k_mma = 32;
            buf_num_max = 6;
            k_blk_mma = 2;
            k_mma_max = k_mma * 4;
        }
    }

    m_mma_max = m_mma * 8;
    n_mma_max = n_mma * 8;
}

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
        int cta_size_in_thd)
{
    int m_blk_num = m_warp / 8;
    int n_blk_num = n_warp / 8;

    int regs_a_v1 = m_blk_num * (k_per_step / k_mma);
    int regs_b_v1 = n_blk_num * (k_per_step / k_mma);

    int regs_c_v1 = DivUp(m_cta * n_cta, cta_size_in_thd * GetPadSize(type) / 4);

    int regs_a_idx = (m_blk_num + 1) * 4;
    int regs_b_idx =  n_blk_num * 2;
    int regs_c_idx =  n_blk_num * 2  + 4;

    int regs_idx = Max(regs_a_idx + regs_b_idx, regs_c_idx);

    int regs_common = 20;

    int regs_per_thd = regs_a_v1 + regs_b_v1 + regs_c_v1 + regs_idx + regs_common;

    return regs_per_thd;
}

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
        int set_size_in_thd)
{
    int regs_a_v4 = DivUp( m_cta * k_cta * type_size, _INT4_TO_16BYTE_ * cta_size_in_thd);
    int regs_b_v4 = DivUp( n_cta * k_cta * type_size, _INT4_TO_16BYTE_ * cta_size_in_thd);

    int regs_c_v4 = 0;
    if( type == ppl::common::DATATYPE_FLOAT16 ) {
        regs_c_v4 = (m_cta * n_cta * type_size) / (_INT4_TO_16BYTE_ * set_size_in_thd);
    } else if( type == ppl::common::DATATYPE_INT8 ) {
        regs_c_v4 = (m_cta * n_cta * _INT_TO_4BYTE_) / (_INT4_TO_16BYTE_ * set_size_in_thd);
    }

    int regs_a_v1 = regs_a_v4 * _4INT_TO_INT4_;
    int regs_b_v1 = regs_b_v4 * _4INT_TO_INT4_;
    int regs_c_v1 = regs_c_v4 * _4INT_TO_INT4_;

    int regs_a_idx = regs_a_v4;
    int regs_b_idx = regs_b_v4;

    int m_thd = m_warp / 8;
    int n_thd = n_warp / 8;
    int reg_buf = (k_per_set / k_mma == 1) ? 1 : 2; // single or double register buffer

    int regs_sa_v1 = m_thd * k_blk_mma * reg_buf;
    int regs_sb_v1 = n_thd * k_blk_mma * reg_buf;

    int regs_common_idx = 41;
    int regs_per_thd = 0;

    if(buf_num <= 2)
        regs_per_thd = regs_c_v1 + regs_a_idx + regs_b_idx + regs_sa_v1 + regs_sb_v1 + regs_common_idx + regs_a_v1 + regs_b_v1;
    else if(buf_num > 2)
        regs_per_thd = regs_c_v1 + regs_a_idx + regs_b_idx + regs_sa_v1 + regs_sb_v1 + regs_common_idx;

    return regs_per_thd;
}

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
        int cta_size_in_thd)
{
    int regs_a_v4 = DivUp( m_cta * k_cta * type_size, _INT4_TO_16BYTE_ * cta_size_in_thd);
    int regs_b_v4 = DivUp( n_cta * k_cta * type_size, _INT4_TO_16BYTE_ * cta_size_in_thd);

    int regs_c_v4 = 0;
    if( type == ppl::common::DATATYPE_FLOAT16 ) {
        regs_c_v4 = (m_cta * n_cta * type_size) / (_INT4_TO_16BYTE_ * cta_size_in_thd);
    } else if( type == ppl::common::DATATYPE_INT8 ) {
        regs_c_v4 = (m_cta * n_cta * _INT_TO_4BYTE_) / (_INT4_TO_16BYTE_ * cta_size_in_thd);
    }

    int regs_a_v1 = regs_a_v4 * _4INT_TO_INT4_;
    int regs_b_v1 = regs_b_v4 * _4INT_TO_INT4_;
    int regs_c_v1 = regs_c_v4 * _4INT_TO_INT4_;

    int regs_a_idx = regs_a_v4;
    int regs_b_idx = regs_b_v4;

    int m_thd = m_warp / 8;
    int n_thd = n_warp / 8;
    int reg_buf = (k_cta / k_mma == 1) ? 1 : 2; // single or double register buffer

    int regs_sa_v1 = m_thd * k_blk_mma * reg_buf;
    int regs_sb_v1 = n_thd * k_blk_mma * reg_buf;

    int regs_common_idx = 20;
    int regs_per_thd = 0;

    if(buf_num <= 2)
        regs_per_thd = regs_c_v1 + regs_a_idx + regs_b_idx + regs_sa_v1 + regs_sb_v1 + regs_common_idx + regs_a_v1 + regs_b_v1;
    else if(buf_num > 2)
        regs_per_thd = regs_c_v1 + regs_a_idx + regs_b_idx + regs_sa_v1 + regs_sb_v1 + regs_common_idx;

    return regs_per_thd;
}

int GetIdxnSmemUsage(
        int m_cta,
        int cta_size_in_thd)
{
    int smem_per_cta = (m_cta + cta_size_in_thd) * _INT4_TO_4INT_ * _INT_TO_4BYTE_; // in byte
    
    return smem_per_cta;
}

int Get2spkSmemUsage(
        ppl::common::datatype_t type,
        int type_size,
        int m_cta,
        int n_cta,
        int k_cta,
        int set_num,
        int buf_num)
{
    int smem_a = m_cta * k_cta * buf_num * type_size;
    int smem_b = n_cta * k_cta * buf_num * type_size;

    int smem_c = 0;
    if( type == ppl::common::DATATYPE_FLOAT16 ) {
        smem_c = m_cta * n_cta * type_size;
    } else if( type == ppl::common::DATATYPE_INT8 ) {
        smem_c = m_cta * n_cta * _INT_TO_4BYTE_;
    }

    int smem_per_cta = Max(smem_a + smem_b, smem_c * set_num);
    
    return smem_per_cta;
}

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
        int cta_size_in_warp)
{
    int smem_a = m_cta * k_cta * buf_num * type_size;
    int smem_b = n_cta * k_cta * buf_num * type_size;

    int smem_r = 0;
    if( type == ppl::common::DATATYPE_FLOAT16 ) {
        if(m_warp == 8)
            smem_r = m_cta * n_mma * cta_size_in_warp * type_size * 2;
        else 
            smem_r = m_cta * n_mma * cta_size_in_warp * type_size;
    } else if( type == ppl::common::DATATYPE_INT8 ) {
        smem_r = m_cta * n_mma * cta_size_in_warp * _INT_TO_4BYTE_;
    }

    int smem_per_cta = Max(smem_a + smem_b, smem_r);
    
    return smem_per_cta;
}

int GetTileKSize(
        int num_chl_per_grp_pad,
        int kloop_num)
{
    int k_cta = 0;
    if(kloop_num == 1) {
        if(num_chl_per_grp_pad > 32 && num_chl_per_grp_pad <= 48) {
            k_cta = 32;
        } else if(num_chl_per_grp_pad > 48 && num_chl_per_grp_pad <= 96) {
            k_cta = 64;
        } else if(num_chl_per_grp_pad > 96 && num_chl_per_grp_pad <= 256) {
            k_cta = 128;
        } else if(num_chl_per_grp_pad > 256) {
            k_cta = 256;
        }

    } else {
        if(num_chl_per_grp_pad > 32 && num_chl_per_grp_pad <= 48) {
            k_cta = 32;
        } else if(num_chl_per_grp_pad > 48 && num_chl_per_grp_pad <= 96) {
            k_cta = 64;
        } else if(num_chl_per_grp_pad > 96 && num_chl_per_grp_pad <= 256) {
            k_cta = 128;
        } else if(num_chl_per_grp_pad > 256) {
            k_cta = 256;
        }
    }

    return k_cta;
}


float GetWarpOccupySMScore(
        int warp_num_per_sm,
        int cta_num_per_sm 
        )
{
    if(warp_num_per_sm >= 0 && warp_num_per_sm <= 4)
        return 0.6 - 0.15 * (cta_num_per_sm - 1);
    else if(warp_num_per_sm > 4 && warp_num_per_sm < 8)
        return 0.8 - 0.15 * (cta_num_per_sm - 1);
    else if(warp_num_per_sm >= 8 && warp_num_per_sm <= 12)
        return 1 - 0.15 * (cta_num_per_sm - 1);
    else if(warp_num_per_sm > 12 && warp_num_per_sm < 16)
        return 0.8 - 0.15 * (cta_num_per_sm - 1);
    else // if(warp_num_per_sm >= 16)
        return 0.6 - 0.15 * (cta_num_per_sm - 1);
}

float GetEfficiencyScore(
        int m_cta,
        int n_cta,
        int k_cta,
        int kloop_total,
        int m_conv,
        int n_conv,
        int k_conv)
{
    int workload_conv   = m_conv * n_conv * k_conv;
    int workload_kernel = m_cta * DivUp(m_conv, m_cta) * \
                          n_cta * DivUp(n_conv, n_cta) * \
                          k_cta * kloop_total;

    float eff_score = 1.0 * workload_conv / workload_kernel;

    return eff_score;
}

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
        float& cta_launch_times)
{
    int cta_num_limit_by_thds = max_thds_per_sm / cta_size_in_thd;
    int cta_num_limit_by_regs = max_regs_per_sm / regs_per_cta;
    int cta_num_limit_by_smem = max_smem_per_sm / smem_per_cta; 

    int cta_num_per_sm      = Min(max_ctas_per_sm, Min(cta_num_limit_by_thds, Min(cta_num_limit_by_regs, cta_num_limit_by_smem)));
    int cta_num_per_launch  = cta_num_per_sm * sm_num;

    int warp_num_per_sm     = cta_num_per_sm * cta_size_in_warp;
    // int warp_num_per_launch = warp_num_per_sm * sm_num;

    cta_launch_times = 1.f * cta_num / cta_num_per_launch;

    // main part
    float main_score_occ = 0.f;
    if(cta_launch_times > 1) {
        main_score_occ = GetWarpOccupySMScore(warp_num_per_sm, cta_num_per_sm);
    }

    // tail part
    int tail_cta_num = cta_num % cta_num_per_launch;

    int max_cta_num_on_sm  = DivUp(tail_cta_num, sm_num);
    int min_cta_num_on_sm  = tail_cta_num / sm_num;

    int max_warp_num_on_sm = max_cta_num_on_sm * cta_size_in_warp;
    int min_warp_num_on_sm = min_cta_num_on_sm * cta_size_in_warp;

    int sm_num_of_max_occupy;
    int sm_num_of_min_occupy;
    if(tail_cta_num % sm_num != 0) {
        sm_num_of_max_occupy = (tail_cta_num - min_cta_num_on_sm * sm_num) / (max_cta_num_on_sm - min_cta_num_on_sm);
        sm_num_of_min_occupy = sm_num - sm_num_of_max_occupy;
    } else {
        sm_num_of_max_occupy = sm_num;
        sm_num_of_min_occupy = 0;
    }

    float sm_num_of_max_occupy_pct = 1.f * sm_num_of_max_occupy / sm_num;
    float sm_num_of_min_occupy_pct = 1.f * sm_num_of_min_occupy / sm_num;

    float tail_score_occ = sm_num_of_max_occupy_pct * GetWarpOccupySMScore(max_warp_num_on_sm, max_cta_num_on_sm) + \
                           sm_num_of_min_occupy_pct * GetWarpOccupySMScore(min_warp_num_on_sm, min_cta_num_on_sm);

    int cta_launch_times_ceil = std::ceil(cta_launch_times);

    // 2 scenarios: main + tail
    float score_occ = (1.f * (cta_launch_times_ceil - 1) / cta_launch_times_ceil) * main_score_occ + \
                       (1.f / cta_launch_times_ceil) * tail_score_occ;

    return score_occ;
}

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
        int latency_dram
        )
{
    int warp_num_per_pb = DivUp(cta_size_in_warp, PB_NUM_PER_SM);

    int cycles_mma = cpi_mma * (m_warp / m_mma) * (n_warp / n_mma) * (k_per_step/ k_mma) * warp_num_per_pb + latency_mma;

    int cycles_mem = 0;

    int mr_flt_total = 0;
    int mr_flt_l2  = 0;
    int mr_flt_l1d = 0;

    int mr_input_total = 0;
    int mr_input_l2 = 0;
    int mr_input_l1d = 0;
    
    if(k_per_step == 8) {
        mr_flt_total = DivUp(n_cta * k_per_step * type_size, _INT_TO_4BYTE_ * WARP_SIZE);
        mr_flt_l2  = mr_flt_total;
        mr_flt_l1d = 0;

        mr_input_total = DivUp(m_cta * k_per_step * type_size, _INT_TO_4BYTE_  * WARP_SIZE);
        mr_input_l2 = DivUp(DivUp(m_cta, out_w) * Min(out_w, m_cta) * (k_per_step >> 2) * type_size, _INT_TO_4BYTE_  * WARP_SIZE);
        mr_input_l1d = mr_input_total - mr_input_l2;

        cycles_mem = cpi_ldg32_l1d * (mr_flt_l1d + mr_input_l1d) + cpi_ldg32_l2 * (mr_flt_l2 + mr_input_l2) + latency_l2_cache;
    }
    else if(k_per_step == 16) {
        mr_flt_total = DivUp(n_cta * k_per_step * type_size, _INT2_TO_8BYTE_ * WARP_SIZE);
        mr_flt_l2  = mr_flt_total;
        mr_flt_l1d = 0;

        mr_input_total = DivUp(m_cta * k_per_step * type_size, _INT2_TO_8BYTE_  * WARP_SIZE);
        mr_input_l2 = DivUp(DivUp(m_cta, out_w) * Min(out_w, m_cta) * (k_per_step >> 2) * type_size, _INT2_TO_8BYTE_  * WARP_SIZE);
        mr_input_l1d = mr_input_total - mr_input_l2;

        cycles_mem = cpi_ldg64_l1d * (mr_flt_l1d + mr_input_l1d) + cpi_ldg64_l2 * (mr_flt_l2 + mr_input_l2) + latency_l2_cache;
    }
    else if(k_per_step == 32) {
        mr_flt_total = DivUp(n_cta * k_per_step * type_size, _INT4_TO_16BYTE_ * WARP_SIZE);
        mr_flt_l2  = mr_flt_total;
        mr_flt_l1d = 0;

        mr_input_total = DivUp(m_cta * k_per_step * type_size, _INT4_TO_16BYTE_  * WARP_SIZE);
        mr_input_l2 = DivUp(DivUp(m_cta, out_w) * Min(out_w, m_cta) * (k_per_step >> 2) * type_size, _INT4_TO_16BYTE_  * WARP_SIZE);
        mr_input_l1d = mr_input_total - mr_input_l2;

        cycles_mem = cpi_ldg128_l1d * (mr_flt_l1d + mr_input_l1d) + cpi_ldg128_l2 * (mr_flt_l2 + mr_input_l2) + latency_l2_cache;
    }

    float ratio = 200.f / (Max(cycles_mma, cycles_mem) * std::ceil(cta_launch_times));

    return  ratio;
}

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
        int latency_dram
        )
{
    // mma part
    int warp_num_per_pb = DivUp(cta_size_in_warp, PB_NUM_PER_SM);

    int mma_issue_cycles_per_buf = cpi_mma * (m_warp / m_mma) * (n_warp / n_mma) * (k_per_set / k_mma) * warp_num_per_pb;

    // memory part
    int mr_flt_total = DivUp(n_cta * k_cta * type_size, _INT4_TO_16BYTE_ * WARP_SIZE);
    int mr_flt_l2  = mr_flt_total;
    int mr_flt_l1d = 0;

    int mr_input_total = DivUp(m_cta * k_cta * type_size, _INT4_TO_16BYTE_ * WARP_SIZE);
    int mr_input_l2 = DivUp(DivUp(m_cta, out_w) * Min(out_w, m_cta) * k_cta * type_size, _INT4_TO_16BYTE_ * WARP_SIZE);
    int mr_input_l1d = mr_input_total - mr_input_l2;

    int mem_issue_cycles_per_buf = cpi_ldg128_l1d * (mr_flt_l1d + mr_input_l1d) + cpi_ldg128_l2 * (mr_flt_l2 + mr_input_l2);

    int cycles_ideal = DivUp(m_conv, m_mma) * \
                       DivUp(n_conv, n_mma) * \
                       DivUp(k_conv, k_mma) * \
                       cpi_mma / \
                       (sm_num * PB_NUM_PER_SM);

    // stage cycles
    int overlap_stage_num = 0;
    int waiting_stage_num = 0;

    int cycles_per_overlap_stage = Max(mma_issue_cycles_per_buf, mem_issue_cycles_per_buf);
    int cycles_per_waiting_stage = Max(mma_issue_cycles_per_buf + latency_mma, mem_issue_cycles_per_buf + latency_l2_cache);

    int min_buf = DivUp(mem_issue_cycles_per_buf, mma_issue_cycles_per_buf);

    if(buf_num == 1) {
        overlap_stage_num = 0;
        waiting_stage_num = kloop_num;

    } else if(buf_num <= min_buf) {
        overlap_stage_num = buf_num;
        waiting_stage_num = kloop_num - overlap_stage_num;

    } else {
        if(mem_issue_cycles_per_buf > mma_issue_cycles_per_buf)
            overlap_stage_num = Min(kloop_num, std::ceil(buf_num / (1.f * mem_issue_cycles_per_buf / mma_issue_cycles_per_buf  - 1)) );
        else
            overlap_stage_num = Min(kloop_num, std::ceil(buf_num / (1.f * mma_issue_cycles_per_buf / mem_issue_cycles_per_buf  - 1)) );

        if(overlap_stage_num < kloop_num)
            waiting_stage_num = kloop_num - overlap_stage_num;
        else
            waiting_stage_num = 0;
    }

    int cycles_reduce = cpi_sts32  * (m_cta / m_mma) * (n_cta / n_mma) * buf_num + \
                        cpi_lds128 * (m_cta / m_mma) * (n_cta / n_mma) * buf_num / _INT4_TO_4INT_;

    int cycles_kernel = overlap_stage_num * cycles_per_overlap_stage + \
                        waiting_stage_num * cycles_per_waiting_stage + \
                        cycles_reduce;

    float ratio = 1.f * cycles_ideal / cycles_kernel;

    return  ratio;
}

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
        int latency_dram
        )
{
    // mma part
    int warp_num_per_pb = DivUp(cta_size_in_warp, PB_NUM_PER_SM);

    int mma_issue_cycles_per_buf = cpi_mma * (m_warp / m_mma) * (n_warp / n_mma) * (k_cta / k_mma) * warp_num_per_pb;

    // memory part
    int mr_flt_total = DivUp(m_cta * k_cta * type_size, _INT4_TO_16BYTE_ * WARP_SIZE);
    int mr_flt_l2  = mr_flt_total;
    int mr_flt_l1d = 0;

    int mr_input_total = DivUp(n_cta * k_cta * type_size, _INT4_TO_16BYTE_ * WARP_SIZE);
    int mr_input_l2 = DivUp(DivUp(n_cta, out_w) * Min(out_w, n_cta) * k_cta * type_size, _INT4_TO_16BYTE_ * WARP_SIZE);
    int mr_input_l1d = mr_input_total - mr_input_l2;

    int mem_issue_cycles_per_buf = cpi_ldg128_l1d * (mr_flt_l1d + mr_input_l1d) + cpi_ldg128_l2 * (mr_flt_l2 + mr_input_l2);

    int cycles_ideal = DivUp(m_conv, m_mma) * \
                       DivUp(n_conv, n_mma) * \
                       DivUp(k_conv, k_mma) * \
                       cpi_mma / \
                       (sm_num * PB_NUM_PER_SM);


    // stage cycles
    int overlap_stage_num = 0;
    int waiting_stage_num = 0;

    int cycles_per_overlap_stage = Max(mma_issue_cycles_per_buf, mem_issue_cycles_per_buf);
    int cycles_per_waiting_stage = Max(mma_issue_cycles_per_buf + latency_mma, mem_issue_cycles_per_buf + latency_l2_cache);

    int min_buf = DivUp(mem_issue_cycles_per_buf, mma_issue_cycles_per_buf);

    if(buf_num == 1) {
        overlap_stage_num = 0;
        waiting_stage_num = kloop_num;

    } else if(buf_num <= min_buf) {
        overlap_stage_num = buf_num;
        waiting_stage_num = kloop_num - overlap_stage_num;

    } else {
        if(mem_issue_cycles_per_buf > mma_issue_cycles_per_buf)
            overlap_stage_num = Min(kloop_num, std::ceil(buf_num / (1.f * mem_issue_cycles_per_buf / mma_issue_cycles_per_buf  - 1)) );
        else
            overlap_stage_num = Min(kloop_num, std::ceil(buf_num / (1.f * mma_issue_cycles_per_buf / mem_issue_cycles_per_buf  - 1)) );

        if(overlap_stage_num < kloop_num)
            waiting_stage_num = kloop_num - overlap_stage_num;
        else
            waiting_stage_num = 0;
    }

    int cycles_kernel = overlap_stage_num * cycles_per_overlap_stage + \
                        waiting_stage_num * cycles_per_waiting_stage;

    float ratio = 1.f * cycles_ideal / (cycles_kernel * cta_launch_times);

    return  ratio;
}
