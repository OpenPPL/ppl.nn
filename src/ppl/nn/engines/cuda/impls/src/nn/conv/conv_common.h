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

#ifndef __PPLCUDA_CONV_COMMON_H__
#define __PPLCUDA_CONV_COMMON_H__

#include <string>
#include <string.h>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <cuda_fp16.h>

#include "kernel_type.h"

#include "ppl/common/types.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "cudakernel/common/common.h"

#define WARP_SIZE               32

#define _2HALF_TO_INT_          2
#define _4CHAR_TO_INT_          4
#define _4INT_TO_INT4_          4
#define _INT_TO_4BYTE_          4
#define _INT2_TO_8BYTE_         8
#define _INT4_TO_16BYTE_        16
#define _INT4_TO_4INT_          4
#define _INT4_TO_4FLOAT_        4
#define _INT4_TO_8HALF_         8
#define _INT4_TO_16CHAR_        16
#define _C2_                    2
#define _C4_                    4
#define _C8_                    8
#define _BYTE128_               128
#define _BYTE1024_              1024

#define Max(x, y)         (((x) > (y))  ? (x) : (y))
#define Min(x, y)         (((x) < (y))  ? (x) : (y))

#define MAX_SPLIT_SIZE          18

#define MAX_STATIC_SMEM_SIZE_PER_CTA    (48  * _BYTE1024_)

#define SM75_MAX_SMEM_SIZE_PER_CTA      (64  * _BYTE1024_)
#define SM80_MAX_SMEM_SIZE_PER_CTA      (164 * _BYTE1024_)

#define SM75_MAX_DYN_SMEM_SIZE_PER_CTA  (64  * _BYTE1024_)
#define SM80_MAX_DYN_SMEM_SIZE_PER_CTA  (163 * _BYTE1024_)
#define SM86_MAX_DYN_SMEM_SIZE_PER_CTA  (99  * _BYTE1024_)
#define SM87_MAX_DYN_SMEM_SIZE_PER_CTA  (163 * _BYTE1024_)

#define ADD_KERNEL(_ktype, _kname, _lut_kptr, _spk_kptr, _idx_kptr) \
    kernel_container.push_back(kernel_info_t(kernel_container.size(), _ktype, _kname, _lut_kptr, _spk_kptr, _idx_kptr));

enum {
    CONV_2SPK_F1  = 0,
    CONV_2SPK_F3  = 1,
    CONV_2SPK_FN  = 2,
    CONV_2SPK_FS  = 3,
    CONV_IDXN_C2  = 4,
    CONV_IDXN_C4  = 5,
    CONV_IDXN_C8  = 6,
    CONV_IDXN_C32 = 7,
    CONV_IDXN_C64 = 8,
    CONV_SWZL_F1 = 9,
    CONV_SWZL_F3 = 10,
    CONV_SWZL_FN = 11,
    CONV_KTYPE_NUM,
};

typedef uint32_t conv_ktype_t;

struct kernel_info_t {
    int kid;

    int tile_m_per_cta;
    int tile_n_per_cta;
    int tile_k_per_cta;

    int tile_m_per_warp;
    int tile_n_per_warp;
    int tile_k_per_warp;

    int tile_k_per_step; // for idxn conv
    int tile_k_per_set; // for 2spk conv

    int flt_size; // for 2spk conv
    int flt_pad_size; // for idxn conv

    int cta_size_in_thd;
    int buf_num;
    int smem_size; // smem size in byte

    std::string kname;

    int karch_major;
    int karch_minor;
    ppl::common::datatype_t kprec;

    conv_ktype_t ktype;

    lut_kernel_t* lut_kptr;
    spk_kernel_t* spk_kptr;
    idx_kernel_t* idx_kptr;
    int8_lut_kernel_t* int8_lut_kptr;
    int8_spk_kernel_t* int8_spk_kptr;
    int8_idx_kernel_t* int8_idx_kptr;

    kernel_info_t()
    {
        kname    = "";
        kid      = -1;
        karch_major = 100;
        karch_minor = 100;
        kprec    = ppl::common::DATATYPE_UNKNOWN;
        ktype    = CONV_KTYPE_NUM;
        lut_kptr = NULL;
        spk_kptr = NULL;
        idx_kptr = NULL;
        int8_lut_kptr = NULL;
        int8_spk_kptr = NULL;
        int8_idx_kptr = NULL;

        tile_m_per_cta = -1;
        tile_n_per_cta = -1;
        tile_k_per_cta = -1;

        tile_m_per_warp = -1;
        tile_n_per_warp = -1;
        tile_k_per_warp = -1;

        tile_k_per_step = -1;
        tile_k_per_set  = -1;

        flt_size     = -1;
        flt_pad_size = -1;

        cta_size_in_thd = -1;
        buf_num = -1;
        smem_size = -1;
    }

    kernel_info_t(int kid_, conv_ktype_t ktype_, const char kname_[], int8_lut_kernel_t * lut_kptr_, int8_spk_kernel_t * spk_kptr_, int8_idx_kernel_t * idx_kptr_)
    {
        kid      = kid_;
        ktype    = ktype_;
        kname    = std::string(kname_);
        int8_lut_kptr = lut_kptr_;
        int8_spk_kptr = spk_kptr_;
        int8_idx_kptr = idx_kptr_;

        parse_kname();
    }

    kernel_info_t(int kid_, conv_ktype_t ktype_, const char kname_[], lut_kernel_t * lut_kptr_, spk_kernel_t * spk_kptr_, idx_kernel_t idx_kptr_)
    {
        kid      = kid_;
        ktype    = ktype_;
        kname    = std::string(kname_);
        lut_kptr = lut_kptr_;
        spk_kptr = spk_kptr_;
        idx_kptr = idx_kptr_;

        parse_kname();
    }

    kernel_info_t(int kid_, conv_ktype_t ktype_, const char kname_[])
    {
        kid   = kid_;
        ktype = ktype_;
        kname = std::string(kname_);

        parse_kname();
    }

    kernel_info_t(struct algo_param_t& algo_param)
    {
        kid   = algo_param.kid;

        if(algo_param.splitk == 1)
            kname = algo_param.algo_name;
        else if(algo_param.splitk > 1 && algo_param.splitk < 10)
            kname = algo_param.algo_name.substr(0, algo_param.algo_name.size() - 5);
        else if(algo_param.splitk >= 10 && algo_param.splitk < 100)
            kname = algo_param.algo_name.substr(0, algo_param.algo_name.size() - 6);
        else if(algo_param.splitk >= 100 && algo_param.splitk < 1000)
            kname = algo_param.algo_name.substr(0, algo_param.algo_name.size() - 7);

        if(algo_param.conv_type == "idxn") {
            if (algo_param.tiles.k_per_step == 8)
                ktype = CONV_IDXN_C2;
            else if (algo_param.tiles.k_per_step == 16)
                ktype = CONV_IDXN_C4;
            else if (algo_param.tiles.k_per_step == 32 && strstr(algo_param.mma_shape.c_str(), "hmma"))
                ktype = CONV_IDXN_C8;
            else if (algo_param.tiles.k_per_step == 32 && strstr(algo_param.mma_shape.c_str(), "imma"))
                ktype = CONV_IDXN_C32;
            else if (algo_param.tiles.k_per_step == 64)
                ktype = CONV_IDXN_C64;

        } else if(algo_param.conv_type == "2spk") {
            if (algo_param.tiles.flt_size == 1)
                ktype = CONV_2SPK_F1;
            else if (algo_param.tiles.flt_size == 3)
                ktype = CONV_2SPK_F3;
            else if (algo_param.tiles.flt_size == 0)
                ktype = CONV_2SPK_FN;
            else if (algo_param.tiles.flt_size == 11)
                ktype = CONV_2SPK_FS;

        } else if(algo_param.conv_type == "swzl") {
            if (algo_param.tiles.flt_size == 1)
                ktype = CONV_SWZL_F1;
            else if (algo_param.tiles.flt_size == 3)
                ktype = CONV_SWZL_F3;
            else if (algo_param.tiles.flt_size == 0)
                ktype = CONV_SWZL_FN;
        }

        parse_kname();
    }

    void parse_kname()
    {
        std::stringstream kname_str(kname);
        std::vector<std::string> kname_substrs;
        std::string substr;

        while (std::getline(kname_str, substr, '_')) {
            kname_substrs.push_back(substr);
        }

        if (strstr(kname_substrs[0].c_str(), "Sm80")) {
            karch_major = 8;
            karch_minor = 0;
        } else if (strstr(kname_substrs[0].c_str(), "Sm75")) {
            karch_major = 7;
            karch_minor = 5;
        }

        if (strstr(kname_substrs[0].c_str(), "Fp16"))
            kprec = ppl::common::DATATYPE_FLOAT16;
        else if (strstr(kname_substrs[0].c_str(), "Int8"))
            kprec = ppl::common::DATATYPE_INT8;

        if (ktype == CONV_IDXN_C2 || ktype == CONV_IDXN_C4 || ktype == CONV_IDXN_C8 || ktype == CONV_IDXN_C32 || ktype == CONV_IDXN_C64) {
            sscanf(kname_substrs[3].c_str(), "b%dx%d", &tile_m_per_cta, &tile_n_per_cta);
            sscanf(kname_substrs[4].c_str(), "w%dx%d", &tile_m_per_warp, &tile_n_per_warp);
            sscanf(kname_substrs[5].c_str(), "k%d",    &tile_k_per_cta);
            sscanf(kname_substrs[6].c_str(), "s%d",    &tile_k_per_step);
    
            if(tile_k_per_step == 8)  flt_pad_size = 2; // only for fp16
            else if(tile_k_per_step == 16)  flt_pad_size = 4;
            else if(tile_k_per_step == 32) flt_pad_size = 8;
            else if(tile_k_per_step == 64) flt_pad_size = 16; // only for int8
            else flt_pad_size = -1;
    
            cta_size_in_thd = (tile_m_per_cta / tile_m_per_warp) * \
                              (tile_n_per_cta / tile_n_per_warp) * \
                              WARP_SIZE;

            smem_size = (tile_m_per_cta + cta_size_in_thd) * _INT4_TO_4INT_ * _INT_TO_4BYTE_;
        } else if (ktype == CONV_2SPK_F1 || ktype == CONV_2SPK_F3 || ktype == CONV_2SPK_FN || ktype == CONV_2SPK_FS) {
            if (strstr(kname_substrs[3].c_str(), "f1"))
                flt_size = 1;
            else if (strstr(kname_substrs[3].c_str(), "f3"))
                flt_size = 3;
            else if (strstr(kname_substrs[3].c_str(), "fn"))
                flt_size = 0;
            else if (strstr(kname_substrs[3].c_str(), "fs"))
                flt_size = 11;
            else
                flt_size = -1;

            sscanf(kname_substrs[4].c_str(), "b%dx%d", &tile_m_per_cta, &tile_n_per_cta);
            sscanf(kname_substrs[5].c_str(), "w%dx%d", &tile_m_per_warp, &tile_n_per_warp);
            sscanf(kname_substrs[6].c_str(), "k%d", &tile_k_per_cta);
            sscanf(kname_substrs[7].c_str(), "s%d", &tile_k_per_set);
            sscanf(kname_substrs[8].c_str(), "buf%d", &buf_num);

            cta_size_in_thd = (tile_m_per_cta / tile_m_per_warp) *
                              (tile_n_per_cta / tile_n_per_warp) *
                              (tile_k_per_cta / tile_k_per_set) *
                              WARP_SIZE;

            int smem_a_v1 = 0;
            int smem_b_v1 = 0;
            int smem_r_v1 = 0;

            if (strstr(kname_substrs[0].c_str(), "Int8")) {
                smem_a_v1 = tile_m_per_cta * tile_k_per_cta * buf_num / _4CHAR_TO_INT_;
                smem_b_v1 = tile_n_per_cta * tile_k_per_cta * buf_num / _4CHAR_TO_INT_;
                smem_r_v1 = tile_m_per_cta * tile_n_per_cta * (tile_k_per_cta / tile_k_per_set);
            } else if (strstr(kname_substrs[0].c_str(), "Fp16")) {
                smem_a_v1 = tile_m_per_cta * tile_k_per_cta * buf_num / _2HALF_TO_INT_;
                smem_b_v1 = tile_n_per_cta * tile_k_per_cta * buf_num / _2HALF_TO_INT_;
                smem_r_v1 = tile_m_per_cta * tile_n_per_cta * (tile_k_per_cta / tile_k_per_set) / _2HALF_TO_INT_;
            }

            smem_size = Max(smem_a_v1 + smem_b_v1, smem_r_v1) * _INT_TO_4BYTE_;
        } else if( ktype == CONV_SWZL_F1 || ktype == CONV_SWZL_F3 || ktype == CONV_SWZL_FN ) {
            if(      strstr(kname_substrs[3].c_str(), "f1") ) flt_size = 1;
            else if( strstr(kname_substrs[3].c_str(), "f3") ) flt_size = 3;
            else if( strstr(kname_substrs[3].c_str(), "fn") ) flt_size = 0;
            else flt_size = -1;
    
            sscanf(kname_substrs[4].c_str(), "b%dx%d", &tile_m_per_cta,  &tile_n_per_cta);
            sscanf(kname_substrs[5].c_str(), "w%dx%d", &tile_m_per_warp, &tile_n_per_warp);
            sscanf(kname_substrs[6].c_str(), "k%d",    &tile_k_per_cta);
            sscanf(kname_substrs[7].c_str(), "buf%d",  &buf_num);
    
            cta_size_in_thd = (tile_m_per_cta / tile_m_per_warp) * \
                              (tile_n_per_cta / tile_n_per_warp) * \
                              WARP_SIZE;

            int smem_a_v1 = 0;
            int smem_b_v1 = 0;
            int smem_r_v1 = 0;

            if (strstr(kname_substrs[0].c_str(), "Int8")) {
                smem_a_v1 = tile_m_per_cta * tile_k_per_cta * buf_num / _4CHAR_TO_INT_;
                smem_b_v1 = tile_n_per_cta * tile_k_per_cta * buf_num / _4CHAR_TO_INT_;
    
                if (strstr(kname_substrs[1].c_str(), "imma16816") || strstr(kname_substrs[1].c_str(), "imma16832") ) {
                    const int TILE_N_PER_MMA = 16;
    
                    smem_r_v1 = tile_m_per_cta * TILE_N_PER_MMA * (cta_size_in_thd / WARP_SIZE);
                } else if (strstr(kname_substrs[1].c_str(), "imma8816")) {
                    const int TILE_N_PER_MMA = 8;
    
                    if(tile_m_per_warp == 8)
                        smem_r_v1 = tile_m_per_cta * TILE_N_PER_MMA * (cta_size_in_thd / WARP_SIZE) * 2;
                    else if (tile_m_per_warp == 16 || tile_m_per_warp == 32 || tile_m_per_warp ==64)
                        smem_r_v1 = tile_m_per_cta * TILE_N_PER_MMA * (cta_size_in_thd / WARP_SIZE);
                }
            } else if (strstr(kname_substrs[0].c_str(), "Fp16")) {
                smem_a_v1 = tile_m_per_cta * tile_k_per_cta * buf_num / _2HALF_TO_INT_;
                smem_b_v1 = tile_n_per_cta * tile_k_per_cta * buf_num / _2HALF_TO_INT_;
    
                const int TILE_N_PER_MMA = 16;
    
                if(tile_m_per_warp == 8)
                    smem_r_v1 = tile_m_per_cta * TILE_N_PER_MMA * (cta_size_in_thd / WARP_SIZE) * 2 / _2HALF_TO_INT_;
                else if (tile_m_per_warp == 16 || tile_m_per_warp == 32 || tile_m_per_warp ==64)
                    smem_r_v1 = tile_m_per_cta * TILE_N_PER_MMA * (cta_size_in_thd / WARP_SIZE) / _2HALF_TO_INT_;
            }

            smem_size = Max(smem_a_v1 + smem_b_v1, smem_r_v1) * _INT_TO_4BYTE_;
        }
    }

    bool CheckSMemSizeFeasible(cudaDeviceProp& device_prop)
    {
        if (device_prop.major == 7 && device_prop.minor == 5)
            return (smem_size <= SM75_MAX_DYN_SMEM_SIZE_PER_CTA);

        if (device_prop.major == 8 && device_prop.minor == 0)
            return (smem_size <= SM80_MAX_DYN_SMEM_SIZE_PER_CTA);

        if (device_prop.major == 8 && device_prop.minor == 6)
            return (smem_size <= SM86_MAX_DYN_SMEM_SIZE_PER_CTA);

        if (device_prop.major == 8 && device_prop.minor == 7)
            return (smem_size <= SM87_MAX_DYN_SMEM_SIZE_PER_CTA);

        return false;
    }

    bool CheckGpuArchFeasible(cudaDeviceProp& device_prop)
    {
        return device_prop.major > karch_major || (device_prop.major == karch_major && device_prop.minor >= karch_minor);
    }

    void AdaptLutKernelSMemSize();
    void AdaptSpkKernelSMemSize();
    void AdaptInt8LutKernelSMemSize();
    void AdaptInt8SpkKernelSMemSize();

    bool CheckKernelTypeFeasible(int flt_height, int flt_width, int num_chl_per_grp, int splitk)
    {
        if (num_chl_per_grp > 0 && num_chl_per_grp <= 2) {
            if (ktype == CONV_IDXN_C2 && splitk == 1) {
                int num_chl_per_grp_pad = Align(num_chl_per_grp, 2);

                int kloop_num  = DivUp(flt_height * flt_width * num_chl_per_grp_pad, tile_k_per_cta);
                int kloop_time = DivUp(kloop_num * (tile_k_per_cta / flt_pad_size), cta_size_in_thd);

                return (kloop_time == 1);
            } else
                return false;
        }
        else if (num_chl_per_grp > 2 && num_chl_per_grp <= 4) {
            if (ktype == CONV_IDXN_C4 && splitk == 1) {
                int num_chl_per_grp_pad = Align(num_chl_per_grp, 4);

                int kloop_num  = DivUp(flt_height * flt_width * num_chl_per_grp_pad, tile_k_per_cta);
                int kloop_time = DivUp(kloop_num * (tile_k_per_cta / flt_pad_size), cta_size_in_thd);

                return (kloop_time == 1);
            } else
                return false;
        } else if (num_chl_per_grp > 4 && num_chl_per_grp <= 32) {
            if (ktype == CONV_IDXN_C32 && splitk == 1) {
                int num_chl_per_grp_pad = Align(num_chl_per_grp, 8);

                int kloop_num  = DivUp(flt_height * flt_width * num_chl_per_grp_pad, tile_k_per_cta);
                int kloop_time = DivUp(kloop_num * (tile_k_per_cta / flt_pad_size), cta_size_in_thd);

                return (kloop_time == 1);
            } else
                return false;
        } else if (flt_height == 1 && flt_width == 1) {
            return (ktype == CONV_2SPK_F1 || ktype == CONV_SWZL_F1) ? true : false;
        } else if (flt_height == 3 && flt_width == 3) {
            return (ktype == CONV_2SPK_F3 || ktype == CONV_SWZL_F3 || ktype == CONV_2SPK_FS) ? true : false;
        } else if (flt_height * flt_width < 128) {
            return (ktype == CONV_2SPK_FN || ktype == CONV_SWZL_FN || ktype == CONV_2SPK_FS) ? true : false;
        }

        return false;
    }

    bool CheckKernelTypeFeasibleInt8(int flt_height, int flt_width, int num_chl_per_grp, int splitk)
    {
        if (ktype == CONV_IDXN_C4 || ktype == CONV_IDXN_C8 || ktype == CONV_IDXN_C64) {
            if (num_chl_per_grp > 0 && num_chl_per_grp <= 4) {
                if (ktype == CONV_IDXN_C4 && splitk == 1) {
                    int num_chl_per_grp_pad = Align(num_chl_per_grp, 4);

                    int kloop_num  = DivUp(flt_height * flt_width * num_chl_per_grp_pad, tile_k_per_cta);
                    int kloop_time = DivUp(kloop_num * (tile_k_per_cta / flt_pad_size), cta_size_in_thd);

                    return (kloop_time == 1);
                } else
                    return false;
            }
            else if (num_chl_per_grp > 4 && num_chl_per_grp <= 8) {
                if (ktype == CONV_IDXN_C8 && splitk == 1) {
                    int num_chl_per_grp_pad = Align(num_chl_per_grp, 8);

                    int kloop_num  = DivUp(flt_height * flt_width * num_chl_per_grp_pad, tile_k_per_cta);
                    int kloop_time = DivUp(kloop_num * (tile_k_per_cta / flt_pad_size), cta_size_in_thd);

                    return (kloop_time == 1);
                } else
                    return false;
            } else if (num_chl_per_grp > 8 && num_chl_per_grp <= 64) {
                if (ktype == CONV_IDXN_C64 && splitk == 1) {
                    int num_chl_per_grp_pad = Align(num_chl_per_grp, 16);

                    int kloop_num  = DivUp(flt_height * flt_width * num_chl_per_grp_pad, tile_k_per_cta);
                    int kloop_time = DivUp(kloop_num * (tile_k_per_cta / flt_pad_size), cta_size_in_thd);

                    return (kloop_time == 1);
                } else
                    return false;
            }
        } else if (flt_height == 1 && flt_width == 1) {
            return (ktype == CONV_2SPK_F1 || ktype == CONV_SWZL_F1) ? true : false;
        } else if (flt_height == 3 && flt_width == 3) {
            return (ktype == CONV_2SPK_F3 || ktype == CONV_SWZL_F3 || ktype == CONV_2SPK_FS) ? true : false;
        } else if (flt_height * flt_width < 128) {
            return (ktype == CONV_2SPK_FN || ktype == CONV_SWZL_FN || ktype == CONV_2SPK_FS) ? true : false;
        }

        return false;
    }

    __inline__ bool CheckSplitkFeasible(int num_chl_per_grp, int splitk)
    {
        if (splitk > 1 && splitk * tile_k_per_cta >= Align(num_chl_per_grp, tile_k_per_cta))
            return false;
        else
            return true;
    }

    __inline__ bool CheckSplitfFeasible(int splitf, int splitk)
    {
        if (ktype == CONV_2SPK_FS) {
            if (splitf == 1)
                return false;
            else if (splitf * splitk > MAX_SPLIT_SIZE)
                return false;
            else
                return true;
        } else
            return true;
    }

    __inline__ bool CheckQuickSelectFeasible(algo_param_t algo_param, int num_chl_per_grp, int grid_size, int flt_hw, int splitk, int splitf, int device_id)
    {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device_id);

        if (kname.at(kname.length() - 1) == '2') // Delete Buf2
            return false;

        if ((ktype == CONV_2SPK_FN || ktype == CONV_SWZL_FN) && (flt_hw == 1 || flt_hw == 9)) // Delete FN for f1 and f3 case
            return false;

        if (splitk != 1 && num_chl_per_grp / splitk <= 128) // Filt splitk for small chl size
            return false;
        if (splitf != 1 && splitk != 1) 
            return false;
        if (splitf != 1 && grid_size / device_prop.multiProcessorCount > 4 && flt_hw * num_chl_per_grp <= 512)
            return false;

        return true;
    }
};

///////////////////////////////////////////////////////////////
// assist function
///////////////////////////////////////////////////////////////

__inline__ int GetPadSize(ppl::common::datatype_t type)
{
    unsigned int pad_size = 0;
    if( type == ppl::common::DATATYPE_FLOAT32 )
	    pad_size = _INT4_TO_4FLOAT_;
    else if( type == ppl::common::DATATYPE_FLOAT16 )
	    pad_size = _INT4_TO_8HALF_;
    else if( type == ppl::common::DATATYPE_INT8 )
	    pad_size = _INT4_TO_16CHAR_;

    return pad_size;
}

__inline__ bool isFuseSupport(conv_param_t &conv_param)
{
    int num_flt_per_grp     = conv_param.num_flt / conv_param.num_grp;
    int num_flt_per_grp_pad = Align(num_flt_per_grp, _C8_);

    // prelu and concat are not supported when group padding
    return num_flt_per_grp == num_flt_per_grp_pad || conv_param.num_grp == 1;
}

__inline__ uint64_t GetCvtInputSize(
        ppl::common::datatype_t type,
        conv_param_t &conv_param,
        unsigned int num_chl_per_grp_pad)
{
    uint64_t cvt_input_size = (uint64_t)conv_param.in_num * conv_param.in_height * \
                                  conv_param.in_width * num_chl_per_grp_pad * \
                                  conv_param.num_grp;
    unsigned int bytes = ppl::common::GetSizeOfDataType(type);

    return Align(cvt_input_size * bytes, _BYTE128_);
}

__inline__ uint64_t getCvtOutputSize(
        ppl::common::datatype_t type,
        conv_param_t &conv_param,
        unsigned int num_flt_per_grp_pad)
{
    uint64_t cvt_output_size  = (uint64_t)conv_param.in_num * conv_param.out_height * \
                                    conv_param.out_width * num_flt_per_grp_pad * \
                                    conv_param.num_grp;
    uint64_t bytes            = ppl::common::GetSizeOfDataType(type);

    return Align(cvt_output_size * bytes, _BYTE128_);
}

__inline__ uint64_t GetMaxSplitSize(
    ppl::common::datatype_t type,
    conv_param_t &conv_param,
    unsigned int num_flt_per_grp_pad)
{
    uint64_t split_size = conv_param.out_height * conv_param.out_width *
                          num_flt_per_grp_pad * conv_param.num_grp *
                          conv_param.in_num;
    if (type==ppl::common::DATATYPE_INT8)
        type=ppl::common::DATATYPE_FLOAT32;
    unsigned int bytes = ppl::common::GetSizeOfDataType(type);

    return split_size * bytes * MAX_SPLIT_SIZE;
}

__inline__ uint64_t GetSplitKFSize(
    ppl::common::datatype_t type,
    conv_param_t &conv_param,
    unsigned int num_flt_per_grp_pad,
    int splitf,
    int splitk)
{
    uint64_t split_size = conv_param.out_height * conv_param.out_width *
                          num_flt_per_grp_pad * conv_param.num_grp *
                          conv_param.in_num;
    if (type==ppl::common::DATATYPE_INT8)
        type=ppl::common::DATATYPE_FLOAT32;
    unsigned int bytes = ppl::common::GetSizeOfDataType(type);

    return split_size * bytes * splitf * splitk;
}

void PPLCUDAConvolutionCvtFlt(
    cudaStream_t &stream,
    void *output,
    const void *input,
    ppl::common::datatype_t type,
    conv_param_t &conv_param);

void PPLCUDAConvolutionCvtInput(
    cudaStream_t &stream,
    void *output,
    const void *input,
    ppl::common::datatype_t type,
    conv_param_t &conv_param);

void PPLCUDAConvolutionCvtOutput(
    cudaStream_t &stream,
    void *output,
    const void *input,
    ppl::common::datatype_t type,
    conv_param_t &conv_param);

void PPLCUDAConvolutionCvtBias(
    cudaStream_t &stream,
    void *output,
    const void *input,
    ppl::common::datatype_t type,
    conv_param_t &conv_param);

// sm75 kernels
// fp16
void Initialize2spkSM75FP16Hmma1688ConvF1KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM75FP16Hmma1688ConvF3KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM75FP16Hmma1688ConvFNKernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM75FP16Hmma1688ConvFSKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeIdxnSM75FP16Hmma1688ConvKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeSwzlSM75FP16Hmma1688ConvF1KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM75FP16Hmma1688ConvF3KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM75FP16Hmma1688ConvFNKernelContainer(std::vector<kernel_info_t> & kernel_container);

// int8
void Initialize2spkSM75Int8Imma8816ConvF1KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM75Int8Imma8816ConvF3KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM75Int8Imma8816ConvFNKernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM75Int8Imma8816ConvFSKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeIdxnSM75Int8Imma8816ConvKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeSwzlSM75Int8Imma8816ConvF1KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM75Int8Imma8816ConvF3KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM75Int8Imma8816ConvFNKernelContainer(std::vector<kernel_info_t> & kernel_container);

// sm80 kernels
// fp16
void Initialize2spkSM80FP16Hmma1688ConvF1KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80FP16Hmma1688ConvF3KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80FP16Hmma1688ConvFNKernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80FP16Hmma1688ConvFSKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeSwzlSM80FP16Hmma1688ConvF1KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80FP16Hmma1688ConvF3KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80FP16Hmma1688ConvFNKernelContainer(std::vector<kernel_info_t> & kernel_container);

void Initialize2spkSM80FP16Hmma16816ConvF1KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80FP16Hmma16816ConvF3KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80FP16Hmma16816ConvFNKernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80FP16Hmma16816ConvFSKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeIdxnSM80FP16Hmma16816ConvKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeSwzlSM80FP16Hmma16816ConvF1KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80FP16Hmma16816ConvF3KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80FP16Hmma16816ConvFNKernelContainer(std::vector<kernel_info_t> & kernel_container);

// int8
void Initialize2spkSM80Int8Imma8816ConvF1KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80Int8Imma8816ConvF3KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80Int8Imma8816ConvFNKernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80Int8Imma8816ConvFSKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeSwzlSM80Int8Imma8816ConvF1KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80Int8Imma8816ConvF3KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80Int8Imma8816ConvFNKernelContainer(std::vector<kernel_info_t> & kernel_container);

void Initialize2spkSM80Int8Imma16816ConvF1KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80Int8Imma16816ConvF3KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80Int8Imma16816ConvFNKernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80Int8Imma16816ConvFSKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeIdxnSM80Int8Imma16816ConvKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeSwzlSM80Int8Imma16816ConvF1KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80Int8Imma16816ConvF3KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80Int8Imma16816ConvFNKernelContainer(std::vector<kernel_info_t> & kernel_container);

void Initialize2spkSM80Int8Imma16832ConvF1KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80Int8Imma16832ConvF3KernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80Int8Imma16832ConvFNKernelContainer(std::vector<kernel_info_t> &kernel_container);
void Initialize2spkSM80Int8Imma16832ConvFSKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeIdxnSM80Int8Imma16832ConvKernelContainer(std::vector<kernel_info_t> &kernel_container);

void InitializeSwzlSM80Int8Imma16832ConvF1KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80Int8Imma16832ConvF3KernelContainer(std::vector<kernel_info_t> & kernel_container);
void InitializeSwzlSM80Int8Imma16832ConvFNKernelContainer(std::vector<kernel_info_t> & kernel_container);

#endif
