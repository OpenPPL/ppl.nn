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
#define _4INT_TO_INT4_          4
#define _INT_TO_4BYTE_          4
#define _INT4_TO_4INT_          4
#define _INT4_TO_4FLOAT_        4
#define _INT4_TO_8HALF_         8
#define _C2_                    2
#define _C4_                    4
#define _C8_                    8
#define _BYTE128_               128

#define Max(x, y)         (((x) > (y))  ? (x) : (y))

#define MAX_SPLIT_SIZE          18

#define ADD_KERNEL(_ktype, _kname, _lut_kptr, _spk_kptr, _idx_kptr) \
    kernel_container.push_back(kernel_info_t(kernel_container.size(), _ktype, _kname, _lut_kptr, _spk_kptr, _idx_kptr));

enum {
     CONV_2SPK_F1 = 0,
     CONV_2SPK_F3 = 1,
     CONV_2SPK_FN = 2,
     CONV_2SPK_FS = 3,
     CONV_IDXN_C2 = 4,
     CONV_IDXN_C4 = 5,
     CONV_IDXN_C32 = 6,
     CONV_KTYPE_NUM,
};

typedef uint32_t conv_ktype_t;

struct kernel_info_t 
{
    int kid;

    std::string kname;

    conv_ktype_t ktype;

    lut_kernel_t* lut_kptr;
    spk_kernel_t* spk_kptr;
    idx_kernel_t* idx_kptr;

    int tile_m_per_cta;
    int tile_n_per_cta;
    int tile_k_per_cta;

    int tile_m_per_warp;
    int tile_n_per_warp;
    int tile_k_per_warp;

    int tile_k_per_step; // for idxn conv
    int tile_k_per_set;  // for 2spk conv

    int flt_size; // for 2spk conv 
    int flt_pad_size; // for idxn conv 

    int cta_size_in_thd;

    kernel_info_t()
    {
        kname    = "";
        kid      = -1;
        ktype    = CONV_KTYPE_NUM;
        lut_kptr = NULL;
        spk_kptr = NULL;
        idx_kptr = NULL;

        tile_m_per_cta  = -1;
        tile_n_per_cta  = -1;
        tile_k_per_cta  = -1;

        tile_m_per_warp = -1;
        tile_n_per_warp = -1;
        tile_k_per_warp = -1;

        tile_k_per_step = -1;
        tile_k_per_set  = -1;

        flt_size = -1;
        flt_pad_size = -1;

        cta_size_in_thd = -1;
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

    void parse_kname()
    {
        std::stringstream kname_str(kname);
        std::vector<std::string> kname_substrs;
        std::string substr;
    
        while(std::getline(kname_str, substr, '_')) {
           kname_substrs.push_back(substr);
        }
    
        if( ktype == CONV_IDXN_C2 || ktype == CONV_IDXN_C4 || ktype == CONV_IDXN_C32 ) {
            sscanf(kname_substrs[3].c_str(), "b%dx%d", &tile_m_per_cta,  &tile_n_per_cta);
            sscanf(kname_substrs[4].c_str(), "w%dx%d", &tile_m_per_warp, &tile_n_per_warp);
            sscanf(kname_substrs[5].c_str(), "k%d",    &tile_k_per_cta);
            sscanf(kname_substrs[6].c_str(), "s%d",    &tile_k_per_step);
    
            if(tile_k_per_step == 8)  flt_pad_size = 2;
            else if(tile_k_per_step == 16) flt_pad_size = 4;
            else if(tile_k_per_step == 32) flt_pad_size = 8;
            else flt_pad_size = -1;
    
            cta_size_in_thd = (tile_m_per_cta / tile_m_per_warp) * \
                              (tile_n_per_cta / tile_n_per_warp) * \
                              WARP_SIZE;
        } else if( ktype == CONV_2SPK_F1 || ktype == CONV_2SPK_F3 || ktype == CONV_2SPK_FN || ktype == CONV_2SPK_FS ) {
            if(      strstr(kname_substrs[3].c_str(), "f1") ) flt_size = 1;
            else if( strstr(kname_substrs[3].c_str(), "f3") ) flt_size = 3;
            else if( strstr(kname_substrs[3].c_str(), "fn") ) flt_size = 0;
            else if( strstr(kname_substrs[3].c_str(), "fs") ) flt_size = 1;
            else flt_size = -1;
    
            sscanf(kname_substrs[4].c_str(), "b%dx%d", &tile_m_per_cta,  &tile_n_per_cta);
            sscanf(kname_substrs[5].c_str(), "w%dx%d", &tile_m_per_warp, &tile_n_per_warp);
            sscanf(kname_substrs[6].c_str(), "k%d",    &tile_k_per_cta);
            sscanf(kname_substrs[7].c_str(), "s%d",    &tile_k_per_set);
    
            cta_size_in_thd = (tile_m_per_cta / tile_m_per_warp) * \
                              (tile_n_per_cta / tile_n_per_warp) * \
                              (tile_k_per_cta / tile_k_per_set)  * \
                              WARP_SIZE;
        }
    }

    bool CheckKernelTypeFeasible(int flt_height, int flt_width, int num_chl_per_grp, int splitk)
    {
        if(num_chl_per_grp > 0 && num_chl_per_grp <= 2) {
            if(ktype == CONV_IDXN_C2 && splitk == 1) {
                int num_chl_per_grp_pad = Align(num_chl_per_grp, 2);

	            int kloop_num  = DivUp(flt_height * flt_width * num_chl_per_grp_pad, tile_k_per_cta);
                int kloop_time = DivUp(kloop_num * (tile_k_per_cta / flt_pad_size), cta_size_in_thd);

                return (kloop_time == 1);
            } else
                return false;
        }
        else if(num_chl_per_grp > 2 && num_chl_per_grp <= 4) {
            if(ktype == CONV_IDXN_C4 && splitk == 1) {
                int num_chl_per_grp_pad = Align(num_chl_per_grp, 4);

	            int kloop_num  = DivUp(flt_height * flt_width * num_chl_per_grp_pad, tile_k_per_cta);
                int kloop_time = DivUp(kloop_num * (tile_k_per_cta / flt_pad_size), cta_size_in_thd);

                return (kloop_time == 1);
            } else
                return false;
        } else if(num_chl_per_grp > 4 && num_chl_per_grp <= 32) {
            if(ktype == CONV_IDXN_C32 && splitk == 1) {
                int num_chl_per_grp_pad = Align(num_chl_per_grp, 8);

	            int kloop_num  = DivUp(flt_height * flt_width * num_chl_per_grp_pad, tile_k_per_cta);
                int kloop_time = DivUp(kloop_num * (tile_k_per_cta / flt_pad_size), cta_size_in_thd);

                return (kloop_time == 1);
            } else
                return false;
        } else if(flt_height == 1 && flt_width == 1) {
            return (ktype == CONV_2SPK_F1) ? true : false;
        } else if(flt_height == 3 && flt_width == 3) {
            return (ktype == CONV_2SPK_F3 || ktype == CONV_2SPK_FS) ? true : false;
        } else if(flt_height * flt_width < 128) {
            return (ktype == CONV_2SPK_FN || ktype == CONV_2SPK_FS) ? true : false;
        }

        return false;
    }

    __inline__ bool CheckSplitkFeasible(int num_chl_per_grp, int splitk)
    {
        if(splitk > 1 && splitk * tile_k_per_cta >= Align(num_chl_per_grp, tile_k_per_cta))
            return false;
        else
            return true;
    }

    __inline__ bool CheckSplitfFeasible(int splitf, int splitk)
    {
        if(ktype == CONV_2SPK_FS) {
            if(splitf == 1)
                return false;
            else if(splitf * splitk > MAX_SPLIT_SIZE)
                return false;
            else
                return true;
        } else
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

    return pad_size;
}

__inline__ bool isFuseSupport(conv_param_t &conv_param)
{
    int num_flt_per_grp     = conv_param.num_flt/conv_param.num_grp;
    int num_flt_per_grp_pad = Align(num_flt_per_grp, _C8_);

    // prelu and concat are not supported when group padding 
    return num_flt_per_grp == num_flt_per_grp_pad || conv_param.num_grp == 1;
}

__inline__ unsigned int GetCvtInputSize(
        ppl::common::datatype_t type,
        conv_param_t &conv_param,
        unsigned int num_chl_per_grp_pad)
{
    unsigned int cvt_input_size = conv_param.in_num   * conv_param.in_height * \
                                  conv_param.in_width * num_chl_per_grp_pad * \
                                  conv_param.num_grp;
    unsigned int bytes          = ppl::common::GetSizeOfDataType(type);

    return Align(cvt_input_size * bytes, _BYTE128_);
}

__inline__ unsigned int getCvtOutputSize(
        ppl::common::datatype_t type,
        conv_param_t &conv_param,
        unsigned int num_flt_per_grp_pad)
{
    unsigned int cvt_output_size  = conv_param.in_num    * conv_param.out_height * \
                                    conv_param.out_width * num_flt_per_grp_pad * \
                                    conv_param.num_grp;
    unsigned int bytes            = ppl::common::GetSizeOfDataType(type);

    return Align(cvt_output_size * bytes, _BYTE128_);
}

__inline__ uint64_t GetMaxSplitSize(
        ppl::common::datatype_t type,
        conv_param_t &conv_param,
        unsigned int num_flt_per_grp_pad)
{
    uint64_t split_size = conv_param.out_height * conv_param.out_width * \
			              num_flt_per_grp_pad   * conv_param.num_grp   * \
                          conv_param.in_num; 
    unsigned int bytes  = ppl::common::GetSizeOfDataType(type);

    return split_size * bytes * MAX_SPLIT_SIZE;
}

__inline__ uint64_t GetSplitKFSize(
        ppl::common::datatype_t type,
        conv_param_t &conv_param,
        unsigned int num_flt_per_grp_pad,
        int splitf,
        int splitk)
{
    uint64_t split_size = conv_param.out_height * conv_param.out_width * \
			              num_flt_per_grp_pad   * conv_param.num_grp   * \
                          conv_param.in_num; 
    unsigned int bytes  = ppl::common::GetSizeOfDataType(type);

    return split_size * bytes * splitf * splitk;
}

void PPLCUDAConvolutionCvtFlt(
        cudaStream_t &stream,
        void* output,
        const void* input, 
        ppl::common::datatype_t type,
        conv_param_t &conv_param);

void PPLCUDAConvolutionCvtInput(
        cudaStream_t &stream,
        void* output,
        const void* input,
        ppl::common::datatype_t type,
        conv_param_t &conv_param);

void PPLCUDAConvolutionCvtOutput(
        cudaStream_t &stream,
        void* output,
        const void* input, 
	    ppl::common::datatype_t type, 
	    conv_param_t &conv_param);

void PPLCUDAConvolutionCvtBias(
        cudaStream_t &stream,
        void* output,
        const void* input, 
	    ppl::common::datatype_t type, 
	    conv_param_t &conv_param);

void Initialize2spkConvF1KernelContainer(std::vector<kernel_info_t> & kernel_container);
void Initialize2spkConvF3KernelContainer(std::vector<kernel_info_t> & kernel_container);
void Initialize2spkConvFNKernelContainer(std::vector<kernel_info_t> & kernel_container);
void Initialize2spkConvFSKernelContainer(std::vector<kernel_info_t> & kernel_container);
                   
void InitializeIdxnConvKernelContainer(std::vector<kernel_info_t> & kernel_container);

#endif
