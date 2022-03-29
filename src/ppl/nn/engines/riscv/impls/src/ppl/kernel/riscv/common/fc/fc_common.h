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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_FC_FC_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_FC_FC_COMMON_H_

#include <vector>
#include <cstring>
#include "ppl/common/log.h"
#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

struct fc_tunning_param {
    int64_t m_blk;
    int64_t n_blk;
    int64_t k_blk;
    int64_t num_thread;
};

template<typename T>
using fc_common_gemm_kernel = void (*)(const T*, const T*, const T*, T*, int32_t, int32_t, int32_t);

template <typename T>
static void fc_common_load_tile(
    const T* src,
    T* src_tile,
    int32_t src_h,
    int32_t src_w,
    int32_t tile_h_beg,
    int32_t tile_h_len,
    int32_t tile_w_beg,
    int32_t tile_w_len
) {
    for (int32_t mi = 0; mi < tile_h_len; mi += 1) {
        auto src_ = src + (mi + tile_h_beg) * src_w + tile_w_beg;
        auto src_tile_ = src_tile + mi * tile_w_len;
        memcpy(src_tile_, src_, tile_w_len * sizeof(T));
    }
}

template <typename T>
static void fc_common_store_tile(
    const T* dst_tile,
    T* dst,
    int32_t dst_h,
    int32_t dst_w,
    int32_t tile_h_beg,
    int32_t tile_h_len,
    int32_t tile_w_beg,
    int32_t tile_w_len
) {
    for (int32_t mi = 0; mi < tile_h_len; mi += 1) {
        auto dst_ = dst + (mi + tile_h_beg) * dst_w + tile_w_beg;
        auto dst_tile_ = dst_tile + mi * tile_w_len;
        memcpy(dst_, dst_tile_, tile_w_len * sizeof(T));
    }
}

template <typename T, int64_t c_blk>
void fc_common_cvt_flt_to_nxcx(
    const T* flt,
    T* flt_cvt,

    int32_t num_outs,
    int32_t channels)
{
    int64_t pad_channels = round_up(channels, c_blk);
    int64_t pad_num_outs = round_up(num_outs, c_blk);

    memset(flt_cvt, (T)0, pad_channels * pad_num_outs * sizeof(T));
    for (int64_t i = 0; i < num_outs; i += 1) {
        for (int64_t j = 0; j < channels; j += 1) {
            flt_cvt[i / c_blk * pad_channels * c_blk + j * c_blk + i % c_blk] = flt[i * channels + j];
        }
    }
}


template <typename T, int64_t atom_n, int64_t atom_k>
uint64_t fc_common_cal_temp_buffer_size(
    int32_t m,
    int32_t n,
    int32_t k,
    const fc_tunning_param &tunning_param
) {
    fc_tunning_param pad_param = {
        round_up(tunning_param.m_blk, 1),
        round_up(tunning_param.n_blk, atom_n),
        round_up(tunning_param.k_blk, atom_k)
    };

    uint64_t data_size = sizeof(T);
    uint64_t src_tile_size = pad_param.m_blk * pad_param.k_blk * data_size;
    uint64_t flt_tile_size = pad_param.k_blk * pad_param.n_blk * data_size;
    uint64_t dst_tile_size = pad_param.m_blk * pad_param.n_blk * data_size;
    
    return src_tile_size + flt_tile_size + dst_tile_size;
}

template <typename T, int64_t atom_oc, int64_t atom_ic>
ppl::common::RetCode fc_common_blocking_execute(
    const T* src,
    const T* flt,
    const T* bias,
    T* dst,
    void* temp_buffer,
    int32_t batch,
    int32_t num_channels,
    int32_t num_outs,
    const fc_tunning_param &tunning_param,
    const fc_common_gemm_kernel<T> tile_kernel_func
) {
    fc_tunning_param pad_param = {
        round_up(tunning_param.m_blk, 1),
        round_up(tunning_param.n_blk, atom_oc),
        round_up(tunning_param.k_blk, atom_ic)
    };

    T* src_tile = (T*)temp_buffer;
    T* flt_tile = src_tile + pad_param.m_blk * pad_param.k_blk;
    T* dst_tile = flt_tile + pad_param.k_blk * pad_param.n_blk;

    int64_t pad_num_channels = round_up(num_channels, atom_ic);
    int64_t pad_num_outs = round_up(num_outs, atom_oc);

    for (int64_t n_tile_beg = 0; n_tile_beg < pad_num_outs; n_tile_beg += pad_param.n_blk) {
        int64_t n_tile_len = min(pad_num_outs - n_tile_beg, pad_param.n_blk);
        auto bias_ = bias + n_tile_beg;

        for (int64_t m_tile_beg = 0; m_tile_beg < batch; m_tile_beg += pad_param.m_blk) {
            int64_t m_tile_len = min(batch - m_tile_beg, pad_param.m_blk);
            
            for (int64_t k_tile_beg = 0; k_tile_beg < pad_num_channels; k_tile_beg += pad_param.k_blk) {
                int64_t k_tile_len = min(pad_num_channels - k_tile_beg, pad_param.k_blk);
                
                // load src tile
                fc_common_load_tile(
                    src,
                    src_tile,
                    batch,
                    pad_num_channels,
                    m_tile_beg,
                    m_tile_len,
                    k_tile_beg,
                    k_tile_len
                );

                // load flt tile
                fc_common_load_tile(
                    flt,
                    flt_tile,
                    pad_num_outs / atom_oc,
                    pad_num_channels * atom_oc,
                    n_tile_beg / atom_oc,
                    n_tile_len / atom_oc,
                    k_tile_beg * atom_oc,
                    k_tile_len * atom_oc
                );

                // tile kernel
                tile_kernel_func(
                    src_tile,
                    flt_tile,
                    bias_,
                    dst_tile,
                    m_tile_len,
                    n_tile_len,
                    k_tile_len
                );

            }
            // store dst tile
            fc_common_store_tile(
                dst_tile,
                dst,
                batch,
                pad_num_outs,
                m_tile_beg,
                m_tile_len,
                n_tile_beg,
                n_tile_len
            );
        }
    }

    return ppl::common::RC_SUCCESS;
}

}}}; //  namespace ppl::kernel::riscv

#endif //  __ST_PPL_KERNEL_RISCV_COMMON_FC_FC_COMMON_H_
