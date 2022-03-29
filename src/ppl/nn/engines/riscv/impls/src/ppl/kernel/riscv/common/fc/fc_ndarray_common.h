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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_FC_FC_NDARRAY_COMMON_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_FC_FC_NDARRAY_COMMON_H_

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
using fc_common_gemm_kernel_func_t = void (*)(const T*, const T*, const T*, T*, int32_t, int32_t, int32_t);

template<typename T>
using fc_common_select_gemm_kernel_func_t = fc_common_gemm_kernel_func_t<T> (*)(int32_t m, int32_t n);

template <typename T>
static void fc_ndarray_common_load_src(
    const T* src,
    T* src_tile,
    int32_t src_h,
    int32_t src_w,
    int32_t tile_h,
    int32_t tile_w,
    int32_t h_beg,
    int32_t h_len,
    int32_t w_beg,
    int32_t w_len
) {
    auto src_ = src + h_beg * src_w + w_beg;
    auto src_tile_ = src_tile;
    int64_t w_diff = tile_w - w_len;
    for (int64_t hi = 0; hi < h_len; hi += 1) {
        memcpy(src_tile_, src_, w_len * sizeof(T));
        src_tile_ += w_len;
        memset(src_tile_, 0.f, w_diff * sizeof(T));
        src_tile_ += w_diff;
        src_ += src_w;
    }
}

template <typename T>
static void fc_ndarray_common_store_dst(
    const T* dst_tile,
    T* dst,
    int32_t dst_h,
    int32_t dst_w,
    int32_t tile_h,
    int32_t tile_w,
    int32_t h_beg,
    int32_t h_len,
    int32_t w_beg,
    int32_t w_len
) {
    auto dst_ = dst + h_beg * dst_w + w_beg;
    auto dst_tile_ = dst_tile;
    for (int64_t hi = 0; hi < h_len; hi += 1) {
        memcpy(dst_, dst_tile_, w_len * sizeof(T));
        dst_ += dst_w;
        dst_tile_ += tile_w;
    }
}

template <typename T, int64_t ic_blk, int64_t oc_blk>
void fc_ndarray_common_cvt_flt_to_nxcx(
    const T* flt,
    T* flt_cvt,

    int32_t num_outs,
    int32_t channels)
{
    int64_t pad_channels = round_up(channels, ic_blk);
    int64_t pad_num_outs = round_up(num_outs, oc_blk);

    memset(flt_cvt, (T)0, pad_channels * pad_num_outs * sizeof(T));
    for (int64_t i = 0; i < num_outs; i += 1) {
        for (int64_t j = 0; j < channels; j += 1) {
            flt_cvt[i / oc_blk * pad_channels * oc_blk + j * oc_blk + i % oc_blk] = flt[i * channels + j];
        }
    }
}


template <typename T, int64_t atom_n, int64_t atom_k, int64_t flt_atom_oc>
uint64_t fc_ndarray_common_cal_temp_buffer_size(
    int32_t m,
    int32_t n,
    int32_t k,
    const fc_tunning_param &tunning_param
) {
    fc_tunning_param pad_param = {
        round_up(tunning_param.m_blk, 1),
        round_up(tunning_param.n_blk, flt_atom_oc),
        round_up(tunning_param.k_blk, atom_k)
    };

    uint64_t data_size = sizeof(T);
    uint64_t src_trans_size = m * round_up(k, atom_k) * data_size;
    uint64_t src_tile_size = pad_param.m_blk * pad_param.k_blk * data_size;
    uint64_t flt_tile_size = pad_param.n_blk * pad_param.k_blk * data_size;
    uint64_t dst_tile_size = pad_param.m_blk * pad_param.n_blk * data_size;
    
    return src_tile_size + dst_tile_size;
}

template <typename T, int64_t atom_oc, int64_t atom_ic, int64_t flt_atom_oc>
ppl::common::RetCode fc_ndarray_common_blocking_execute(
    const T* src,
    const T* flt,
    const T* bias,
    T* dst,
    void* temp_buffer,
    int32_t batch,
    int32_t num_channels,
    int32_t num_outs,
    const fc_tunning_param &tunning_param,
    const fc_common_select_gemm_kernel_func_t<T> first_tile_select_kernel_func,
    const fc_common_select_gemm_kernel_func_t<T> tile_select_kernel_func
) {
    fc_tunning_param pad_param = {
        round_up(tunning_param.m_blk, 1),
        round_up(tunning_param.n_blk, flt_atom_oc),
        round_up(tunning_param.k_blk, atom_ic)
    };

    int64_t pad_num_channels = round_up(num_channels, atom_ic);
    int64_t pad_num_outs = round_up(num_outs, atom_oc);

    T* src_tile = (T*)temp_buffer;
    // T* flt_tile = src_tile + pad_param.m_blk * pad_param.k_blk;
    // T* dst_tile = flt_tile + pad_param.n_blk * pad_param.k_blk;
    T* dst_tile = src_tile + pad_param.m_blk * pad_param.k_blk;

    // loop n
    for (int64_t n_tile_beg = 0; n_tile_beg < pad_num_outs; n_tile_beg += pad_param.n_blk) {
        int64_t pad_n_tile_len = min(pad_num_outs - n_tile_beg, pad_param.n_blk);
        int64_t n_tile_len = min(num_outs - n_tile_beg, pad_param.n_blk);
        auto bias_ = bias + n_tile_beg;

        // loop m
        for (int64_t m_tile_beg = 0; m_tile_beg < batch; m_tile_beg += pad_param.m_blk) {
            int64_t m_tile_len = min(batch - m_tile_beg, pad_param.m_blk);
            
            // loop k
            {
                int64_t k_tile_beg = 0;
                {
                    int64_t pad_k_tile_len = min(pad_num_channels - k_tile_beg, pad_param.k_blk);
                    int64_t k_tile_len = min(num_channels - k_tile_beg, pad_param.k_blk);

                    auto first_tile_kernel_func = first_tile_select_kernel_func(m_tile_len, n_tile_len);
                    auto flt_tile = flt + n_tile_beg * pad_num_channels + k_tile_beg * flt_atom_oc;

                    // load src tile
                    fc_ndarray_common_load_src(
                        src,
                        src_tile,
                        batch,          // src_h
                        num_channels,   // src_w
                        m_tile_len,     // src_tile_h
                        pad_k_tile_len, // src_tile_w
                        m_tile_beg,
                        m_tile_len,
                        k_tile_beg,
                        k_tile_len
                    );

                    // tile kernel
                    first_tile_kernel_func(
                        src_tile,
                        flt_tile,
                        bias_,
                        dst_tile,
                        m_tile_len,
                        pad_n_tile_len,
                        pad_k_tile_len
                    );

                    k_tile_beg += pad_param.k_blk;
                }

                auto tile_kernel_func = tile_select_kernel_func(m_tile_len, n_tile_len);
                for (; k_tile_beg < pad_num_channels; k_tile_beg += pad_param.k_blk) {
                    int64_t pad_k_tile_len = min(pad_num_channels - k_tile_beg, pad_param.k_blk);
                    int64_t k_tile_len = min(num_channels - k_tile_beg, pad_param.k_blk);

                    auto flt_tile = flt + n_tile_beg * pad_num_channels + k_tile_beg * flt_atom_oc;

                    // load src tile
                    fc_ndarray_common_load_src(
                        src,
                        src_tile,
                        batch,          // src_h
                        num_channels,   // src_w
                        m_tile_len,     // src_tile_h
                        pad_k_tile_len, // src_tile_w
                        m_tile_beg,
                        m_tile_len,
                        k_tile_beg,
                        k_tile_len
                    );

                    // tile kernel
                    tile_kernel_func(
                        src_tile,
                        flt_tile,
                        bias_,
                        dst_tile,
                        m_tile_len,
                        pad_n_tile_len,
                        pad_k_tile_len
                    );
                }
            }

            // store dst tile
            fc_ndarray_common_store_dst(
                dst_tile,
                dst,
                batch,          // dst_h
                num_outs,       // dst_w
                m_tile_len,     // dst_tile_h
                pad_n_tile_len, // dst_tile_w
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

#endif //  __ST_PPL_KERNEL_RISCV_COMMON_FC_FC_NDARRAY_COMMON_H_
