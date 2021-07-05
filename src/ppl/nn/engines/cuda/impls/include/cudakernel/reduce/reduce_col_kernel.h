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

#ifndef PPLCUDA_REDUCE_REDUCE_COL_KERNEL_H_
#define PPLCUDA_REDUCE_REDUCE_COL_KERNEL_H_
#include "cudakernel/reduce/reduce_helper.h"
#include "cudakernel/reduce/block_warp_reduce.h"
#include "cudakernel/common/atomic.h"
#include "cudakernel/math/operators.h"
#include <cuda_fp16.h>
#include <cuda.h>

template <typename T, class Operator, bool MultiBlock, int ReduceSize>
__device__ void ppl_reduce_col_sr(
    Operator op,
    PPLReduceDimDes des,
    ReduceParam param)
{
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::srctype src_type;
    typedef typename Operator::dsttype dst_type;
    typedef typename Operator::OpMath math;
    int64_t n_outer    = des.n_outer;
    int64_t n_reduce   = des.n_reduce;
    int64_t n_inner    = des.n_inner;
    int64_t n_elements = des.num_elements;

    int64_t outer_stride       = n_reduce * n_inner;
    int64_t non_reduce         = n_outer * n_inner;
    int64_t block_size         = blockDim.x * blockDim.y;
    int64_t grid_stride        = block_size * gridDim.x;
    int64_t tid                = blockIdx.x * block_size + threadIdx.y * blockDim.x + threadIdx.x;
    int64_t multi_block_offset = blockIdx.y * n_elements * n_inner;

    for (int64_t idx = tid; idx < non_reduce; idx += grid_stride) {
        int64_t out_idx = idx / n_inner;
        int64_t in_idx  = idx % n_inner;
        int64_t offset  = out_idx * outer_stride + in_idx + multi_block_offset;
        T val           = (T)op.InitVal();
        for (int i = 0; i < n_elements; i++) {
            val = op.compute(val, (T)op.fetch(offset + i * n_inner));
        }
        if (MultiBlock) {
            if (param == ReduceMean) {
                val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
            }
            PPLAtomicWrite<dst_type, Operator>(op.dst + idx, static_cast<dst_type>(val), op);
        } else {
            if (param == ReduceMean) {
                val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
            }
            op.out(idx, static_cast<dst_type>(val));
        }
    }
}

template <typename T, class Operator, bool MultiBlock, int ReduceSize>
__inline __device__ void ppl_reudce_col_mrsi(
    Operator op,
    PPLReduceDimDes des,
    ReduceParam param)
{
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::srctype src_type;
    typedef typename Operator::dsttype dst_type;

    typedef typename Operator::OpMath math;

    __shared__ T shared_data[32][33];

    int64_t n_outer      = des.n_outer;
    int64_t n_reduce     = des.n_reduce;
    int64_t n_inner      = des.n_inner;
    int64_t num_elements = des.num_elements;

    int64_t outer_stride = n_reduce * n_inner;
    // int64_t non_reduce = n_outer * n_inner;
    int64_t block_size   = blockDim.x * blockDim.y;
    int64_t grid_stride  = gridDim.x;
    // int64_t multi_block_offset = blockIdx.y * num_elements * n_inner;

    int64_t lane    = threadIdx.x;
    int64_t warp    = threadIdx.y;
    int64_t n_warps = blockDim.y;
    int64_t tid     = threadIdx.x + threadIdx.y * blockDim.x;

    int64_t lane_offset0 = -((warp * 32) % n_inner);
    if (lane_offset0 < 0)
        lane_offset0 += n_inner;

    int64_t block_end = CudaMin(num_elements * n_inner, outer_stride);
    for (int64_t bx = blockIdx.x; bx < n_outer; bx += grid_stride) {
        for (int w = warp; w < 32; w += n_warps) {
            shared_data[w][lane] = (T)op.InitVal();
        }
        __syncthreads();

        int lane_offset   = lane_offset0;
        int64_t bx_offset = bx * outer_stride;
        T val             = (T)op.InitVal();

        for (int64_t i = 0; i < block_end; i += block_size) {
            int64_t idx  = i + tid;
            int64_t idx0 = idx + lane_offset;

            val = op.compute(val, lane + lane_offset < 32 && idx0 < block_end ? (T)op.fetch(idx0 + bx_offset) : (T)op.InitVal());
            if (lane + lane_offset >= n_inner && lane < n_inner) {
                int64_t idx1 = idx0 - n_inner;
                if (idx1 < block_end)
                    op.compute(val, op.fetch(idx1 + bx_offset));
            }
            lane_offset -= (block_size % n_inner);
            if (lane_offset < 0)
                lane_offset += n_inner;
        }

        shared_data[warp][lane] = val;
        __syncthreads();

        for (int i = warp; i < 32; i += n_warps) {
            val = shared_data[lane][i];
            for (int j = i + n_inner; j < 32; j += n_inner) {
                op.compute(val, shared_data[lane][j]);
            }
            warp_reduce_unroll<T, Operator, ReduceSize>(op, nullptr, val);
            if (MultiBlock) {
                if (threadIdx.x == 0 && i < n_inner) {
                    if (param == ReduceMean) {
                        val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
                    }
                    PPLAtomicWrite<dst_type, Operator>(op.dst + bx_offset + i, static_cast<dst_type>(val), op);
                }
            } else {
                if (threadIdx.x == 0 && i < n_inner) {
                    if (param == ReduceMean) {
                        val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
                    }
                    op.out(bx_offset + i, static_cast<dst_type>(val));
                }
            }
        }
    }
}

template <typename T, class Operator, bool MultiBlock, int ReduceSize>
__inline __device__ void ppl_reduce_col_mrli(
    Operator op,
    PPLReduceDimDes des,
    ReduceParam param)
{
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::srctype src_type;
    typedef typename Operator::dsttype dst_type;

    typedef typename Operator::OpMath math;
    __shared__ T shared_data[32][33];

    int64_t n_outer      = des.n_outer;
    int64_t n_reduce     = des.n_reduce;
    int64_t n_inner      = des.n_inner;
    int64_t num_elements = des.num_elements;

    int64_t outer_stride = n_reduce * n_inner;
    // int64_t non_reduce = n_outer * n_inner;
    // int64_t block_size =  blockDim.x * blockDim.y;
    int64_t grid_stride  = gridDim.x;

    int64_t multi_block_offset = blockIdx.y * num_elements * n_inner;
    int64_t tid                = threadIdx.y * blockDim.x + threadIdx.x;
    int64_t lane               = tid & 31;
    int64_t warp               = tid >> 5;
    int64_t n_warps            = (blockDim.x * blockDim.y + 31) >> 5;
    int64_t num_block_inner    = (n_inner + 31) / 32;
    // int64_t inner_stride = (n_inner + num_block_inner - 1) / num_block_inner;
    for (int64_t bx = blockIdx.x; bx < n_outer * num_block_inner; bx += grid_stride) {
        for (int w = warp; w < 32; w += n_warps) {
            shared_data[w][lane] = (T)op.InitVal();
        }
        __syncthreads();

        int64_t out_idx      = bx / num_block_inner;
        int64_t inner_idx    = bx % num_block_inner;
        int64_t outer_offset = out_idx * outer_stride;

        int64_t inner = inner_idx * 32 + lane;
        T val         = (T)op.InitVal();

        if (inner < n_inner) {
            for (int64_t i = warp; i < num_elements && blockIdx.y * num_elements + i < n_reduce; i += n_warps) {
                int64_t in_offset = i * n_inner + inner + outer_offset + multi_block_offset;
                val               = op.compute(val, op.fetch(in_offset));
            }
        }
        shared_data[warp][lane] = val;
        __syncthreads();

        for (int i = warp; i < 32; i += n_warps) {
            val = shared_data[lane][i];
            warp_reduce_unroll<T, Operator, ReduceSize>(op, nullptr, val);
            if (MultiBlock) {
                if (lane == 0 && i + inner < n_inner) {
                    if (param == ReduceMean) {
                        val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
                    }
                    PPLAtomicWrite<dst_type, Operator>(op.dst + outer_offset + i + inner, static_cast<dst_type>(val), op);
                }
            } else {
                if (lane == 0 && i + inner < n_inner) {
                    if (param == ReduceMean) {
                        val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
                    }
                    op.out(outer_offset + i + inner, static_cast<dst_type>(val));
                }
            }
        }
    }
}

template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__device__ void ppl_reduce_cols(
    Operator op,
    PPLReduceDimDes des,
    ReduceParam param)
{
    if (des.n_reduce < 1024 && des.split_k_num == 1) {
        ppl_reduce_col_sr<T, Operator, MultiBlock, ReduceSize>(op, des, param);
    } else if (des.n_inner < 32) {
        ppl_reudce_col_mrsi<T, Operator, MultiBlock, ReduceSize>(op, des, param);
    } else {
        ppl_reduce_col_mrli<T, Operator, MultiBlock, ReduceSize>(op, des, param);
    }
}

#endif