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

#ifndef PPLCUDA_REDUCE_REDUCE_ROW_KERNEL_H_
#define PPLCUDA_REDUCE_REDUCE_ROW_KERNEL_H_
#include "cudakernel/reduce/reduce_helper.h"
#include "cudakernel/reduce/block_warp_reduce.h"
#include "cudakernel/common/atomic.h"
#include "cudakernel/math/operators.h"
#include <stdio.h>
/*
 *  [BX, 1]
 *  [Gx, 1]
 *  length reduce length
 *  num_elements
 */
template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__inline__ __device__ void ppl_reduce_all(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    typedef typename Operator::srctype src_type;
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::dsttype dst_type;

    __shared__ acc_type shared_data[ReduceSize];
    T val = (T)op.InitVal();

    int64_t grid_stride = blockIdx.x * des.num_elements;
    for (int64_t tid = threadIdx.x; tid < des.num_elements; tid += blockDim.x) {
        acc_type tmp = (tid + grid_stride) < des.n_reduce ? (acc_type)op.fetch(tid + grid_stride) : (acc_type)op.InitVal();
        val          = op.compute(val, tmp);
    }
    shared_data[threadIdx.x] = val;

    __syncthreads();
    block_reduce_row<T, Operator, ReduceSize>(op, shared_data, val);
    __syncthreads();
    if (MultiBlock) {
        if (threadIdx.x == 0) {
            if (param == ReduceMean) {
                val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
            }
            PPLAtomicWrite<dst_type, Operator>(op.dst, static_cast<dst_type>(val), op);
        }
    }

    else {
        if (threadIdx.x == 0) {
            if (param == ReduceMean) {
                val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
            }
            op.out(0, static_cast<dst_type>(val));
        }
    }
}

template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__inline __device__ void ppl_reduce_row_li(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    typedef typename Operator::srctype src_type;
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::dsttype dst_type;

    __shared__ acc_type shared_data[ReduceSize];
    int64_t tid                = threadIdx.x + threadIdx.y * blockDim.x;
    T val                      = (T)op.InitVal();
    int64_t offset             = blockIdx.x * des.n_reduce;
    int64_t blocksize          = blockDim.x * blockDim.y;
    int64_t multi_block_offset = blockIdx.y * des.num_elements;
    for (int64_t i = tid; i < des.num_elements; i += blocksize) {
        acc_type tmp = multi_block_offset + i < des.n_reduce ? (acc_type)op.fetch(i + offset + multi_block_offset) : (acc_type)op.InitVal();
        val          = op.compute(val, tmp);
    }
    shared_data[tid] = val;
    __syncthreads();
    block_reduce_row<T, Operator, ReduceSize>(op, shared_data, val);
    __syncthreads();
    if (MultiBlock) {
        if (param == ReduceMean) {
            shared_data[0] = Math<acc_type, dst_type, acc_type>::div(shared_data[0], (long long)des.n_reduce);
        }
        PPLAtomicWrite<dst_type, Operator>(op.dst + blockIdx.x, static_cast<dst_type>(shared_data[0]), op);
    } else {
        if (param == ReduceMean) {
            shared_data[0] = Math<acc_type, dst_type, acc_type>::div(shared_data[0], (long long)des.n_reduce);
        }
        op.out(blockIdx.x, shared_data[0]);
    }
}

/*

*/

template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__inline__ __device__ void ppl_reduce_row_si(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    typedef typename Operator::srctype src_type;
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::dsttype dst_type;

    const uint64_t block_size  = blockDim.x * blockDim.y;
    const uint64_t grid_stride = gridDim.x * block_size;
    const int base             = threadIdx.x + threadIdx.y * blockDim.x + blockIdx.x * block_size;
    for (uint64_t outer = base; outer < des.n_outer; outer += grid_stride) {
        T val           = (T)op.InitVal();
        uint64_t offset = outer * des.n_reduce;
        for (uint64_t i = 0; i < des.n_reduce; i++) {
            T tmp = (T)op.fetch(offset + i);
            val   = op.compute(val, tmp);
        }
        if (param == ReduceMean) {
            val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
        }
        op.out(outer, val);
    }
}
//[32,32]
template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__inline__ __device__ void ppl_reduce_row_mi(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    typedef typename Operator::srctype src_type;
    typedef typename Operator::acctype acc_type;
    typedef typename Operator::dsttype dst_type;

    const uint64_t grid_stride = gridDim.x * blockDim.y;
    const int base             = threadIdx.y + blockIdx.x * blockDim.y;
    for (uint64_t outer = base; outer < des.n_outer; outer += grid_stride) {
        T val           = (T)op.InitVal();
        uint64_t offset = outer * des.n_reduce;
        for (uint64_t i = threadIdx.x; i < des.n_reduce; i += blockDim.x) {
            val = op.compute(val, (T)op.fetch(offset + i));
        }
        warp_reduce_unroll<T, Operator, ReduceSize>(op, nullptr, val);
        if (threadIdx.x == 0) {
            if (param == ReduceMean) {
                val = Math<acc_type, dst_type, acc_type>::div(val, (long long)des.n_reduce);
            }
            op.out(outer, val);
        }
    }
}

template <typename T, class Operator, int ReduceSize, bool MultiBlock>
__device__ __inline__ void ppl_reduce_rows(Operator op, PPLReduceDimDes des, ReduceParam param)
{
    if (des.n_reduce < 32 && des.split_k_num == 1) {
        ppl_reduce_row_si<T, Operator, ReduceSize, MultiBlock>(op, des, param);
    } else if (des.n_reduce < 1024 && des.split_k_num == 1) {
        ppl_reduce_row_mi<T, Operator, ReduceSize, MultiBlock>(op, des, param);
    } else {
        ppl_reduce_row_li<T, Operator, ReduceSize, MultiBlock>(op, des, param);
    }
}

#endif