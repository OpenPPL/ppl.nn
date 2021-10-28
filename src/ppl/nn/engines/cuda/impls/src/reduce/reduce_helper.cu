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

#include "cudakernel/reduce/reduce_helper.h"
#include "cudakernel/reduce/reduce.h"

ReduceMode GetReduceMode(PPLReduceDimDes des)
{
    // if (des.n_reduce == 1) return ReduceNon;
    if (des.n_inner * des.n_outer == 1)
        return ReduceAll;
    if (des.n_inner == 1)
        return ReduceRow;
    return ReduceCol;
}

void GetSplitNum(
    int64_t bx,
    int64_t by,
    int64_t block_reduce,
    int64_t& split_num,
    bool& multi_block)
{
    while (bx * by < LEASTBLOCKNUM && block_reduce > BLOCKSIZE) {
        by <<= 1;
        block_reduce = DivUp(2, block_reduce);
    }
    split_num   = by;
    multi_block = !(split_num == 1);
}

std::pair<dim3, dim3> ComputeReduceRowConfigure(
    ReduceParam param,
    ReduceMode mode,
    int64_t& num_elements,
    bool& multi_block,
    PPLReduceDimDes& des)
{
    dim3 block_dim(32, BLOCKSIZE / 32), grid_dim(DivUp(BLOCKSIZE, des.n_outer), 1);
    if (des.n_reduce < 32)
        return {block_dim, grid_dim};
    if (des.n_reduce < 1024) {
        grid_dim.x = DivUp(BLOCKSIZE / 32, des.n_outer);
        return {block_dim, grid_dim};
    }
    grid_dim.x = des.n_outer;
    int64_t bx = des.n_outer, by = 1, split_num = 1, block_reduce = des.n_reduce;
    GetSplitNum(bx, by, block_reduce, split_num, multi_block);
    grid_dim.y       = split_num;
    des.num_elements = DivUp(split_num, des.n_reduce);
    return {block_dim, grid_dim};
}

std::pair<dim3, dim3> ComputeReduceAllConfigure(
    ReduceParam param,
    ReduceMode mode,
    int64_t& num_elements,
    bool& multi_block,
    PPLReduceDimDes& des)
{
    dim3 block_dim, grid_dim;
    int64_t bx = 1, by = 1, split_num = 1, block_reduce = des.n_reduce;

    GetSplitNum(bx, by, block_reduce, split_num, multi_block);
    block_dim.x      = BLOCKSIZE;
    grid_dim.x       = split_num;
    des.num_elements = DivUp(split_num, des.n_reduce);
    return {block_dim, grid_dim};
}

std::pair<dim3, dim3> ComputeReduceColConfigure(
    ReduceParam param,
    ReduceMode mode,
    int64_t& num_elements,
    bool& multi_block,
    PPLReduceDimDes& des)
{
    dim3 block_dim(32, BLOCKSIZE / 32), grid_dim(DivUp(BLOCKSIZE, des.n_outer * des.n_inner), 1);
    if (des.n_reduce < 1024)
        return {block_dim, grid_dim};

    int64_t bx = des.n_outer, by = 1, split_num = 1, block_reduce = des.n_reduce;
    if (des.n_inner < 32) {
        // GetSplitNum(bx, by, block_reduce, split_num, multi_block);
        grid_dim.x = bx;
        grid_dim.y = split_num;
        return {block_dim, grid_dim};
    }

    bx = des.n_outer * DivUp(32, des.n_inner);
    GetSplitNum(bx, by, block_reduce, split_num, multi_block);
    grid_dim.x       = bx;
    grid_dim.y       = split_num;
    des.num_elements = DivUp(split_num, des.n_reduce);
    des.split_k_num  = split_num;
    return {block_dim, grid_dim};
}