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

#ifndef PPLCUDA_REDUCE_REDUCE_HELPER_H_
#define PPLCUDA_REDUCE_REDUCE_HELPER_H_
#include <utility>
#include <cudakernel/common/macro.h>

struct PPLReduceDimDes {
    int64_t n_inner;
    int64_t n_reduce;
    int64_t n_outer;

    PPLReduceDimDes(int64_t n_inner, int64_t n_reduce, int64_t n_outer)
    {
        this->n_inner  = n_inner;
        this->n_outer  = n_outer;
        this->n_reduce = n_reduce;
        num_elements   = n_reduce;
        split_k_num    = 1;
    }

    int64_t num_elements;
    int64_t split_k_num = 1;
};

enum ReduceMode {
    ReduceNon = 0,
    ReduceAll = 1,
    ReduceRow = 2,
    ReduceCol = 3
};

enum ReduceParam {
    ReduceSum  = 0,
    ReduceMax  = 1,
    ReduceMin  = 2,
    ReduceProd = 3,
    ReduceMean = 4
};

inline constexpr int64_t DivUp(int64_t divisor, int64_t dividend)
{
    return (dividend + divisor - 1) / divisor;
}
const int BLOCKSIZE     = 1024;
const int LEASTBLOCKNUM = 128;

void GetSplitNum(int64_t bx, int64_t by, int64_t block_reduce, int64_t &split_num, bool &multi_block);

std::pair<dim3, dim3> ComputeReduceRowConfigure(ReduceParam param, ReduceMode mode, int64_t &num_elements, bool &multi_block, PPLReduceDimDes &des);

std::pair<dim3, dim3> ComputeReduceAllConfigure(ReduceParam param, ReduceMode mode, int64_t &num_elements, bool &multi_block, PPLReduceDimDes &des);

std::pair<dim3, dim3> ComputeReduceColConfigure(ReduceParam param, ReduceMode mode, int64_t &num_elements, bool &multi_block, PPLReduceDimDes &des);

#endif