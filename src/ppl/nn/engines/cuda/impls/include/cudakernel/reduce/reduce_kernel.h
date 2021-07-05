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

#ifndef PPLCUDA_REDUCE_REDUCE_KERNEL_H_
#define PPLCUDA_REDUCE_REDUCE_KERNEL_H_
#include "cudakernel/reduce/reduce_row_kernel.h"
#include "cudakernel/reduce/reduce_col_kernel.h"
#include "cudakernel/common/atomic.h"
#include "cudakernel/math/operators.h"

template <typename T, class Operator, int ReduceSize, bool MultiBlock, int ReduceMode>
__global__ void ppl_reduce(
    Operator op,
    PPLReduceDimDes des,
    ReduceParam param)
{
    if (ReduceMode == 1) {
        ppl_reduce_all<T, Operator, ReduceSize, MultiBlock>(op, des, param);
        return;
    } else if (ReduceMode == 2) {
        ppl_reduce_rows<T, Operator, ReduceSize, MultiBlock>(op, des, param);
        return;
    } else if (ReduceMode == 3) {
        ppl_reduce_cols<T, Operator, ReduceSize, MultiBlock>(op, des, param);
        return;
    }
}

#endif