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