#include "cudakernel/nn/cumsum.h"
#include "cudakernel/common/common.cuh"


constexpr inline int ceil_div(int n, int m) {
  return (n + m - 1) / m;
}

template<typename scalar_t>
__global__ void tensor_kernel_scan_outer_dim(scalar_t *tgt_, const scalar_t *src_,
                                              const int num_orows, const int num_irows, const int row_size,
                                              const scalar_t init)
{
  for (int orow = blockIdx.x; orow < num_orows; orow += gridDim.x) {
    for (int irow = blockIdx.y * blockDim.x + threadIdx.x; irow < num_irows; irow += gridDim.y * blockDim.x) {
      const scalar_t *src = src_ + orow * row_size * num_irows + irow;
      scalar_t *tgt = tgt_ + orow * row_size * num_irows + irow;
      scalar_t acc = init;

      for (int col = 0; col < row_size; ++col) {
        acc = acc + *src;
        *tgt = acc;

        src += num_irows;
        tgt += num_irows; 
      }
    }
  }
}

/* Perform an inclusive scan along the innermost dimension of a tensor.
 *
 * - num_rows is the size of the flattened outer dimensions;
 * - row_size is the size of the innermost dimension;
 *
 * The outer dimensions of the tensor are considered as a single dimension, i.e. the tensor is
 * considered as having 'num_rows' rows of size 'row_size'.
 * Each thread block processes one or more sets of contiguous rows (processing multiple rows
 * per thread block is quicker than processing a single row, especially for short rows).
 */
template<typename T, int num_threads_x, int num_threads_y>
__device__ void tensor_kernel_scan_innermost_dim_impl(T* row_buf, T *tgt_, const T *src_,
                                      const int num_rows, const int row_size,
                                      T init){
  for (int block_row = blockIdx.x * blockDim.y;
       block_row < num_rows;
       block_row += blockDim.y * gridDim.x) {
    int row = block_row + threadIdx.y;
    T block_total = init;

    const T *row_src = src_ + row * row_size;
    T *row_tgt = tgt_ + row * row_size;

    // Perform scan on one block at a time, keeping track of the total value of
    // all blocks processed so far.
    for (int block_col = 0; block_col < row_size; block_col += 2 * num_threads_x) {
      // Load data into shared memory (two values per thread).
      int col1 = block_col + threadIdx.x;
      int col2 = block_col + num_threads_x + threadIdx.x;
      if (row < num_rows) {
        if (col1 < row_size) {
          row_buf[threadIdx.x] = row_src[col1];
        } else {
          row_buf[threadIdx.x] = init;
        }

        if (col2 < row_size) {
          row_buf[num_threads_x + threadIdx.x] = row_src[col2];
        } else {
          row_buf[num_threads_x + threadIdx.x] = init;
        }

        // Add the total value of all previous blocks to the first value of this block.
        if (threadIdx.x == 0) {
          row_buf[0] = row_buf[0] + block_total;
        }
      }
      __syncthreads();

      // Parallel reduction (up-sweep). 
      for (int s = num_threads_x, d = 1; s >= 1; s >>= 1, d <<= 1) {
        if (row < num_rows && threadIdx.x < s) {
          int offset = (2 * threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = row_buf[offset] + row_buf[offset + d];
        }
        __syncthreads();
      }

      // Down-sweep.
      for (int s = 2, d = num_threads_x / 2; d >= 1; s <<= 1, d >>= 1) {
        if (row < num_rows && threadIdx.x < s - 1) {
          int offset = 2 * (threadIdx.x + 1) * d - 1;
          row_buf[offset + d] = row_buf[offset] + row_buf[offset + d];
        }
        __syncthreads();
      }

      // Write back to output.
      if (row < num_rows) {
        if (col1 < row_size) row_tgt[col1] = row_buf[threadIdx.x];
        if (col2 < row_size) row_tgt[col2] = row_buf[num_threads_x + threadIdx.x];
      }
      block_total = row_buf[2 * num_threads_x - 1];
      __syncthreads();
    }
  }
}

template <
    typename T,
    int num_threads_x,
    int num_threads_y>
__global__ void tensor_kernel_scan_innermost_dim(
    T* tgt_,
    const T* src_,
    int num_rows,
    int row_size,
    T init) {
  __shared__ T sbuf[num_threads_y][2 * num_threads_x];
  T* row_buf = sbuf[threadIdx.y];

  tensor_kernel_scan_innermost_dim_impl<T, num_threads_x, num_threads_y>(
      row_buf, tgt_, src_, num_rows, row_size, init);
}

template<typename T>
__device__ T ScanWarp(T val) {
  int32_t lane = threadIdx.x & 31;
  T tmp = __shfl_up_sync(0xffffffff, val, 1);
  if (lane >= 1) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 2);
  if (lane >= 2) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 4);
  if (lane >= 4) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 8);
  if (lane >= 8) {
    val += tmp;
  }
  tmp = __shfl_up_sync(0xffffffff, val, 16);
  if (lane >= 16) {
    val += tmp;
  }
  return val;
}


template<typename T>
__device__ __forceinline__ T ScanBlock(T val) {
  int32_t warp_id = threadIdx.x >> 5;
  int32_t lane = threadIdx.x & 31;
  __shared__ T warp_sum[32];
  // scan each warp
  val = ScanWarp(val);
  __syncthreads();
  // write sum of each warp to warp_sum
  if (lane == 31) {
    warp_sum[warp_id] = val;
  }
  __syncthreads();
  // use a single warp to scan warp_sum
  if (warp_id == 0) {
    warp_sum[lane] = ScanWarp(warp_sum[lane]);
  }
  __syncthreads();
  // add base
  if (warp_id > 0) {
    val += warp_sum[warp_id - 1];
  }
  __syncthreads();
  return val;
}








template<typename T>
__global__ void ReducePartSumKernelSinglePass(const T* input,
                                              T* g_part_sum, int n,
                                              int part_size) {
  // this block process input[part_begin:part_end]
  size_t part_begin = blockIdx.x * part_size;
  size_t part_end = (blockIdx.x + 1) * part_size < n ? (blockIdx.x + 1) * part_size : n;
  // part_sum
  T part_sum = 0;
  for (size_t i = part_begin + threadIdx.x; i < part_end; i += blockDim.x) {
    part_sum += input[i];
  }
  part_sum = BlockReduceSum(part_sum);
  if (threadIdx.x == 0) {
    g_part_sum[blockIdx.x] = part_sum;
  }
}

template<typename T>
__global__ void ScanWithBaseSumSinglePass(const T* input,
                                          T* g_base_sum, T* output,
                                          int n, int part_size) {
  // base sum
  __shared__ T base_sum;
  if (threadIdx.x == 0) {
    if (blockIdx.x == 0) {
      base_sum = 0;
    } else {
      base_sum = g_base_sum[blockIdx.x - 1];
    }
  }
  __syncthreads();
  // this block process input[part_begin:part_end]
  size_t part_begin = blockIdx.x * part_size;
  size_t part_end = (blockIdx.x + 1) * part_size;
  for (size_t i = part_begin + threadIdx.x; i < part_end; i += blockDim.x) {
    T val = i < n ? input[i] : 0;
    val = ScanBlock<T>(val);
    if (i < n) {
      output[i] = val + base_sum;
    }
    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) {
      base_sum += val;
    }
    __syncthreads();
  }
}


ppl::common::RetCode PPLCUDACumsumForwardImp(
    cudaStream_t stream,
    int axis,
    const ppl::nn::TensorShape* input_shape,
    const void* input,
    void* output)
{
  int num_elems = input_shape->CalcElementsIncludingPadding();
  int num_dims = input_shape->GetDimCount();
  int row_size = input_shape->GetDim(axis);

  #define CASE(TYPE)           \
  if(row_size == num_elems) {       \
    size_t part_num = 1024;           \
    size_t part_size = (num_elems + part_num - 1) / part_num;        \
    TYPE* part_sum = nullptr;                                   \
    cudaMalloc((void**)part_sum, part_num*sizeof(TYPE));            \
    ReducePartSumKernelSinglePass<TYPE><<<part_num, 1024, 0, stream>>>((const TYPE*)input, (TYPE*)part_sum, num_elems, part_size);   \
    ScanWithBaseSumSinglePass<TYPE><<<1, 1024, 0, stream>>>(         \
        (const TYPE*)part_sum, (TYPE*)part_sum, (TYPE*)part_sum, part_num, part_num);          \
    ScanWithBaseSumSinglePass<TYPE><<<part_num, 1024, 0, stream>>>(               \
        (const TYPE*)input, (TYPE*)part_sum, (TYPE*)output, num_elems, part_size);      \
  } else if(axis == num_dims - 1) {                 \
      int num_rows = num_elems / row_size;              \
      dim3 threads(16, 32);                      \
      dim3 grid(ceil_div(num_rows, threads.y));            \
      tensor_kernel_scan_innermost_dim<TYPE, 16, 32><<<grid, threads, 0, stream>>>(          \
      (TYPE*)output, (const TYPE*)input, num_rows, row_size, 0);            \
  } else {                                                                    \
      int num_orows = 1;                                        \
      int num_irows = 1;                                     \
      for(int i = 0; i < axis; i++)                        \
          num_orows *= input_shape->GetDim(i);              \
      for(int i = axis + 1; i < num_dims; i++)                    \
          num_irows *= input_shape->GetDim(i);                  \
      dim3 threads(num_irows < 512 ? num_irows : 512);                            \
      dim3 grid(num_orows, ceil_div(num_irows, threads.x));                    \
      tensor_kernel_scan_outer_dim<TYPE><<<grid, threads, 0, stream>>>(               \
          (TYPE*)output, (const TYPE*)input,                                \
          num_orows, num_irows, row_size, 0);                            \
  }                                                                     \
  return ppl::common::RC_SUCCESS;         
  if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT32) {
    CASE(float)
  } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT16) {
    CASE(int16_t)
  } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT32) {
    CASE(int32_t)
  } else if (input_shape->GetDataType() == ppl::common::DATATYPE_INT64) {
    CASE(int64_t)
  } else {
    return ppl::common::RC_UNSUPPORTED;
  }
}
