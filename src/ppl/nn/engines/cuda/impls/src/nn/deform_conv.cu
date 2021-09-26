#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "cudakernel/common/common.h"
#include "cudakernel/gemm/gemm.h"
#include "cudakernel/nn/deform_conv.h"

const int kMaxParallelImgs = 32;
int get_greatest_divisor_below_bound(int n, int bound) {
  for (int k = bound; k > 1; --k) {
    if (n % k == 0) {
      return k;
    }
  }
  return 1;
}

template <typename scalar_t, typename index_t>
__device__ float bilinear_interpolate(
    const scalar_t* in,
    index_t height,
    index_t width,
    float h,
    float w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  index_t h_low = floor(h);
  index_t w_low = floor(w);
  index_t h_high = h_low + 1;
  index_t w_high = w_low + 1;

  scalar_t lh = h - h_low;
  scalar_t lw = w - w_low;
  scalar_t hh = 1 - lh, hw = 1 - lw;

  scalar_t v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = in[h_low * width + w_low];
  scalar_t v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = in[h_low * width + w_high];
  scalar_t v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = in[h_high * width + w_low];
  scalar_t v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = in[h_high * width + w_high];

  scalar_t w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}
template <>
__device__ float bilinear_interpolate<__half, int>(
    const __half* in,
    int height,
    int width,
    float h,
    float w) {
  if (h <= -1 || height <= h || w <= -1 || width <= w) {
    return 0;
  }

  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  float lh = h - h_low;
  float lw = w - w_low;
  float hh = 1 - lh, hw = 1 - lw;

  float v1 = 0;
  if (h_low >= 0 && w_low >= 0)
    v1 = __half2float(in[h_low * width + w_low]);
  float v2 = 0;
  if (h_low >= 0 && w_high <= width - 1)
    v2 = __half2float(in[h_low * width + w_high]);
  float v3 = 0;
  if (h_high <= height - 1 && w_low >= 0)
    v3 = __half2float(in[h_high * width + w_low]);
  float v4 = 0;
  if (h_high <= height - 1 && w_high <= width - 1)
    v4 = __half2float(in[h_high * width + w_high]);

  float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

  float val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);
  return val;
}

template <typename scalar_t, typename index_t>
__global__ void deformable_im2col_kernel(
    index_t n,
    const scalar_t* input_ptr,
    const scalar_t* offset_ptr,
    const scalar_t* mask_ptr,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int batch_sz,
    int in_channels,
    int n_offset_grps,
    int out_h,
    int out_w,
    bool use_mask,
    scalar_t* columns_ptr) {
  for(index_t index = blockIdx.x*blockDim.x + threadIdx.x; index < n; index += gridDim.x*blockDim.x){
    const index_t out_x = index % out_w;
    const index_t out_y = (index / out_w) % out_h;
    const index_t out_b = (index / (out_w * out_h)) % batch_sz;
    const index_t in_c = index / (out_w * out_h * batch_sz);
    const index_t out_c = in_c * weight_h * weight_w;

    index_t c_per_offset_grp = in_channels / n_offset_grps;
    const index_t grp_idx = in_c / c_per_offset_grp;

    columns_ptr +=
        (out_c * (batch_sz * out_h * out_w) + out_b * (out_h * out_w) +
         out_y * out_w + out_x);

    input_ptr +=
        (out_b * (in_channels * height * width) + in_c * (height * width));

    offset_ptr += (out_b * n_offset_grps + grp_idx) * 2 * weight_h * weight_w *
        out_h * out_w;

    if (use_mask) {
      mask_ptr += (out_b * n_offset_grps + grp_idx) * weight_h * weight_w *
          out_h * out_w;
    }

    for (int i = 0; i < weight_h; ++i) {
      for (int j = 0; j < weight_w; ++j) {
        const index_t mask_idx = i * weight_w + j;
        const index_t offset_idx = 2 * mask_idx;

        scalar_t mask_value = 1;
        if (use_mask) {
          mask_value =
              mask_ptr[mask_idx * (out_h * out_w) + out_y * out_w + out_x];
        }

        const scalar_t offset_h =
            offset_ptr[offset_idx * (out_h * out_w) + out_y * out_w + out_x];
        const scalar_t offset_w = offset_ptr
            [(offset_idx + 1) * (out_h * out_w) + out_y * out_w + out_x];
        const float y =
            (out_y * stride_h - pad_h) + i * dilation_h + (float)offset_h;
        const float x =
            (out_x * stride_w - pad_w) + j * dilation_w + (float)offset_w;
        *columns_ptr =
            (__half)((float)mask_value * bilinear_interpolate(input_ptr, height, width, y, x));
        columns_ptr += batch_sz * out_h * out_w;
      }
    }
  }

}


void deformable_im2col(
    const void *input,
    const void *data_offset,
    const void *data_mask,
    int in_channels,
    int height,
    int width,
    int weight_h,
    int weight_w,
    int pad_h,
    int pad_w,
    int stride_h,
    int stride_w,
    int dilation_h,
    int dilation_w,
    int out_h,
    int out_w,
    int parallel_imgs,
    int deformable_group,
    bool use_mask,
    void *data_col) {
  int64_t num_kernels = (int64_t)in_channels * out_h * out_w * parallel_imgs;

  const unsigned int threads = 512;
  const unsigned int blocks = DivUp(num_kernels, threads);

  bool use_64bits_indexing = false;
  use_64bits_indexing |= num_kernels > (1 << 31);
  use_64bits_indexing |=
      ((int64_t)in_channels * weight_h * weight_w * parallel_imgs * out_h * out_w >
       (1 << 31));

  if (use_64bits_indexing) {
          deformable_im2col_kernel<__half, int64_t><<<blocks, threads>>>(
              num_kernels,
              (const __half*)input,
              (const __half*)data_offset,
              (const __half*)data_mask,
              height,
              width,
              weight_h,
              weight_w,
              pad_h,
              pad_w,
              stride_h,
              stride_w,
              dilation_h,
              dilation_w,
              parallel_imgs,
              in_channels,
              deformable_group,
              out_h,
              out_w,
              use_mask,
              (__half *)data_col);

  } else {
          deformable_im2col_kernel<__half, int><<<blocks, threads>>>(
              num_kernels,
              (const __half*)input,
              (const __half*)data_offset,
              (const __half*)data_mask,
              height,
              width,
              weight_h,
              weight_w,
              pad_h,
              pad_w,
              stride_h,
              stride_w,
              dilation_h,
              dilation_w,
              parallel_imgs,
              in_channels,
              deformable_group,
              out_h,
              out_w,
              use_mask,
              (__half*)data_col);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("error in deformable_im2col: %s\n", cudaGetErrorString(err));
  }
}

template <typename T>
__global__ void transpose_with_pad(
    const T *input, 
    T *output, 
    const int batch, 
    const int input_h, 
    const int input_w,
    const int output_h, 
    const int output_w)
{
    __shared__ T smem[32][33];
    int i_h         = blockIdx.y * 32 + threadIdx.y;
    int i_w         = blockIdx.x * 32 + threadIdx.x;
    int o_w         = blockIdx.y * 32 + threadIdx.x;
    int o_h         = blockIdx.x * 32 + threadIdx.y;
    bool inBound0   = i_h < input_h && i_w < input_w;
    int64_t index   = (blockIdx.z * input_h + i_h) * input_w + i_w;
    bool inBound1   = o_h < output_h && o_w < output_w;
    int64_t o_index = (blockIdx.z * output_h + o_h) * output_w + o_w;

    T value                        = inBound0 ? input[index] : (T)0;
    smem[threadIdx.x][threadIdx.y] = value;
    __syncthreads();
    value = smem[threadIdx.y][threadIdx.x];

    if (inBound1) {
        output[o_index] = value;
    }
}


__global__ void pad_kernel(const __half *input, __half *output, int h, int w, int pad_w){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int out_w = tid % pad_w;
    int out_h = tid / pad_w;
    bool in_range = tid < h*pad_w;
    __half value = (in_range && out_w < w) ? input[out_h*w + out_w] : (__half)0;
    if(in_range)    output[out_h*pad_w + out_w] = value;
}

ppl::common::RetCode PPLCUDADeformConvModifyWeights(
    const cudaStream_t &stream,
    const ppl::nn::TensorShape *flt_shape,
    const void *in_flt,
    void *out_flt){
    
    int pad_size = 16 / sizeof(__half);
    int n = flt_shape->GetDim(0);
    int c = flt_shape->GetDim(1);
    int r = flt_shape->GetDim(2);
    int s = flt_shape->GetDim(3);
    int size = c*r*s;
    if (size % pad_size == 0){
        cudaMemcpy(out_flt, in_flt, n*size*sizeof(__half), cudaMemcpyDeviceToDevice);
        return ppl::common::RC_SUCCESS;
    }
    int aligned_size = Align(size, pad_size);
    int threads = 256;
    int blocks = DivUp(size, threads);
    pad_kernel<<<blocks, threads>>>((const __half*)in_flt, (__half*)out_flt, n, size, aligned_size);
    return ppl::common::RC_SUCCESS;
}

int64_t PPLCUDADeformConvGetBufSize(
    const ppl::nn::TensorShape *input_shape,
    const ppl::nn::TensorShape *flt_shape,
    const ppl::nn::TensorShape *output_shape){

    const int batch = input_shape->GetDim(0);
    const int in_c = input_shape->GetDim(1);
    const int in_h = input_shape->GetDim(2);
    const int in_w = input_shape->GetDim(3);
    const int out_c = output_shape->GetDim(1);
    const int out_h = output_shape->GetDim(2);
    const int out_w = output_shape->GetDim(3);
    const int kernel_h = flt_shape->GetDim(2);
    const int kernel_w = flt_shape->GetDim(3);
    int n_parallel_imgs = get_greatest_divisor_below_bound(batch, kMaxParallelImgs);
    const int pad_size = 16 / sizeof(__half);
    int size = 0;
    int K = in_c*kernel_h*kernel_w;
    int N = n_parallel_imgs*out_h*out_w;
    int M = out_c;
    int K_pad = Align(K, pad_size);
    int N_pad = Align(N, pad_size);
    size += K * N;// columns
    size += N_pad * K_pad;// transpose&pad columns, FIXME should be N x K_pad
    size += M * N_pad;// pre-transpose output
    return (int64_t)size * sizeof(__half);
}
//need flt  nchw layout, chw pad 8
//src: (batch/n_parallel_imgs, n_parallel_imgs, in_channels, in_h, in_w)
//offset: (batch/n_parallel_imgs, n_parallel_imgs, offset_group*2*kernel_h*kernel_w, out_h, out_w)
//mask: (batch/n_parallel_imgs, n_parallel_imgs, offset_group*kernel_h*kernel_w, out_h, out_w)
//columns: (in_channels*kernel_h*kernel_w, n_parallel_imgs*out_h*out_w)
//but we need shape of NxK columns
//output: (batch/n_parallel_imgs, out_c, n_parralel_imgs, out_h, out_w)

ppl::common::RetCode PPLCUDADeformConvForward(
    const cudaStream_t &stream,
    const ppl::nn::TensorShape *output_shape,
    const ppl::nn::TensorShape *input_shape,
    void *output,
    const void *input,
    const void *flt,
    const void *offset,
    const void *mask,
    const void *bias,
    const int group,
    const int offset_group,
    const int channels,
    const int num_output,
    const int stride_h,
    const int stride_w,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    bool use_mask,
    void *temp_buffer){
    const int pad_size = 16 / sizeof(__half);
    if(channels % group != 0 || num_output % group != 0)
        return ppl::common::RC_INVALID_VALUE;

    
    const int batch = input_shape->GetDim(0);
    const int in_c = input_shape->GetDim(1);
    const int in_h = input_shape->GetDim(2);
    const int in_w = input_shape->GetDim(3);

    int n_parallel_imgs = get_greatest_divisor_below_bound(batch, kMaxParallelImgs);

    const int out_c = output_shape->GetDim(1);
    const int out_h = output_shape->GetDim(2);
    const int out_w = output_shape->GetDim(3);
    
    const int ic_per_grp = channels / group;
    const int oc_per_grp = num_output / group;

    // NN -> NT
    int col_trans_in_h = ic_per_grp*kernel_h*kernel_w;
    int col_trans_in_w = n_parallel_imgs*out_h*out_w;
    int col_trans_out_h = Align(col_trans_in_w, pad_size);
    //int col_trans_out_h = col_trans_in_w;// FIXME
    int col_trans_out_w = Align(col_trans_in_h, pad_size);

    int M = oc_per_grp;
    int N = col_trans_out_h;
    int K = col_trans_out_w;
#define FAKE_GEMM_PARAM  \
            ppl::nn::TensorShape shape_a, shape_b, shape_c; \
            shape_a.SetDimCount(2); \
            shape_b.SetDimCount(2); \
            shape_c.SetDimCount(2); \
            shape_a.SetDim(0, M); shape_a.SetDim(1, K); \
            shape_b.SetDim(0, N); shape_b.SetDim(1, K); \
            shape_c.SetDim(0, M); shape_c.SetDim(1, N); \
            shape_a.SetDataType(ppl::common::DATATYPE_FLOAT16); \
            shape_b.SetDataType(ppl::common::DATATYPE_FLOAT16); \
            shape_c.SetDataType(ppl::common::DATATYPE_FLOAT16); \
            ppl::nn::common::GemmParam gemm_param; \
            fuse_param_t fuse_param; \
            gemm_param.bias_term = 0; \
            gemm_param.transA = 0;   gemm_param.transB = 1; \
            gemm_param.alpha  = 1.f; gemm_param.beta   = 1.f; \
            gemm_param.N      = N; \
            int kid = 0;
    FAKE_GEMM_PARAM
#undef FAKE_GEMM_PARAM

    __half *columns = reinterpret_cast<__half*>(temp_buffer);//in_c*r*s,  n_parallel_imgs*out_h*out_w
    __half *trans_columns = columns + group*col_trans_in_h * col_trans_in_w;
    __half *output_buf = trans_columns + group * N * K;
    for (int64_t b = 0; b < batch / n_parallel_imgs; ++b) {
        deformable_im2col(
            (__half*)input + b * n_parallel_imgs * in_c * in_h * in_w,
            (__half*)offset + b * n_parallel_imgs * offset_group * 2 * kernel_h * kernel_w * out_h * out_w,
            (__half*)mask + b * n_parallel_imgs * offset_group * kernel_h * kernel_w * out_h * out_w,
            in_c,
            in_h, in_w,
            kernel_h, kernel_w,
            pad_h, pad_w,
            stride_h, stride_w,
            dilation_h, dilation_w,
            out_h, out_w,
            n_parallel_imgs, offset_group,
            mask != nullptr,
            columns);
        {

        dim3 threads(32,32,1);
        dim3 grid;
        grid.x = DivUp(col_trans_in_w, 32);
        grid.y = DivUp(col_trans_in_h, 32);
        grid.z = group;
        transpose_with_pad<__half><<<grid, threads, 0, stream>>>(columns, trans_columns, group,
                                                      col_trans_in_h, col_trans_in_w,
                                                      col_trans_out_h, col_trans_out_w);
        }

        for(int g = 0; g < group; g++){
            //FIXME flt nchw pad 8
            __half *tmp_a = (__half*)flt + g * oc_per_grp * col_trans_out_w;
            __half *tmp_b = (__half*)trans_columns + g * col_trans_out_h * col_trans_out_w;
            __half *tmp_output = (__half*)output_buf + b * out_c * col_trans_out_h + g * oc_per_grp * col_trans_out_h;

            __half *tmp_bias = bias? (__half*)bias + g*oc_per_grp : NULL;
            PPLCUDAGemmForwardImp(
                stream, &shape_a, tmp_a, &shape_b, tmp_b,
                tmp_bias, &shape_c, tmp_output,
                gemm_param, temp_buffer, fuse_param, kid);

            
        }
    }

    // output nhwc pad 8 layout FIXME 
    //if(n_parallel_imgs > 1){
    const int o_trans_in_h = out_c;
    const int o_trans_in_w = N;//n_parallel_imgs*out_h*out_w;
    const int o_trans_out_w_pad = Align(o_trans_in_h, pad_size);
    const int o_trans_out_h = n_parallel_imgs*out_h*out_w;//o_trans_out_w;
    dim3 out_threads(32,32,1);
    dim3 out_grid;
    out_grid.x = DivUp(o_trans_in_w, 32);
    out_grid.y = DivUp(o_trans_in_h, 32);
    out_grid.z = batch / n_parallel_imgs;
    transpose_with_pad<__half><<<out_grid, out_threads, 0, stream>>>(output_buf, (__half*)output, batch/n_parallel_imgs, 
                                                          o_trans_in_h, o_trans_in_w, 
                                                          o_trans_out_h, o_trans_out_w_pad);
    //}
    return ppl::common::RC_SUCCESS;
    
}
