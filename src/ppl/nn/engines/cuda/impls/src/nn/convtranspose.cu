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

#include "cudakernel/nn/convtranspose.h"
#include "cudakernel/math/math.h"
#include "cudakernel/memory/transpose.h"
#include "cudakernel/common/common.h"
#include "ppl/nn/params/onnx/transpose_param.h"
#include "cudakernel/nn/conv/conv_fp16.h"
#include "conv_common.h"
#include "cudakernel/gemm/gemm.h"
#include <float.h>
#include <cuda_fp16.h>

#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)


uint64_t PPLConvTransposeGetFilterBufSizeCudaFp16(
    const ppl::nn::TensorShape* weight_shape)
{
    int M = weight_shape->GetDim(1);
    M *= weight_shape->GetDim(2);
    M *= weight_shape->GetDim(3);
    size_t K = weight_shape->GetDim(0);

    size_t padM = Align(M, 1);
    size_t padK = Align(K, 8);

    size_t dst = Align(padM * padK * sizeof(__half), 16);
    return dst * 2; // transpose buf
}

uint64_t PPLConvTransposeGetCompilationBufSizeCuda(
    ppl::nn::TensorShape* input_shape,
    ppl::nn::TensorShape* output_shape,
    const ppl::nn::onnx::ConvTransposeParam* param)
{
    auto type     = input_shape->GetDataType();
    int pad_size = GetPadSize(type); // ldg 128 bytes
    int batch     = input_shape->GetDim(0);
    int in_c      = input_shape->GetDim(1);
    int in_h      = input_shape->GetDim(2);
    int in_w      = input_shape->GetDim(3);
    int out_c     = output_shape->GetDim(1);
    int out_h     = output_shape->GetDim(2);
    int out_w     = output_shape->GetDim(3);
    int kernel_h  = param->kernel_shape[0];
    int kernel_w  = param->kernel_shape[1];
    int pad_h     = param->pads[0];
    int pad_w     = param->pads[1];
    int stride_h  = param->strides[0];
    int stride_w  = param->strides[1];
    int hole_h    = param->dilations[0];
    int hole_w    = param->dilations[1];
    uint64_t size = 0;

    if (type != ppl::common::DATATYPE_FLOAT16) {
        return 0;
    }
    if (stride_h == 1 && stride_w == 1){
        //fake conv
        conv_param_t conv_param;
        conv_param.in_num = batch;
        conv_param.num_chl = in_c;
        conv_param.num_flt = out_c;
        conv_param.num_chl_pad = Align(in_c, 8);
        conv_param.num_flt_pad = Align(out_c, 8);
        conv_param.num_grp = 1;
        conv_param.in_height = in_h;
        conv_param.in_width = in_w;
        conv_param.flt_height = kernel_h;
        conv_param.flt_width = kernel_w;
        conv_param.pad_height = kernel_h - 1- pad_h;
        conv_param.pad_width = kernel_w - 1- pad_w;
        conv_param.stride_height = 1;
        conv_param.stride_width = 1;
        conv_param.hole_height = 1;
        conv_param.hole_width = 1;
        conv_param.out_height = in_h + 2*(kernel_h-1-pad_h) - (kernel_h-1);
        conv_param.out_width = in_w + 2*(kernel_w-1-pad_w) - (kernel_w-1);

        size += PPLCUDAConvolutionGetCompilationBufSize(ppl::common::DATATYPE_FLOAT16, conv_param);
    }
    else {
        int in_c_pad  = Align(in_c, pad_size);
        int out_c_pad = Align(out_c, pad_size);
        int cvt_in_h = in_h + DivUp(kernel_h-stride_h, stride_h);
        int cvt_in_w = in_w + DivUp(kernel_w-stride_w, stride_w);
        int kernel_u = DivUp(kernel_h, stride_h);
        int kernel_v = DivUp(kernel_w, stride_w);
        int pattern_num = stride_h*stride_w;
        size += batch*cvt_in_h*cvt_in_w*kernel_u*kernel_v*in_c_pad*sizeof(__half);
        size += batch*cvt_in_h*cvt_in_w*stride_h*stride_w*out_c_pad*sizeof(__half);
        size += pattern_num*out_c_pad*kernel_u*kernel_v*in_c_pad*sizeof(__half);
    }
    return size;
}


uint64_t PPLConvTransposeGetBufSizeCuda(
    ppl::nn::TensorShape* input_shape,
    ppl::nn::TensorShape* output_shape,
    const ppl::nn::onnx::ConvTransposeParam* param)
{
    auto type     = input_shape->GetDataType();
    int pad_size = GetPadSize(type); // ldg 128 bytes
    int batch     = input_shape->GetDim(0);
    int in_c      = input_shape->GetDim(1);
    int in_h      = input_shape->GetDim(2);
    int in_w      = input_shape->GetDim(3);
    int out_c     = output_shape->GetDim(1);
    int out_h     = output_shape->GetDim(2);
    int out_w     = output_shape->GetDim(3);
    int kernel_h  = param->kernel_shape[0];
    int kernel_w  = param->kernel_shape[1];
    int pad_h     = param->pads[0];
    int pad_w     = param->pads[1];
    int stride_h  = param->strides[0];
    int stride_w  = param->strides[1];
    int hole_h    = param->dilations[0];
    int hole_w    = param->dilations[1];
    uint64_t size = 0;

    if (stride_h == 1 && stride_w == 1)
        return 0;
    if (stride_h != 1 || stride_w != 1){
        int in_c_pad  = Align(in_c, pad_size);
        int out_c_pad = Align(out_c, pad_size);
        int cvt_in_h = in_h + DivUp(kernel_h-stride_h, stride_h);
        int cvt_in_w = in_w + DivUp(kernel_w-stride_w, stride_w);
        int kernel_u = DivUp(kernel_h, stride_h);
        int kernel_v = DivUp(kernel_w, stride_w);
        size += batch*cvt_in_h*cvt_in_w*kernel_u*kernel_v*in_c_pad;
        size += batch*cvt_in_h*cvt_in_w*stride_h*stride_w*out_c_pad;
    }
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        size *= sizeof(__half);
    } else {
        return 0;
    }
    return size;
}


/*
   if stride != 1
   half
   mapped by output shape
   DivUp(out hw, cta_size)
   nhwc and nhwuvc are both paded
*/
__global__ void nhwc2nhwuvc(const void *input, void *cvt_input,
        const int batch, const int in_height, const int in_width, const int in_c_v4,
        const int out_height, const int out_width,
        const int stride_h, const int stride_w){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int c_id  = tid % in_c_v4;
    int nhw_id = tid / in_c_v4;
    int w_id = nhw_id % out_width;
    int h_id = nhw_id / out_width % out_height;
    int n_id = nhw_id / out_width / out_height;

    int in_w_id = w_id / stride_w;
    int in_h_id = h_id / stride_h;

    bool in_range = n_id < batch &&
                    h_id % stride_h == 0 &&
                    w_id % stride_w == 0;
    int in_off = n_id * in_height * in_width * in_c_v4 +
                 in_h_id * in_width * in_c_v4 +
                 in_w_id * in_c_v4 + c_id;
    int4 zeros = {0,0,0,0};
    ((int4*)cvt_input)[tid] = in_range? ((int4*)input)[in_off] : zeros; 
}
/*
   v = (s+stride_w-1)/stride_w
   u = (r+stride_h-1)/stride_h
*/
__global__ void new_nhwc2nhw_ker_c(int4 *cvt_input, const int4 *input,
        const int batch, const int in_height, const int in_width, const int in_c_v4,
        const int out_height, const int out_width,
        const int flt_height, const int flt_width,
        const int stride_height, const int stride_width,
        const int kernel_u, const int kernel_v){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int c_id = tid % in_c_v4;
    int v_id = tid / in_c_v4 % kernel_v;
    int u_id = tid / in_c_v4 / kernel_v % kernel_u;
    int ow_id = tid / in_c_v4 / kernel_v / kernel_u % out_width;
    int oh_id = tid / in_c_v4 / kernel_v / kernel_u / out_width % out_height;
    int n_id = tid / in_c_v4 / kernel_v / kernel_u / out_width / out_height;
    if (n_id >= batch)    return;
    int out_off = tid;

    int iw_id = ow_id + v_id - (out_width - in_width);
    int ih_id = oh_id + u_id - (out_height - in_height);
    int in_off = n_id * in_height*in_width*in_c_v4 +
                 ih_id * in_width*in_c_v4 +
                 iw_id * in_c_v4 +
                 c_id;
    bool in_range = iw_id >= 0 && iw_id < in_width &&
                    ih_id >= 0 && ih_id < in_height;

    int4 zeros{0,0,0,0};
    //FIXME out_off = tid
    cvt_input[out_off] = in_range? input[in_off] : zeros;

}
/*
   v = (s+1)/stride_w
   u = (r+1)/stride_h
   v = (s+stride_w-1)/stride_w
   u = (r+stride_h-1)/stride_h
   pattern number = stride_h*stride_w
   convert to (k*pattern) * (u*v*c)
*/
__global__ void flt_krsc2pkuvc(int4 *cvt_flt, const int4 *flt,
        const int out_channel, const int out_channel_pad, const int in_channel_v4,
        const int flt_height, const int flt_width,
        const int stride_h, const int stride_w,
        const int kernel_u, const int kernel_v,
        const int pattern_num){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int c_id = tid % in_channel_v4;
    int v_id = tid / in_channel_v4 % kernel_v;
    int u_id = tid / in_channel_v4 / kernel_v % kernel_u;
    int k_id = tid / in_channel_v4 / kernel_v / kernel_u % out_channel_pad;
    int p_id = tid / in_channel_v4 / kernel_v / kernel_u / out_channel_pad;
    // reverse uv
    p_id = pattern_num-1 - p_id;
    if (p_id < 0)    return;

    int out_off = p_id * out_channel_pad * kernel_u * kernel_v * in_channel_v4 +
                  k_id * kernel_u * kernel_v * in_channel_v4 +
                  u_id * kernel_v * in_channel_v4 +
                  v_id * in_channel_v4 +
                  c_id;
    int in_base = k_id * flt_height * flt_width * in_channel_v4 + c_id;
    int start_r = p_id / stride_w;
    int start_s = p_id % stride_w;
    int ir_id = start_r + u_id*stride_h;
    int is_id = start_s + v_id*stride_w;
    int4 element = {0,0,0,0};
    if (k_id < out_channel && ir_id < flt_height && is_id < flt_width){
        int rs_id = ir_id*flt_width + is_id;
        int in_off = in_base +
                     rs_id * in_channel_v4;
        element = flt[in_off];
    }

    cvt_flt[out_off] = element;
}

/*
   
   inh + (r-stride_h)
   inw + (s-stride_w)
   uv = stride_h*stride_w = pattern num
*/
template<int AlignInt4>
__global__ void nhwuvc2nhwc(int4 *output, const int4 *gemm_output,
        const int batch, const int out_height, const int out_width,
        const int channel_v4, const int input_height, const int input_width,
        const int flt_h, const int flt_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int4 *bias, const int has_relu){
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int c_id = tid % channel_v4;
    const int &kernel_u = stride_h;
    const int &kernel_v = stride_w;
    int v_id  = tid / channel_v4 % kernel_v;
    int u_id  = tid / channel_v4 / kernel_v % kernel_u;
    int iw_id = tid / channel_v4 / kernel_v / kernel_u % input_width;
    int ih_id = tid / channel_v4 / kernel_v / kernel_u / input_width;// % input_height;
    int n_id  = blockIdx.y;
    int in_off = n_id * input_height * input_width * channel_v4 * stride_h * stride_w +
                 tid;
    if (ih_id >= input_height)  return;
    
    int local_h = -u_id;
    int local_w = -v_id;
    int oh_id = ih_id * stride_h + local_h + pad_h;
    int ow_id = iw_id * stride_w + local_w + pad_w;
    int out_off = n_id * out_height * out_width * channel_v4 +
                  oh_id * out_width * channel_v4 +
                  ow_id * channel_v4 +
                  c_id;
    if(oh_id >= 0 && oh_id < out_height && ow_id < out_width && ow_id >= 0){
        int4 data = gemm_output[in_off];
        __half2 *h2_data = (__half2*)&data;
        if (bias){
            int4 bias_data = bias[c_id];
            __half2 *h2_bias = (__half2*)&bias_data;
#pragma unroll
            for(int i = 0; i < 4; i++){
                h2_data[i] = __hadd2(h2_data[i], h2_bias[i]);
            }
        }
        //fuse
        if (has_relu){
            int *data_v1 = (int*)&data;
#pragma unroll
            for(int i = 0; i < 4; i++){
                data_v1[i] = __vmaxs2(data_v1[i], 0);
            }
        }
        output[out_off] = data;
    }
#endif
}

//input aligned with int2
//output aligned with int4
template<>
__global__ void nhwuvc2nhwc<0>(int4 *output, const int4 *gemm_output,
        const int batch, const int out_height, const int out_width,
        const int channel_v4, const int input_height, const int input_width,
        const int flt_h, const int flt_w,
        const int pad_h, const int pad_w,
        const int stride_h, const int stride_w,
        const int4 *bias, const int has_relu){
#if __CUDA_ARCH__ >= 600 && __CUDACC_VER_MAJOR__ >= 9
    constexpr int channel_v2 = 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int c_id = tid % channel_v2;// ===0
    const int &kernel_u = stride_h;
    const int &kernel_v = stride_w;
    int v_id  = tid / channel_v2 % kernel_v;
    int u_id  = tid / channel_v2 / kernel_v % kernel_u;
    int iw_id = tid / channel_v2 / kernel_v / kernel_u % input_width;
    int ih_id = tid / channel_v2 / kernel_v / kernel_u / input_width;// % input_height;
    int n_id  = blockIdx.y;
    int in_off = n_id * input_height * input_width * channel_v4 * stride_h * stride_w +
                 tid;
    if (ih_id >= input_height)  return;
    
    int local_h = -u_id;
    int local_w = -v_id;
    int oh_id = ih_id * stride_h + local_h + pad_h;
    int ow_id = iw_id * stride_w + local_w + pad_w;
    int out_off = n_id * out_height * out_width * channel_v2 +
                  oh_id * out_width * channel_v2 +
                  ow_id * channel_v2 +
                  c_id;
    if(oh_id >= 0 && oh_id < out_height && ow_id < out_width && ow_id >= 0){
        int2 data = ((int2*)gemm_output)[in_off];
        __half2 *h2_data = (__half2*)&data;
        if (bias){
            int2 bias_data = ((int2*)bias)[c_id];
            __half2 *h2_bias = (__half2*)&bias_data;
#pragma unroll
            for(int i = 0; i < 2; i++){
                h2_data[i] = __hadd2(h2_data[i], h2_bias[i]);
            }
        }
        //fuse
        if (has_relu){
            int *data_v1 = (int*)&data;
#pragma unroll
            for(int i = 0; i < 2; i++){
                data_v1[i] = __vmaxs2(data_v1[i], 0);
            }
        }
        int4 out_data{0,0,0,0};
        out_data.x = data.x;
        out_data.y = data.y;
        output[out_off] = out_data;
    }
#endif
}

/*
   //kcrs2crsk_pad
   origin flt: ckrs
   pre flt: crsk_pad
   and reverse rs order
*/
__global__ void reverse_flt(void *flt, void *rev_flt, const int C, const int R, const int S, const int K_v4_pad){
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int k_id = tid % K_v4_pad;
    int rs_id = (tid / K_v4_pad) % (R*S);
    int c_id = (tid / K_v4_pad) / (R*S);
    bool in_range = c_id < C;

    int rev_rs_id = R*S - rs_id - 1;
    int out_off = c_id * R*S*K_v4_pad +
                  rev_rs_id * K_v4_pad +
                  k_id;
    if(in_range)    ((int4*)rev_flt)[out_off] = ((int4*)flt)[tid];
}


ppl::common::RetCode PPLCUDAConvTransposeCvt(
    int device_id,
    cudaStream_t stream,
    const void* in_filter,
    void* temp_buffer,
    void* out_filter,
    const ppl::nn::TensorShape* filter_shape,
    const ppl::nn::onnx::ConvTransposeParam* param)
{
    int conv_in_c  = filter_shape->GetDim(1);
    int conv_out_c = filter_shape->GetDim(0);
    int kernel_h = param->kernel_shape[0];
    int kernel_w = param->kernel_shape[1];
    int pad_h    = param->pads[0];
    int pad_w    = param->pads[1];
    int stride_h = param->strides[0];
    int stride_w = param->strides[1];
    int hole_h   = param->dilations[0];
    int hole_w   = param->dilations[1];
    int num_filters      = conv_out_c;
    int M                = conv_in_c * kernel_h * kernel_w;
    int K                = num_filters;
    int padM             = Align(M, 1);
    int padK             = Align(K, 8);

    ppl::nn::onnx::TransposeParam trans_param;
    trans_param.perm.push_back(1);
    trans_param.perm.push_back(0);

    ppl::nn::TensorShape a_shape, out_a_shape;
    a_shape.SetDataType(filter_shape->GetDataType());
    out_a_shape.SetDataType(filter_shape->GetDataType());
    a_shape.Reshape({K, M});
    out_a_shape.Reshape({padM, padK});

    __half* trans_flt = (__half*)temp_buffer;
    //k_pad-crs 2 crsk_pad
    ppl::common::RetCode status = PPLCUDATransposeForwardImp(device_id,
                                                             stream,
                                                             trans_param,
                                                             &a_shape,
                                                             in_filter,
                                                             &out_a_shape,
                                                             trans_flt);

    //crsk_pad 2 rev crsk_pad
    int cta_size = 256;
    int rev_grid = DivUp(M*padK, cta_size);
    int4 *rev_flt = (int4*)temp_buffer + padM*padK/(sizeof(int4)/sizeof(__half));
    if (stride_h == 1 && stride_w == 1){
        rev_flt = (int4*)out_filter;
    }
    reverse_flt<<<rev_grid, cta_size, 0, stream>>>(trans_flt, rev_flt, conv_in_c, kernel_h, kernel_w, padK/8);

    if (stride_h == 1 && stride_w == 1)
        return ppl::common::RC_SUCCESS;
        
    //rsc2puvc
    int conv_out_c_v4 = DivUp(conv_out_c, 8);
    int conv_in_c_pad = Align(conv_in_c, 8);
    int kernel_v = DivUp(kernel_w, stride_w);
    int kernel_u = DivUp(kernel_h, stride_h);
    int pattern_num = stride_h*stride_w;
    int cvt_cta_size = 128;
    dim3 cvt_grid;
    if (conv_in_c <= 4 && ((pattern_num&1)==0) ) {
        conv_in_c_pad = 4; 
    }
    cvt_grid.x = DivUp(conv_in_c_pad*pattern_num*kernel_u*kernel_v*conv_out_c_v4, cvt_cta_size);
    cvt_grid.y = 1;//conv_in_c;
    cvt_grid.z = 1;
    flt_krsc2pkuvc<<<cvt_grid, cvt_cta_size, 0, stream>>>(
            (int4 *)out_filter, (const int4 *)rev_flt,
            conv_in_c, conv_in_c_pad, conv_out_c_v4,
            kernel_h, kernel_w, stride_h, stride_w,
            kernel_u, kernel_v, pattern_num);
    return ppl::common::RC_SUCCESS;
}

/*
   flt: reversed crsk_pad
   input: nhwc_pad
*/
ppl::common::RetCode PPLCUDAConvTransposeForward(
    int device_id,
    cudaStream_t stream,
    ppl::nn::cuda::CUDAModule* module,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    const void* rev_flt,
    const void* bias,
    ppl::nn::TensorShape* output_shape,
    void* output,
    const ppl::nn::onnx::ConvTransposeParam* param,
    algo_param_t algo_param,
    fuse_param_t &fuse_param,
    void* temp_buffer)
{
    int batch    = input_shape->GetDim(0);
    int in_c     = input_shape->GetDim(1);
    int in_h     = input_shape->GetDim(2);
    int in_w     = input_shape->GetDim(3);
    int out_c    = output_shape->GetDim(1);
    int out_h    = output_shape->GetDim(2);
    int out_w    = output_shape->GetDim(3);
    int kernel_h = param->kernel_shape[0];
    int kernel_w = param->kernel_shape[1];
    int pad_h    = param->pads[0];
    int pad_w    = param->pads[1];
    int stride_h = param->strides[0];
    int stride_w = param->strides[1];
    int hole_h   = param->dilations[0];
    int hole_w   = param->dilations[1];
    if (hole_h != 1 || hole_w != 1)
        return ppl::common::RC_UNSUPPORTED;

    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        void* cvt_input  = (__half*)temp_buffer;
        int in_c_v4 = DivUp(in_c, 8);
        int in_c_pad = Align(in_c, 8);
        int out_c_pad = Align(out_c, 8);
        int out_c_v4 = DivUp(out_c, 8);
        int cvt_in_h = in_h;
        int cvt_in_w = in_w;
        int cvt_in_size_v4 = 0;
        // nhwc to stride-dilated nhwc
        if(stride_h != 1 || stride_w != 1){
            cvt_in_h = in_h + DivUp(kernel_h-stride_h, stride_h);
            cvt_in_w = in_w + DivUp(kernel_w-stride_w, stride_w);
            int kernel_u = DivUp(kernel_h, stride_h);
            int kernel_v = DivUp(kernel_w, stride_w);
            int pattern_num = stride_h*stride_w;
            cvt_in_size_v4 = batch*cvt_in_h*cvt_in_w*kernel_u*kernel_v*in_c_v4;
            constexpr int cta_size = 256;
            int grid_size = DivUp(cvt_in_size_v4, cta_size);

            new_nhwc2nhw_ker_c<<<grid_size, cta_size, 0, stream>>>(
                            (int4 *)cvt_input, (const int4 *)input,
                            batch, in_h, in_w, in_c_v4,
                            cvt_in_h, cvt_in_w,
                            kernel_h, kernel_w,
                            stride_h, stride_w,
                            kernel_u, kernel_v);


            //gemm
            ppl::nn::onnx::GemmParam gemm_param;
            fuse_param_t gemm_fuse_param;
            int M     = batch*cvt_in_h*cvt_in_w;
            int K_pad = kernel_u*kernel_v*in_c_pad;
            if (out_c <= 4 && ((pattern_num&1)==0)) {
                out_c_pad = 4;
            }
            int N_pad = out_c_pad*pattern_num;

            gemm_param.transA    = 0;
            gemm_param.transB    = 1;
            gemm_param.alpha     = 1.f;
            gemm_param.beta      = 0.f;
            ppl::nn::TensorShape a_shape, b_shape, c_shape;
            a_shape.SetDataType(input_shape->GetDataType());
            b_shape.SetDataType(input_shape->GetDataType());
            c_shape.SetDataType(output_shape->GetDataType());
            a_shape.Reshape( {M,     K_pad} );
            b_shape.Reshape( {N_pad, K_pad} );
            c_shape.Reshape( {M,     N_pad} );
            const void *gemm_bias = NULL;
            void *gemm_buf = NULL;
            void *gemm_output = (int4*)temp_buffer + cvt_in_size_v4;

            PPLCUDAGemmForwardImp(device_id, stream, module, &a_shape, cvt_input, &b_shape, rev_flt, 
                    gemm_bias, &c_shape, gemm_output, gemm_param, gemm_buf, gemm_fuse_param, algo_param);


            //cvt gemm_output to nhwc
            int has_relu = fuse_param.has_activation;
            constexpr int cvt_cta_size = 256;
            dim3 cvt_grid;
            cvt_grid.x = DivUp(cvt_in_h*cvt_in_w*pattern_num*out_c_v4, cvt_cta_size);
            cvt_grid.y = batch;
            cvt_grid.z = 1;

            int pad_height = kernel_h-1 - DivUp(kernel_h-stride_h, stride_h)*stride_h - pad_h;
            int pad_width  = kernel_w-1 - DivUp(kernel_w-stride_w, stride_w)*stride_w - pad_w;

            if (out_c_pad != 4) {
                nhwuvc2nhwc<1><<<cvt_grid, cvt_cta_size, 0, stream>>>(
                        (int4 *)output, (const int4 *)gemm_output,
                        batch, out_h, out_w, out_c_v4,
                        cvt_in_h, cvt_in_w,
                        kernel_h, kernel_w,
                        pad_height, pad_width,
                        stride_h, stride_w,
                        (const int4 *)bias, has_relu);
            }
            else {
                const int out_c_v2 = 1;//DivUp(out_c, 4);
                cvt_grid.x = DivUp(cvt_in_h*cvt_in_w*pattern_num*out_c_v2, cvt_cta_size);
                nhwuvc2nhwc<0><<<cvt_grid, cvt_cta_size, 0, stream>>>(
                        (int4 *)output, (const int4 *)gemm_output,
                        batch, out_h, out_w, out_c_v2,
                        cvt_in_h, cvt_in_w,
                        kernel_h, kernel_w,
                        pad_height, pad_width,
                        stride_h, stride_w,
                        (const int4 *)bias, has_relu);
            }

        }
        else{
            cvt_input = (void*)input;

            //fake conv
            conv_param_t conv_param;
            conv_param.in_num = batch;
            conv_param.num_chl = in_c;
            conv_param.num_flt = out_c;
            conv_param.num_chl_pad = Align(in_c, 8);
            conv_param.num_flt_pad = Align(out_c, 8);
            conv_param.num_grp = 1;
            conv_param.in_height = cvt_in_h;
            conv_param.in_width = cvt_in_w;
            conv_param.flt_height = kernel_h;
            conv_param.flt_width = kernel_w;
            conv_param.pad_height = kernel_h - 1- pad_h;
            conv_param.pad_width = kernel_w - 1- pad_w;
            conv_param.stride_height = 1;
            conv_param.stride_width = 1;
            conv_param.hole_height = 1;
            conv_param.hole_width = 1;
            conv_param.out_height = cvt_in_h + 2*(kernel_h-1-pad_h) - (kernel_h-1);
            conv_param.out_width = cvt_in_w + 2*(kernel_w-1-pad_w) - (kernel_w-1);
            conv_param.has_bias = NULL!=bias;
            
#ifdef PPLNN_ENABLE_CUDA_JIT
            PPLCUDAConvolutionForwardJitImp(
                device_id, stream, module->GetKernelFunc(), input_shape->GetDataType(),
                (int4*)cvt_input, (int4*)rev_flt, (int4*)output, (int4*)bias,
                (int4*)temp_buffer, algo_param, conv_param, fuse_param);
#else
            PPLCUDAConvolutionForwardImp(
                device_id, stream, ppl::common::DATATYPE_FLOAT16,
                (int4 *)cvt_input, (int4*)rev_flt, (int4*)output,
                (int4*)bias, (int4*)temp_buffer,
                algo_param, conv_param, fuse_param);
#endif
            return ppl::common::RC_SUCCESS;
        }
    }
    else{
        return ppl::common::RC_UNSUPPORTED;
    }
    return ppl::common::RC_SUCCESS;
}


double PPLCUDAConvTransposeSelectKernel(
    int device_id,
    cudaStream_t& stream,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    const void* rev_flt,
    const void* bias,
    void* temp_buffer,
    ppl::nn::TensorShape* output_shape,
    void* output,
    const ppl::nn::onnx::ConvTransposeParam* param,
    algo_param_t& algo_param)
{
    int batch    = input_shape->GetDim(0);
    int in_c     = input_shape->GetDim(1);
    int in_h     = input_shape->GetDim(2);
    int in_w     = input_shape->GetDim(3);
    int out_c    = output_shape->GetDim(1);
    int out_h    = output_shape->GetDim(2);
    int out_w    = output_shape->GetDim(3);
    int kernel_h = param->kernel_shape[0];
    int kernel_w = param->kernel_shape[1];
    int pad_h    = param->pads[0];
    int pad_w    = param->pads[1];
    int stride_h = param->strides[0];
    int stride_w = param->strides[1];
    int hole_h   = param->dilations[0];
    int hole_w   = param->dilations[1];
    double min_time = FLT_MAX;
    if (hole_h != 1 || hole_w != 1)
        return min_time;

    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        int in_c_v4 = DivUp(in_c, 8);
        int in_c_pad = Align(in_c, 8);
        int out_c_pad = Align(out_c, 8);
        int cvt_in_h = in_h;
        int cvt_in_w = in_w;
        // nhwc to stride-dilated nhwc
        if(stride_h != 1 || stride_w != 1){
            int4* cvt_input  = (int4*)temp_buffer;
            cvt_in_h = in_h + DivUp(kernel_h-stride_h, stride_h);
            cvt_in_w = in_w + DivUp(kernel_w-stride_w, stride_w);
            int kernel_u = DivUp(kernel_h, stride_h);
            int kernel_v = DivUp(kernel_w, stride_w);
            int pattern_num = stride_h*stride_w;
            int cvt_in_size_v4 = batch*cvt_in_h*cvt_in_w*kernel_u*kernel_v*in_c_v4;
            constexpr int cta_size = 256;
            int grid_size = DivUp(cvt_in_size_v4, cta_size);
            new_nhwc2nhw_ker_c<<<grid_size, cta_size, 0, stream>>>(
                            (int4 *)cvt_input, (const int4 *)input,
                            batch, in_h, in_w, in_c_v4,
                            cvt_in_h, cvt_in_w,
                            kernel_h, kernel_w,
                            stride_h, stride_w,
                            kernel_u, kernel_v);

            //rsc2puvc
            int cvt_cta_size = 128;
            dim3 cvt_grid;
            if (out_c <= 4 && ((pattern_num&1)==0)) {
                out_c_pad = 4; 
            }
            cvt_grid.x = DivUp(out_c_pad*pattern_num*kernel_u*kernel_v*in_c_v4, cvt_cta_size);
            cvt_grid.y = 1;//conv_in_c;
            cvt_grid.z = 1;
            int4 *cvt_flt = cvt_input + cvt_in_size_v4;
            flt_krsc2pkuvc<<<cvt_grid, cvt_cta_size, 0, stream>>>(
                    cvt_flt, (const int4 *)rev_flt,
                    out_c, out_c_pad, in_c_v4,
                    kernel_h, kernel_w, stride_h, stride_w,
                    kernel_u, kernel_v, pattern_num);

            //gemm
            fuse_param_t gemm_fuse_param;
            int M     = batch*cvt_in_h*cvt_in_w;
            int K_pad = kernel_u*kernel_v*in_c_pad;
            int N_pad = out_c_pad*pattern_num;
            ppl::nn::TensorShape a_shape, b_shape, c_shape;
            a_shape.SetDataType(input_shape->GetDataType());
            b_shape.SetDataType(input_shape->GetDataType());
            c_shape.SetDataType(output_shape->GetDataType());
            a_shape.Reshape( {M,     K_pad} );
            b_shape.Reshape( {N_pad, K_pad} );
            c_shape.Reshape( {M,     N_pad} );
            const void *gemm_bias = NULL;
            void *gemm_buf = NULL;
            void *gemm_output = (__half*)cvt_flt + N_pad * K_pad;


#ifdef PPLNN_ENABLE_CUDA_JIT
            //fake conv
            conv_param_t gemm_conv_param;
            gemm_conv_param.in_num = M;
            gemm_conv_param.num_chl = K_pad;
            gemm_conv_param.num_flt = N_pad;
            gemm_conv_param.num_chl_pad = K_pad;
            gemm_conv_param.num_flt_pad = N_pad;
            gemm_conv_param.num_grp = 1;
            gemm_conv_param.in_height = 1;
            gemm_conv_param.in_width = 1;
            gemm_conv_param.flt_height = 1;
            gemm_conv_param.flt_width = 1;
            gemm_conv_param.pad_height = 0;
            gemm_conv_param.pad_width = 0;
            gemm_conv_param.stride_height = 1;
            gemm_conv_param.stride_width = 1;
            gemm_conv_param.hole_height = 1;
            gemm_conv_param.hole_width = 1;
            gemm_conv_param.out_height = 1;
            gemm_conv_param.out_width = 1;
            gemm_conv_param.has_bias = false;

            min_time = PPLCUDAGemmJITSelectKernel(
                                device_id, stream,
                                input_shape->GetDataType(), &a_shape, (void*)cvt_input, &b_shape,
                                (void*)rev_flt, (void*)gemm_bias, &c_shape, (void*)gemm_output,
                                (void*)gemm_buf, gemm_conv_param, gemm_fuse_param, algo_param);
#else
            ppl::nn::onnx::GemmParam gemm_param;
            gemm_param.transA    = 0;
            gemm_param.transB    = 1;
            gemm_param.alpha     = 1.f;
            gemm_param.beta      = 1.f;

            min_time = PPLCUDAGemmSelectKernel(device_id, stream,
                                &a_shape, cvt_input, &b_shape, rev_flt,
                                gemm_bias, &c_shape, gemm_output,
                                gemm_buf, gemm_param, gemm_fuse_param,
                                algo_param);
#endif
        }
        else{
            void *cvt_input = (void*)input;

            //fake conv
            conv_param_t conv_param;
            conv_param.in_num = batch;
            conv_param.num_chl = in_c;
            conv_param.num_flt = out_c;
            conv_param.num_chl_pad = Align(in_c, 8);
            conv_param.num_flt_pad = Align(out_c, 8);
            conv_param.num_grp = 1;
            conv_param.in_height = cvt_in_h;
            conv_param.in_width = cvt_in_w;
            conv_param.flt_height = kernel_h;
            conv_param.flt_width = kernel_w;
            conv_param.pad_height = kernel_h - 1- pad_h;
            conv_param.pad_width = kernel_w - 1- pad_w;
            conv_param.stride_height = 1;
            conv_param.stride_width = 1;
            conv_param.hole_height = 1;
            conv_param.hole_width = 1;
            conv_param.out_height = cvt_in_h + 2*(kernel_h-1-pad_h) - (kernel_h-1);
            conv_param.out_width = cvt_in_w + 2*(kernel_w-1-pad_w) - (kernel_w-1);
            conv_param.has_bias = NULL!=bias;

            //algo_param_t algo_param;
            fuse_param_t fuse_param;

            void *d_temp_buf = temp_buffer;
#ifdef PPLNN_ENABLE_CUDA_JIT
            min_time = PPLCUDAConvolutionJitSelectKernel(
                            device_id, stream, input_shape->GetDataType(),
                            (int4*)cvt_input, (int4*)rev_flt, (int4*)output, (int4*)bias,
                            (int4*)d_temp_buf, algo_param, conv_param, fuse_param);
#else

            min_time = PPLCUDAConvolutionSelectKernel(
                            device_id, stream, input_shape->GetDataType(),
                            (int4 *)cvt_input, (int4*)rev_flt, (int4*)output,
                            (int4*)bias, (int4*)d_temp_buf,
                            algo_param, conv_param, fuse_param);
#endif
        }
    }

    return min_time;
}
