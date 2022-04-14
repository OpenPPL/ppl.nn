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
#include <float.h>
#include <cuda_fp16.h>

#define CUDA_KERNEL_LOOP(i, n)                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
         i < (n);                                       \
         i += blockDim.x * gridDim.x)


uint64_t pplConvTransposeGetFilterBufSizeCudaFp16(
    const int num_filters,
    const int num_channels,
    const int filter_height,
    const int filter_width)
{
    int M = num_filters;
    M *= filter_height;
    M *= filter_width;
    size_t K = num_channels;

    size_t padM = Align(M, 1);
    size_t padK = Align(K, 8);

    size_t dst = Align(padM * padK * sizeof(__half), 128);
    return dst * 2; // transpose buf
}

uint64_t pplConvTransposeGetTempBufSizeCudaFp32(
    const int group,
    const int in_c,
    const int in_h,
    const int in_w,
    const int out_c,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int hole_h,
    const int hole_w)
{
    size_t M = out_c * kernel_h * kernel_w;
    size_t N = in_w * in_h;
    size_t K = in_c;

    size_t padN = Align(N, 8);
    size_t padK = Align(K, 8);

    // for trans buf
    return 2 * Align(padN * padK * sizeof(__half), 128) +
           Align(M * N * sizeof(__half), 128) + Align(M * padN * sizeof(__half), 128);
}

uint64_t PPLConvTransposeGetBufSizeCuda(
    ppl::nn::TensorShape* input_shape,
    ppl::nn::TensorShape* output_shape,
    const ppl::nn::onnx::ConvTransposeParam* param)
{
    int batch     = input_shape->GetDim(0);
    int in_c      = input_shape->GetDim(1);
    int in_h      = input_shape->GetDim(2);
    int in_w      = input_shape->GetDim(3);
    int out_c     = output_shape->GetDim(1);
    int out_h     = output_shape->GetDim(2);
    int out_w     = output_shape->GetDim(3);
    //int group     = param->group;
    int kernel_h  = param->kernel_shape[0];
    int kernel_w  = param->kernel_shape[1];
    int pad_h     = param->pads[0];
    int pad_w     = param->pads[1];
    int stride_h  = param->strides[0];
    int stride_w  = param->strides[1];
    int hole_h    = param->dilations[0];
    int hole_w    = param->dilations[1];
    uint64_t size = 0;
    int cvt_h = stride_h * (in_h-1) + 1;
    int cvt_w = stride_w * (in_w-1) + 1;
    if (input_shape->GetDataType() == ppl::common::DATATYPE_FLOAT16) {
        size = (uint64_t)batch*cvt_h*cvt_w*Align(in_c, 8)*sizeof(__half);
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
    ppl::common::RetCode status = PPLCUDATransposeForwardImp(stream,
                                                             trans_param,
                                                             &a_shape,
                                                             in_filter,
                                                             &out_a_shape,
                                                             trans_flt);

    int cta_size = 256;
    int rev_grid = DivUp(M*padK, cta_size);
    //crsk_pad 2 rev crsk_pad
    reverse_flt<<<rev_grid, cta_size, 0, stream>>>(trans_flt, out_filter, conv_in_c, kernel_h, kernel_w, padK/8);

    return ppl::common::RC_SUCCESS;
}

/*
   flt: reversed crsk_pad
   input: nhwc_pad
*/
ppl::common::RetCode PPLCUDAConvTransposeForward(
    cudaStream_t stream,
    ppl::nn::cuda::CUDAModule* module,
    ppl::nn::TensorShape* input_shape,
    const void* input,
    const void* rev_flt,
    const void* bias,
    const ppl::nn::onnx::ConvTransposeParam* param,
    algo_param_t algo_param,
    void* temp_buffer,
    ppl::nn::TensorShape* output_shape,
    void* output)
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
        int cvt_in_h = stride_h * (in_h-1) + 1;
        int cvt_in_w = stride_w * (in_w-1) + 1;
        // nhwc to stride-dilated nhwc
        if(stride_h != 1 || stride_w != 1){
            int cta_size = 256;
            dim3 grid(DivUp(batch*cvt_in_h*cvt_in_w*in_c_v4, cta_size));
            nhwc2nhwuvc<<<grid, cta_size, 0, stream>>>(input, cvt_input,
                        batch, in_h, in_w, in_c_v4,
                        cvt_in_h, cvt_in_w,
                        stride_h, stride_w);
        }
        else{
            cvt_input = (void*)input;
        }

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


        int4 *d_temp_buf = (int4*)cvt_input + batch*cvt_in_h*cvt_in_w*in_c_v4;
        PPLCUDAConvolutionForwardImp(
            stream, ppl::common::DATATYPE_FLOAT16,
            (int4 *)cvt_input, (int4*)rev_flt, (int4*)output,
            (int4*)bias, (int4*)d_temp_buf,
            algo_param, conv_param, fuse_param);
        return ppl::common::RC_SUCCESS;
    }
    else{
        return ppl::common::RC_UNSUPPORTED;
    }
}


double PPLCUDAConvTransposeSelectKernel(
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
        void* cvt_input  = (__half*)temp_buffer;
        int in_c_v4 = DivUp(in_c, 8);
        int cvt_in_h = stride_h * (in_h-1) + 1;
        int cvt_in_w = stride_w * (in_w-1) + 1;
        // nhwc to stride-dilated nhwc
        if(stride_h != 1 || stride_w != 1){
            int cta_size = 256;
            dim3 grid(DivUp(batch*cvt_in_h*cvt_in_w*in_c_v4, cta_size));
            nhwc2nhwuvc<<<grid, cta_size, 0, stream>>>(input, cvt_input,
                        batch, in_h, in_w, in_c_v4,
                        cvt_in_h, cvt_in_w,
                        stride_h, stride_w);
        }
        else{
            cvt_input = (void*)input;
        }

        //fake conv
        fuse_param_t fuse_param;
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

        int4 *d_temp_buf = (int4*)cvt_input + batch*cvt_in_h*cvt_in_w*in_c_v4;
        min_time = PPLCUDAConvolutionSelectKernel(
                            stream, ppl::common::DATATYPE_FLOAT16,
                            (int4 *)cvt_input, (int4*)rev_flt, (int4*)output,
                            (int4*)bias, (int4*)d_temp_buf,
                            algo_param, conv_param, fuse_param);
    }
    return min_time;
}
