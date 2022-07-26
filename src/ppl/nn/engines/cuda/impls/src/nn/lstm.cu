#include <cuda.h>
#include <cuda_fp16.h>
#include "cudakernel/common/common.h"
#include "cudakernel/gemm/gemm.h"
#include "cudakernel/nn/lstm.h"
#include <stdio.h>

__device__ float sigmoidf(float a)
{
    return expf(a) / (1 + expf(a));
}

//(seq, batch, dir, 4*hidden)
//(dir, batch, 4*hidden)
// P: (dir, 3*hidden)

// output: (dir, batch, hidden)
__global__ void fuse_gate(
    const void *hidden,
    const void *X_in,
    const void *bias,
    const void *ceil,
    const void *P,
    const int num_direction,
    const int batch,
    const int hidden_size,
    void *out_c,
    void *out_h)
{
    int tid       = blockIdx.x * blockDim.x + threadIdx.x;
    int h_id      = tid % (hidden_size);
    int hb_id     = tid / (hidden_size);
    bool in_range = tid < batch * hidden_size;

    int x_off = hb_id * num_direction * 4 * hidden_size + h_id;
    int h_off = hb_id * 4 * hidden_size + h_id;

    if (!in_range)
        return;

    float x1 = ((__half *)X_in)[x_off];
    float x2 = ((__half *)X_in)[x_off + 1 * hidden_size];
    float x3 = ((__half *)X_in)[x_off + 2 * hidden_size];
    float x4 = ((__half *)X_in)[x_off + 3 * hidden_size];

    float h1 = ((__half *)hidden)[h_off];
    float h2 = ((__half *)hidden)[h_off + 1 * hidden_size];
    float h3 = ((__half *)hidden)[h_off + 2 * hidden_size];
    float h4 = ((__half *)hidden)[h_off + 3 * hidden_size];

    float xb1 = bias ? (float)((__half *)bias)[h_id] : 0.f;
    float xb2 = bias ? (float)((__half *)bias)[h_id + 1 * hidden_size] : 0.f;
    float xb3 = bias ? (float)((__half *)bias)[h_id + 2 * hidden_size] : 0.f;
    float xb4 = bias ? (float)((__half *)bias)[h_id + 3 * hidden_size] : 0.f;

    float hb1 = bias ? (float)((__half *)bias)[h_id + 4 * hidden_size] : 0.f;
    float hb2 = bias ? (float)((__half *)bias)[h_id + 5 * hidden_size] : 0.f;
    float hb3 = bias ? (float)((__half *)bias)[h_id + 6 * hidden_size] : 0.f;
    float hb4 = bias ? (float)((__half *)bias)[h_id + 7 * hidden_size] : 0.f;

    float c_pre = ceil ? (float)((__half *)ceil)[tid] : 0.f;
    float pi    = P ? (float)((__half *)P)[h_id] : 0.f;
    float po    = P ? (float)((__half *)P)[h_id + 1 * hidden_size] : 0.f;
    float pf    = P ? (float)((__half *)P)[h_id + 2 * hidden_size] : 0.f;

    float gi = (x1 + xb1) + (h1 + hb1) + pi * c_pre;
    float go = (x2 + xb2) + (h2 + hb2) + po * c_pre;
    float gf = (x3 + xb3) + (h3 + hb3) + pf * c_pre;
    float gc = (x4 + xb4) + (h4 + hb4);

    gf           = sigmoidf(gf);
    gi           = sigmoidf(gi);
    gc           = tanhf(gc);
    float c      = gf * c_pre + gi * gc;
    go           = go + po * c;
    float output = sigmoidf(go);
    float ht     = output * tanhf(c);

    ((__half *)out_h)[tid] = (__half)ht;
    ((__half *)out_c)[tid] = (__half)c;
}

int64_t PPLCUDALstmGetRuntimeBufSize(
    const ppl::nn::TensorShape *X_shape,
    const unsigned int direction,
    const int64_t hidden_size)
{
    int seq_len       = X_shape->GetDim(0); // max seq_len
    int batch         = X_shape->GetDim(1);
    int num_direction = direction == RnnDirection::bidirectional ? 2 : 1;
    int64_t size      = 0;
    size += seq_len * batch * num_direction * 4 * hidden_size; // X_in
    size += batch * 4 * hidden_size; // hidden_buf
    size += batch * hidden_size; // ceil_buf
    return size * sizeof(__half);
}

/*
X: (sequence_len, batch, input_size)
W: (direction, 4*hidden_size, input_size)
=(seq, batch, dir, 4*hidden)

c: (direction, batch, hidden_size)

h: (direction, batch, hidden_size)
R: (direction, 4*hidden_size, hidden_size))
=(dir, batch, 4*hidden)

Bias: (dir, 2*4*hidden)
y: (sequence_len, dir, batch, hidden_size)
y_h: (dir, batch, hidden_size)
y_c: (dir, batch, hidden_size)
*/
ppl::common::RetCode PPLCUDALstmForwardImp(
    int device_id,
    cudaStream_t stream,
    ppl::nn::cuda::CUDAModule *module,
    const ppl::nn::TensorShape *X_shape,
    const void *X,
    const void *X_weight,
    const void *R_weight,
    const void *P_weight,
    const void *bias,
    const void *sequence_lens, // FIXME: batch-wise output is different
    const void *initial_h,
    const void *initial_c,
    const unsigned int direction,
    const int64_t hidden_size,
    void *temp_buffer,
    void *Y,
    void *Y_h,
    void *Y_c)
{
    int seq_len       = X_shape->GetDim(0); // max seq_len
    int batch         = X_shape->GetDim(1);
    int input_size    = X_shape->GetDim(2);
    int num_direction = direction == RnnDirection::bidirectional ? 2 : 1;

    //(seq, batch, dir, 4*hidden)
    ppl::nn::TensorShape input_shape, weight_shape, output_shape;
    int M = seq_len * batch;
    int K = input_size;
    int N = num_direction * 4 * hidden_size;
    if (sequence_lens) {
        printf("error: lstm sequence_lens are different.\n");
        return ppl::common::RC_UNSUPPORTED;
    }
    if (K % 8 != 0 || hidden_size % 8 != 0) {
        printf("error: lstm input size or hidden_size is not aligned.\n");
        return ppl::common::RC_UNSUPPORTED;
    }
#define GET_GEMM_PARAM                                       \
    input_shape.Reshape({M, K});                             \
    weight_shape.Reshape({N, K});                            \
    output_shape.Reshape({M, N});                            \
    input_shape.SetDataType(ppl::common::DATATYPE_FLOAT16);  \
    weight_shape.SetDataType(ppl::common::DATATYPE_FLOAT16); \
    output_shape.SetDataType(ppl::common::DATATYPE_FLOAT16); \
    fuse_param_t fuse_param;                                 \
    ppl::nn::onnx::GemmParam gemm_param;                     \
    gemm_param.transA    = 0;                                \
    gemm_param.transB    = 1;                                \
    gemm_param.alpha     = 1.f;                              \
    gemm_param.beta      = 1.f;                              \
    void *tmp_buf        = NULL;
    GET_GEMM_PARAM

    __half *X_in = (__half *)temp_buffer;
    algo_param_t algo_param;
    algo_param.UseDefaultF1Kernel();
    PPLCUDAGemmForwardImp(
        device_id, stream, module, &input_shape, X, &weight_shape, X_weight, NULL, &output_shape, X_in, gemm_param, tmp_buf, fuse_param, algo_param);

    __half *hidden_buf = (__half *)X_in + M * N;
    __half *ceil_buf   = hidden_buf + batch * 4 * hidden_size;
    int reverse        = direction == RnnDirection::reverse ? 1 : 0;

    for (int d = 0; d < num_direction; d++) {
        bool rev       = (reverse || d == 1);
        int dir        = rev ? -1 : 1;
        __half *tR     = (__half *)R_weight + d * 4 * hidden_size * hidden_size;
        __half *P      = P_weight ? (__half *)P_weight + d * 3 * hidden_size : NULL;
        __half *t_bias = (__half *)bias + d * 8 * hidden_size;

        for (int i = 0; i < seq_len; i++) {
            int pre_idx         = rev ? seq_len - i : i - 1;
            int cur_idx         = pre_idx + dir;
            __half *pre_hidden  = i == 0 ? (__half *)initial_h : (__half *)Y + pre_idx * num_direction * batch * hidden_size + d * batch * hidden_size;
            __half *post_hidden = hidden_buf;
            if (initial_h != nullptr) {
                int M = batch;
                int K = hidden_size;
                int N = 4 * hidden_size;
                GET_GEMM_PARAM
                PPLCUDAGemmForwardImp(
                    device_id, stream, module, &input_shape, pre_hidden, &weight_shape, tR, NULL, &output_shape, post_hidden, gemm_param, tmp_buf, fuse_param, algo_param);
            } else {
                cudaMemset(post_hidden, 0, 4 * hidden_size * sizeof(__half));
            }

            __half *out_h = (__half *)Y + cur_idx * num_direction * batch * hidden_size + d * batch * hidden_size;
            __half *pre_c = i == 0 ? (__half *)initial_c : ceil_buf;
            __half *out_c = ceil_buf;
            __half *Xt    = X_in + cur_idx * num_direction * batch * 4 * hidden_size + d * batch * 4 * hidden_size;

            dim3 grid;
            const int threads = 512;
            grid.x            = DivUp(batch * hidden_size, threads);
            grid.y            = 1;
            grid.z            = 1;
            fuse_gate<<<grid, threads, 0, stream>>>(post_hidden, Xt, t_bias, pre_c, P, num_direction, batch, hidden_size, out_c, out_h);

            if (Y_h && i == seq_len - 1) {
                cudaMemcpyAsync((__half *)Y_h + d * batch * hidden_size, out_h, batch * hidden_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
            }
            if (Y_c && i == seq_len - 1) {
                cudaMemcpyAsync((__half *)Y_c + d * batch * hidden_size, out_c, batch * hidden_size * sizeof(__half), cudaMemcpyDeviceToDevice, stream);
            }
        }
    }

    return ppl::common::RC_SUCCESS;
}
