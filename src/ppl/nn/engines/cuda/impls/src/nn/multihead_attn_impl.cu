#include "cudakernel/common/common.cuh"

// Multihead Split qkv
/*
param :
    in : [T, B, 3, H, D]
    bias : [3, H, D]
    q : [B, H, T, D]
    k : [B, H, T, D]
    v : [B, H, T, D]
    T : time step
    B : Batch
    H : num head
    D : dimention for one head, H * D = embed_dim
thread :
    grid : (BH, T, 3)
    block : (D)
*/
template<typename Tin, typename TBias, typename TCompute = float>
__global__ void SplitQKVKernel(const Tin* in, const TBias* bias, Tin* q, Tin* k,
                                Tin* v, const int T, const int B, const int H,
                                const int D, const TCompute q_scale) {
        auto bi = blockIdx.x / H;
        auto hi = blockIdx.x % H;
        const Tin* cur_in = in + (blockIdx.y * B * 3 * H * D + + bi * 3 * H * D +
                                    blockIdx.z * H * D + hi * D);
        const TBias* cur_bias = bias + (blockIdx.z * H * D + hi * D);
        if (blockIdx.z == 0) {
            // split Q [B, H, T, D]
            Tin* o = q + (blockIdx.x * T * D + blockIdx.y * D);
            for(auto di = threadIdx.x; di < D; di += blockDim.x) {
                o[di] =
                    (Tin)(((TCompute)__ldg(cur_in + di) + (TCompute)__ldg(cur_bias + di)) *
                        q_scale);
            }
        } else if (blockIdx.z == 2) {
            // split V
            Tin* o = v + (blockIdx.x * T * D + blockIdx.y * D);
            for(auto di = threadIdx.x; di < D; di += blockDim.x) {
                o[di] =
                    (Tin)(((TCompute)__ldg(cur_in + di) + (TCompute)__ldg(cur_bias + di)));
            }
        } else {
            // split K 
            Tin* o = k + (blockIdx.x * T * D + blockIdx.y * D);
            for(auto di = threadIdx.x; di < D; di += blockDim.x) {
                o[di] =
                    (Tin)(((TCompute)__ldg(cur_in + di) + (TCompute)__ldg(cur_bias + di)));
            }
        }
    }
//support no bias
template<typename Tin, typename TCompute = float>
__global__ void SplitQKVKernel(const Tin* in, Tin* q, Tin* k,
                                Tin* v, const int T, const int B, const int H,
                                const int D, const TCompute q_scale) {
        auto bi = blockIdx.x / H;
        auto hi = blockIdx.x % H;
        const Tin* cur_in = in + (blockIdx.y * B * 3 * H * D + + bi * 3 * H * D +
                                    blockIdx.z * H * D + hi * D);
        if (blockIdx.z == 0) {
            // split Q [B, H, T, D]
            Tin* o = q + (blockIdx.x * T * D + blockIdx.y * D);
            for(auto di = threadIdx.x; di < D; di += blockDim.x) {
                o[di] = (Tin)((TCompute)__ldg(cur_in + di) * q_scale);
            }
        } else if (blockIdx.z == 2) {
            // split V
            Tin* o = v + (blockIdx.x * T * D + blockIdx.y * D);
            for(auto di = threadIdx.x; di < D; di += blockDim.x) {
                o[di] = (Tin)(TCompute)__ldg(cur_in + di);
            }
        } else {
            // split K 
            Tin* o = k + (blockIdx.x * T * D + blockIdx.y * D);
            for(auto di = threadIdx.x; di < D; di += blockDim.x) {
                o[di] = (Tin)(TCompute)__ldg(cur_in + di);
            }
        }
    }
template <typename Tin, typename TBias>
void SplitQKV(const Tin* in, const TBias* bias, Tin* q, Tin* k, Tin* v,
                const int T, const int B, const int H, const int D,
                cudaStream_t stream) {
    float q_scale = 1.0 / std::sqrt(static_cast<float>(D));
    dim3 grid(B * H, T, 3);
    dim3 block(GetBlockSize(D));
    if(bias == nullptr) {
        SplitQKVKernel<Tin, float><<<grid, block>>>(in, q, k, v, T, B, H, D, q_scale);
    } else {
        SplitQKVKernel<Tin, TBias, float><<<grid, block>>>(in, bias, q, k, v, T, B, H, D, q_scale);
    }
}



