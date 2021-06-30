#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA__CUDA_COMMON_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA__CUDA_COMMON_H_

#include <cuda_runtime.h>

namespace ppl { namespace nn {

struct CudaCtxParam {
    int device_id;
    cudaStream_t stream = nullptr;
};

bool PPLCudaComputeCapabilityRequired(int major, int minor, int device);
bool PPLCudaComputeCapabilityEqual(int major, int minor, int device);

}} // namespace ppl::nn

#endif
