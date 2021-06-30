
#include "cuda_common.h"

namespace ppl { namespace nn {

bool PPLCUDAComputeCapabilityRequired(int major, int minor, int device) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);
    return device_prop.major > major || (device_prop.major == major && device_prop.minor >= minor);
}

bool PPLCUDAComputeCapabilityEqual(int major, int minor, int device) {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, device);
    return (device_prop.major == major && device_prop.minor == minor);
}

}} // namespace ppl::nn
