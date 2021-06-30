#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_DEFAULT_CUDA_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_DEFAULT_CUDA_DEVICE_H_

#include "ppl/common/allocator.h"

#include "ppl/nn/engines/cuda/cuda_device.h"

namespace ppl { namespace nn { namespace cuda {

class DefaultCudaDevice final : public CudaDevice {
public:
    DefaultCudaDevice();
    virtual ~DefaultCudaDevice();

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc*) override;
    void Free(BufferDesc*) override;

private:
    std::unique_ptr<ppl::common::Allocator> allocator_;
};

}}} // namespace ppl::nn::cuda

#endif
