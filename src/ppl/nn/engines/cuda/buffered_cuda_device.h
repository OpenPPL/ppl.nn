#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_BUFFERED_CUDA_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_BUFFERED_CUDA_DEVICE_H_

#include <cuda.h>

#include "ppl/nn/runtime/policy_defs.h"
#include "ppl/nn/utils/buffer_manager.h"
#include "ppl/nn/engines/cuda/cuda_device.h"

namespace ppl { namespace nn { namespace cuda {

class BufferedCudaDevice final : public CudaDevice {
public:
    ~BufferedCudaDevice();

    ppl::common::RetCode Init(MemoryManagementPolicy mm_policy = MM_BETTER_PERFORMANCE);

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc*) override;
    void Free(BufferDesc*) override;

    ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) override;
    void FreeTmpBuffer(BufferDesc* buffer) override;

private:
    std::unique_ptr<ppl::common::Allocator> allocator_;
    std::unique_ptr<utils::BufferManager> buffer_manager_;
    BufferDesc shared_tmp_buffer_;
    uint64_t tmp_buffer_size_ = 0;
};

}}} // namespace ppl::nn::cuda

#endif
