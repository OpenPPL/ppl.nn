#ifndef _ST_HPC_PPL_NN_UTILS_COMPACT_BUFFER_MANAGER_H_
#define _ST_HPC_PPL_NN_UTILS_COMPACT_BUFFER_MANAGER_H_

#include "ppl/common/compact_memory_manager.h"
#include "ppl/nn/utils/buffer_manager.h"

namespace ppl { namespace nn { namespace utils {

class CompactBufferManager final : public BufferManager {
public:
    CompactBufferManager(ppl::common::Allocator* ar, uint64_t block_size = 1048576)
        : BufferManager("CompactBufferManager"), mgr_(ar, block_size) {}

    ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) override;
    void Free(BufferDesc* buffer) override;
    uint64_t GetAllocatedBytes() const override {
        return mgr_.GetAllocatedBytes();
    }

private:
    ppl::common::CompactMemoryManager mgr_;
};

}}} // namespace ppl::nn::utils

#endif
