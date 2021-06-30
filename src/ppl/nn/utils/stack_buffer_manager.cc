#include "ppl/nn/utils/stack_buffer_manager.h"
using namespace ppl::common;

namespace ppl { namespace nn { namespace utils {

StackBufferManager::~StackBufferManager() {
    for (auto b : buffer_list_) {
        if (b.addr) {
            allocator_->Free(b.addr);
        }
    }
}

RetCode StackBufferManager::Realloc(uint64_t bytes, BufferDesc* buffer) {
    if (bytes == 0) {
        Free(buffer);
        buffer->addr = nullptr;
        return RC_SUCCESS;
    }

    if (buffer->addr == nullptr) {
        if (!buffer_stack_.empty()) {
            uint64_t managed_buffer_id;
            if (use_bestfit_) {
                managed_buffer_id = FindBestEmptyBuffer(bytes);
            } else {
                managed_buffer_id = buffer_stack_.back();
                buffer_stack_.pop_back();
            }
            BufferAddrAndSize& managed_buffer = buffer_list_[managed_buffer_id];
            ReallocManagedBuffer(bytes, &managed_buffer);
            if (managed_buffer.addr == nullptr) {
                return RC_OUT_OF_MEMORY;
            }
            buffer->addr = managed_buffer.addr;
            buffer->desc = managed_buffer_id;
        } else {
            BufferAddrAndSize managed_buffer;
            AllocManagedBuffer(bytes, &managed_buffer);
            if (managed_buffer.addr == nullptr) {
                return RC_OUT_OF_MEMORY;
            }
            buffer->addr = managed_buffer.addr;
            buffer->desc = buffer_list_.size();
            buffer_list_.push_back(managed_buffer);
        }
    } else {
        if (buffer->desc > buffer_list_.size()) {
            return RC_INVALID_VALUE;
        }
        BufferAddrAndSize& managed_buffer = buffer_list_[buffer->desc];
        ReallocManagedBuffer(bytes, &managed_buffer);
        if (managed_buffer.addr == nullptr) {
            buffer->addr = nullptr;
            return RC_OUT_OF_MEMORY;
        }
        buffer->addr = managed_buffer.addr;
    }

    return RC_SUCCESS;
}

void StackBufferManager::Free(BufferDesc* buffer) {
    if (buffer->addr == nullptr || buffer->desc > buffer_list_.size()) {
        return;
    }
    buffer_stack_.push_back(buffer->desc);
    buffer->addr = nullptr;
}

}}} // namespace ppl::nn::utils
