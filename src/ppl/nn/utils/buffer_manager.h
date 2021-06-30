#ifndef _ST_HPC_PPL_NN_UTILS_BUFFER_MANAGER_H_
#define _ST_HPC_PPL_NN_UTILS_BUFFER_MANAGER_H_

#include <string>

#include "ppl/common/retcode.h"
#include "ppl/nn/common/buffer_desc.h"

namespace ppl { namespace nn { namespace utils {

class BufferManager {
public:
    BufferManager(const std::string& name) : name_(name) {}
    virtual ~BufferManager() {}
    const std::string& GetName() const {
        return name_;
    }
    virtual ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) = 0;
    virtual void Free(BufferDesc* buffer) = 0;
    virtual uint64_t GetAllocatedBytes() const = 0;

private:
    const std::string name_;
};

}}} // namespace ppl::nn::utils

#endif
