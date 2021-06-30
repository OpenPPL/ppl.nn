#ifndef _ST_HPC_PPL_NN_COMMON_BUFFER_DESC_H_
#define _ST_HPC_PPL_NN_COMMON_BUFFER_DESC_H_

#include <stdint.h>
#include <functional>

namespace ppl { namespace nn {

struct BufferDesc final {
    BufferDesc(void* a = nullptr) : addr(a) {}

    /** pointer to data area */
    void* addr;

    /** used by engines with different meanings. this union is invalid if `addr` is nullptr. */
    union {
        uint64_t desc;
        void* info;
    };
};

class BufferDescGuard {
public:
    BufferDescGuard(BufferDesc* buffer, const std::function<void(BufferDesc*)>& deleter)
        : buffer_(buffer), deleter_(deleter) {}
    ~BufferDescGuard() {
        deleter_(buffer_);
    }

private:
    BufferDesc* buffer_;
    std::function<void(BufferDesc*)> deleter_;
};

}} // namespace ppl::nn

#endif
