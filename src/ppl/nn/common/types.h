#ifndef _ST_HPC_PPL_NN_COMMON_TYPES_H_
#define _ST_HPC_PPL_NN_COMMON_TYPES_H_

#include <stdint.h>
#include <memory>
#include <functional>

namespace ppl { namespace nn {

enum {
    /** tensors that can be modified or reused */
    TENSORTYPE_NORMAL,
    /** tensors that are reserved and cannot be reused */
    TENSORTYPE_RESERVED,
};
typedef uint32_t tensortype_t;

typedef std::unique_ptr<void, std::function<void(void*)>> VoidPtr;

static const uint32_t INVALID_NODEID = UINT32_MAX;
static const uint32_t INVALID_EDGEID = UINT32_MAX;

typedef uint32_t nodeid_t;
typedef uint32_t edgeid_t;

}} // namespace ppl::nn

#endif
