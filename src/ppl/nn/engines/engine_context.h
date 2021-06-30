#ifndef _ST_HPC_PPL_NN_ENGINES_ENGINE_CONTEXT_H_
#define _ST_HPC_PPL_NN_ENGINES_ENGINE_CONTEXT_H_

#include "ppl/nn/common/device.h"

namespace ppl { namespace nn {

/**
   @class EngineContext
   @brief resources needed by a `Runtime` instance
   @note Each `Runtime` has only one `EngineContext` and an `EngineContext`
   is used only by one `Runtime` instance.
*/
class EngineContext {
public:
    virtual ~EngineContext() {}

    /** @brief get device instance used by `Runtime` */
    virtual Device* GetDevice() = 0;
};

}} // namespace ppl::nn

#endif
