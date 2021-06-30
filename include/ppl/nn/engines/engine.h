#ifndef _ST_HPC_PPL_NN_ENGINES_ENGINE_H_
#define _ST_HPC_PPL_NN_ENGINES_ENGINE_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/common.h"

namespace ppl { namespace nn {

/**
   @class Engine
   @brief collection of op implementations
*/
class PPLNN_PUBLIC Engine {
public:
    virtual ~Engine() {}

    /** @brief get engine's name */
    virtual const char* GetName() const = 0;

    /** @brief various configurations. see `engines/<engname>/<engname>_options.h` for details. */
    virtual ppl::common::RetCode Configure(uint32_t, ...) = 0;
};

}} // namespace ppl::nn

#endif
