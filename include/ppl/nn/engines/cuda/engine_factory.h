#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_ENGINE_FACTORY_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_ENGINE_FACTORY_H_

#include "ppl/nn/engines/engine.h"

namespace ppl { namespace nn {

class PPLNN_PUBLIC CudaEngineFactory final {
public:
    static Engine* Create();
};

}} // namespace ppl::nn

#endif
