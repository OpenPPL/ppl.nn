#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_ONNX_RUNTIME_BUILDER_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_ONNX_RUNTIME_BUILDER_H_

#include "ppl/nn/common/common.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/runtime/runtime_options.h"

namespace ppl { namespace nn {

class PPLNN_PUBLIC OnnxRuntimeBuilder {
public:
    virtual ~OnnxRuntimeBuilder() {}
    virtual Runtime* CreateRuntime(const RuntimeOptions&) = 0;
};

}} // namespace ppl::nn

#endif
