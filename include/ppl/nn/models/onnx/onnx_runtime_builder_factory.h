#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_ONNX_RUNTIME_BUILDER_FACTORY_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_ONNX_RUNTIME_BUILDER_FACTORY_H_

#include "ppl/nn/common/common.h"
#include "ppl/nn/models/onnx/onnx_runtime_builder.h"
#include "ppl/nn/engines/engine.h"
#include <vector>
#include <memory>

namespace ppl { namespace nn {

class PPLNN_PUBLIC OnnxRuntimeBuilderFactory final {
public:
    static OnnxRuntimeBuilder* Create(const char* model_file, std::vector<std::unique_ptr<Engine>>&&);

    static OnnxRuntimeBuilder* Create(const char* model_buf, uint64_t buf_len, std::vector<std::unique_ptr<Engine>>&&);
};

}} // namespace ppl::nn

#endif
