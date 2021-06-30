#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_RUNTIME_BUILDER_IMPL_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_RUNTIME_BUILDER_IMPL_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/models/onnx/onnx_runtime_builder.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/utils/shared_resource.h"
#include "ppl/nn/runtime/runtime.h"
#include "ppl/nn/runtime/runtime_options.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include "ppl/nn/runtime/runtime_aux_info.h"

namespace ppl { namespace nn { namespace onnx {

class RuntimeBuilderImpl final : public OnnxRuntimeBuilder {
public:
    RuntimeBuilderImpl();
    ~RuntimeBuilderImpl();
    ppl::common::RetCode Init(std::vector<std::unique_ptr<EngineImpl>>&&, const char* model_buf, size_t buf_len);
    ppl::common::RetCode Init(std::vector<std::unique_ptr<EngineImpl>>&&, const char* model_file);
    Runtime* CreateRuntime(const RuntimeOptions&) override;

private:
    ir::Graph graph_;
    std::shared_ptr<utils::SharedResource> resource_;
    std::shared_ptr<RuntimeGraphInfo> graph_info_;
    std::shared_ptr<RuntimeAuxInfo> aux_info_;
};

}}} // namespace ppl::nn::onnx

#endif
