#include "ppl/nn/common/logger.h"
#include "ppl/nn/optimizers/utils.h"
#include "ppl/nn/runtime/runtime_impl.h"
#include "ppl/nn/models/onnx/model_parser.h"
#include "ppl/nn/models/onnx/runtime_builder_impl.h"
#include "ppl/common/file_mapping.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn { namespace onnx {

RuntimeBuilderImpl::RuntimeBuilderImpl() {
    resource_ = make_shared<utils::SharedResource>();
    graph_info_ = make_shared<RuntimeGraphInfo>();
    aux_info_ = make_shared<RuntimeAuxInfo>();
}

RuntimeBuilderImpl::~RuntimeBuilderImpl() {
    aux_info_.reset();
    graph_info_.reset();
    resource_.reset();
}

RetCode RuntimeBuilderImpl::Init(vector<unique_ptr<EngineImpl>>&& engines, const char* model_buf, size_t buf_len) {
    resource_->engines.reserve(engines.size());
    for (auto e = engines.begin(); e != engines.end(); ++e) {
        auto impl = unique_ptr<EngineImpl>(static_cast<EngineImpl*>(e->release()));
        resource_->engines.emplace_back(std::move(impl));
    }

    auto status = ModelParser::Parse(model_buf, buf_len, &graph_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "parse graph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = utils::ProcessGraph(resource_.get(), &graph_, graph_info_.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "process graph failed: " << GetRetCodeStr(status);
        return status;
    }

    status = GenerateRuntimeAuxInfo(*graph_info_, aux_info_.get());
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "GenerateRuntimeAuxInfo failed: " << GetRetCodeStr(status);
        return status;
    }

    return RC_SUCCESS;
}

RetCode RuntimeBuilderImpl::Init(vector<unique_ptr<EngineImpl>>&& engines, const char* model_file) {
    ppl::common::FileMapping fm;
    if (fm.Init(model_file) != RC_SUCCESS) {
        LOG(ERROR) << "Init filemapping from file [" << model_file << "] error.";
        return RC_INVALID_VALUE;
    }
    return Init(std::move(engines), fm.Data(), fm.Size());
}

Runtime* RuntimeBuilderImpl::CreateRuntime(const RuntimeOptions& options) {
    auto runtime = new RuntimeImpl();
    if (!runtime) {
        return nullptr;
    }

    auto status = runtime->Init(options, graph_.topo, graph_info_, aux_info_, resource_);
    if (status != RC_SUCCESS) {
        LOG(ERROR) << "init runtime failed: " << GetRetCodeStr(status);
        delete runtime;
        return nullptr;
    }

    return runtime;
}

}}} // namespace ppl::nn::onnx
