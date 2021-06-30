#ifndef _ST_HPC_PPL_NN_ENGINES_X86_ENGINE_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_ENGINE_H_

#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/engines/x86/x86_device.h"
#include "ppl/nn/engines/x86/x86_options.h"

namespace ppl { namespace nn { namespace x86 {

class X86Engine final : public EngineImpl {
public:
    X86Engine();
    ppl::common::RetCode Configure(uint32_t, ...) override;
    EngineContext* CreateEngineContext(const std::string& graph_name, const EngineContextOptions&) override;
    bool CanRunOp(const ir::Node*) const override;
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph*, RuntimePartitionInfo*) override;

private:
    ppl::common::RetCode DoOptimize(ir::Graph*, utils::SharedResource*, RuntimePartitionInfo*);

private:
    /*
     * some of them may visit class members.
     * defined as member functions can avoid exporting unnecessary APIs
     */
    static ppl::common::RetCode DisableAVX512(X86Engine*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(X86Engine*, va_list);
    static ConfHandlerFunc conf_handlers_[X86_CONF_MAX];

private:
    X86Device device_;
};

}}} // namespace ppl::nn::x86

#endif
