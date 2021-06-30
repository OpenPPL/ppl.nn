#ifndef _ST_HPC_PPL_NN_ENGINES_ENGINE_IMPL_H_
#define _ST_HPC_PPL_NN_ENGINES_ENGINE_IMPL_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/engine.h"
#include "ppl/nn/engines/engine_context.h"
#include "ppl/nn/engines/engine_context_options.h"

namespace ppl { namespace nn {

namespace utils {
struct SharedResource;
}

struct RuntimePartitionInfo;

/**
   @class EngineImpl
   @brief engine implementation interface
*/
class EngineImpl : public Engine {
public:
    /** @param name engine's name */
    EngineImpl(const std::string& name) : name_(name) {}

    virtual ~EngineImpl() {}

    const char* GetName() const override final {
        return name_.c_str();
    }

    /**
       @brief create an `EngineContext` instance for a `Runtime` instance of
       graph named `graph_name`.
    */
    virtual EngineContext* CreateEngineContext(const std::string& graph_name, const EngineContextOptions&) = 0;

    /** @brief tells whether this engine can run an op specified by `node`. */
    virtual bool CanRunOp(const ir::Node* node) const = 0;

    /**
       @brief optimize the compute graph `graph` and fill `info`
       @param graph graph to be optimized and can be modified
       @note DO NOT modify input and output edges
    */
    virtual ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph* graph, RuntimePartitionInfo* info) = 0;

private:
    const std::string name_;
};

}} // namespace ppl::nn

#endif
