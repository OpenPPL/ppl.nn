#ifndef _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_H_
#define _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/runtime/runtime_graph.h"
#include "ppl/nn/runtime/profiler.h"

namespace ppl { namespace nn {

class Scheduler {
public:
    virtual ~Scheduler() {}
    virtual ppl::common::RetCode Init(const ir::GraphTopo*, const RuntimeAuxInfo*, RuntimeGraph*) = 0;
    virtual ppl::common::RetCode Run(Profiler*) = 0;
};

}} // namespace ppl::nn

#endif
