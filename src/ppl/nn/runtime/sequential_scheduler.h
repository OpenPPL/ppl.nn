#ifndef _ST_HPC_PPL_NN_RUNTIME_SEQUENTIAL_SCHEDULER_H_
#define _ST_HPC_PPL_NN_RUNTIME_SEQUENTIAL_SCHEDULER_H_

#include "ppl/nn/runtime/scheduler.h"
#include "ppl/common/object_pool.h"
#include "ppl/nn/runtime/tensor_sequence.h"

namespace ppl { namespace nn {

class SequentialScheduler final : public Scheduler {
public:
    ppl::common::RetCode Init(const ir::GraphTopo* topo, const RuntimeAuxInfo* aux_info, RuntimeGraph* g) override;
    ppl::common::RetCode Run(Profiler*) override;

private:
    const ir::GraphTopo* topo_;
    const RuntimeAuxInfo* aux_info_;
    RuntimeGraph* graph_;

    /** object reference count. this vector is read-only after being created. */
    std::vector<uint32_t> const_object_refcount_;

    /** used to hold objects that are used during Run() */
    std::vector<EdgeObject*> edgeid2object_;

    /** used to accelerlate tensor allocations */
    ppl::common::ObjectPool<TensorImpl> tensor_pool_;

    /** used to accelerlate tensor sequence allocations */
    ppl::common::ObjectPool<TensorSequence> tensor_sequence_pool_;
};

}} // namespace ppl::nn

#endif
