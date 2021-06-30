#ifndef _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_COMMON_H_
#define _ST_HPC_PPL_NN_RUNTIME_SCHEDULER_COMMON_H_

#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/runtime_graph.h"
#include "ppl/nn/runtime/profiler.h"
#include <stdint.h>
#include <vector>

namespace ppl { namespace nn { namespace utils {

/** @brief calculate each object's reference count in `topo`. */
std::vector<uint32_t> InitObjectRefcount(const ir::GraphTopo* topo);

/** @brief put inputs/extra_inputs/outputs/constants into a vector */
std::vector<EdgeObject*> InitObjectInUse(const ir::GraphTopo* topo, RuntimeGraph* graph);

ppl::common::RetCode ExecuteKernel(KernelImpl*, KernelExecContext*, bool needs_output_barrier,
                                   const std::function<ppl::common::RetCode(EdgeObject*)>& release_object_func,
                                   Profiler*);

}}} // namespace ppl::nn::utils

#endif
