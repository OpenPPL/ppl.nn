#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/barrier.h"
#include <vector>

namespace ppl { namespace nn {

/**
   @class RuntimeGraph
   @brief data used in runtime stage
*/
struct RuntimeGraph {
    void Clear() {
        inputs.clear();
        extra_inputs.clear();
        constants.clear();
        outputs.clear();
        tensors.clear();
        nodeid2kernel.clear();
    }

    /** input tensors of the graph except constants */
    std::vector<TensorImpl*> inputs;

    /** extra inputs used by this graph */
    std::vector<TensorImpl*> extra_inputs;

    /** constant tensors */
    std::vector<TensorImpl*> constants;

    /** output tensors of the graph */
    std::vector<TensorImpl*> outputs;

    /** union of inputs/extra_inputs/constants/outputs */
    std::map<edgeid_t, TensorImpl> tensors;

    /** kernels list where the subscriptor is KernelImpl::GetNode()::GetId() */
    std::vector<std::unique_ptr<KernelImpl>> nodeid2kernel;

    /** whether a kernel needs to be synchronized before getting its outputs */
    std::vector<bool> kernel_barrier_flag;

    /** barriers for EdgeObjects before getting their contents */
    std::vector<std::shared_ptr<Barrier>> edgeid2barrier;
};

}} // namespace ppl::nn

#endif
