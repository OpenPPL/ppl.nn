#ifndef _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_INFO_H_
#define _ST_HPC_PPL_NN_RUNTIME_RUNTIME_GRAPH_INFO_H_

#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/runtime/opt_kernel.h"
#include "ppl/nn/runtime/runtime_constant_info.h"
#include <vector>
#include <map>

namespace ppl { namespace nn {

class EngineImpl;

struct RuntimeKernelInfo {
    RuntimeKernelInfo() : engine(nullptr) {}
    RuntimeKernelInfo(RuntimeKernelInfo&&) = default;
    RuntimeKernelInfo& operator=(RuntimeKernelInfo&&) = default;

    EngineImpl* engine;
    std::unique_ptr<OptKernel> op;
};

struct RuntimeGraphInfo {
    std::map<edgeid_t, TensorShape> shapes;
    std::vector<std::pair<edgeid_t, RuntimeConstantInfo>> constants;
    std::vector<RuntimeKernelInfo> kernels; // sorted topologically
};

}} // namespace ppl::nn

#endif
