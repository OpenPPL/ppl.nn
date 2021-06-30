#ifndef _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPT_GRAPH_H_
#define _ST_HPC_PPL_NN_ENGINES_X86_OPTIMIZER_OPT_GRAPH_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/engines/x86/x86_device.h"
#include "ppl/nn/runtime/runtime_partition_info.h"
#include "ppl/nn/engines/x86/optimizer/opt_kernel.h"
#include <memory>

namespace ppl { namespace nn { namespace x86 {

class OptGraph final {
public:
    ppl::common::RetCode Init(ir::Graph*, utils::SharedResource*, RuntimePartitionInfo*);
    ppl::common::RetCode DoOptimize(X86Device*);

private:
    ppl::common::RetCode InitKernels(const ir::Graph* graph);
    ppl::common::RetCode InitTensorImpls();
    ppl::common::RetCode AddReorderOp(const OptKernelOptions& options, const edgeid_t& edge_id, const nodeid_t& node_id,
                                      const int32_t& reorder_type, const ppl::common::dataformat_t& reorder_in_format,
                                      const ppl::common::dataformat_t& reorder_out_format);
    ppl::common::RetCode LayoutOptimize(const OptKernelOptions& options);
    ppl::common::RetCode FuseReorderOp();
    ppl::common::RetCode TryToInferType(X86Device* device);
    ppl::common::RetCode TryToInferDims(X86Device* device);
    bool FuseConvActivation();
    bool FuseConvAdd();
    bool FuseChannelShuffle();
    bool FuseBNReLU();
    bool FuseArithmeticReLU();
    bool FuseFcActivation();

private:
    utils::SharedResource* resource_ = nullptr;
    ir::Graph* graph_ = nullptr;
    RuntimePartitionInfo* info_ = nullptr;
    std::map<edgeid_t, std::unique_ptr<TensorImpl>> tensor_impls_;
};

}}} // namespace ppl::nn::x86

#endif
