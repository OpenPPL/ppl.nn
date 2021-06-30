#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_ENGINE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_ENGINE_H_

#include <map>

#include "ppl/common/types.h"
#include "ppl/nn/engines/engine_impl.h"
#include "ppl/nn/engines/cuda/cuda_options.h"
#include "ppl/nn/engines/cuda/buffered_cuda_device.h"
#include "ppl/nn/quantization/quant_param_parser.h"
#include "ppl/nn/runtime/runtime_options.h"

#define MAX_NODE_SIZE 1000

using namespace std;

namespace ppl { namespace nn { namespace cuda {

struct CudaArgs {
    bool quick_select = false;
    ppl::common::datatype_t kernel_default_type = 0;
    std::map<std::string, ppl::common::dataformat_t> output_formats;
    std::map<std::string, ppl::common::datatype_t> output_types;
    std::map<std::string, ppl::common::datatype_t> node_types;
    std::map<std::string, std::vector<uint32_t>> input_dims;
};

class CudaEngine final : public EngineImpl {
public:
    CudaEngine() : EngineImpl("cuda") {}
    ppl::common::RetCode Init();
    ppl::common::RetCode Configure(uint32_t, ...) override;
    EngineContext* CreateEngineContext(const std::string& graph_name, const EngineContextOptions&) override;
    bool CanRunOp(const ir::Node*) const override;
    ppl::common::RetCode ProcessGraph(utils::SharedResource*, ir::Graph*, RuntimePartitionInfo*) override;

private:
    ppl::common::RetCode DoOptimize(ir::Graph*, utils::SharedResource*, RuntimePartitionInfo*);

private:
    /*
      some of them may visit class members.
      defined as member functions can avoid exporting unnecessary APIs
     */
    static ppl::common::RetCode SetOutputFormat(CudaEngine*, va_list);
    static ppl::common::RetCode SetOutputType(CudaEngine*, va_list);
    static ppl::common::RetCode SetCompilerInputDims(CudaEngine*, va_list);
    static ppl::common::RetCode SetKernelDefaultType(CudaEngine*, va_list);
    static ppl::common::RetCode SetAlgorithm(CudaEngine*, va_list);
    static ppl::common::RetCode SetNodeType(CudaEngine*, va_list);

    typedef ppl::common::RetCode (*ConfHandlerFunc)(CudaEngine*, va_list);
    static ConfHandlerFunc conf_handlers_[CUDA_CONF_MAX];

private:
    BufferedCudaDevice device_;
    CudaArgs cuda_flags_;
};

}}} // namespace ppl::nn::cuda

#endif
