#ifdef PPLNN_ENABLE_FP8

#include "online_cast_kernel.h"

#include "ppl/kernel/llm/cuda/pmx/f8f8/cast.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {

ppl::common::RetCode F8F8OnlineCastKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);
    

    if (input->GetShape()->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
        LOG(ERROR) << "currently only support fp16 input";
        return ppl::common::RC_UNSUPPORTED;
    }

    const auto dim_count = input->GetShape()->GetDimCount();
    const int64_t quant_dim = input->GetShape()->GetDim(dim_count - 1);
    const int64_t batch = input->GetShape()->CalcElementsToDimensionIncludingPadding(dim_count - 1);

    return ppl::kernel::llm::cuda::pmx::f8f8::cast_fp16(
        GetStream(),
        input->GetBufferPtr(),
        batch,
        quant_dim,
        output->GetBufferPtr()
    );
}

}}}}} // namespace ppl::nn::llm::cuda::pmx

#endif
