#ifdef PPLNN_ENABLE_FP8

#include "column_parallel_linear_kernel.h"

#include "ppl/common/cuda/nccl_utils.h"
#include "ppl/common/destructor.h"

#include "ppl/kernel/llm/cuda/pmx/f8f8/column_parallel_linear.h"

namespace ppl { namespace nn { namespace llm { namespace cuda { namespace opmx {


ppl::common::RetCode F8F8ColumnParallelLinearKernel::DoExecute(KernelExecContext* ctx) {
    PPLNN_LLM_CUDA_DEBUG_TRACE("Entry LlmCudaKernel: [%s]\n", GetName().c_str());

    PPLNN_LLM_CUDA_REQUIRED_INPUT(input, 0);
    PPLNN_LLM_CUDA_REQUIRED_INPUT(weight, 1);
    PPLNN_LLM_CUDA_OPTIONAL_INPUT(bias, 2);
    PPLNN_LLM_CUDA_REQUIRED_OUTPUT(output, 0);

    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [input]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(input);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Input [weight]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(weight);
    if (bias) {
        PPLNN_LLM_CUDA_DEBUG_TRACE("Input [bias]:\n");
        PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(bias);
    }

    PPLNN_LLM_CUDA_DEBUG_TRACE("in_features: %d\n", param_->in_features);
    PPLNN_LLM_CUDA_DEBUG_TRACE("out_features: %d\n", param_->out_features);
    PPLNN_LLM_CUDA_DEBUG_TRACE("bias_term: %d\n", param_->bias_term);
    PPLNN_LLM_CUDA_DEBUG_TRACE("gather_output: %d\n", param_->gather_output);

    PPLNN_LLM_CUDA_RESHAPE_OUTPUTS();

    PPLNN_LLM_CUDA_REALLOC_TENSOR_BUFFER(output);
    PPLNN_LLM_CUDA_DEBUG_TRACE("Output [output]:\n");
    PPLNN_LLM_CUDA_TENSOR_PRINT_DEBUG_MSG(output);

    auto input_shape = input->GetShape();
    auto weight_shape = weight->GetShape();
    auto output_shape = output->GetShape();

    TensorShape *bias_shape = nullptr;
    void *bias_data = nullptr;
    if (param_->bias_term) {
        if (!bias) {
            LOG(ERROR) << "bias_term == true but bias not found.";
            return ppl::common::RC_NOT_FOUND;
        }
        bias_shape = bias->GetShape();
        bias_data = bias->GetBufferPtr();

        if (bias_shape->GetDataType() != ppl::common::DATATYPE_FLOAT16) {
            LOG(ERROR) << "currently only support fp16 bias";
            return ppl::common::RC_UNSUPPORTED;
        }
    }
 
    if (input_shape->GetDataType() != ppl::common::DATATYPE_FLOAT8E4M3) {
        LOG(ERROR) << "only support float8 input";
        return ppl::common::RC_UNSUPPORTED;
    }
    if (weight_shape->GetDataType() != ppl::common::DATATYPE_FLOAT8E4M3) {
        LOG(ERROR) << "only support float8 weight";
        return ppl::common::RC_UNSUPPORTED;
    }

    auto cublas_handle = GetCublasHandle();
    auto nccl_param = GetTensorParallelNcclParam();

    uint64_t gather_buffer_size = 0;
    void *gather_buffer = nullptr;
    if (param_->gather_output && nccl_param->size > 1) {
        gather_buffer_size = output_shape->CalcBytesIncludingPadding();
    }

    BufferDesc tmp_buffer_desc;
    auto status = GetCudaDevice()->AllocTmpBuffer(gather_buffer_size, &tmp_buffer_desc);
    if (status != ppl::common::RC_SUCCESS) {
        LOG(ERROR) << "alloc tmp buffer size[" << gather_buffer_size << "] for kernel[" << GetName()
                << "] failed: " << ppl::common::GetRetCodeStr(status);
        return status;
    }
    ppl::common::Destructor __tmp_buffer_guard([this, &tmp_buffer_desc]() -> void {
        GetCudaDevice()->FreeTmpBuffer(&tmp_buffer_desc);
    });
    gather_buffer = tmp_buffer_desc.addr;

    status = ppl::kernel::llm::cuda::pmx::f8f8::column_parallel_linear(
        GetStream(),
        cublas_handle,
        nullptr,
        input_shape,
        input->GetBufferPtr(),
        weight_shape,
        weight->GetBufferPtr(),
        bias_shape,
        bias_data,
        param_->in_features,
        param_->out_features,
        nccl_param,
        param_->gather_output,
        gather_buffer,
        GetCudaDevice()->GetCublasWorkspaceSize(),
        GetCudaDevice()->GetCublasWorkspace(),
        output_shape,
        output->GetBufferPtr()
    );

    if (status != ppl::common::RC_SUCCESS) {
        return status;
    }

    if (input_shape->GetPadding1(0) > 0) {
        output_shape->SetPadding1(0, 0);
    }

    return ppl::common::RC_SUCCESS;
}

}}}}} // namespace ppl::nn::llm::cuda::pmx

#endif