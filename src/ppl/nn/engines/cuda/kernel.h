#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_KERNEL_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_KERNEL_H_

#include "ppl/nn/runtime/kernel_impl.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/engines/cuda/macros.h"
#include "ppl/nn/engines/cuda/cuda_device.h"
#include "ppl/nn/engines/cuda/cuda_common_param.h"
#include "ppl/common/sys.h"

namespace ppl { namespace nn { namespace cuda {

class CudaKernel : public KernelImpl {
public:
    CudaKernel(const ir::Node* node) : KernelImpl(node) {}
    virtual ~CudaKernel();

    ppl::common::RetCode Init();

    cudaStream_t GetStream() const {
        auto cuda_device = static_cast<const CudaDevice*>(GetDevice());
        return cuda_device->GetStream();
    }

    void SetCommonParam(const CudaCommonParam* p) {
        common_param_ = p;
    }

    void SetReshapeFunc(const std::function<ppl::common::RetCode(InputOutputInfo*)>& f) {
        reshape_func_ = f;
    }

    ppl::common::RetCode Reshape(KernelExecContext* ctx) const {
        return reshape_func_(ctx);
    }

    ppl::common::RetCode Execute(KernelExecContext*) override final;

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
public:
    uint64_t GetExecutionTime() const override final;

private:
    cudaEvent_t exec_begin_event_, exec_end_event_;
#endif

protected:
    virtual bool CanDoExecute(const KernelExecContext&) const;
    virtual ppl::common::RetCode DoExecute(KernelExecContext*) = 0;
    virtual ppl::common::RetCode BeforeExecute(KernelExecContext*);

    virtual uint64_t CalcTmpBufferSize(const KernelExecContext& ctx) const {
        return 0;
    }

    CudaDevice* GetCudaDevice() {
        return reinterpret_cast<CudaDevice*>(GetDevice());
    }

private:
    const CudaCommonParam* common_param_ = nullptr;
    std::function<ppl::common::RetCode(InputOutputInfo*)> reshape_func_;
    cudaEvent_t finished_event_;
};

}}} // namespace ppl::nn::cuda

#endif
