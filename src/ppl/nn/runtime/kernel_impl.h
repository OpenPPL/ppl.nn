#ifndef _ST_HPC_PPL_NN_RUNTIME_KERNEL_IMPL_H_
#define _ST_HPC_PPL_NN_RUNTIME_KERNEL_IMPL_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/runtime/kernel_exec_context.h"
#include <string>

namespace ppl { namespace nn {

/**
   @class KernelImpl
   @note KernelImpl may run on different threads when SCHED_PARALLEL is specified
*/
class KernelImpl {
public:
    KernelImpl(const ir::Node* node) : node_(node), device_(nullptr) {}
    KernelImpl(KernelImpl&&) = default;
    KernelImpl& operator=(KernelImpl&&) = default;
    virtual ~KernelImpl() {}

    /** @brief get associated node in the compute graph */
    const ir::Node* GetNode() const {
        return node_;
    }

    /** @brief get kernel's name */
    const std::string& GetName() const {
        return node_->GetName();
    }

    /** @brief get kernel's type */
    const ir::Node::Type& GetType() const {
        return node_->GetType();
    }

    /** @brief set the device that this kernel uses in execution. */
    void SetDevice(Device* d) {
        device_ = d;
    }

    /** @brief get the device where this kernel runs. */
    Device* GetDevice() const {
        return device_;
    }

    /**
       @brief evaluate this op.
       @param ctx contexts needed during execution
    */
    virtual ppl::common::RetCode Execute(KernelExecContext* ctx) = 0;

    /**
       @brief get id of the task queue where this kernel was executed on device. default is 0.
       @note task queue id may be unspecified before Execute() and changed after Execute().
    */
    virtual uint32_t GetTaskQueueId() const {
        return 0;
    }

#ifdef PPLNN_ENABLE_KERNEL_PROFILING
public:
    /** @brief get execution time in microseconds */
    virtual uint64_t GetExecutionTime() const {
        return 0;
    }
#endif

private:
    /** assiciated node in the compute graph */
    const ir::Node* node_;

    /** device which this kernel runs on */
    Device* device_;

private:
    KernelImpl(const KernelImpl&) = delete;
    KernelImpl& operator=(const KernelImpl&) = delete;
};

}} // namespace ppl::nn

#endif
