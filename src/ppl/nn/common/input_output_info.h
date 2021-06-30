#ifndef _ST_HPC_PPL_NN_COMMON_INPUT_OUTPUT_INFO_H_
#define _ST_HPC_PPL_NN_COMMON_INPUT_OUTPUT_INFO_H_

#include "ppl/nn/ir/node.h"
#include "ppl/nn/common/device.h"
#include "ppl/nn/runtime/edge_object.h"
#include <vector>

namespace ppl { namespace nn {

/**
   @class InputOutputInfo
   @brief wrapper for getting input/output tensors of a node/kernel
*/
class InputOutputInfo {
public:
    virtual ~InputOutputInfo() {}

    void SetNode(const ir::Node* node) {
        node_ = node;
    }
    void SetDevice(Device* device) {
        device_ = device;
    }

    /**
       @brief set a getter function which requires an edgeid `eid` and `etype`
       and returns the corresponding object.
    */
    void SetAcquireObjectFunc(
        const std::function<EdgeObject*(edgeid_t eid, uint32_t etype, Device* device)>& acquire_object) {
        acquire_object_func_ = acquire_object;
    }

    uint32_t GetInputCount() const {
        return node_->GetInputCount();
    }

    template <typename T>
    T* GetInput(uint32_t idx) const {
        auto eid = node_->GetInput(idx);
        return static_cast<T*>(acquire_object_func_(eid, EdgeObjectType<T>::value, device_));
    }

    uint32_t GetExtraInputCount() const {
        return node_->GetExtraInputCount();
    }

    template <typename T>
    T* GetExtraInput(uint32_t idx) const {
        auto eid = node_->GetExtraInput(idx);
        return static_cast<T*>(acquire_object_func_(eid, EdgeObjectType<T>::value, device_));
    }

    uint32_t GetOutputCount() const {
        return node_->GetOutputCount();
    }

    template <typename T>
    T* GetOutput(uint32_t idx) const {
        auto eid = node_->GetOutput(idx);
        return static_cast<T*>(acquire_object_func_(eid, EdgeObjectType<T>::value, device_));
    }

protected:
    const ir::Node* node_ = nullptr;
    Device* device_ = nullptr;
    std::function<EdgeObject*(edgeid_t, uint32_t, Device*)> acquire_object_func_;
};

}} // namespace ppl::nn

#endif
