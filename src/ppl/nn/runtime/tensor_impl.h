#ifndef _ST_HPC_PPL_NN_RUNTIME_TENSOR_IMPL_H_
#define _ST_HPC_PPL_NN_RUNTIME_TENSOR_IMPL_H_

#include "ppl/nn/runtime/tensor.h"
#include "ppl/nn/runtime/edge_object.h"
#include "ppl/nn/common/tensor_buffer_info.h"
#include "ppl/nn/common/types.h"

namespace ppl { namespace nn {

class TensorImpl;

template <>
struct EdgeObjectType<TensorImpl> final {
    static const uint32_t value = EdgeObject::T_TENSOR;
};

class TensorImpl final : public EdgeObject, public Tensor {
public:
    TensorImpl(const ir::Edge* edge, tensortype_t t) : EdgeObject(edge, EdgeObjectType<TensorImpl>::value), type_(t) {}

    TensorImpl(TensorImpl&&) = default;
    TensorImpl& operator=(TensorImpl&&) = default;

    const char* GetName() const override {
        return GetEdge()->GetName().c_str();
    }

    tensortype_t GetType() const {
        return type_;
    }

    bool IsBufferOwner() const {
        return buffer_info_.IsBufferOwner();
    }

    ppl::common::RetCode SetDevice(Device* dev) {
        return buffer_info_.SetDevice(dev);
    }

    Device* GetDevice() const {
        return buffer_info_.GetDevice();
    }

    void SetBuffer(const BufferDesc& buf, Device* device = nullptr, bool is_buffer_owner = false) {
        buffer_info_.SetBuffer(buf, device, is_buffer_owner);
    }

    /**
       @brief move buffer from tensor `another`. old buffer of this tensor will be freed(or detached).
       @note this tensor will inherits the ownership of `another`.
    */
    void TransferBufferFrom(TensorImpl* another) {
        buffer_info_.SetBuffer(another->GetBufferDesc(), another->GetDevice(), another->IsBufferOwner());
        another->DetachBuffer();
    }

    BufferDesc DetachBuffer() {
        return buffer_info_.DetachBuffer();
    }

    void FreeBuffer() {
        buffer_info_.FreeBuffer();
    }

    ppl::common::RetCode ReallocBuffer() override {
        return buffer_info_.ReallocBuffer();
    }

    template <typename T = void>
    T* GetBufferPtr() const {
        return buffer_info_.GetBufferPtr<T>();
    }

    BufferDesc& GetBufferDesc() {
        return buffer_info_.GetBufferDesc();
    }
    const BufferDesc& GetBufferDesc() const {
        return buffer_info_.GetBufferDesc();
    }

    TensorShape& GetShape() override {
        return buffer_info_.GetShape();
    }
    const TensorShape& GetShape() const override {
        return buffer_info_.GetShape();
    }

    ppl::common::RetCode CopyToHost(void* dst) const override;
    ppl::common::RetCode CopyFromHost(const void* src) override;

    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc) const override;
    ppl::common::RetCode ConvertFromHost(const void* src, const TensorShape& src_desc) override;

private:
    tensortype_t type_;
    TensorBufferInfo buffer_info_;

private:
    TensorImpl(const TensorImpl&) = delete;
    TensorImpl& operator=(const TensorImpl&) = delete;
};

}} // namespace ppl::nn

#endif
