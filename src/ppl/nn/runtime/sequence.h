#ifndef _ST_HPC_PPL_NN_RUNTIME_SEQUENCE_H_
#define _ST_HPC_PPL_NN_RUNTIME_SEQUENCE_H_

#include "ppl/nn/runtime/edge_object.h"
#include <vector>

namespace ppl { namespace nn {

template <typename T>
class Sequence final : public EdgeObject {
public:
    Sequence(const ir::Edge* edge) : EdgeObject(edge, EdgeObjectType<Sequence<T>>::value) {}

    uint32_t GetElementCount() const {
        return elements_.size();
    }
    T* GetElement(uint32_t idx) {
        return &elements_[idx];
    }
    const T* GetElement(uint32_t idx) const {
        return &elements_[idx];
    }
    void EmplaceBack(T&& value) {
        elements_.emplace_back(std::move(value));
    }

private:
    std::vector<T> elements_;
};

}} // namespace ppl::nn

#endif
