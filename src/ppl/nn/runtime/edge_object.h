#ifndef _ST_HPC_PPL_NN_RUNTIME_EDGE_OBJECT_H_
#define _ST_HPC_PPL_NN_RUNTIME_EDGE_OBJECT_H_

#include "ppl/nn/ir/edge.h"

namespace ppl { namespace nn {

class EdgeObject {
public:
    /** EdgeObject types */
    enum {
        T_UNKNOWN,
        T_EDGE_OBJECT,
        T_TENSOR,
        T_TENSOR_SEQUENCE,
    };

public:
    EdgeObject(const ir::Edge* edge, uint32_t type) : edge_(edge), type_(type) {}
    virtual ~EdgeObject() {}
    EdgeObject(EdgeObject&&) = default;
    EdgeObject& operator=(EdgeObject&&) = default;
    EdgeObject(const EdgeObject&) = default;
    EdgeObject& operator=(const EdgeObject&) = default;

    const ir::Edge* GetEdge() const {
        return edge_;
    }
    uint32_t GetObjectType() const {
        return type_;
    }

private:
    const ir::Edge* edge_;
    uint32_t type_;
};

template <typename T>
struct EdgeObjectType final {
    static const uint32_t value = EdgeObject::T_UNKNOWN;
};

template <>
struct EdgeObjectType<EdgeObject> final {
    static const uint32_t value = EdgeObject::T_EDGE_OBJECT;
};

}} // namespace ppl::nn

#endif
