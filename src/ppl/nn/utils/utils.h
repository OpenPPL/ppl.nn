#ifndef _ST_HPC_PPL_NN_UTILS_UTILS_H_
#define _ST_HPC_PPL_NN_UTILS_UTILS_H_

#include "ppl/nn/ir/graph.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/runtime/runtime_graph_info.h"
#include <map>

namespace ppl { namespace nn { namespace utils {

/**
   @brief copy buffer to tensor
   @note `dst` MUST have the same dims as `src_shape` and `src_buf` is synchronized
*/
ppl::common::RetCode CopyBuffer(const BufferDesc& src_buf, const TensorShape& src_shape, const Device* src_device,
                                TensorImpl* dst, Device* tmp_cpu_device = nullptr);

/**
   @brief copy one tensor to another
   @note `dst` MUST have the same dims as `src` and `src` is synchronized
*/
static inline ppl::common::RetCode CopyTensorBuffer(const TensorImpl& src, TensorImpl* dst,
                                                    Device* tmp_cpu_device = nullptr) {
    return CopyBuffer(src.GetBufferDesc(), src.GetShape(), src.GetDevice(), dst, tmp_cpu_device);
}

ppl::common::RetCode LoadConstants(const ir::Graph&, Device*, std::map<edgeid_t, RuntimeConstantInfo>*);

ppl::common::RetCode GenericLoadConstant(edgeid_t eid, const ir::Constant& constant, const TensorShape& shape,
                                         Device* device, RuntimeConstantInfo* info);

void IrShape2TensorShape(const ir::Shape&, TensorShape*);

static inline bool IsPplConverterNode(const ir::Node* node) {
    auto& type = node->GetType();
    return (type.name == "Converter" && type.domain == "ppl");
}

static inline ir::Node::Type MakePplConverterNodeType() {
    return ir::Node::Type("ppl", "Converter");
}

}}} // namespace ppl::nn::utils

#endif
