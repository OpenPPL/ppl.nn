#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_MODEL_PARSER_H_
#define _ST_HPC_PPL_NN_MODELS_ONNX_MODEL_PARSER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/graph.h"

namespace ppl { namespace nn { namespace onnx {

class ModelParser final {
public:
    static ppl::common::RetCode Parse(const char* model_buf, size_t buf_len, ir::Graph* graph);

    static ppl::common::RetCode Parse(const char* model_file, ir::Graph* graph);
};

}}} // namespace ppl::nn::onnx

#endif
