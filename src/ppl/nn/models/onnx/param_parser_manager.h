#ifndef _ST_HPC_PPL_NN_MODELS_ONNX_PARAM_PARSER_MANAGER_H
#define _ST_HPC_PPL_NN_MODELS_ONNX_PARAM_PARSER_MANAGER_H

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/graph.h"
#include "ppl/nn/models/onnx/generated/onnx.pb.h"

namespace ppl { namespace nn { namespace onnx {

typedef void* (*CreateParamFunc)();
typedef ppl::common::RetCode (*ParseParamFunc)(const ::onnx::NodeProto&, void* param, ir::Node*, ir::GraphTopo*);
typedef void (*DeleteParamFunc)(void* param);

struct ParserInfo {
    CreateParamFunc create_param;
    ParseParamFunc parse_param;
    DeleteParamFunc destroy_param;
};

class ParamParserManager {
public:
    static ParamParserManager* Instance() {
        static ParamParserManager mgr;
        return &mgr;
    }

    void Register(const std::string& domain, const std::string& op_type, const ParserInfo&);
    const ParserInfo* Find(const std::string& domain, const std::string& op_type) const;

private:
    std::map<std::string, std::map<std::string, ParserInfo>> domain_type_parser_;

private:
    ParamParserManager();
};

}}} // namespace ppl::nn::onnx

#endif
