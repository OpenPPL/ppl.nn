#ifndef _ST_HPC_PPL_NN_MODELS_OP_INFO_MANAGER_H
#define _ST_HPC_PPL_NN_MODELS_OP_INFO_MANAGER_H

#include "ppl/common/retcode.h"
#include "ppl/nn/ir/graph.h"

namespace ppl { namespace nn {

typedef bool (*ParamEqualFunc)(void* param_0, void* param_1);

struct OpInfo {
    ParamEqualFunc param_equal;
};

class OpInfoManager {
public:
    static OpInfoManager* Instance() {
        static OpInfoManager mgr;
        return &mgr;
    }

    const OpInfo* Find(const std::string& domain, const std::string& op_type) const;
    void Register(const std::string& domain, const std::string& op_type, const OpInfo&);

private:
    std::map<std::string, std::map<std::string, OpInfo>> info_;
};

}} // namespace ppl::nn

#endif
