#include "ppl/nn/models/op_info_manager.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

const OpInfo* OpInfoManager::Find(const string& domain, const string& op_type) const {
    auto type_ref = info_.find(domain);
    if (type_ref != info_.end()) {
        auto ref = type_ref->second.find(op_type);
        if (ref != type_ref->second.end()) {
            return &(ref->second);
        }
    }
    return nullptr;
}

void OpInfoManager::Register(const string& domain, const string& op_type, const OpInfo& info) {
    info_[domain][op_type] = info;
}

}} // namespace ppl::nn
