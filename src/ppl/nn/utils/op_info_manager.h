// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#ifndef _ST_HPC_PPL_NN_UTILS_OP_INFO_MANAGER_H
#define _ST_HPC_PPL_NN_UTILS_OP_INFO_MANAGER_H

#include "ppl/common/retcode.h"
#include "ppl/nn/common/logger.h"
#include <map>
#include <string>
#include <vector>

namespace ppl { namespace nn { namespace utils {

/** [first, last] included */
struct VersionRange final {
    VersionRange(uint64_t f = 0, uint64_t l = 0) : first(f), last(l) {}
    uint64_t first;
    uint64_t last;
};

template <typename T>
class OpInfoManager final {
public:
    ppl::common::RetCode Register(const std::string& domain, const std::string& type, const VersionRange& ver,
                                  const T& item) {
        auto& versions = info_[domain][type];
        auto iter = versions.begin();
        for (; iter != versions.end(); ++iter) {
            auto& cur = iter->first;
            if (ver.first < cur.first) {
                if (ver.last < cur.first) {
                    /*
                      [ver.first, ver.last], [cur.first, cur.last]
                    */
                    break;
                }

                LOG(ERROR) << "Register for [" << domain << ", " << type << "] failed: VersionRange[" << ver.first
                           << ", " << ver.last << "] overlapped with [" << cur.first << ", " << cur.last << "]";
                return ppl::common::RC_EXISTS;
            } else if (ver.first <= cur.last) {
                LOG(ERROR) << "Register for [" << domain << ", " << type << "] failed: VersionRange[" << ver.first
                           << ", " << ver.last << "] overlapped with [" << cur.first << ", " << cur.last << "]";
                return ppl::common::RC_EXISTS;
            }
        }

        versions.insert(iter, std::make_pair(ver, item));
        return ppl::common::RC_SUCCESS;
    }

    const T* Find(const std::string& domain, const std::string& type, uint64_t version) const {
        auto domain_iter = info_.find(domain);
        if (domain_iter == info_.end()) {
            return nullptr;
        }

        auto type_iter = domain_iter->second.find(type);
        if (type_iter == domain_iter->second.end()) {
            return nullptr;
        }

        for (auto ver_iter = type_iter->second.begin(); ver_iter != type_iter->second.end(); ++ver_iter) {
            auto& cur_ver = ver_iter->first;
            if (cur_ver.first <= version && version <= cur_ver.last) {
                return &ver_iter->second;
            }
        }

        return nullptr;
    }

    void Remove(const std::string& domain, const std::string& type) {
        auto domain_iter = info_.find(domain);
        if (domain_iter != info_.end()) {
            auto& type_info = domain_iter->second;
            type_info.erase(type);
            if (type_info.empty()) {
                info_.erase(domain_iter);
            }
        }
    }

private:
    std::map<std::string, std::map<std::string, std::vector<std::pair<VersionRange, T>>>> info_;
};

}}} // namespace ppl::nn::utils

#endif
