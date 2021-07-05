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

#ifndef _ST_HPC_PPL_NN_UTILS_VECTOR_UTILS_H_
#define _ST_HPC_PPL_NN_UTILS_VECTOR_UTILS_H_

#include <vector>
#include <functional>
#include <algorithm>

namespace ppl { namespace nn { namespace utils {

template <typename T, typename UnaryPredicate>
void VectorRemoveAllIf(std::vector<T>& v, const UnaryPredicate& p) {
    v.erase(remove_if(v.begin(), v.end(), p), v.end());
}

template <typename T, typename UnaryPredicate>
void VectorRemoveOneIf(std::vector<T>& v, const UnaryPredicate& p) {
    for (uint32_t i = 0; i < v.size(); ++i) {
        if (p(v[i])) {
            for (uint32_t j = i + 1; j < v.size(); ++j) {
                v[j - 1] = std::move(v[j]);
            }
            v.pop_back();
            return;
        }
    }
}

template <typename T>
uint32_t VectorFind(const std::vector<T>& vec, const T& value) {
    for (uint32_t i = 0; i < vec.size(); ++i) {
        if (vec[i] == value) {
            return i;
        }
    }
    return vec.size();
}

template <typename T>
uint32_t VectorAddUnique(std::vector<T>& vec, const T& value) {
    auto idx = VectorFind(vec, value);
    if (idx == vec.size()) {
        vec.push_back(value);
        return vec.size() - 1;
    }

    return idx;
}

template <typename T>
class VecIter {
private:
    static bool DefaultFilter(const T&) {
        return true;
    }

public:
    VecIter(const std::vector<T>* vec) : idx_(0), vec_(vec), filter_(DefaultFilter) {}
    VecIter(const VecIter&) = default;
    VecIter(VecIter&&) = default;
    VecIter& operator=(const VecIter&) = default;
    VecIter& operator=(VecIter&&) = default;

    void Reset(const std::function<bool(const T&)>& f) {
        filter_ = f;
        idx_ = 0;
        while (idx_ < vec_->size() && !f(vec_->at(idx_))) {
            ++idx_;
        }
    }

    bool IsValid() const {
        return idx_ < vec_->size();
    }
    const T& Get() const {
        return vec_->at(idx_);
    }

    void Forward() {
        while (true) {
            ++idx_;
            if (idx_ >= vec_->size()) {
                return;
            }
            if (filter_(vec_->at(idx_))) {
                return;
            }
        }
    }

private:
    uint32_t idx_;
    const std::vector<T>* vec_;
    std::function<bool(const T&)> filter_;
};

}}} // namespace ppl::nn::utils

#endif
