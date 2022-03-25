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

#ifndef _ST_HPC_PPL_NN_UTILS_DESTRUCTOR_H_
#define _ST_HPC_PPL_NN_UTILS_DESTRUCTOR_H_

#include <functional>

namespace ppl { namespace nn { namespace utils {

class Destructor final {
public:
    Destructor(const std::function<void()>& f) : f_(f) {}
    Destructor(std::function<void()>&& f) : f_(std::move(f)) {}
    Destructor(Destructor&&) = default;
    Destructor& operator=(Destructor&&) = default;
    ~Destructor() {
        f_();
    }

private:
    std::function<void()> f_;

private:
    Destructor(const Destructor&) = delete;
    Destructor& operator=(const Destructor&) = delete;
};

}}} // namespace ppl::nn::utils

#endif
