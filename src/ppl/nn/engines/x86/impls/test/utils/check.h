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

#ifndef __ST_FP_H_
#define __ST_FP_H_

#include <stdint.h>

template <typename T>
bool check_array_error(T* input, T* ref, uint64_t len, T eps)
{
    for (uint64_t i = 0; i < len; ++i) {
        double err = double(input[i] - ref[i]);
        if (std::abs(err / ref[i]) > eps && std::abs(err) > eps) {
            std::cerr << "error[" << i << "]=" << input[i] << " ref:" << ref[i];
            return false;
        }
    }
    std::cerr << "pass";
    return true;
}

#endif