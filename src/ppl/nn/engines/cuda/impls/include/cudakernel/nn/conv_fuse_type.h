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

#ifndef PPLCUDA_KERNEL_INCLUDE_CONV_FUSE_TYPE_H_
#define PPLCUDA_KERNEL_INCLUDE_CONV_FUSE_TYPE_H_

struct ConvFuse {
    ConvFuse()
    {
        ifReLU           = false;
        ifEltwise        = false;
        ifEltwiseReLU    = false;
        ifConcat         = false;
        ifPReLU          = false;
        ifReLUMax        = false;
        ifEltwiseReLUMax = false;
        ifEltwisePReLU   = false;
        reluMax          = 0.0f;
        eltwiseReluMax   = 0.0f;
        concatOffset     = 0;
        concatStride     = 0;
        preDataGrp       = nullptr;
        concatOutData    = nullptr;
        negeData         = nullptr;
        negeEltData      = nullptr;
    }

    bool ifReLU;
    bool ifEltwise;
    bool ifEltwiseReLU;
    bool ifConcat;
    bool ifPReLU;
    bool ifReLUMax;
    bool ifEltwiseReLUMax;
    bool ifEltwisePReLU;
    float reluMax;
    float eltwiseReluMax;
    int concatOffset;
    int concatStride;
    void* preDataGrp;
    void* concatOutData;
    void* negeData;
    void* negeEltData;
};

#endif // PPLCUDA_KERNEL_INCLUDE_CONV_FUSE_TYPE_H_