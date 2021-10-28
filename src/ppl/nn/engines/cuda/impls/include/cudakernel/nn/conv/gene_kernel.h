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

#ifndef __PPLCUDA_CONV_GENE_KERNEL_H_
#define __PPLCUDA_CONV_GENE_KERNEL_H_

#include <string>

#include "ppl/common/types.h"
#include "ppl/common/retcode.h"
#include "cudakernel/nn/conv/conv_fp16.h"

ppl::common::RetCode Gene2spkKernel(std::string& file_res, std::string& kname, int cta_y, int cta_x, int warp_y, int warp_x, int k_size, int s_size, int splitk, int splitf, int buf_size, int declare_times);
ppl::common::RetCode GeneIdxnKernel(std::string& file_res, std::string& kname, int cta_y, int cta_x, int warp_y, int warp_x, int k_size, int s_size, int declare_times);
ppl::common::RetCode ReplaceFusionFor2spk(std::string& file_res, fuse_info_t fuse_info);
ppl::common::RetCode ReplaceFusionForIdxn(std::string& file_res, fuse_info_t fuse_info);

#endif
