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

#ifndef __PPLCUDA_INIT_LUT_H__
#define __PPLCUDA_INIT_LUT_H__

#define MAX_LUT_SIZE 128

#define MAX_SPLITK_SIZE 8

struct lut_t {
    int idx[MAX_LUT_SIZE];
    lut_t(){};
};

void InitializeInputLut(
    int& in_lut_size,
    int* in_lut,
    int flt_height,
    int flt_width,
    int in_height,
    int in_width,
    int pad_height,
    int pad_width,
    int hole_height,
    int hole_width,
    int num_chl_per_grp_pad,
    int num_grp,
    int num_chl_per_step,
    int pad_size);

void InitializeFilterLut(
    int& flt_lut_size,
    int* flt_lut,
    int flt_height,
    int flt_width,
    int num_chl_per_grp_pad,
    int num_chl_per_step,
    int pad_size);

void InitializeNumChlPerSpk(
    int& num_chl_per_spk_head,
    int& num_chl_per_spk_tail,
    int num_chl,
    int num_grp,
    int pad_size,
    int num_chl_per_step,
    int splitk);

#endif
