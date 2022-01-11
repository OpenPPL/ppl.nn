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

#include "ppl/kernel/arm_server/conv2d/neon/fp32/n4cx_sgemm/n4cx_sgemm.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

#include "n4cx_sgemm_m4nx_header.inc"
#include "n4cx_sgemm_m8nx_header.inc"

const sgemm_n4cx_kernel_func_t sgemm_n4cx_kernel_m4nx_fp32_func_table[12][3][6] = {
    {{sgemm_n4cx_kernel_m4nx_fp32_func<1, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<1, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<1, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<1, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<2, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<2, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<2, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<2, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<3, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<3, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<3, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<3, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<4, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<4, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<4, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<4, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<5, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<5, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<5, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<5, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<6, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<6, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<6, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<6, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<7, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<7, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<7, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<7, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<8, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<8, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<8, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<8, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<9, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<9, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<9, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<9, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<10, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<10, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<10, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<10, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<11, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<11, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<11, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<11, 2, 6>}},
    {{sgemm_n4cx_kernel_m4nx_fp32_func<12, 0, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 0, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 0, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 0, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 0, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 0, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<12, 1, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 1, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 1, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 1, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 1, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 1, 6>},
     {sgemm_n4cx_kernel_m4nx_fp32_func<12, 2, 0>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 2, 1>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 2, 2>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 2, 4>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 2, 5>,
      sgemm_n4cx_kernel_m4nx_fp32_func<12, 2, 6>}}};

const sgemm_n4cx_kernel_func_t sgemm_n4cx_kernel_m8nx_fp32_func_table[10][3][6] = {
    {{sgemm_n4cx_kernel_m8nx_fp32_func<1, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<1, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<1, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<1, 2, 6>}},
    {{sgemm_n4cx_kernel_m8nx_fp32_func<2, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<2, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<2, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<2, 2, 6>}},
    {{sgemm_n4cx_kernel_m8nx_fp32_func<3, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<3, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<3, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<3, 2, 6>}},
    {{sgemm_n4cx_kernel_m8nx_fp32_func<4, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<4, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<4, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<4, 2, 6>}},
    {{sgemm_n4cx_kernel_m8nx_fp32_func<5, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<5, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<5, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<5, 2, 6>}},
    {{sgemm_n4cx_kernel_m8nx_fp32_func<6, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<6, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<6, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<6, 2, 6>}},
    {{sgemm_n4cx_kernel_m8nx_fp32_func<7, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<7, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<7, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<7, 2, 6>}},
    {{sgemm_n4cx_kernel_m8nx_fp32_func<8, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<8, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<8, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<8, 2, 6>}},
    {{sgemm_n4cx_kernel_m8nx_fp32_func<9, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<9, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<9, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<9, 2, 6>}},
    {{sgemm_n4cx_kernel_m8nx_fp32_func<10, 0, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 0, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 0, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 0, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 0, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 0, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<10, 1, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 1, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 1, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 1, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 1, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 1, 6>},
     {sgemm_n4cx_kernel_m8nx_fp32_func<10, 2, 0>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 2, 1>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 2, 2>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 2, 4>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 2, 5>,
      sgemm_n4cx_kernel_m8nx_fp32_func<10, 2, 6>}}};

#define FUSE_T()   0 // none
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   1 // relu
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   2 // relu6
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   4 // sum
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   5 // sum & relu
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   6 // sum & relu6
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m4nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T

#define FUSE_T()   0 // none
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   1 // relu
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   2 // relu6
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   4 // sum
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   5 // sum & relu
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T
#define FUSE_T()   6 // sum & relu6
#define INIT_T()   0
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   1
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#define INIT_T()   2
#define N_BLOCK0() 12
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 11
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 10
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 9
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 8
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 7
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 6
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 5
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 4
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 3
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 2
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#define N_BLOCK0() 1
#include "n4cx_sgemm_m8nx_kernel.inc"
#undef N_BLOCK0
#undef INIT_T
#undef FUSE_T

}}}}; // namespace ppl::kernel::arm_server::neon
