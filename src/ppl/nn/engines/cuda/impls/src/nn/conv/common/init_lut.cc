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

#include <stdlib.h>
#include <stdio.h>

#include "cudakernel/common/common.h"
#include "common/init_lut.h"

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
    int in_chl_per_step,
    int pad_size)
{
    int flt_size           = flt_height * flt_width;
    int in_chl             = (num_chl_per_grp_pad / pad_size) * num_grp;
    int in_chl_per_step_v8 = in_chl_per_step / pad_size;

    in_lut_size = flt_size + 1;

    int* tmp_lut = (int*)malloc(in_lut_size * sizeof(int));

    for (int i = 0; i < in_lut_size; i++) {
        int flt_c = i / flt_size;
        int flt_h = (i % flt_size) / flt_width;
        int flt_w = (i % flt_size) % flt_width;

        int in_c = flt_c;
        int in_h = flt_h * hole_height;
        int in_w = flt_w * hole_width;

        int lut_id = flt_c * flt_width * flt_height + flt_h * flt_width + flt_w;

        tmp_lut[lut_id] = in_h * in_chl * in_width + in_w * in_chl +
                          in_c * in_chl_per_step_v8 - (pad_height * in_width + pad_width) * in_chl;
    }

    in_lut[0] = tmp_lut[0];
    for (int lut_id = 1; lut_id < in_lut_size; lut_id++) {
        in_lut[lut_id] = tmp_lut[lut_id] - tmp_lut[lut_id - 1];
    }

    free(tmp_lut);
}

void InitializeFilterLut(
    int& flt_lut_size,
    int* flt_lut,
    int flt_height,
    int flt_width,
    int num_chl_per_grp_pad,
    int flt_chl_per_step,
    int pad_size)
{
    int flt_size            = flt_height * flt_width;
    int flt_chl             = num_chl_per_grp_pad / pad_size;
    int flt_chl_per_step_v8 = flt_chl_per_step / pad_size;

    flt_lut_size = flt_size + 1;

    int* tmp_lut = (int*)malloc(flt_lut_size * sizeof(int));

    for (int i = 0; i < flt_lut_size; i++) {
        int flt_c = i / flt_size;
        int flt_h = (i % flt_size) / flt_width;
        int flt_w = (i % flt_size) % flt_width;

        int lut_id = flt_c * flt_width * flt_height + flt_h * flt_width + flt_w;

        tmp_lut[lut_id] = flt_h * flt_chl * flt_width + flt_w * flt_chl + flt_c * flt_chl_per_step_v8;
    }

    flt_lut[0] = tmp_lut[0];
    for (int lut_id = 1; lut_id < flt_lut_size; lut_id++) {
        flt_lut[lut_id] = tmp_lut[lut_id] - tmp_lut[lut_id - 1];
    }

    free(tmp_lut);
}

void InitializeAbsChlLut(
    int& abs_chl_lut_size,
    int* abs_chl_lut,
    int num_chl,
    int num_grp,
    int pad_size,
    int num_chl_per_step,
    int splitk)
{
    abs_chl_lut_size = splitk;

    int num_chl_per_grp              = num_chl / num_grp;
    int num_chl_per_grp_pad          = Align(num_chl_per_grp, pad_size);
    int num_chl_per_grp_pad_step     = Align(num_chl_per_grp, num_chl_per_step);
    int num_chl_per_grp_pad_per_step = num_chl_per_grp_pad_step / num_chl_per_step;

    for (int i = 0; i < splitk; i++)
        abs_chl_lut[i] = num_chl_per_grp_pad_per_step / splitk;

    for (int i = 0; i < (num_chl_per_grp_pad_per_step % splitk); i++)
        abs_chl_lut[i] += 1;

    for (int i = 0; i < splitk; i++)
        abs_chl_lut[i] = abs_chl_lut[i] * num_chl_per_step;

    abs_chl_lut[splitk - 1] = abs_chl_lut[splitk - 1] - (num_chl_per_grp_pad_step - num_chl_per_grp_pad);
}

void InitializeChlLut(
    int& chl_lut_size,
    int* chl_lut,
    int num_chl,
    int num_grp,
    int pad_size,
    int num_chl_per_step,
    int splitk)
{
    chl_lut_size = splitk + 1;

    int abs_chl_lut_size;
    int abs_chl_lut[MAX_SPLITK_SIZE];

    InitializeAbsChlLut(abs_chl_lut_size, abs_chl_lut, num_chl, num_grp, pad_size, num_chl_per_step, splitk);

    chl_lut[0] = 0;
    for (int i = 1; i < splitk + 1; i++)
        chl_lut[i] = chl_lut[i - 1] + abs_chl_lut[i - 1];
}

void InitializeKloopLut(
    int& kloop_lut_size,
    int* kloop_lut,
    int num_chl,
    int num_grp,
    int pad_size,
    int num_chl_per_step,
    int splitk,
    int splitf,
    int flt_hw)
{
    kloop_lut_size = splitk;

    int abs_chl_lut_size;
    int abs_chl_lut[MAX_SPLITK_SIZE];

    InitializeAbsChlLut(abs_chl_lut_size, abs_chl_lut, num_chl, num_grp, pad_size, num_chl_per_step, splitk);

    for (int i = 0; i < splitk; i++) {
        kloop_lut[i] = ((splitf == 1) ? flt_hw : 1) * DivUp(abs_chl_lut[i], num_chl_per_step);
    }
}
