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

#include <iostream>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include <random>
#include <chrono>

#include <float.h>
#include <string.h>
#include <inttypes.h>

#if defined(__linux__) && defined(PPL_USE_X86_OMP)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <omp.h>
#endif

#include "ppl/kernel/x86/fp32/conv2d.h"
#include "ppl/kernel/x86/fp32/pd_conv2d.h"
#include "ppl/kernel/x86/fp32/reorder.h"
#include "ppl/kernel/x86/common/math.h"
#include "ppl/kernel/x86/common/macros.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/common/tensor_shape.h"
#include "simple_flags.h"
#include "utils/check.h"

// #define ENABLE_DEBUG_TAG
#ifdef ENABLE_DEBUG_TAG
#define DEBUG_TAG(X) fprintf(stderr, "," #X)
#else
#define DEBUG_TAG(X)
#endif

#define CASE_STRING_FMT() \
    "g%" PRId64 \
    "_mb%" PRId64 \
    "_ic%" PRId64 "ih%" PRId64 "iw%" PRId64 \
    "_oc%" PRId64 "oh%" PRId64 "ow%" PRId64 \
    "_kh%" PRId64 "kw%" PRId64 "sh%" PRId64 "sw%" PRId64 "ph%" PRId64 "pw%" PRId64 "dh%" PRId64 "dw%" PRId64 \
    "_kh%" PRId64 "kw%" PRId64 "sh%" PRId64 "sw%" PRId64 "ph%" PRId64 "pw%" PRId64 "dh%" PRId64 "dw%" PRId64 \
    "_n%s"

Define_bool_opt("--help", Flag_help, false, "show these help information");
Define_string(cfg, "", "(required) conv config file, format:" CASE_STRING_FMT());
Define_int32(loop_cfg, 1, "(1) loop config file times");
Define_int32(mb, 0, "(0) custom batch");
Define_int32(warm_up, 2, "(2) warm up iterations");
Define_int32(min_iter, 4, "(4) min benchmark iterations");
Define_float(min_second, 0.5f, "(0.5) min benchmark seconds");
Define_bool(validate, false, "(false) do result validation");
Define_float(eps, 1e-6f, "(1e-6) rel error trunk for validation");
#ifdef PPL_USE_X86_AVX512
Define_bool(disable_avx512, false, "(false) disable avx512 for auto select algo");
#else
static bool Flag_disable_avx512 = true;
#endif

/*

config file format(mkl format):
^BEG
# comment...
# comment...
case strings...\n
case strings...\n
case strings...\n
...\n
^EOF

*/

int main(int argc, char **argv) {
    simple_flags::parse_args(argc, argv);
    if (Flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    int32_t num_threads = 1;
#if defined(__linux__) && defined(PPL_USE_X86_OMP)
    num_threads = omp_get_max_threads();
#pragma omp parallel
    {
#define handle_error_en(en, msg) do { errno = en; perror(msg); exit(EXIT_FAILURE); } while (0)
        int i = omp_get_thread_num();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(i, &cpuset);
        if (int s = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) != 0) {
            handle_error_en(s, "pthread_setaffinity_np");
        }
#undef handle_error_en
    }
#endif

    if (Flag_validate) {
        Flag_warm_up = 0;
        Flag_min_iter = 1;
        Flag_min_second = 0;
    }

    std::cerr << "==============================================================\n";
    fprintf(
        stderr,
        "num_threads=%d\nwarm_up=%d\nmin_iter=%d\nmin_second=%f\nvalidate=%d\neps=%f\n",
        num_threads, Flag_warm_up, Flag_min_iter, Flag_min_second, Flag_validate, Flag_eps
    );

for (int64_t lcfg = 0; lcfg < Flag_loop_cfg; ++lcfg) {

    std::cerr << "==============================================================\n";
    std::cerr << "read config\n";

    std::ifstream cfgfile;
    {
        cfgfile.open(Flag_cfg, std::ios_base::in | std::ios_base::binary);
        if (!cfgfile.is_open()) {
            std::cerr << "cannot open config file\n";
            simple_flags::print_args_info();
            return -1;
        }
    }

    std::cerr << "==============================================================\n";
    std::cerr << "begin tests\n";
    std::cerr << "%line_no,%case_string,%mops,%mbs,%min_ms,%max_gflops,%max_gbps,%avg_ms,%avg_gflops,%avg_gbps,%avg_scv_ms,%avg_sdw_ms,%acc\n";

    char line[512];
    int line_no = 0;
    int case_no = 0;
    double all_case_gflops = 0.;
    double all_case_us = 0.;
    while (cfgfile.getline(line, 512, '\n')) {
        ++line_no;

        // skip comment
        if (line[0] == '#' || line[0] == '\0') {
            continue;
        }

        char case_name[100];
        ppl::kernel::x86::conv2d_param cv_param;
        ppl::kernel::x86::conv2d_param dw_param;
        int64_t batch;
        int64_t src_h;
        int64_t src_w;
        int64_t dst_h;
        int64_t dst_w;
        int64_t cv_dh;
        int64_t cv_dw;
        int64_t dw_dh;
        int64_t dw_dw;
        if (25 != sscanf(
            line,
            CASE_STRING_FMT() "\n",
            &cv_param.group, &batch,
            &cv_param.channels, &src_h, &src_w,
            &cv_param.num_output, &dst_h, &dst_w,
            &cv_param.kernel_h, &cv_param.kernel_w,
            &cv_param.stride_h, &cv_param.stride_w,
            &cv_param.pad_h, &cv_param.pad_w,
            &cv_dh, &cv_dw,
            &dw_param.kernel_h, &dw_param.kernel_w,
            &dw_param.stride_h, &dw_param.stride_w,
            &dw_param.pad_h, &dw_param.pad_w,
            &dw_dh, &dw_dw,
            case_name
        )) {
            std::cerr << line_no << "," << line << ",invalid format\n";
            continue;
        }
        cv_param.dilation_h = cv_dh + 1;
        cv_param.dilation_w = cv_dw + 1;
        cv_param.fuse_flag = 0;

        dw_param.group = cv_param.num_output;
        dw_param.channels = cv_param.num_output;
        dw_param.num_output = cv_param.num_output;
        dw_param.dilation_h = dw_dh + 1;
        dw_param.dilation_w = dw_dw + 1;
        dw_param.fuse_flag = 0;

        if (Flag_mb > 0) {
            batch = Flag_mb;
        }

        fprintf(
            stderr,
            "%d," CASE_STRING_FMT(),
            line_no,
            cv_param.group, batch,
            cv_param.channels, src_h, src_w,
            cv_param.num_output, dst_h, dst_w,
            cv_param.kernel_h, cv_param.kernel_w,
            cv_param.stride_h, cv_param.stride_w,
            cv_param.pad_h, cv_param.pad_w,
            cv_dh, cv_dw,
            dw_param.kernel_h, dw_param.kernel_w,
            dw_param.stride_h, dw_param.stride_w,
            dw_param.pad_h, dw_param.pad_w,
            dw_dh, dw_dw,
            case_name
        );

        const int64_t ext_cv_kernel_h = (cv_param.kernel_h - 1) * cv_param.dilation_h + 1;
        const int64_t ext_cv_kernel_w = (cv_param.kernel_w - 1) * cv_param.dilation_w + 1;
        const int64_t ext_dw_kernel_h = (dw_param.kernel_h - 1) * dw_param.dilation_h + 1;
        const int64_t ext_dw_kernel_w = (dw_param.kernel_w - 1) * dw_param.dilation_w + 1;
        const int64_t assume_inter_h = ((src_h + 2 * cv_param.pad_h - ext_cv_kernel_h) / cv_param.stride_h + 1);
        const int64_t assume_inter_w = ((src_w + 2 * cv_param.pad_w - ext_cv_kernel_w) / cv_param.stride_w + 1);
        const int64_t assume_dst_h = ((assume_inter_h + 2 * dw_param.pad_h - ext_dw_kernel_h) / dw_param.stride_h + 1);
        const int64_t assume_dst_w = ((assume_inter_w + 2 * dw_param.pad_w - ext_dw_kernel_w) / dw_param.stride_w + 1);
        if (dst_h != assume_dst_h || dst_w != assume_dst_w) {
            std::cerr << "," << "dst_h(" << dst_h << ") and dst_w(" << dst_w << ") not match assume(" << assume_dst_h << ", " << assume_dst_w << ")\n";
            continue;
        }

        if (cv_param.channels % cv_param.group != 0 || cv_param.num_output % cv_param.group  != 0) {
            std::cerr << "," << "channels and num_output cannot divide by group\n";
            continue;
        }

DEBUG_TAG(A);
        ppl::common::GenericCpuAllocator allocator(PPL_X86_CACHELINE_BYTES());

        auto isa = ppl::common::GetCpuISA();
        if (Flag_disable_avx512) {
            isa &= ~(ppl::common::ISA_X86_AVX512);
        }

        auto cv_algoinfo = ppl::kernel::x86::conv2d_fp32_algo_selector::select_algo(ppl::common::DATAFORMAT_NDARRAY, cv_param, isa);
        auto dw_algoinfo = ppl::kernel::x86::conv2d_fp32_algo_selector::select_algo(ppl::common::DATAFORMAT_N16CX, dw_param, isa);

        auto pd_conv_algo_info = ppl::kernel::x86::pd_conv2d_algo_selector::select_algo(cv_algoinfo, dw_algoinfo, cv_param, dw_param);
        auto pd_mgr = ppl::kernel::x86::pd_conv2d_algo_selector::gen_algo(cv_param, dw_param, pd_conv_algo_info, &allocator);

        if (pd_conv_algo_info.algo_type == ppl::kernel::x86::pd_conv2d_fp32_algo::UNKNOWN || !pd_mgr) {
            delete pd_mgr;
            std::cerr << "," << "unsupported case: "
                << (pd_conv_algo_info.algo_type == ppl::kernel::x86::pd_conv2d_fp32_algo::UNKNOWN)
                << "," << (!pd_mgr) << "\n";
            continue;
        }
        
DEBUG_TAG(B);

        const int32_t wei_mod = 7;
        const int32_t src_mod = 5;
        const int32_t wei_shift = -3;
        const int32_t src_shift = -2;
        const float wei_scale = Flag_validate ? 1.0 : 0.1;
        const float src_scale = Flag_validate ? 1.0 : 0.1;

        const int64_t ic = cv_param.channels / cv_param.group;
        const int64_t oc = cv_param.num_output / cv_param.group;
        const float gops = 
            (cv_param.group * batch * ic * oc * src_h * src_w +
             dw_param.group * batch * dw_param.kernel_h * dw_param.kernel_w * dst_h * dst_w) * 2.0f / 1e9f;

DEBUG_TAG(C);
        ppl::common::TensorShape src_shape;
        src_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        src_shape.SetDataFormat(ppl::common::DATAFORMAT_N16CX);
        src_shape.Reshape({batch, cv_param.channels, src_h, src_w});

        ppl::common::TensorShape inter_shape;
        inter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        inter_shape.SetDataFormat(ppl::common::DATAFORMAT_N16CX);
        inter_shape.Reshape({batch, cv_param.num_output, assume_inter_h, assume_inter_w});

        ppl::common::TensorShape dst_shape;
        dst_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        dst_shape.SetDataFormat(ppl::common::DATAFORMAT_N16CX);
        dst_shape.Reshape({batch, dw_param.group, dst_h, dst_w});

        ppl::common::TensorShape cv_filter_shape;
        cv_filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        cv_filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        cv_filter_shape.Reshape({cv_param.num_output, cv_param.channels / cv_param.group, cv_param.kernel_h, cv_param.kernel_w});

        ppl::common::TensorShape cv_bias_shape;
        cv_bias_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        cv_bias_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        cv_bias_shape.Reshape({cv_param.num_output});

        ppl::common::TensorShape dw_filter_shape;
        dw_filter_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        dw_filter_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        dw_filter_shape.Reshape({dw_param.group, 1, dw_param.kernel_h, dw_param.kernel_w});

        ppl::common::TensorShape dw_bias_shape;
        dw_bias_shape.SetDataType(ppl::common::DATATYPE_FLOAT32);
        dw_bias_shape.SetDataFormat(ppl::common::DATAFORMAT_NDARRAY);
        dw_bias_shape.Reshape({dw_param.group});

        const float mbs = ((float)src_shape.CalcBytesExcludingPadding() +
                          dst_shape.CalcBytesExcludingPadding() +
                          cv_filter_shape.CalcBytesExcludingPadding() +
                          cv_bias_shape.CalcBytesExcludingPadding() +
                          dw_filter_shape.CalcBytesExcludingPadding() +
                          dw_bias_shape.CalcBytesExcludingPadding()) / 1024 / 1024;

DEBUG_TAG(D);
        float *src = nullptr;
        float *dst = nullptr;
        float *inter = nullptr;
        float *dst_ref = nullptr;
        float *cv_filter = nullptr;
        float *cv_bias = nullptr;
        float *dw_filter = nullptr;
        float *dw_bias = nullptr;
        void *pd_temp_buffer = nullptr;
        void *cv_temp_buffer = nullptr;
        void *dw_temp_buffer = nullptr;
        src = (float*)allocator.Alloc(src_shape.CalcBytesIncludingPadding());
        cv_filter = (float*)allocator.Alloc(cv_filter_shape.CalcBytesIncludingPadding());
        cv_bias = (float*)allocator.Alloc(cv_bias_shape.CalcBytesIncludingPadding());
        dw_filter = (float*)allocator.Alloc(dw_filter_shape.CalcBytesIncludingPadding());
        dw_bias = (float*)allocator.Alloc(dw_bias_shape.CalcBytesIncludingPadding());
        if (!src || !cv_filter || !cv_bias || !dw_filter || !dw_bias) {
            std::cerr << "," << "input tensors out of memory\n";
            return -1;
        }
DEBUG_TAG(E);
        for (uint64_t i = 0; i < cv_filter_shape.CalcElementsIncludingPadding(); ++i) {
            cv_filter[i] = (rand() % wei_mod + wei_shift) * wei_scale;
        }
        for (uint64_t i = 0; i < cv_bias_shape.CalcElementsIncludingPadding(); ++i) {
            cv_bias[i] = (rand() % wei_mod + wei_shift) * wei_scale * 10.0f;
        }
        for (uint64_t i = 0; i < dw_filter_shape.CalcElementsIncludingPadding(); ++i) {
            dw_filter[i] = (rand() % wei_mod + wei_shift) * wei_scale;
        }
        for (uint64_t i = 0; i < dw_bias_shape.CalcElementsIncludingPadding(); ++i) {
            dw_bias[i] = (rand() % wei_mod + wei_shift) * wei_scale * 10.0f;
        }
        for (uint64_t i = 0; i < src_shape.CalcElementsIncludingPadding(); ++i) {
            src[i] = (rand() % src_mod + src_shift) * src_scale;
        }
DEBUG_TAG(G);
        dst = (float*)allocator.Alloc(dst_shape.CalcBytesIncludingPadding());
        if (!dst) {
            std::cerr << "," << "dst out of memory\n";
                return -1;
        }
        memset(dst, 0, dst_shape.CalcBytesIncludingPadding());
DEBUG_TAG(H);
        inter = (float*)allocator.Alloc(inter_shape.CalcBytesIncludingPadding());
        dst_ref = (float*)allocator.Alloc(dst_shape.CalcBytesIncludingPadding());
        if (!inter || !dst_ref) {
            std::cerr << "," << "inter/dst_ref out of memory\n";
            return -1;
        }
        memset(inter, 0, inter_shape.CalcBytesIncludingPadding());
        memset(dst_ref, 0, dst_shape.CalcBytesIncludingPadding());

DEBUG_TAG(J);
        if (ppl::common::RC_SUCCESS != pd_mgr->gen_cvt_weights(cv_filter, cv_bias, dw_filter, dw_bias)) {
            std::cerr << "," << "gen_cvt_weights failed\n";
            return -1;
        }

DEBUG_TAG(K);
        auto pd_exe = pd_mgr->gen_executor();
        pd_exe->set_src_shape(&src_shape);
        pd_exe->set_dst_shape(&dst_shape);

        if (ppl::common::RC_SUCCESS != pd_exe->prepare()) {
            std::cerr << "," << "pd prepare failed\n";
            return -1;
        }
DEBUG_TAG(L);
        const uint64_t pd_temp_buffer_size = pd_exe->cal_temp_buffer_size();
        pd_temp_buffer = allocator.Alloc(pd_temp_buffer_size);
        if (!pd_temp_buffer) {
            std::cerr << "," << "pd_temp_buffer out of memory\n";
                return -1;
        }
        memset(pd_temp_buffer, 0, pd_temp_buffer_size);
        pd_exe->set_temp_buffer(pd_temp_buffer);
DEBUG_TAG(M);
        pd_exe->set_src(src);
        pd_exe->set_dst(dst);

DEBUG_TAG(N);
        for (int32_t i = 0; i < Flag_warm_up; ++i) {
            if (ppl::common::RC_SUCCESS != pd_exe->execute()) {
                std::cerr << "," << "pd execute failed\n";
                return -1;
            }
        }

        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point mid;
        std::chrono::high_resolution_clock::time_point end;
        double tot_exe_us = 0.;
        double min_exe_us = DBL_MAX;
        int64_t tot_exe_iter = 0;

        for (; tot_exe_iter < Flag_min_iter || tot_exe_us < Flag_min_second * 1e6; ++tot_exe_iter) {
            start = std::chrono::high_resolution_clock::now();
            pd_exe->execute();
            end = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e3;
            tot_exe_us += dur;
            if (dur < min_exe_us) {
                min_exe_us = dur;
            }
        }

        double avg_exe_us = tot_exe_us / tot_exe_iter;
        double max_gflops = gops / (min_exe_us / 1e6);
        double avg_gflops = gops / (avg_exe_us / 1e6);
        double max_gbps = mbs / 1024 / (min_exe_us / 1e6);
        double avg_mbs = mbs / 1024 / (avg_exe_us / 1e6);

        auto cv_exe = pd_exe->conv2d_executor();
        auto dw_exe = pd_exe->depthwise_conv2d_executor();

        cv_exe->set_src_shape(&src_shape);
        cv_exe->set_dst_shape(&inter_shape);
        if (ppl::common::RC_SUCCESS != cv_exe->prepare()) {
            std::cerr << "," << "cv prepare failed\n";
            return -1;
        }
        const uint64_t cv_temp_buffer_size = cv_exe->cal_temp_buffer_size();
        cv_temp_buffer = allocator.Alloc(cv_temp_buffer_size);
        if (!cv_temp_buffer) {
            std::cerr << "," << "cv_temp_buffer out of memory\n";
                return -1;
        }
        memset(cv_temp_buffer, 0, cv_temp_buffer_size);
        cv_exe->set_temp_buffer(cv_temp_buffer);
        cv_exe->set_src(src);
        cv_exe->set_dst(inter);

        dw_exe->set_src_shape(&inter_shape);
        dw_exe->set_dst_shape(&dst_shape);
        if (ppl::common::RC_SUCCESS != dw_exe->prepare()) {
            std::cerr << "," << "cv_dw prepare failed\n";
            return -1;
        }
        const uint64_t dw_temp_buffer_size = dw_exe->cal_temp_buffer_size();
        dw_temp_buffer = allocator.Alloc(dw_temp_buffer_size);
        if (!dw_temp_buffer) {
            std::cerr << "," << "dw_temp_buffer out of memory\n";
                return -1;
        }
        memset(dw_temp_buffer, 0, dw_temp_buffer_size);
        dw_exe->set_temp_buffer(dw_temp_buffer);
        dw_exe->set_src(inter);
        dw_exe->set_dst(dst_ref);

        for (int32_t i = 0; i < Flag_warm_up; ++i) {
            if (ppl::common::RC_SUCCESS != cv_exe->execute()) {
                std::cerr << "," << "pd execute failed\n";
                return -1;
            }
            if (ppl::common::RC_SUCCESS != dw_exe->execute()) {
                std::cerr << "," << "pd execute failed\n";
                return -1;
            }
        }

        double sp_exe_us = 0.;
        double cv_exe_us = 0.;
        double dw_exe_us = 0.;
        int64_t sp_exe_iter = 0;

        for (; sp_exe_iter < Flag_min_iter || sp_exe_us < Flag_min_second * 1e6; ++sp_exe_iter) {
            start = std::chrono::high_resolution_clock::now();
            cv_exe->execute();
            mid = std::chrono::high_resolution_clock::now();
            dw_exe->execute();
            end = std::chrono::high_resolution_clock::now();
            double dur = std::chrono::duration_cast<std::chrono::nanoseconds>(mid - start).count() / 1e3;
            cv_exe_us += dur;
            double dw_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - mid).count() / 1e3;
            dw_exe_us += dw_dur;
            double sp_dur = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() / 1e3;
            sp_exe_us += sp_dur;
        }

        sp_exe_us /= sp_exe_iter;
        cv_exe_us /= sp_exe_iter;
        dw_exe_us /= sp_exe_iter;

        fprintf(stderr, ",%.3f,%.3f,%.3f,%.2f,%.2f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f",
            gops * 1000, mbs, min_exe_us / 1e3, max_gflops, max_gbps, avg_exe_us / 1e3, avg_gflops, avg_mbs, cv_exe_us / 1e3, dw_exe_us / 1e3, sp_exe_us / avg_exe_us);

        ++case_no;
        all_case_gflops += avg_gflops;
        all_case_us += avg_exe_us;

DEBUG_TAG(O);
        if (Flag_validate) {
            std::cerr << ",";
            check_array_error(dst, dst_ref, dst_shape.CalcElementsIncludingPadding(), Flag_eps);
        }

DEBUG_TAG(Y);
        pd_mgr->release_cvt_weights();
        if (pd_mgr) delete pd_mgr;
        if (pd_exe) delete pd_exe;
        if (src) allocator.Free(src);
        if (cv_filter) allocator.Free(cv_filter);
        if (cv_bias) allocator.Free(cv_bias);
        if (dw_filter) allocator.Free(dw_filter);
        if (dw_bias) allocator.Free(dw_bias);
        if (dst) allocator.Free(dst);
        if (dst_ref) allocator.Free(dst_ref);
        if (pd_temp_buffer) allocator.Free(pd_temp_buffer);
        if (cv_temp_buffer) allocator.Free(cv_temp_buffer);
        if (dw_temp_buffer) allocator.Free(dw_temp_buffer);
DEBUG_TAG(Z);
        std::cerr << "\n";
    }
    std::cerr << "tot time(ms): " << all_case_us / 1e3 << "\t" << "avg gflops: " << all_case_gflops / case_no << "\n";
    cfgfile.close();
}

}
