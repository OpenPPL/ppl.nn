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
#include <fstream>
#include <float.h>
#include <string.h>
#include <chrono>
#include <memory>
#include <inttypes.h>

#if defined(__linux__) && defined(PPL_USE_X86_OMP)
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <pthread.h>
#include <omp.h>
#endif

#include "ppl/kernel/x86/fp32/gemm_v2.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/kernel/x86/common/simd_tools.h"
#include "ppl/kernel/x86/common/internal_include.h"
#include "simple_flags.h"
#include "utils/check.h"

// #define ENABLE_DEBUG_TAG
#ifdef ENABLE_DEBUG_TAG
#define DEBUG_TAG(X) fprintf(stderr, "," #X)
#else
#define DEBUG_TAG(X)
#endif

#define CASE_STRING_FMT() "m%" PRId64 "n%" PRId64 "k%" PRId64 "_trans_A%dtrans_B%d_alpha%fbeta%f_C%d_fuse%d_n%s"

Define_bool_opt("--help", Flag_help, false, "show these help information");
Define_string(cfg, "", "(required) fc config file, format:" CASE_STRING_FMT());
Define_int32(warm_up, 10, "(10) warm up iterations");
Define_int32(min_iter, 20, "(20) min benchmark iterations");
Define_float(min_second, 1.0f, "(1.0) min benchmark seconds");
Define_bool(validate, false, "(false) do result validation");
Define_float(eps, 1e-5f, "(1e-5) rel error trunk for validation");

ppl::common::RetCode gemm_v2_ref_fp32(const ppl::kernel::x86::gemm_v2_param_fp32& param) {
#ifdef PPL_USE_X86_OMP_COLLAPSE
    PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(2)
#else
    PRAGMA_OMP_PARALLEL_FOR()
#endif
    for (int32_t m = 0; m < param.M; m++) {
        for (int32_t n = 0; n < param.N; n++) {
            float sum = 0;
            for (int32_t k = 0; k < param.K; k++) {
                const float a_val = param.trans_A ? param.src_A[k * param.lda + m] : param.src_A[m * param.lda + k];
                const float b_val = param.trans_B ? param.src_B[n * param.ldb + k] : param.src_B[k * param.ldb + n];
                sum += a_val * b_val;
            }
            float c_val = 0;
            if (param.c_type == ppl::kernel::x86::gemm_v2_C_type::SCALAR) {
                c_val = param.src_C[0];
            } else if (param.c_type == ppl::kernel::x86::gemm_v2_C_type::VECTOR_H) {
                c_val = param.src_C[m];
            } else if (param.c_type == ppl::kernel::x86::gemm_v2_C_type::VECTOR_W) {
                c_val = param.src_C[n];
            } else if (param.c_type == ppl::kernel::x86::gemm_v2_C_type::MATRIX) {
                c_val = param.src_C[m * param.ldc + n];
            }
            float result = param.alpha * sum + param.beta * c_val;
            if (param.fuse_flag & ppl::kernel::x86::gemm_v2_fuse_flag::RELU) {
                result = ppl::kernel::x86::max(result, 0.0f);
            }
            param.dst_Y[m * param.ldy + n] = result;
        }
    }

    return ppl::common::RC_SUCCESS;
}

int main(int argc, char **argv) {
    simple_flags::parse_args(argc, argv);
    if (Flag_help) {
        simple_flags::print_args_info();
        return 0;
    }

    ppl::kernel::x86::set_denormals_zero(1);

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
        "num_threads=%d\nwarm_up=%d\nmin_iter=%d\nmin_second=%f\nvalidate=%d\neps=%f\n\n",
        num_threads, Flag_warm_up, Flag_min_iter, Flag_min_second, Flag_validate, Flag_eps
    );
    std::cerr << "==============================================================\n";
    std::cerr << "begin tests\n";
    std::cerr << "line_no,case_string,min_ms,max_gflops,max_gbps,avg_ms,avg_gflops,avg_gbps\n";

    char line[512];
    int line_no = 0;
    int case_no = 0;
    double all_case_gflops = 0.;
    double all_case_gbps = 0.;
    double all_case_us = 0.;

    while (cfgfile.getline(line, 512, '\n')) {
        ++line_no;

        // skip comment
        if (line[0] == '#' || line[0] == '\0') {
            continue;
        }

        char case_name[100];
        int64_t M, N, K;
        int32_t trans_A, trans_B;
        float alpha, beta;
        int32_t c_type;
        int32_t fuse_flag;

        if (10 != sscanf(
            line,
            CASE_STRING_FMT() "\n",
            &M, &N, &K, &trans_A, &trans_B, &alpha, &beta, &c_type, &fuse_flag, case_name
        )) {
            std::cerr << line_no << "," << line << ",invalid format\n";
            continue;
        }

        fprintf(
            stderr,
            "%d," CASE_STRING_FMT(),
            line_no, M, N, K, trans_A, trans_B, alpha, beta, c_type, fuse_flag, case_name
        );
        std::cerr << "\n";
DEBUG_TAG(A);
        const int64_t lda = trans_A ? M : K;
        const int64_t ldb = trans_B ? K : N;
        const int64_t ldy = N;
        const int64_t A_num_elements = M * K;
        const int64_t B_num_elements = K * N;
        const int64_t dst_num_elements = M * N;
        const int64_t A_num_bytes = A_num_elements * sizeof(float);
        const int64_t B_num_bytes = B_num_elements * sizeof(float);
        const int64_t dst_num_bytes = dst_num_elements * sizeof(float);

        int64_t ldc = 0;
        int64_t C_num_elements = 0;
        if (c_type == ppl::kernel::x86::gemm_v2_C_type::SCALAR) {
            C_num_elements = 1;
        } else if (c_type == ppl::kernel::x86::gemm_v2_C_type::VECTOR_H) {
            C_num_elements = M;
        } else if (c_type == ppl::kernel::x86::gemm_v2_C_type::VECTOR_W) {
            C_num_elements = N;
        } else if (c_type == ppl::kernel::x86::gemm_v2_C_type::MATRIX) {
            C_num_elements = M * N;
            ldc = N;
        }
        const int64_t C_num_bytes = C_num_elements * sizeof(float);

        const double gops = (double)M * N * K * 2 / 1e9;
        const double gbs = (double)(A_num_bytes + B_num_bytes + C_num_bytes + dst_num_bytes) / 1e9;
DEBUG_TAG(B);
        ppl::common::GenericCpuAllocator allocator(PPL_X86_CACHELINE_BYTES());

        float* A = (float*)allocator.Alloc(A_num_bytes);
        float* B = (float*)allocator.Alloc(B_num_bytes);
        float* dst = (float*)allocator.Alloc(dst_num_bytes);
        if (!A || !B || !dst) {
            std::cerr << "," << "input & output tensors out of memory\n";
            return -1;
        }
        memset(dst, 0, dst_num_bytes);

        float* C = nullptr;
        if (C_num_bytes > 0) {
            C = (float*)allocator.Alloc(C_num_bytes);
            if (!C) {
                std::cerr << "," << "input tensors out of memory\n";
                return -1;
            }
        }
DEBUG_TAG(C);
        for (int64_t i = 0; i < A_num_elements; i++) {
            A[i] = (float)rand() / INT32_MAX - 0.5f;
        }
        for (int64_t i = 0; i < B_num_elements; i++) {
            B[i] = (float)rand() / INT32_MAX - 0.5f;
        }
        for (int64_t i = 0; i < C_num_elements; i++) {
            C[i] = (float)rand() / INT32_MAX - 0.5f;
        }

        float* dst_ref = nullptr;
        if (Flag_validate) {
            dst_ref = (float*)allocator.Alloc(dst_num_bytes);
            if (!dst_ref) {
                std::cerr << "," << "dst_ref out of memory\n";
                return -1;
            }
            memset(dst_ref, 0, dst_num_bytes);
        }
DEBUG_TAG(D);
        ppl::kernel::x86::gemm_v2_param_fp32 param;
        param.src_A = A;
        param.src_B = B;
        param.src_C = C;
        param.dst_Y = dst;
        param.M = M;
        param.N = N;
        param.K = K;
        param.lda = lda;
        param.ldb = ldb;
        param.ldc = ldc;
        param.ldy = ldy;
        param.alpha = alpha;
        param.beta = beta;
        param.trans_A = trans_A;
        param.trans_B = trans_B;
#ifdef PPL_USE_X86_AVX512
        param.isa_flag = ppl::common::ISA_X86_AVX512;
#else
        param.isa_flag = ppl::common::ISA_X86_FMA;
#endif
        param.fuse_flag = fuse_flag;
        param.c_type = c_type;

        auto executor = std::unique_ptr<ppl::kernel::x86::gemm_v2_executor_fp32>(ppl::kernel::x86::create_gemm_v2_executor_fp32(param));
        if (executor == nullptr) {
            fprintf(stderr, "cannot create executor!\n");
            return -1;
        }
DEBUG_TAG(E);
        const int64_t temp_buffer_bytes = executor->get_buffer_bytes();
        void* temp_buffer = nullptr;
        if (temp_buffer_bytes > 0) {
            temp_buffer = allocator.Alloc(temp_buffer_bytes);
            if (!temp_buffer) {
                std::cerr << "," << "temp_buffer out of memory\n";
                return -1;
            }
        }
        executor->set_temp_buffer(temp_buffer);
DEBUG_TAG(F);
        ppl::common::RetCode ret = executor->execute();
        if (ret != ppl::common::RC_SUCCESS) {
            fprintf(stderr, "execute failed!\n");
            return -1;
        }

        if (Flag_validate) {
            param.dst_Y = dst_ref;
            gemm_v2_ref_fp32(param);
            check_array_error(dst, dst_ref, dst_num_elements, Flag_eps);
        }
DEBUG_TAG(G);
        std::chrono::high_resolution_clock::time_point start;
        std::chrono::high_resolution_clock::time_point end;
        double tot_exe_us = 0.;
        double min_exe_us = DBL_MAX;
        int64_t tot_exe_iter = 0;

        for (; tot_exe_iter < Flag_min_iter || tot_exe_us < Flag_min_second * 1e6; ++tot_exe_iter) {
            start = std::chrono::high_resolution_clock::now();
            executor->execute();
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
        double max_gbps = gbs / (min_exe_us / 1e6);
        double avg_gbps = gbs / (avg_exe_us / 1e6);

        fprintf(stderr, ",%.3f,%.2f,%.2f,%.3f,%.2f,%.2f",
            min_exe_us / 1e3, max_gflops, max_gbps,
            avg_exe_us / 1e3, avg_gflops, avg_gbps);

        ++case_no;
        all_case_gflops += avg_gflops;
        all_case_gbps += avg_gbps;
        all_case_us += avg_exe_us;
DEBUG_TAG(H);
        allocator.Free(A);
        allocator.Free(B);
        allocator.Free(C);
        if (dst_ref) {
            allocator.Free(dst_ref);
        }
        if (temp_buffer) {
            allocator.Free(temp_buffer);
        }
        std::cerr << "\n";
    }
    std::cerr
        << "tot time(ms): "<< all_case_us / 1e3 << "\t"
        << "avg gflops: " << all_case_gflops / case_no  << "\t"
        << "avg gbps: " << all_case_gbps / case_no << "\n";
    cfgfile.close();

    return 0;
}
