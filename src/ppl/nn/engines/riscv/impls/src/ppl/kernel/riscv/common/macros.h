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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_MACROS_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_MACROS_H_

#define PPL_RISCV_PRAGMA(X) _Pragma(#X)

#ifdef PPL_USE_RISCV_OMP
#include <omp.h>
#define PRAGMA_OMP_PARALLEL_FOR_SCHEDULE(TYPE) PPL_RISCV_PRAGMA(omp parallel for schedule(TYPE))
#define PRAGMA_OMP_PARALLEL_FOR() PPL_RISCV_PRAGMA(omp parallel for)
#define PRAGMA_OMP_PARALLEL() PPL_RISCV_PRAGMA(omp parallel)
#define PRAGMA_OMP_FOR_NOWAIT() PPL_RISCV_PRAGMA(omp for nowait)
#define PRAGMA_OMP_FOR() PPL_RISCV_PRAGMA(omp for)
#define PRAGMA_OMP_SINGLE()   PPL_RISCV_PRAGMA(omp single)
#define PPL_OMP_NUM_THREADS() omp_get_num_threads()
#define PPL_OMP_MAX_THREADS() omp_get_max_threads()
#define PPL_OMP_THREAD_ID()   omp_get_thread_num()
#else
#define PRAGMA_OMP_PARALLEL_FOR_SCHEDULE(TYPE)
#define PRAGMA_OMP_PARALLEL_FOR()
#define PRAGMA_OMP_PARALLEL()
#define PRAGMA_OMP_FOR_NOWAIT()
#define PRAGMA_OMP_FOR()
#define PRAGMA_OMP_SINGLE()
#define PPL_OMP_NUM_THREADS() 1
#define PPL_OMP_MAX_THREADS() 1
#define PPL_OMP_THREAD_ID()   0
#endif

#define PPL_RISCV_TENSOR_MAX_DIMS() 8

#endif //  __ST_PPL_KERNEL_RISCV_COMMON_MACROS_H_
