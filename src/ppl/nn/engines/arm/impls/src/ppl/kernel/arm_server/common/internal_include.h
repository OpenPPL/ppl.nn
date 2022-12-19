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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_COMMON_INTERNAL_INCLUDE_H_
#define __ST_PPL_KERNEL_ARM_SERVER_COMMON_INTERNAL_INCLUDE_H_

#include "ppl/kernel/arm_server/common/math.h"
#include "ppl/kernel/arm_server/common/general_include.h"

#define CEIL2(val)   ((val + 1) & (~1))
#define CEIL4(val)   ((val + 3) & (~3))
#define CEIL8(val)   ((val + 7) & (~7))
#define CEIL16(val)  ((val + 15) & (~15))
#define CEIL128(val) ((val + 127) & (~127))

#define FLOOR2(val)  ((val) & (~1))
#define FLOOR4(val)  ((val) & (~3))
#define FLOOR8(val)  ((val) & (~7))
#define FLOOR16(val) ((val) & (~15))

#define CEIL(aval, bval) ( (aval + bval - 1) / bval * bval )
#define FLOOR(aval, bval) ( aval / bval * bval )

#define DIV_CEIL(aval, bval) ( (aval + bval - 1) / bval )
#define DIV_FLOOR(aval, bval) ( aval / bval )

#define LLC_CACHELINE_SIZE() 128

#ifdef PPL_USE_ARM_SERVER_OMP
#include <omp.h>
#define PPL_ARM_PRAGMA(X)                                _Pragma(#X)
#define PRAGMA_OMP_PARALLEL()                            PPL_ARM_PRAGMA(omp parallel)
#define PRAGMA_OMP_PARALLEL_FOR()                        PPL_ARM_PRAGMA(omp parallel for)
#define PRAGMA_OMP_PARALLEL_FOR_SCHEDULE(TYPE)           PPL_ARM_PRAGMA(omp parallel for schedule(TYPE))
#define PRAGMA_OMP_PARALLEL_FOR_SCHEDULE_CHUNK(TYPE, S)  PPL_ARM_PRAGMA(omp parallel for schedule(TYPE, S))
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(LEVEL)          PPL_ARM_PRAGMA(omp parallel for collapse(LEVEL))
#define PRAGMA_OMP_FOR()                                 PPL_ARM_PRAGMA(omp for)
#define PRAGMA_OMP_FOR_NOWAIT()                          PPL_ARM_PRAGMA(omp for nowait)
#define PRAGMA_OMP_FOR_SCHEDULE(TYPE)                    PPL_ARM_PRAGMA(omp for schedule(TYPE))
#define PRAGMA_OMP_FOR_SCHEDULE_CHUNK(TYPE, S)           PPL_ARM_PRAGMA(omp for schedule(TYPE, S))
#define PRAGMA_OMP_FOR_COLLAPSE(LEVEL)                   PPL_ARM_PRAGMA(omp for collapse(LEVEL))
#define PRAGMA_OMP_FOR_COLLAPSE_NOWAIT(LEVEL)            PPL_ARM_PRAGMA(omp for collapse(LEVEL) nowait)
#define PRAGMA_OMP_BARRIER()                             PPL_ARM_PRAGMA(omp barrier)
#define PRAGMA_OMP_SINGLE()                              PPL_ARM_PRAGMA(omp single)
#define PRAGMA_OMP_SINGLE_NOWAIT()                       PPL_ARM_PRAGMA(omp single nowait)
#define PPL_OMP_NUM_THREADS()                            omp_get_num_threads()
#define PPL_OMP_MAX_THREADS()                            omp_get_max_threads()
#define PPL_OMP_THREAD_ID()                              omp_get_thread_num()
#else
#define PRAGMA_OMP_PARALLEL()
#define PRAGMA_OMP_PARALLEL_FOR()
#define PRAGMA_OMP_PARALLEL_FOR_SCHEDULE(TYPE)
#define PRAGMA_OMP_PARALLEL_FOR_SCHEDULE_CHUNK(TYPE, S)
#define PRAGMA_OMP_FOR()
#define PRAGMA_OMP_FOR_NOWAIT()
#define PRAGMA_OMP_FOR_SCHEDULE(TYPE)
#define PRAGMA_OMP_FOR_SCHEDULE_CHUNK(TYPE, S)
#define PRAGMA_OMP_FOR_COLLAPSE(LEVEL)
#define PRAGMA_OMP_FOR_COLLAPSE_NOWAIT(LEVEL)
#define PRAGMA_OMP_PARALLEL_FOR_COLLAPSE(LEVEL)
#define PRAGMA_OMP_BARRIER()
#define PRAGMA_OMP_SINGLE()
#define PRAGMA_OMP_SINGLE_NOWAIT()
#define PPL_OMP_NUM_THREADS() 1
#define PPL_OMP_MAX_THREADS() 1
#define PPL_OMP_THREAD_ID()   0
#endif

#endif
