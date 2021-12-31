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

#include "ppl/nn/engines/riscv/utils/data_trans.h"
#include "ppl/nn/engines/riscv/utils/fp16fp32_cvt.h"
#include <cstring>
#include "ppl/common/log.h"

inline static void transpose_8x8(const __fp16 input[64], __fp16 output[64]) {
    for (int64_t i = 0; i < 8; i += 1) {
        for (int64_t j = 0; j < 8; j += 1) {
            output[j * 8 + i] = input[i * 8 + j];
        }
    }
}

void N8cxToNdarrayFp16(const __fp16* src, int64_t n, int64_t c, int64_t h, int64_t w, __fp16* dst) {
    int64_t pad_c = (c + 8 - 1) / 8 * 8;
    int64_t hw = h * w;
    for (int64_t i = 0; i < n; i += 1) {
        for (int64_t j = 0; j < c; j += 1) {
            for (int64_t k = 0; k < hw; k += 1) {
                int64_t src_idx = i * pad_c * hw + (j / 8) * hw * 8 + k * 8 + (j % 8);
                int64_t dst_idx = i * c * hw + j * hw + k;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

void NdarrayToN8cxFp16(const __fp16* src, int64_t n, int64_t c, int64_t h, int64_t w, __fp16* dst) {
    int64_t pad_c = (c + 8 - 1) / 8 * 8;
    int64_t hw = h * w;

    for (int64_t i = 0; i < n; i += 1) {
        for (int64_t j = 0; j < pad_c; j += 1) {
            if (j >= c) {
                for (int64_t k = 0; k < hw; k += 1) {
                    int64_t dst_idx = i * pad_c * hw + (j / 8) * hw * 8 + k * 8 + (j % 8);
                    dst[dst_idx] = 0.f;
                }
            } else {
                for (int64_t k = 0; k < hw; k += 1) {
                    int64_t src_idx = i * c * hw + j * hw + k;
                    int64_t dst_idx = i * pad_c * hw + (j / 8) * hw * 8 + k * 8 + (j % 8);
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

void N8cxToNdarrayFp32(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst) {
    int64_t pad_c = (c + 8 - 1) / 8 * 8;
    int64_t hw = h * w;
    for (int64_t i = 0; i < n; i += 1) {
        for (int64_t j = 0; j < c; j += 1) {
            for (int64_t k = 0; k < hw; k += 1) {
                int64_t src_idx = i * pad_c * hw + (j / 8) * hw * 8 + k * 8 + (j % 8);
                int64_t dst_idx = i * c * hw + j * hw + k;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

void N4cxToNdarrayFp32(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst) {
    const int64_t atom_c = 4;
    int64_t pad_c = (c + atom_c - 1) / atom_c * atom_c;
    int64_t hw = h * w;
    for (int64_t i = 0; i < n; i += 1) {
        for (int64_t j = 0; j < c; j += 1) {
            for (int64_t k = 0; k < hw; k += 1) {
                int64_t src_idx = i * pad_c * hw + (j / atom_c) * hw * atom_c + k * atom_c + (j % atom_c);
                int64_t dst_idx = i * c * hw + j * hw + k;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

void NdarrayToN8cxFp32(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst) {
    int64_t pad_c = (c + 8 - 1) / 8 * 8;
    int64_t hw = h * w;

    for (int64_t i = 0; i < n; i += 1) {
        for (int64_t j = 0; j < pad_c; j += 1) {
            if (j >= c) {
                for (int64_t k = 0; k < hw; k += 1) {
                    int64_t dst_idx = i * pad_c * hw + (j / 8) * hw * 8 + k * 8 + (j % 8);
                    dst[dst_idx] = 0.f;
                }
            } else {
                for (int64_t k = 0; k < hw; k += 1) {
                    int64_t src_idx = i * c * hw + j * hw + k;
                    int64_t dst_idx = i * pad_c * hw + (j / 8) * hw * 8 + k * 8 + (j % 8);
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

void NdarrayToN4cxFp32(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst) {
    const int64_t atom_c = 4;

    int64_t pad_c = (c + atom_c - 1) / atom_c * atom_c;
    int64_t hw = h * w;

    for (int64_t i = 0; i < n; i += 1) {
        for (int64_t j = 0; j < pad_c; j += 1) {
            if (j >= c) {
                for (int64_t k = 0; k < hw; k += 1) {
                    int64_t dst_idx = i * pad_c * hw + (j / atom_c) * hw * atom_c + k * atom_c + (j % atom_c);
                    dst[dst_idx] = 0.f;
                }
            } else {
                for (int64_t k = 0; k < hw; k += 1) {
                    int64_t src_idx = i * c * hw + j * hw + k;
                    int64_t dst_idx = i * pad_c * hw + (j / atom_c) * hw * atom_c + k * atom_c + (j % atom_c);
                    dst[dst_idx] = src[src_idx];
                }
            }
        }
    }
}

// TODO Optimize
void N8cxFp16ToNdarrayFp32(const __fp16* src, int64_t n, int64_t c, int64_t h, int64_t w, float* dst) {
    int64_t pad_c = (c + 8 - 1) / 8 * 8;
    int64_t hw = h * w;

    for (int64_t i = 0; i < n; i += 1) {
        for (int64_t j = 0; j < c; j += 1) {
            for (int64_t k = 0; k < hw; k += 1) {
                int64_t src_idx = i * pad_c * hw + (j / 8) * hw * 8 + k * 8 + (j % 8);
                int64_t dst_idx = i * c * hw + j * hw + k;
                dst[dst_idx] = (float)(src[src_idx]);
            }
        }
    }
}

// TODO Optimize
void NdarrayFp32ToN8cxFp16(const float* src, int64_t n, int64_t c, int64_t h, int64_t w, __fp16* dst) {
    int64_t pad_c = (c + 8 - 1) / 8 * 8;
    int64_t hw = h * w;

    for (int64_t i = 0; i < n; i += 1) {
        for (int64_t j = 0; j < pad_c; j += 1) {
            if (j >= c) {
                for (int64_t k = 0; k < hw; k += 1) {
                    int64_t dst_idx = i * pad_c * hw + (j / 8) * hw * 8 + k * 8 + (j % 8);
                    dst[dst_idx] = 0.f;
                }
            } else {
                for (int64_t k = 0; k < hw; k += 1) {
                    int64_t src_idx = i * c * hw + j * hw + k;
                    int64_t dst_idx = i * pad_c * hw + (j / 8) * hw * 8 + k * 8 + (j % 8);
                    dst[dst_idx] = (__fp16)(src[src_idx]);
                }
            }
        }
    }
}
