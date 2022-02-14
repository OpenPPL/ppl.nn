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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_MMCV_GRIDSAMPLE_MMCV_GRIDSAMPLE_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_MMCV_GRIDSAMPLE_MMCV_GRIDSAMPLE_H_

#include <cmath>
#include "ppl/kernel/riscv/common/internal_include.h"

namespace ppl { namespace kernel { namespace riscv {

enum grid_sample_interpolation {
    BILINEAR = 0,
    NEAREST  = 1,
    BICUBIC  = 2
};

enum grid_sample_padding {
    ZEROS      = 0,
    BORDER     = 1,
    REFLECTION = 2
};

template <typename T, bool align_corners>
static inline T unnormalize(
    const T coord,
    const int64_t size)
{
    if (align_corners) {
        return ((coord + 1) / 2) * (size - 1);
    } else {
        return ((coord + 1) * size - 1) / 2;
    }
}

// Clip coordinates to between 0 and (clip_limit - 1)
template <typename T>
static inline T clip_coordinates(
    const T in,
    const int64_t clip_limit)
{
    return min(static_cast<T>(clip_limit - 1), max(in, static_cast<T>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values can be represented as ints.
template <typename T>
static inline T reflect_coordinates(
    const T in,
    const int64_t twice_low,
    const int64_t twice_high)
{
    if (twice_high == twice_low) {
        return static_cast<T>(0);
    }
    T min  = static_cast<T>(twice_low) / 2;
    T span = static_cast<T>(twice_high - twice_low) / 2;
    T in_  = std::fabs(in - min);

    T extra       = std::fmod(in_, span);
    int32_t flips = static_cast<int32_t>(std::floor(in_ / span));
    if (flips % 2 == 0) {
        return extra + min;
    } else {
        return span - extra + min;
    }
}

// compute interpolation location for the grid coordinate
template <typename T, bool align_corners, grid_sample_padding padding_mode>
static inline T compute_location(
    const T coord,
    const int64_t size)
{
    if (grid_sample_padding::BORDER == padding_mode) {
        return clip_coordinates(unnormalize<T, align_corners>(coord, size), size);
    } else if (grid_sample_padding::REFLECTION == padding_mode) {
        T coord_;
        if (align_corners) {
            coord_ = reflect_coordinates(unnormalize<T, align_corners>(coord, size), 0, 2 * (size - 1));
        } else {
            coord_ = reflect_coordinates(unnormalize<T, align_corners>(coord, size), -1, 2 * size - 1);
        }
        return clip_coordinates(coord_, size);
    } else {
        return unnormalize<T, align_corners>(coord, size);
    }
}

static inline bool within_bounds_2d(
    const int64_t h,
    const int64_t w,
    const int64_t bound_h,
    const int64_t bound_w)
{
    return h >= 0 && h < bound_h && w >= 0 && w < bound_w;
}

}}}; // namespace ppl::kernel::riscv

#endif