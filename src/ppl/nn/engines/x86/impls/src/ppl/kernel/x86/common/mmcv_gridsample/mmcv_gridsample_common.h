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

#ifndef __ST_PPL_KERNEL_X86_COMMON_MMCV_GRIDSAMPLE_MMCV_GRIDSAMPLE_COMMON_H_
#define __ST_PPL_KERNEL_X86_COMMON_MMCV_GRIDSAMPLE_MMCV_GRIDSAMPLE_COMMON_H_

#include <cmath>

#include "ppl/kernel/x86/common/internal_include.h"

namespace ppl { namespace kernel { namespace x86 {

enum grid_sampler_interpolation {
    BILINEAR = 0,
    NEAREST  = 1,
    BICUBIC  = 2
};
enum grid_sampler_padding {
    ZEROS      = 0,
    BORDER     = 1,
    REFLECTION = 2
};

template <typename scalar_t, bool align_corners>
static inline scalar_t grid_sampler_unnormalize(const scalar_t coord, const int64_t size)
{
    if (align_corners) {
        return ((coord + 1) * 0.5) * (size - 1);
    } else {
        return ((coord + 1) * size - 1) * 0.5;
    }
}

// Clips coordinates to between 0 and clip_limit - 1
template <typename scalar_t>
static inline scalar_t clip_coordinates(const scalar_t in, const int64_t clip_limit)
{
    return min(static_cast<scalar_t>(clip_limit - 1),
                    max(in, static_cast<scalar_t>(0)));
}

// Reflects coordinates until they fall between low and high (inclusive).
// The bounds are passed as twice their value so that half-integer values
// can be represented as ints.
template <typename scalar_t>
static inline scalar_t reflect_coordinates(const scalar_t in, const int64_t twice_low, const int64_t twice_high)
{
    if (twice_low == twice_high) {
        return static_cast<scalar_t>(0);
    }
    scalar_t min   = static_cast<scalar_t>(twice_low) / 2;
    scalar_t span  = static_cast<scalar_t>(twice_high - twice_low) / 2;
    scalar_t in_             = std::fabs(in - min);
    // `fmod` returns same sign as `in`, which is positive after the `fabs` above.
    scalar_t extra = std::fmod(in_, span);
    int32_t flips  = static_cast<int32_t>(std::floor(in_ / span));
    if (flips % 2 == 0) {
        return extra + min;
    } else {
        return span - extra + min;
    }
}

template <typename scalar_t, bool align_corners, grid_sampler_padding padding_mode>
static inline scalar_t compute_coordinates(const scalar_t coord, const int64_t size)
{
    if (padding_mode == grid_sampler_padding::BORDER) {
        return clip_coordinates(coord, size);
    } else if (padding_mode == grid_sampler_padding::REFLECTION) {
        scalar_t coord_;
        if (align_corners) {
            coord_ = reflect_coordinates(coord, 0, 2 * (size - 1));
        } else {
            coord_ = reflect_coordinates(coord, -1, 2 * size - 1);
        }
        return clip_coordinates(coord_, size);
    } else {
        return coord;
    }
}

// Computes the pixel source index value for a grid coordinate
template <typename scalar_t, bool align_corners, grid_sampler_padding padding_mode>
static inline scalar_t grid_sampler_compute_source_index(const scalar_t coord, const int64_t size)
{
    scalar_t coord_ = grid_sampler_unnormalize<scalar_t, align_corners>(coord, size);
    return compute_coordinates<scalar_t, align_corners, padding_mode>(coord_, size);
}

static inline bool within_bounds_2d(const int64_t h, const int64_t w, const int64_t bound_h, const int64_t bound_w)
{
    return h >= 0 && h < bound_h && w >= 0 && w < bound_w;
}

}}}; // namespace ppl::kernel::x86

#endif // __ST_PPL_KERNEL_X86_COMMON_MMCV_GRIDSAMPLE_MMCV_GRIDSAMPLE_COMMON_H_
