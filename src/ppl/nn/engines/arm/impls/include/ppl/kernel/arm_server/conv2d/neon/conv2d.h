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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_FP16_CONV2D_H_
#define __ST_PPL_KERNEL_ARM_SERVER_FP16_CONV2D_H_

#include <string>

#include "ppl/kernel/arm_server/common/general_include.h"
#include "ppl/common/allocator.h"
#include "ppl/common/arm/sysinfo.h"
#include "ppl/common/sys.h"
#include "ppl/nn/engines/arm/engine_options.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

typedef uint32_t conv_fuse_flag_t;
typedef uint32_t conv_pad_type_t;

class conv_fuse_flag {
public:
    enum {
        NONE   = 0,
        RELU   = 1 << 0,
        RELU6  = 1 << 1,
        SUM    = 1 << 2,
        PRELU  = 1 << 3,
        HSWISH = 1 << 4,
    };
};

class conv_pad_type {
public:
    enum {
        ZERO  = 0,
        RFLCT = 1,
        SAME  = 2,
    };
};

struct conv2d_param {
    int64_t kernel_h;
    int64_t kernel_w;
    int64_t stride_h;
    int64_t stride_w;
    int64_t dilation_h;
    int64_t dilation_w;
    int64_t pad_h;
    int64_t pad_w;
    int64_t channels;
    int64_t num_output;
    int64_t group;
    conv_fuse_flag_t fuse_flag;
    conv_pad_type_t pad_type;

    float sparse_level() const
    {
        const int32_t sparse_h = stride_h * dilation_h;
        const int32_t sparse_w = stride_w * dilation_w;
        return float(sparse_h * sparse_w) / float(kernel_h * kernel_w);
    }

    bool is_depthwise() const
    {
        return true &&
               group != 1 &&
               group == channels &&
               group == num_output;
    }

    bool is_pointwise() const
    {
        return true &&
               kernel_h == 1 &&
               kernel_w == 1 &&
               pad_h == 0 &&
               pad_w == 0 &&
               dilation_h == 1 &&
               dilation_w == 1 &&
               !is_depthwise();
    }
};

typedef uint32_t conv2d_algo_t;

// ensure consistent with below algo str
class conv2d_algo {
public:
    static const conv2d_algo_t unknown        = 0;
    static const conv2d_algo_t depthwise      = 1;
    static const conv2d_algo_t gemm           = 2;
    static const conv2d_algo_t tile_gemm      = 3;
    static const conv2d_algo_t direct         = 4;
    static const conv2d_algo_t direct_ndarray = 5;
    static const conv2d_algo_t winograd_b2f3  = 6;
    static const conv2d_algo_t winograd_b4f3  = 7;
};

static inline const char *get_conv_algo_str(const conv2d_algo_t algo)
{
    static const char *algo_str[] = {
        "unknown",
        "depthwise",
        "gemm",
        "tile_gemm",
        "direct",
        "direct_ndarray",
        "winograd_b2f3",
        "winograd_b4f3",
    };
    return algo_str[algo];
}

struct conv2d_algo_info {
    conv2d_algo_t algo_type;
    ppl::common::isa_t isa;
    ppl::common::datatype_t data_type;
    ppl::common::dataformat_t input_format;
    ppl::common::dataformat_t output_format;
};

class conv2d_runtime_executor {
protected:
    const conv2d_param *conv_param_;
    const void *cvt_filter_;
    const void *cvt_bias_;

    const void *src_;
    const ppl::nn::TensorShape *src_shape_;
    void *dst_;
    const ppl::nn::TensorShape *dst_shape_;
    void *sum_;
    void *temp_buffer_;

public:
    conv2d_runtime_executor()
        : conv_param_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , sum_(nullptr)
        , temp_buffer_(nullptr) {}

    conv2d_runtime_executor(const conv2d_param *conv_param, const void *cvt_filter, const void *cvt_bias)
        : conv_param_(conv_param)
        , cvt_filter_(cvt_filter)
        , cvt_bias_(cvt_bias)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , sum_(nullptr)
        , temp_buffer_(nullptr) {}

    virtual uint64_t cal_temp_buffer_size() = 0;
    virtual ppl::common::RetCode prepare()  = 0;
    virtual ppl::common::RetCode execute()  = 0;
    virtual ~conv2d_runtime_executor() {}

    virtual bool init_profiler()
    {
        return false;
    }
    virtual void clear_profiler() {}
    virtual std::string export_profiler()
    {
        return "";
    }

    void set_conv_param(const conv2d_param *conv_param)
    {
        conv_param_ = conv_param;
    }
    const conv2d_param *conv_param() const
    {
        return conv_param_;
    };

    void set_cvt_filter(const void *cvt_filter)
    {
        cvt_filter_ = cvt_filter;
    }
    const void *cvt_filter() const
    {
        return cvt_filter_;
    }

    void set_cvt_bias(const void *cvt_bias)
    {
        cvt_bias_ = cvt_bias;
    }
    const void *cvt_bias() const
    {
        return cvt_bias_;
    }

    void set_src(const void *src)
    {
        src_ = src;
    }
    const void *src() const
    {
        return src_;
    }

    void set_src_shape(const ppl::nn::TensorShape *src_shape)
    {
        src_shape_ = src_shape;
    }
    const ppl::nn::TensorShape *src_shape() const
    {
        return src_shape_;
    };

    void set_dst(void *dst)
    {
        dst_ = dst;
    }
    void *dst() const
    {
        return dst_;
    }

    void set_dst_shape(const ppl::nn::TensorShape *dst_shape)
    {
        dst_shape_ = dst_shape;
    }
    const ppl::nn::TensorShape *dst_shape() const
    {
        return dst_shape_;
    }

    void set_sum(void *sum)
    {
        sum_ = sum;
    }
    void *sum() const
    {
        return sum_;
    }

    void set_temp_buffer(void *temp_buffer)
    {
        temp_buffer_ = temp_buffer;
    }
    void *temp_buffer() const
    {
        return temp_buffer_;
    }
};

class conv2d_offline_manager {
protected:
    conv2d_param param_;
    ppl::common::Allocator *allocator_;

    void *cvt_filter_;
    void *cvt_bias_;
    uint64_t cvt_filter_size_;
    uint64_t cvt_bias_size_;

private:
    conv2d_algo_info algo_info_;

public:
    conv2d_offline_manager()
        : allocator_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0) {}

    conv2d_offline_manager(const conv2d_param &param, ppl::common::Allocator *allocator)
        : allocator_(allocator)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0)
    {
        param_ = param;
    }

    void set_param(const conv2d_param &param)
    {
        param_ = param;
    }
    const conv2d_param &get_param() const
    {
        return param_;
    };

    void set_allocator(ppl::common::Allocator *allocator)
    {
        allocator_ = allocator;
    }
    ppl::common::Allocator *get_allocator()
    {
        return allocator_;
    }

    void set_cvt_filter(const void *cvt_filter, const uint64_t cvt_filter_size)
    {
        cvt_filter_      = const_cast<void *>(cvt_filter);
        cvt_filter_size_ = cvt_filter_size;
    }
    const void *get_cvt_filter() const
    {
        return cvt_filter_;
    }
    uint64_t get_cvt_filter_size() const
    {
        return cvt_filter_size_;
    }

    void set_cvt_bias(const void *cvt_bias, const uint64_t cvt_bias_size)
    {
        cvt_bias_      = const_cast<void *>(cvt_bias);
        cvt_bias_size_ = cvt_bias_size;
    }
    const void *get_cvt_bias() const
    {
        return cvt_bias_;
    }
    uint64_t get_cvt_bias_size() const
    {
        return cvt_bias_size_;
    }

    void release_cvt_weights()
    {
        if (cvt_filter_) {
            allocator_->Free(cvt_filter_);
            cvt_filter_ = nullptr;
        }

        if (cvt_bias_) {
            allocator_->Free(cvt_bias_);
            cvt_bias_ = nullptr;
        }
    }

    conv2d_algo_info &algo_info()
    {
        return algo_info_;
    };

    void set_algo_info(conv2d_algo_info &algo)
    {
        algo_info_ = algo;
    };

    virtual conv2d_algo_t get_algo_type()                                              = 0;
    // schedule param setting rule is simple but fast.
    virtual ppl::common::RetCode fast_init_schedule_param()                            = 0;
    // run all possible schedule params and choose the best.
    virtual ppl::common::RetCode pick_best_schedule_param(
        const ppl::nn::TensorShape &src_shape, void *src, void *cvt_bias,
        const ppl::nn::TensorShape &dst_shape, void *dst,
        const bool tune_sp, double &runtime)                                           = 0;
    virtual bool is_supported()                                                        = 0;
    virtual ppl::common::RetCode try_fuse(conv_fuse_flag_t fuse_type)                  = 0;
    virtual ppl::common::RetCode try_reflect_pad(const std::vector<int>& pads)         = 0;
    virtual ppl::common::RetCode gen_cvt_weights(const void *filter, const void *bias) = 0;
    virtual conv2d_runtime_executor *gen_executor()                                    = 0;

    virtual ~conv2d_offline_manager() {}
};

class conv2d_algo_selector {
public:
    // algo selection is simple but fast. return the final algo manager.
    static conv2d_offline_manager *fast_gen_algo(const ppl::nn::TensorShape &shape, const ppl::nn::arm::EngineOptions &options, const ppl::common::isa_t isa_flags, const conv2d_param &param, ppl::common::Allocator *allocator);
    // run all possible algo & block size, select the best one and return the final algo manager.
    static conv2d_offline_manager *gen_fast_algo(const ppl::nn::TensorShape &shape, const ppl::nn::arm::EngineOptions &options, const ppl::common::isa_t isa_flags, const conv2d_param &param, ppl::common::Allocator *allocator);
};

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
