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

#ifndef __ST_PPL_KERNEL_X86_FP32_FC_H_
#define __ST_PPL_KERNEL_X86_FP32_FC_H_

#include <string>

#include "ppl/kernel/x86/common/general_include.h"
#include "ppl/kernel/x86/common/fc_common.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace x86 {

struct fc_fp32_param {
    int64_t channels;
    int64_t num_output;
    fc_fuse_flag_t fuse_flag;
};

typedef uint32_t fc_fp32_algo_t;

class fc_fp32_algo {
public:
    static const fc_fp32_algo_t UNKNOWN  = 0;
    static const fc_fp32_algo_t STANDARD = 1;
};

struct fc_fp32_algo_info {
    fc_fp32_algo_t algo_type;
    ppl::common::isa_t isa;
};

class fc_fp32_executor {
protected:
    const fc_fp32_param *fc_param_;
    const float *cvt_filter_;
    const float *cvt_bias_;

    const float *src_;
    const ppl::nn::TensorShape *src_shape_;
    float *dst_;
    const ppl::nn::TensorShape *dst_shape_;

    void *temp_buffer_;

public:
    fc_fp32_executor()
        : fc_param_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , temp_buffer_(nullptr) {}

    fc_fp32_executor(const fc_fp32_param *fc_param, const float *cvt_filter, const float *cvt_bias)
        : fc_param_(fc_param)
        , cvt_filter_(cvt_filter)
        , cvt_bias_(cvt_bias)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , temp_buffer_(nullptr) {}

    virtual uint64_t cal_temp_buffer_size() = 0;
    virtual ppl::common::RetCode prepare()  = 0;
    virtual ppl::common::RetCode execute()  = 0;
    virtual ~fc_fp32_executor() {}

    virtual bool init_profiler()
    {
        return false;
    }
    virtual void clear_profiler() {}
    virtual std::string export_profiler()
    {
        return "";
    }

    void set_fc_param(const fc_fp32_param *fc_param)
    {
        fc_param_ = fc_param;
    }
    const fc_fp32_param *fc_param() const
    {
        return fc_param_;
    };

    void set_cvt_filter(const float *cvt_filter)
    {
        cvt_filter_ = cvt_filter;
    }
    const float *cvt_filter() const
    {
        return cvt_filter_;
    }

    void set_cvt_bias(const float *cvt_bias)
    {
        cvt_bias_ = cvt_bias;
    }
    const float *cvt_bias() const
    {
        return cvt_bias_;
    }

    void set_src(const float *src)
    {
        src_ = src;
    }
    const float *src() const
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

    void set_dst(float *dst)
    {
        dst_ = dst;
    }
    float *dst() const
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

    void set_temp_buffer(void *temp_buffer)
    {
        temp_buffer_ = temp_buffer;
    }
    void *temp_buffer() const
    {
        return temp_buffer_;
    }
};

class fc_fp32_manager {
protected:
    fc_fp32_param param_;
    ppl::common::Allocator *allocator_;

    float *cvt_filter_;
    float *cvt_bias_;
    uint64_t cvt_filter_size_;
    uint64_t cvt_bias_size_;

public:
    fc_fp32_manager()
        : allocator_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0) {}

    fc_fp32_manager(const fc_fp32_param &param, ppl::common::Allocator *allocator)
        : allocator_(allocator)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0)
    {
        param_ = param;
    }

    void set_param(const fc_fp32_param &param)
    {
        param_ = param;
    }
    const fc_fp32_param &param() const
    {
        return param_;
    };

    void set_allocator(ppl::common::Allocator *allocator)
    {
        allocator_ = allocator;
    }
    ppl::common::Allocator *allocator()
    {
        return allocator_;
    }

    void set_cvt_filter(const float *cvt_filter, const uint64_t &cvt_filter_size)
    {
        cvt_filter_      = const_cast<float *>(cvt_filter);
        cvt_filter_size_ = cvt_filter_size;
    }
    const float *cvt_filter() const
    {
        return cvt_filter_;
    }
    uint64_t cvt_filter_size() const
    {
        return cvt_filter_size_;
    }

    void set_cvt_bias(const float *cvt_bias, const uint64_t &cvt_bias_size)
    {
        cvt_bias_      = const_cast<float *>(cvt_bias);
        cvt_bias_size_ = cvt_bias_size;
    }
    const float *cvt_bias() const
    {
        return cvt_bias_;
    }
    uint64_t cvt_bias_size() const
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

    virtual ppl::common::RetCode gen_cvt_weights(const float *filter, const float *bias) = 0;
    virtual fc_fp32_executor *gen_executor() = 0;

    virtual ~fc_fp32_manager() {}
};

class fc_algo_selector {
public:
    static fc_fp32_algo_info select_algo(const ppl::common::dataformat_t &src_format, const fc_fp32_param &param, const ppl::common::isa_t &isa_flags);
    static fc_fp32_manager *gen_algo(const fc_fp32_param &param, const fc_fp32_algo_info &algo_info, ppl::common::Allocator *allocator);
};

}}}; // namespace ppl::kernel::x86

#endif
