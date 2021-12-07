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

#ifndef __ST_PPL_KERNEL_X86_FP32_PD_CONV2D_H_
#define __ST_PPL_KERNEL_X86_FP32_PD_CONV2D_H_

#include "ppl/kernel/x86/common/general_include.h"
#include "ppl/kernel/x86/fp32/conv2d.h"

namespace ppl { namespace kernel { namespace x86 {

typedef uint32_t pd_conv2d_fp32_algo_t;

class pd_conv2d_fp32_algo {
public:
    static const pd_conv2d_fp32_algo_t UNKNOWN  = 0;
    static const pd_conv2d_fp32_algo_t GEMM_DIRECT = 1;
    static const pd_conv2d_fp32_algo_t DIRECT = 2;
};

struct pd_conv2d_fp32_algo_info {
    pd_conv2d_fp32_algo_t algo_type;
    ppl::common::isa_t isa;
    ppl::common::dataformat_t input_format;
    ppl::common::dataformat_t output_format;
};

class pd_conv2d_fp32_executor {
protected:
    conv2d_fp32_executor *conv2d_executor_;
    conv2d_fp32_executor *depthwise_conv2d_executor_;

    const float *src_;
    const ppl::nn::TensorShape *src_shape_;
    float *dst_;
    const ppl::nn::TensorShape *dst_shape_;

    void *temp_buffer_;

public:
    pd_conv2d_fp32_executor() 
        : conv2d_executor_(nullptr)
        , depthwise_conv2d_executor_(nullptr)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , temp_buffer_(nullptr) {}
    pd_conv2d_fp32_executor(conv2d_fp32_executor *exec, conv2d_fp32_executor *depthwise_exec)
        : src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , temp_buffer_(nullptr) {
        this->conv2d_executor_ = exec;
        this->depthwise_conv2d_executor_ = depthwise_exec;
    }

    virtual uint64_t cal_temp_buffer_size() = 0;
    virtual ppl::common::RetCode prepare() = 0;
    virtual ppl::common::RetCode execute() = 0;
    virtual ~pd_conv2d_fp32_executor() {}

    void set_conv2d_executor(conv2d_fp32_executor *exec) {
        conv2d_executor_ = exec;
    }
    conv2d_fp32_executor *conv2d_executor() const
    {
        return conv2d_executor_;
    }
    void set_depthwise_conv2d_executor(conv2d_fp32_executor *exec) {
        depthwise_conv2d_executor_ = exec;
    }
    conv2d_fp32_executor *depthwise_conv2d_executor() const
    {
        return depthwise_conv2d_executor_;
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
    }

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

class pd_conv2d_fp32_manager {
protected:
    conv2d_fp32_manager *conv2d_manager_;
    conv2d_fp32_manager *depthwise_conv2d_manager_;

public:
    pd_conv2d_fp32_manager() : conv2d_manager_(nullptr), depthwise_conv2d_manager_(nullptr) {};
    pd_conv2d_fp32_manager(conv2d_fp32_manager *mgr, conv2d_fp32_manager *depthwise_mgr)
    {
        this->conv2d_manager_ = mgr;
        this->depthwise_conv2d_manager_ = depthwise_mgr;
    }

    virtual pd_conv2d_fp32_executor *gen_executor() = 0;

    void set_conv2d_manager(conv2d_fp32_manager *mgr)
    {
        conv2d_manager_ = mgr;
    }
    conv2d_fp32_manager *conv2d_manager()
    {
        return conv2d_manager_;
    }
    void set_depthwise_conv2d_manager(conv2d_fp32_manager *mgr)
    {
        depthwise_conv2d_manager_ = mgr;
    }
    conv2d_fp32_manager *depthwise_conv2d_manager()
    {
        return depthwise_conv2d_manager_;
    }

    ppl::common::RetCode gen_cvt_weights(
        const float *filter,
        const float *bias,
        const float *depthwise_filter,
        const float *depthwise_bias)
    {
        ppl::common::RetCode rc;
        if (conv2d_manager_) {
            rc = conv2d_manager_->gen_cvt_weights(filter, bias);
            if (ppl::common::RC_SUCCESS != rc) {
                return rc;
            }
        }
        if (depthwise_conv2d_manager_) {
            rc = depthwise_conv2d_manager_->gen_cvt_weights(depthwise_filter, depthwise_bias);
            if (ppl::common::RC_SUCCESS != rc) {
                return rc;
            }
            return ppl::common::RC_SUCCESS;
        }
        return ppl::common::RC_OTHER_ERROR;
    }

    void release_cvt_weights()
    {
        if (conv2d_manager_) conv2d_manager_->release_cvt_weights();
        if (depthwise_conv2d_manager_) depthwise_conv2d_manager_->release_cvt_weights();
    }

    virtual ~pd_conv2d_fp32_manager() {};
};

// Post-Depthwise Conv2d
class pd_conv2d_algo_selector {
public:
    static pd_conv2d_fp32_algo_info select_algo(
        const conv2d_fp32_algo_info &algo,
        const conv2d_fp32_algo_info &post_algo,
        const conv2d_fp32_param &param,
        const conv2d_fp32_param &post_param);
    static pd_conv2d_fp32_manager *gen_algo(
        const conv2d_fp32_param &param,
        const conv2d_fp32_param &depthwise_param,
        const pd_conv2d_fp32_algo_info &algo_info,
        ppl::common::Allocator *allocator);
    static pd_conv2d_fp32_manager *gen_algo(
        const pd_conv2d_fp32_algo_info &algo_info,
        conv2d_fp32_manager *mgr,
        conv2d_fp32_manager *depthwise_mgr);
};

}}};

#endif