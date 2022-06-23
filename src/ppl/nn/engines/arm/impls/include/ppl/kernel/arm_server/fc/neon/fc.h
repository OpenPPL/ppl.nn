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

#ifndef __ST_PPL_KERNEL_ARM_SERVER_FC_NEON_FC_H_
#define __ST_PPL_KERNEL_ARM_SERVER_FC_NEON_FC_H_

#include <cstring>
#include <string>

#include "ppl/kernel/arm_server/common/general_include.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace arm_server { namespace neon {

// fp16
#ifdef PPLNN_USE_ARMV8_2_FP16
size_t ppl_arm_server_kernel_fp16_fc_get_converted_filter_size(
    const int64_t num_in,
    const int64_t num_out);

// num_in : c, h, w, <-> c/8, h, w, 8
void ppl_arm_server_kernel_fp16_fc_convert_weights(
    const __fp16 *weights,
    __fp16 *cvt_weights,
    const int64_t num_in,
    const int64_t num_out);
#endif

// fp32

size_t ppl_arm_server_kernel_fp32_fc_get_converted_filter_size(
    const int64_t num_in,
    const int64_t num_out);

// num_in : c, h, w, <-> c/4, h, w, 4
void ppl_arm_server_kernel_fp32_fc_convert_weights(
    const float *weights,
    float *cvt_weights,
    const int64_t num_in,
    const int64_t num_out);

typedef uint32_t fc_fuse_flag_t;

class fc_fuse_flag {
public:
    enum {
        none  = 0,
        relu  = 1 << 0,
    };
};

struct fc_param {
    int64_t channels;
    int64_t num_output;
    fc_fuse_flag_t fuse_flag;
};

typedef uint32_t fc_algo_t;

class fc_algo {
public:
    static const fc_algo_t unknown  = 0;
    static const fc_algo_t standard = 1;
    static const fc_algo_t flatten_nbcx = 2;
};

struct fc_algo_info {
    fc_algo_t algo_type;
    ppl::common::datatype_t dtype;
    ppl::common::isa_t isa;
};

class fc_executor {
protected:
    const struct fc_param *fc_param_;
    void *cvt_filter_;
    void *cvt_bias_;

    void *src_;
    const ppl::nn::TensorShape *src_shape_;
    void *dst_;
    const ppl::nn::TensorShape *dst_shape_;

    void *temp_buffer_;

public:
    fc_executor()
        : fc_param_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , temp_buffer_(nullptr) {}

    fc_executor(const fc_param *fc_param, void *cvt_filter, void *cvt_bias)
        : fc_param_(fc_param)
        , cvt_filter_(cvt_filter)
        , cvt_bias_(cvt_bias)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , temp_buffer_(nullptr) {}

    // virtual uint64_t cal_temp_buffer_size();
    // virtual ppl::common::RetCode prepare();
    // virtual ppl::common::RetCode execute();
    virtual ~fc_executor() {}

    virtual bool init_profiler()
    {
        return false;
    }
    virtual void clear_profiler() {}
    virtual std::string export_profiler()
    {
        return "";
    }

    void set_fc_param(const fc_param *fc_param)
    {
        fc_param_ = fc_param;
    }
    const struct fc_param *fc_param() const
    {
        return fc_param_;
    };

    void set_cvt_filter(void *cvt_filter)
    {
        cvt_filter_ = cvt_filter;
    }
    void *cvt_filter() const
    {
        return cvt_filter_;
    }

    void set_cvt_bias(void *cvt_bias)
    {
        cvt_bias_ = cvt_bias;
    }
    void *cvt_bias() const
    {
        return cvt_bias_;
    }

    void set_src(void *src)
    {
        src_ = src;
    }
    void *src() const
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

    void set_temp_buffer(void *temp_buffer)
    {
        temp_buffer_ = temp_buffer;
    }
    void *temp_buffer() const
    {
        return temp_buffer_;
    }
};

class fc_manager {
protected:
    fc_param param_;
    ppl::common::Allocator *allocator_;

    void *cvt_filter_;
    void *cvt_bias_;
    uint64_t cvt_filter_size_;
    uint64_t cvt_bias_size_;

public:
    fc_manager()
        : allocator_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0) {}

    fc_manager(const fc_param &param, ppl::common::Allocator *allocator)
        : allocator_(allocator)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0)
    {
        param_ = param;
    }

    void set_param(const fc_param &param)
    {
        param_ = param;
    }
    const fc_param &param() const
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

    void set_cvt_filter(void *cvt_filter, const uint64_t &cvt_filter_size)
    {
        cvt_filter_      = cvt_filter;
        cvt_filter_size_ = cvt_filter_size;
    }
    void *cvt_filter() const
    {
        return cvt_filter_;
    }
    uint64_t cvt_filter_size() const
    {
        return cvt_filter_size_;
    }

    void set_cvt_bias(void *cvt_bias, const uint64_t &cvt_bias_size)
    {
        cvt_bias_      = cvt_bias;
        cvt_bias_size_ = cvt_bias_size;
    }
    void *cvt_bias() const
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

    virtual ppl::common::RetCode gen_cvt_weights(
        void *filter, 
        void *bias, 
        ppl::common::datatype_t dtype) {
        if (cvt_bias_ != nullptr || cvt_filter_ != nullptr) {
            return ppl::common::RC_PERMISSION_DENIED;
        }

#ifdef PPLNN_USE_AARCH64
        const int64_t num_output = param_.num_output;
        const int64_t channels = param_.channels;
        if (dtype == ppl::common::DATATYPE_FLOAT32) {
            cvt_bias_size_ = num_output;
            cvt_bias_      = (float *)allocator_->Alloc(cvt_bias_size_ * sizeof(float));
            memcpy(cvt_bias_, bias, num_output * sizeof(float));
            
            cvt_filter_size_ = ppl_arm_server_kernel_fp32_fc_get_converted_filter_size(channels, num_output);
            cvt_filter_      = (float *)allocator_->Alloc(cvt_filter_size_);
            ppl_arm_server_kernel_fp32_fc_convert_weights(
                (float *)filter, (float *)cvt_filter_,
                param_.channels, param_.num_output);

            return ppl::common::RC_SUCCESS;
        }
#ifdef PPLNN_USE_ARMV8_2_FP16
        else if (dtype == ppl::common::DATATYPE_FLOAT16) {
            cvt_bias_size_ = num_output;
            cvt_bias_      = (__fp16 *)allocator_->Alloc(cvt_bias_size_ * sizeof(__fp16));
            memcpy(cvt_bias_, bias, num_output * sizeof(__fp16));
            
            cvt_filter_size_ = ppl_arm_server_kernel_fp16_fc_get_converted_filter_size(channels, num_output);
            cvt_filter_      = (__fp16 *)allocator_->Alloc(cvt_filter_size_);
            ppl_arm_server_kernel_fp16_fc_convert_weights(
                (__fp16 *)filter, (__fp16 *)cvt_filter_,
                param_.channels, param_.num_output);

            return ppl::common::RC_SUCCESS;
        }
#endif
#endif

        return ppl::common::RC_UNSUPPORTED;
    };

    virtual fc_executor *gen_executor() {
        return new fc_executor(&param_, cvt_filter_, cvt_bias_);
    }

    virtual ~fc_manager() {}
};

class fc_algo_selector {
public:
    static fc_algo_info select_algo(const ppl::common::dataformat_t &src_format, const fc_param &param, const ppl::common::datatype_t &dtype,  const ppl::common::isa_t &isa_flags) {
        fc_algo_info algo_info;
        
#ifdef PPLNN_USE_AARCH64
        if (!(dtype == ppl::common::DATATYPE_FLOAT16 && 
            (src_format == ppl::common::DATAFORMAT_N8CX || src_format == ppl::common::DATAFORMAT_NDARRAY)) &&
            !(dtype == ppl::common::DATATYPE_FLOAT32 && 
            (src_format == ppl::common::DATAFORMAT_N4CX || src_format == ppl::common::DATAFORMAT_NDARRAY))    ) {
            algo_info.algo_type = fc_algo::unknown;
            return algo_info;
        }

        algo_info.algo_type = (src_format == ppl::common::DATAFORMAT_NDARRAY) ? fc_algo::standard : fc_algo::flatten_nbcx;
        algo_info.dtype = dtype;
        algo_info.isa = isa_flags;
#else
        algo_info.algo_type = fc_algo::unknown;
#endif

        return algo_info;
    };
    static fc_manager *gen_algo(const fc_param &param, const fc_algo_info &algo_info, ppl::common::Allocator *allocator) {
        return new fc_manager(param, allocator);
    };
};

// fp16
#ifdef PPLNN_USE_ARMV8_2_FP16
size_t ppl_arm_server_kernel_fp16_fc_get_buffer_size(
    const int64_t num_in,
    const int64_t num_out,
    const int64_t num_batch,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m3,
    const int64_t sgemm_k3);

ppl::common::RetCode fc_fp16(
    const __fp16 *cvt_weights,
    const __fp16 *cvt_bias,
    const __fp16 *input,
    __fp16 *output,
    __fp16 *tmp_buffer,
    const int64_t num_in,
    const int64_t num_out,
    const int64_t num_batch,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m2,
    const int64_t sgemm_n2,
    const int64_t sgemm_k3);
#endif

// fp32

size_t ppl_arm_server_kernel_fp32_fc_get_buffer_size(
    const int64_t num_in,
    const int64_t num_out,
    const int64_t num_batch,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_k3);

ppl::common::RetCode fc_fp32(
    const float *cvt_weights,
    const float *cvt_bias,
    const float *input,
    float *output,
    float *tmp_buffer,
    const int64_t num_in,
    const int64_t num_out,
    const int64_t num_batch,
    const int64_t sgemm_m1,
    const int64_t sgemm_n1,
    const int64_t sgemm_k1,
    const int64_t sgemm_m2,
    const int64_t sgemm_n2,
    const int64_t sgemm_k3);

}}}}; // namespace ppl::kernel::arm_server::neon

#endif
