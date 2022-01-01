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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_FC_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_FC_H_

#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/kernel/riscv/common/general_include.h"
#include "ppl/kernel/riscv/common/fc_common.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "ppl/common/sys.h"

namespace ppl { namespace kernel { namespace riscv {

typedef uint64_t fc_fuse_flag_t;

class fc_fuse_flag {
public:
    enum {
        none = 0,
        relu = 1 << 0,
    };
};

struct fc_common_param {
    int64_t channels;
    int64_t num_output;
    fc_fuse_flag_t fuse_flag;
};

typedef uint32_t fc_common_algo_t;

class fc_common_algo {
public:
    static const fc_common_algo_t unknown = 0;
    static const fc_common_algo_t standard = 1;
};

struct fc_common_algo_info {
    fc_common_algo_t algo_type;
};

class fc_base_executor {
public:
    fc_base_executor() {}
    virtual uint64_t cal_temp_buffer_size() = 0;
    virtual ppl::common::RetCode prepare() = 0;
    virtual ppl::common::RetCode execute() = 0;
    virtual void set_temp_buffer(void* temp_buffer) = 0;
    virtual const fc_common_param* fc_param() const = 0;
    virtual ppl::common::RetCode set_src_tensor(ppl::nn::TensorImpl& src_tensor) = 0;
    virtual ppl::common::RetCode set_dst_tensor(ppl::nn::TensorImpl& src_tensor) = 0;
    virtual ~fc_base_executor() {}
};

template <typename T>
class fc_executor : public fc_base_executor {
protected:
    const fc_common_param* fc_param_;
    const T* cvt_filter_;
    const T* cvt_bias_;

    const T* src_;
    const ppl::nn::TensorShape* src_shape_;
    T* dst_;
    const ppl::nn::TensorShape* dst_shape_;

    void* temp_buffer_;

public:
    fc_executor()
        : fc_base_executor()
        , fc_param_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , temp_buffer_(nullptr) {}

    fc_executor(const fc_common_param* fc_param, const T* cvt_filter, const T* cvt_bias)
        : fc_base_executor()
        , fc_param_(fc_param)
        , cvt_filter_(cvt_filter)
        , cvt_bias_(cvt_bias)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , temp_buffer_(nullptr) {}

    virtual uint64_t cal_temp_buffer_size() = 0;
    virtual ppl::common::RetCode prepare() = 0;
    virtual ppl::common::RetCode execute() = 0;
    virtual ~fc_executor() {}

    virtual ppl::common::RetCode set_src_tensor(ppl::nn::TensorImpl& src_tensor) override {
        set_src_shape(src_tensor.GetShape());
        set_src(src_tensor.GetBufferPtr<T>());
        return ppl::common::RC_SUCCESS;
    };
    virtual ppl::common::RetCode set_dst_tensor(ppl::nn::TensorImpl& dst_tensor) override {
        set_dst_shape(dst_tensor.GetShape());
        set_dst(dst_tensor.GetBufferPtr<T>());
        return ppl::common::RC_SUCCESS;
    };

    void set_fc_param(const fc_common_param* fc_param) {
        fc_param_ = fc_param;
    }
    const fc_common_param* fc_param() const override {
        return fc_param_;
    };

    void set_cvt_filter(const T* cvt_filter) {
        cvt_filter_ = cvt_filter;
    }
    const T* cvt_filter() const {
        return cvt_filter_;
    }

    void set_cvt_bias(const T* cvt_bias) {
        cvt_bias_ = cvt_bias;
    }
    const T* cvt_bias() const {
        return cvt_bias_;
    }

    void set_src(const T* src) {
        src_ = src;
    }
    const T* src() const {
        return src_;
    }

    void set_src_shape(const ppl::nn::TensorShape* src_shape) {
        src_shape_ = src_shape;
    }
    const ppl::nn::TensorShape* src_shape() const {
        return src_shape_;
    };

    void set_dst(T* dst) {
        dst_ = dst;
    }
    T* dst() const {
        return dst_;
    }

    void set_dst_shape(const ppl::nn::TensorShape* dst_shape) {
        dst_shape_ = dst_shape;
    }
    const ppl::nn::TensorShape* dst_shape() const {
        return dst_shape_;
    }

    void set_temp_buffer(void* temp_buffer) {
        temp_buffer_ = temp_buffer;
    }
    void* temp_buffer() const {
        return temp_buffer_;
    }
};

class fc_base_manager {
public:
    fc_base_manager() {}
    virtual void release_cvt_weights() = 0;
    virtual fc_base_executor* gen_executor() = 0;
    virtual void set_param(const fc_common_param& param) = 0;
    virtual const fc_common_param& param() const = 0;
    virtual ~fc_base_manager() {}
};

template <typename T>
class fc_manager : public fc_base_manager {
protected:
    fc_common_param param_;
    ppl::common::Allocator* allocator_;

    T* cvt_filter_;
    T* cvt_bias_;
    uint64_t cvt_filter_size_;
    uint64_t cvt_bias_size_;

public:
    fc_manager()
        : fc_base_manager()
        , allocator_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0) {}

    fc_manager(const fc_common_param& param, ppl::common::Allocator* allocator)
        : fc_base_manager()
        , allocator_(allocator)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0) {
        param_ = param;
    }

    void set_param(const fc_common_param& param) override {
        param_ = param;
    }
    const fc_common_param& param() const override {
        return param_;
    };

    void set_allocator(ppl::common::Allocator* allocator) {
        allocator_ = allocator;
    }
    ppl::common::Allocator* allocator() {
        return allocator_;
    }

    void set_cvt_filter(const T* cvt_filter, const uint64_t& cvt_filter_size) {
        cvt_filter_ = const_cast<T*>(cvt_filter);
        cvt_filter_size_ = cvt_filter_size;
    }
    const T* cvt_filter() const {
        return cvt_filter_;
    }
    uint64_t cvt_filter_size() const {
        return cvt_filter_size_;
    }

    void set_cvt_bias(const T* cvt_bias, const uint64_t& cvt_bias_size) {
        cvt_bias_ = const_cast<T*>(cvt_bias);
        cvt_bias_size_ = cvt_bias_size;
    }
    const T* cvt_bias() const {
        return cvt_bias_;
    }
    uint64_t cvt_bias_size() const {
        return cvt_bias_size_;
    }

    void release_cvt_weights() {
        if (cvt_filter_) {
            allocator_->Free(cvt_filter_);
            cvt_filter_ = nullptr;
        }

        if (cvt_bias_) {
            allocator_->Free(cvt_bias_);
            cvt_bias_ = nullptr;
        }
    }

    virtual ppl::common::RetCode gen_cvt_weights(const T* filter, const T* bias) = 0;
    virtual fc_executor<T>* gen_executor() = 0;

    virtual ~fc_manager() {}
};

}}}; // namespace ppl::kernel::riscv

#endif
