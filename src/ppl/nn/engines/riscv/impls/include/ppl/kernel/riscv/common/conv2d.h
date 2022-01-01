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

#ifndef __ST_PPL_KERNEL_RISCV_COMMON_CONV2D_H_
#define __ST_PPL_KERNEL_RISCV_COMMON_CONV2D_H_

#include <string>
#include <chrono>

#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/common/retcode.h"
#include "ppl/common/allocator.h"
#include "ppl/common/sys.h"
#include "ppl/common/log.h"

namespace ppl { namespace kernel { namespace riscv {

struct conv2d_common_param {
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

    __fp16 sparse_level() const {
        // TODO: are there any better index for sparse_level?
        const int32_t sparse_h = stride_h * dilation_h;
        const int32_t sparse_w = stride_w * dilation_w;
        return __fp16(sparse_h * sparse_w) / __fp16(kernel_h * kernel_w);
    }

    bool is_depthwise() const {
        return true && group != 1 && group == channels && group == num_output;
    }

    bool is_pointwise() const {
        return true && kernel_h == 1 && kernel_w == 1 && pad_h == 0 && pad_w == 0 && dilation_h == 1 &&
            dilation_w == 1 && !is_depthwise();
    }
};

typedef uint32_t conv2d_common_algo_t;

class conv2d_common_algo {
public:
    static const conv2d_common_algo_t unknown = 0;
    static const conv2d_common_algo_t naive = 2;
    static const conv2d_common_algo_t depthwise = 3;
    static const conv2d_common_algo_t tile_gemm = 4;
    static const conv2d_common_algo_t gemm = 5;
    static const conv2d_common_algo_t direct_gemm = 6;
    static const conv2d_common_algo_t direct = 7;
    static const conv2d_common_algo_t winograd_b2f3 = 32;
    static const conv2d_common_algo_t winograd_b4f3 = 33;
    static const conv2d_common_algo_t winograd_b6f3 = 34;
};

struct conv2d_common_algo_info {
    conv2d_common_algo_t algo_type;
    ppl::common::dataformat_t input_format;
    ppl::common::dataformat_t output_format;
    ppl::common::datatype_t input_data_type;
    ppl::common::datatype_t output_data_type;
};

class conv2d_base_runtime_executor {
public:
    conv2d_base_runtime_executor() {}

    virtual uint64_t cal_temp_buffer_size() = 0;
    virtual ppl::common::RetCode prepare() = 0;
    virtual ppl::common::RetCode execute() = 0;
    virtual ppl::common::RetCode set_src_tensor(ppl::nn::TensorImpl& src_tensor) = 0;
    virtual ppl::common::RetCode set_dst_tensor(ppl::nn::TensorImpl& src_tensor) = 0;
    virtual void set_temp_buffer(void* temp_buffer) = 0;

    virtual ~conv2d_base_runtime_executor() {}
};

template <typename T>
class conv2d_runtime_executor : public conv2d_base_runtime_executor {
protected:
    const conv2d_common_param* conv_param_;

    const T* cvt_filter_;
    const T* cvt_bias_;

    const T* src_;
    T* dst_;
    const T* sum_src_;

    const ppl::nn::TensorShape* src_shape_;
    const ppl::nn::TensorShape* dst_shape_;
    const ppl::nn::TensorShape* sum_src_shape_;

    void* temp_buffer_;

public:
    conv2d_runtime_executor()
        : conv_param_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , sum_src_(nullptr)
        , sum_src_shape_(nullptr)
        , temp_buffer_(nullptr) {}

    conv2d_runtime_executor(const conv2d_common_param* conv_param, const T* cvt_filter, const T* cvt_bias)
        : conv_param_(conv_param)
        , cvt_filter_(cvt_filter)
        , cvt_bias_(cvt_bias)
        , src_(nullptr)
        , src_shape_(nullptr)
        , dst_(nullptr)
        , dst_shape_(nullptr)
        , sum_src_(nullptr)
        , sum_src_shape_(nullptr)
        , temp_buffer_(nullptr) {}

    virtual uint64_t cal_temp_buffer_size() = 0;
    virtual ppl::common::RetCode prepare() = 0;
    virtual ppl::common::RetCode execute() = 0;
    virtual ~conv2d_runtime_executor() {}

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

    void set_conv_param(const conv2d_common_param* conv_param) {
        conv_param_ = conv_param;
    }
    const conv2d_common_param* conv_param() const {
        return conv_param_;
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

    void set_sum_src(const T* sum_src) {
        sum_src_ = sum_src;
    }
    const T* sum_src() const {
        return sum_src_;
    }

    void set_sum_src_shape(const ppl::nn::TensorShape* sum_src_shape) {
        sum_src_shape_ = sum_src_shape;
    }
    const ppl::nn::TensorShape* sum_src_shape() const {
        return sum_src_shape_;
    }

    void set_temp_buffer(void* temp_buffer) override {
        temp_buffer_ = temp_buffer;
    }
    void* temp_buffer() const {
        return temp_buffer_;
    }
};

class conv2d_base_offline_manager {
protected:
    conv2d_common_algo_info algo_info_;

public:
    conv2d_base_offline_manager() {}

    conv2d_common_algo_info& algo_info() {
        return algo_info_;
    }
    void set_algo_info(conv2d_common_algo_info& algo) {
        algo_info_ = algo;
    }
    virtual conv2d_base_runtime_executor* gen_executor() = 0;
    virtual void release_cvt_weights() = 0;
    virtual ~conv2d_base_offline_manager() {}
};

template <typename T>
class conv2d_offline_manager : public conv2d_base_offline_manager {
protected:
    conv2d_common_param param_;
    ppl::common::Allocator* allocator_;

    T* cvt_filter_;
    T* cvt_bias_;
    uint64_t cvt_filter_size_;
    uint64_t cvt_bias_size_;

public:
    conv2d_offline_manager()
        : conv2d_base_offline_manager()
        , allocator_(nullptr)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0) {}

    conv2d_offline_manager(const conv2d_common_param& param, const conv2d_common_algo_info& algo_info,
                           ppl::common::Allocator* allocator)
        : conv2d_base_offline_manager()
        , allocator_(allocator)
        , cvt_filter_(nullptr)
        , cvt_bias_(nullptr)
        , cvt_filter_size_(0)
        , cvt_bias_size_(0) {
        param_ = param;
        algo_info_ = algo_info;
    }

    void set_param(const conv2d_common_param& param) {
        param_ = param;
    }
    const conv2d_common_param& param() const {
        return param_;
    }
    void set_allocator(ppl::common::Allocator* allocator) {
        allocator_ = allocator;
    }
    ppl::common::Allocator* allocator() {
        return allocator_;
    }

    void set_cvt_filter(const T* cvt_filter, const uint64_t cvt_filter_size) {
        cvt_filter_ = const_cast<T*>(cvt_filter);
        cvt_filter_size_ = cvt_filter_size;
    }
    const T* cvt_filter() const {
        return cvt_filter_;
    }
    uint64_t cvt_filter_size() const {
        return cvt_filter_size_;
    }

    void set_cvt_bias(const T* cvt_bias, const uint64_t cvt_bias_size) {
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

    virtual ppl::common::RetCode fast_init_tunning_param() = 0;
    virtual ppl::common::RetCode pick_best_tunning_param(const T* src, const T* filter, T* dst,
                                                         ppl::nn::TensorShape& src_shape,
                                                         ppl::nn::TensorShape& dst_shape) = 0;
    virtual bool is_supported() = 0;
    virtual ppl::common::RetCode gen_cvt_weights(const T* filter, const T* bias) = 0;
    virtual conv2d_base_runtime_executor* gen_executor() = 0;
    virtual ~conv2d_offline_manager() {}

protected:
    double profile_tunning_param(const T* src, const T* filter, T* dst, ppl::nn::TensorShape& src_shape,
                                 ppl::nn::TensorShape& dst_shape) {
        const int32_t exe_count = 1;

        conv2d_offline_manager<T>& offline_manager = *this;
        std::vector<T> zero_bias(offline_manager.param().num_output, 0.0f);
        // std::vector<T> tmp_buffer;
        offline_manager.gen_cvt_weights(filter, zero_bias.data());

        conv2d_runtime_executor<T>* executor =
            dynamic_cast<conv2d_runtime_executor<T>*>(offline_manager.gen_executor());
        auto start = std::chrono::high_resolution_clock::now();
        {
            executor->set_src_shape(&src_shape);
            executor->set_src(src);
            executor->set_dst_shape(&dst_shape);
            executor->set_dst(dst);

            ppl::common::RetCode rc;
            if (ppl::common::RC_SUCCESS != (rc = executor->prepare())) {
                LOG(ERROR) << "Prepare failed while the offline manager is picking the best tunning param: "
                           << ppl::common::GetRetCodeStr(rc);
            }
            uint64_t tmp_buffer_size = executor->cal_temp_buffer_size();
            auto tmp_buffer = (T*)offline_manager.allocator()->Alloc(tmp_buffer_size * sizeof(T));
            // tmp_buffer.resize(tmp_buffer_size / sizeof(T));
            executor->set_temp_buffer(tmp_buffer);
            for (int32_t i = 0; i < exe_count; i++) {
                if (ppl::common::RC_SUCCESS != (rc = executor->execute())) {
                    LOG(ERROR) << "Execute failed while the offline manager is picking the best tunning param: "
                               << ppl::common::GetRetCodeStr(rc);
                }
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        offline_manager.allocator()->Free(executor->temp_buffer());
        offline_manager.release_cvt_weights();
        delete executor;

        return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count(); // microseconds
    }
};

}}} // namespace ppl::kernel::riscv

#endif // __ST_PPL_KERNEL_RISCV_COMMON_CONV2D_H_
