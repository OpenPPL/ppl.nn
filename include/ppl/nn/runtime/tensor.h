#ifndef _ST_HPC_PPL_NN_RUNTIME_TENSOR_H_
#define _ST_HPC_PPL_NN_RUNTIME_TENSOR_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/common/common.h"

namespace ppl { namespace nn {

class PPLNN_PUBLIC Tensor {
public:
    virtual ~Tensor() {}

    /** @brief get tensor's name */
    virtual const char* GetName() const = 0;

    /** @brief get tensor's shape */
    virtual TensorShape& GetShape() = 0;

    /** @brief get tensor's shape */
    virtual const TensorShape& GetShape() const = 0;

    /** @brief rellocate a buffer according to its shape */
    virtual ppl::common::RetCode ReallocBuffer() = 0;

    /**
       @brief copy tensor's data to `dst`, which points to a host memory
       @note `dst` MUST have enough space.
    */
    virtual ppl::common::RetCode CopyToHost(void* dst) const = 0;

    /**
       @brief copy tensor's data from `dst`, which points to a host memory
       @note `dst` MUST have enough space.
    */
    virtual ppl::common::RetCode CopyFromHost(const void* src) = 0;

    /** @brief convert tensor's data to `dst` with shape `dst_desc` */
    virtual ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc) const = 0;

    /** @brief convert tensor's data from `dst` with shape `dst_desc` */
    virtual ppl::common::RetCode ConvertFromHost(const void* src, const TensorShape& src_desc) = 0;
};

}} // namespace ppl::nn

#endif
