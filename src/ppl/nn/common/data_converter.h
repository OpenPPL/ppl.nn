#ifndef _ST_HPC_PPL_NN_COMMON_DATA_CONVERTER_H_
#define _ST_HPC_PPL_NN_COMMON_DATA_CONVERTER_H_

#include "ppl/common/retcode.h"
#include "ppl/nn/common/tensor_shape.h"
#include "ppl/nn/common/buffer_desc.h"

namespace ppl { namespace nn {

class DataConverter {
public:
    virtual ~DataConverter() {}

    /**
       @brief convert data described by `src_desc` from `src` to `dst` described by `dst_desc`
       @param dst points to cpu memory
       @param dst_desc shape of `dst`
       @param src points to data area of a device
       @param src_desc shape of `src`
    */
    virtual ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                               const TensorShape& src_desc) const = 0;

    /**
       @brief convert data described by `src_desc` from `src` to `dst` described by `dst_desc`
       @param dst points to data area of a device
       @param dst_desc shape of `dst`
       @param src points to cpu memory
       @param src_desc shape of `src`
    */
    virtual ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                                 const TensorShape& src_desc) const = 0;

    /**
       @brief convert data described by `src_desc` from `src` to `dst` described by `dst_desc`
       @param dst pointer to data area of a device
       @param dst_desc shape of `dst`
       @param src pointer to data area of a device
       @param src_desc shape of `src`
    */
    virtual ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                         const TensorShape& src_desc) const = 0;
};

}} // namespace ppl::nn

#endif
