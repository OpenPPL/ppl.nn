#ifndef _ST_HPC_PPL_NN_UTILS_GENERIC_CPU_DATA_CONVERTER_H_
#define _ST_HPC_PPL_NN_UTILS_GENERIC_CPU_DATA_CONVERTER_H_

#include "ppl/nn/common/data_converter.h"

namespace ppl { namespace nn { namespace utils {

class GenericCpuDataConverter final : public DataConverter {
public:
    ppl::common::RetCode ConvertToHost(void* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                       const TensorShape& src_desc) const override;

    ppl::common::RetCode ConvertFromHost(BufferDesc* dst, const TensorShape& dst_desc, const void* src,
                                         const TensorShape& src_desc) const override;

    ppl::common::RetCode Convert(BufferDesc* dst, const TensorShape& dst_desc, const BufferDesc& src,
                                 const TensorShape& src_desc) const override;
};

}}} // namespace ppl::nn::utils

#endif
