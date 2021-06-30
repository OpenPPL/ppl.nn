#include "ppl/nn/runtime/tensor_impl.h"
#include "ppl/nn/common/logger.h"
using namespace std;
using namespace ppl::common;

namespace ppl { namespace nn {

RetCode TensorImpl::CopyToHost(void* dst) const {
    return buffer_info_.GetDevice()->CopyToHost(dst, buffer_info_.GetBufferDesc(), buffer_info_.GetShape());
}

RetCode TensorImpl::CopyFromHost(const void* src) {
    return buffer_info_.GetDevice()->CopyFromHost(&buffer_info_.GetBufferDesc(), src, buffer_info_.GetShape());
}

RetCode TensorImpl::ConvertToHost(void* dst, const TensorShape& dst_desc) const {
    auto converter = buffer_info_.GetDevice()->GetDataConverter();
    return converter->ConvertToHost(dst, dst_desc, buffer_info_.GetBufferDesc(), buffer_info_.GetShape());
}

RetCode TensorImpl::ConvertFromHost(const void* src, const TensorShape& src_desc) {
    auto converter = buffer_info_.GetDevice()->GetDataConverter();
    return converter->ConvertFromHost(&buffer_info_.GetBufferDesc(), buffer_info_.GetShape(), src, src_desc);
}

}} // namespace ppl::nn
