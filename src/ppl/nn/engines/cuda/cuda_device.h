#ifndef _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_DEVICE_H_
#define _ST_HPC_PPL_NN_ENGINES_CUDA_CUDA_DEVICE_H_

#include "ppl/nn/common/device.h"

#include <map>
#include <random>

#include "ppl/nn/engines/cuda/data_converter.h"
#include "ppl/nn/engines/cuda/cuda_common.h"

namespace ppl { namespace nn { namespace cuda {

class CudaDevice : public Device {
public:
    CudaDevice();
    virtual ~CudaDevice();

    virtual ppl::common::RetCode Realloc(uint64_t bytes, BufferDesc* buffer) = 0;

    ppl::common::RetCode Realloc(const TensorShape& shape, BufferDesc* buffer) override final {
        return Realloc(shape.GetBytesIncludingPadding(), buffer);
    }

    virtual ppl::common::RetCode ReallocWithRandomValue(uint64_t bytes, BufferDesc* buffer) {
        auto status = Realloc(bytes, buffer);
        if (status != ppl::common::RC_SUCCESS) {
            return status;
        }
        std::default_random_engine eng;
        std::uniform_real_distribution<float> dis(-2.64f, 2.64f);
        std::unique_ptr<float[]> host_random_data(new float[bytes / sizeof(float)]);
        for (size_t i = 0; i < bytes / sizeof(float); ++i) {
            host_random_data[i] = dis(eng);
        }
        cudaMemcpyAsync(buffer->addr, host_random_data.get(), bytes, cudaMemcpyHostToDevice, context_.stream);
        return status;
    }

    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyFromHost(BufferDesc* dst, const void* src, const TensorShape&) const override final;

    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode CopyToHost(void* dst, const BufferDesc& src, const TensorShape&) const override final;

    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, uint64_t bytes) const override final;
    ppl::common::RetCode Copy(BufferDesc* dst, const BufferDesc& src, const TensorShape&) const override final;

    const DataConverter* GetDataConverter() const override final {
        return &data_converter_;
    }

    virtual ppl::common::RetCode AllocTmpBuffer(uint64_t bytes, BufferDesc* buffer) {
        return Realloc(bytes, buffer);
    }
    virtual void FreeTmpBuffer(BufferDesc* buffer) {
        Free(buffer);
    }

    std::shared_ptr<Barrier> CreateBarrier() override final;

    cudaStream_t GetStream() const {
        return context_.stream;
    }
    int GetDeviceId() const {
        return context_.device_id;
    }

    std::map<edgeid_t, BufferDesc>* GetEdge2Buffer() {
        return &edge2buffer_;
    }

private:
    CudaCtxParam context_;
    CudaDataConverter data_converter_;
    std::map<edgeid_t, BufferDesc> edge2buffer_;
};

}}} // namespace ppl::nn::cuda

#endif
