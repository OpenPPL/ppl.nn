#include "tensor_buffer_info_tools.h"
#include "ppl/nn/utils/generic_cpu_device.h"
#include "gtest/gtest.h"
using namespace ppl::nn;
using namespace ppl::common;
using namespace ppl::nn::test;

TEST(TensorBufferInfoTest, empty) {
    utils::GenericCpuDevice device;
    TensorBufferInfo info;
    EXPECT_EQ(RC_SUCCESS, info.SetDevice(&device));
    EXPECT_EQ(&device, info.GetDevice());
    EXPECT_EQ(RC_SUCCESS, info.SetDevice(&device));
    EXPECT_FALSE(info.IsBufferOwner());
}

TEST(TensorBufferInfoTest, with_buffer) {
    utils::GenericCpuDevice device;
    auto info = GenRandomTensorBufferInfo(&device);
    EXPECT_EQ(&device, info.GetDevice());
    EXPECT_NE(RC_SUCCESS, info.SetDevice(&device)); // cannot set device if buffer is not empty
    info.FreeBuffer();
}

TEST(TensorBufferInfoTest, setbuffer) {
    utils::GenericCpuDevice device;
    TensorBufferInfo info;

    BufferDesc buffer;
    device.Realloc(1000, &buffer);
    info.SetBuffer(buffer, &device, true);
    EXPECT_EQ(buffer.addr, info.GetBufferPtr());

    auto buf = info.DetachBuffer();
    EXPECT_EQ(nullptr, info.GetBufferPtr());
    device.Free(&buf);
}
