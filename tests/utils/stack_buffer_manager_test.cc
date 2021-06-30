#include "ppl/nn/utils/stack_buffer_manager.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "gtest/gtest.h"
using namespace ppl::nn;
using namespace ppl::common;

TEST(StackBufferManagerTest, all_and_free) {
    const uint64_t alignment = 128;
    const uint64_t bytes_needed = 1243;

    GenericCpuAllocator ar(alignment);
    utils::StackBufferManager mgr(&ar);

    BufferDesc buffer;
    auto status = mgr.Realloc(bytes_needed, &buffer);
    EXPECT_EQ(RC_SUCCESS, status);
    EXPECT_NE(nullptr, buffer.addr);
    mgr.Free(&buffer);

    EXPECT_LE(bytes_needed, mgr.GetAllocatedBytes());
}
