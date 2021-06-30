#include "ppl/nn/utils/compact_buffer_manager.h"
#include "ppl/common/generic_cpu_allocator.h"
#include "gtest/gtest.h"
using namespace ppl::nn;
using namespace ppl::common;

TEST(CompactBufferManagerTest, alloc_and_free) {
    const uint64_t bytes_needed = 1000;
    const uint64_t block_size = 1024;
    const uint64_t alignment = 128;

    GenericCpuAllocator ar(alignment);
    utils::CompactBufferManager mgr(&ar, block_size);
    BufferDesc buffer;
    auto status = mgr.Realloc(bytes_needed, &buffer);
    EXPECT_EQ(RC_SUCCESS, status);
    EXPECT_NE(nullptr, buffer.addr);
    EXPECT_LE(bytes_needed, buffer.desc);
    EXPECT_EQ(0, (uintptr_t)(buffer.addr) % alignment);
    mgr.Free(&buffer);
    EXPECT_EQ(mgr.GetAllocatedBytes(), block_size);
}
