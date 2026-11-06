#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/upload_memory_recovery.h"

namespace {

struct Budget {
    std::uint64_t limit = 32;
    std::uint64_t allocated = 0;
    unsigned released = 0;
    unsigned deferred = 0;
};

struct Buffer {
    Budget& budget;
    std::uint64_t size;
    bool released = false;
    Buffer(Budget& budget_, std::uint64_t size_) : budget{budget_}, size{size_} {
        budget.allocated += size;
    }
    ~Buffer() {
        if (!released) {
            ++budget.deferred;
        }
    }
    std::uint64_t Size() const { return size; }
    bool Mapped() const { return true; }
    void Release() {
        budget.allocated -= size;
        ++budget.released;
        released = true;
    }
};

struct Chunk {
    std::unique_ptr<Buffer> buffer;
    std::uint64_t retire_tick = 0;
};

struct Fixture {
    Budget budget;
    std::vector<Chunk> free;
    std::deque<Chunk> retiring;
    std::vector<std::uint64_t> waits;
    std::uint64_t done = 0;
    unsigned attempts = 0;

    Chunk Make(std::uint64_t size, std::uint64_t tick = 0) {
        return {std::make_unique<Buffer>(budget, size), tick};
    }
    Chunk Recover(std::uint64_t size, std::uint64_t current, bool reuse = true) {
        return Vulkan::RecoverUploadMemory(
            size, current, reuse, free, retiring, [&] { return done; },
            [&](std::uint64_t tick) {
                waits.push_back(tick);
                done = tick;
            },
            [](std::unique_ptr<Buffer> buffer) { buffer->Release(); },
            [&] {
                ++attempts;
                return budget.allocated + size <= budget.limit ? Make(size) : Chunk{};
            });
    }
};

}

TEST(UploadMemoryRecovery, ReleasesAllocationsBeforeRetrying) {
    Fixture f;
    f.budget.limit = 16;
    f.free.push_back(f.Make(8));
    f.free.push_back(f.Make(8));
    auto result = f.Recover(16, 5);
    ASSERT_TRUE(result.buffer);
    EXPECT_EQ(f.attempts, 1u);
    EXPECT_EQ(f.budget.released, 2u);
    EXPECT_EQ(f.budget.deferred, 0u);
    EXPECT_EQ(f.budget.allocated, 16u);
}

TEST(UploadMemoryRecovery, ReusesCompletedChunkWithoutAllocation) {
    Fixture f;
    f.budget.limit = 16;
    f.retiring.push_back(f.Make(16, 3));
    auto result = f.Recover(16, 5);
    ASSERT_TRUE(result.buffer);
    EXPECT_EQ(f.waits, (std::vector<std::uint64_t>{3}));
    EXPECT_EQ(f.attempts, 0u);
    EXPECT_EQ(f.budget.released, 0u);
    EXPECT_TRUE(f.retiring.empty());
}

TEST(UploadMemoryRecovery, NeverWaitsForOrFreesUnsubmittedChunks) {
    Fixture f;
    f.budget.limit = 16;
    f.retiring.push_back(f.Make(16, 5));
    auto result = f.Recover(16, 5);
    EXPECT_FALSE(result.buffer);
    EXPECT_TRUE(f.waits.empty());
    EXPECT_EQ(f.retiring.size(), 1u);
    EXPECT_EQ(f.budget.released, 0u);
    EXPECT_EQ(f.budget.deferred, 0u);
}

TEST(UploadMemoryRecovery, ReclaimsSubmittedChunksBehindUnsubmittedOnes) {
    Fixture f;
    f.budget.limit = 24;
    f.retiring.push_back(f.Make(8, 5));
    f.retiring.push_back(f.Make(8, 3));
    f.retiring.push_back(f.Make(8, 4));
    auto result = f.Recover(16, 5);
    ASSERT_TRUE(result.buffer);
    EXPECT_EQ(f.waits, (std::vector<std::uint64_t>{4}));
    EXPECT_EQ(f.budget.released, 2u);
    EXPECT_EQ(f.budget.allocated, 24u);
    ASSERT_EQ(f.retiring.size(), 1u);
    EXPECT_EQ(f.retiring.front().retire_tick, 5u);
}

TEST(UploadMemoryRecovery, HonorsDisabledReuseAndPropagatesAllocationFailure) {
    Fixture f;
    f.budget.limit = 8;
    f.free.push_back(f.Make(8));
    auto result = f.Recover(16, 5, false);
    EXPECT_FALSE(result.buffer);
    EXPECT_EQ(f.budget.released, 1u);
    EXPECT_EQ(f.budget.allocated, 0u);
    EXPECT_EQ(f.attempts, 1u);
}
