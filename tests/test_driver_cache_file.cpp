// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/driver_cache_file.h"

using Vulkan::DriverCacheFile;

TEST(DriverCacheFile, OnlyALaterSaveReplacesTheFile) {
    DriverCacheFile file;
    u32 replaced = 0;
    const auto replace = [&] {
        ++replaced;
        return true;
    };
    EXPECT_TRUE(file.Commit(10, replace));
    EXPECT_EQ(file.Kept(), 10u);
    EXPECT_FALSE(file.Commit(5, replace)) << "an older save finishing late";
    EXPECT_FALSE(file.Commit(10, replace)) << "the same save again";
    EXPECT_TRUE(file.Commit(12, replace));
    EXPECT_EQ(file.Kept(), 12u);
    EXPECT_EQ(replaced, 2u) << "the refused ones never touched the file";
}

TEST(DriverCacheFile, ASaveStillWritingIsNotKept) {
    // The review's case (FIX-GT7-023): an automatic save asked for at 32 pipelines is still
    // writing when the window closes, with no pipeline built since.
    DriverCacheFile file;
    const u32 asked_at = 32;
    const u32 at_exit = 32;
    EXPECT_TRUE(file.Kept() < at_exit) << "the exit save must write its own";
    EXPECT_TRUE(file.Commit(at_exit, [] { return true; }));
    bool late_replaced = false;
    EXPECT_FALSE(file.Commit(asked_at, [&] {
        late_replaced = true;
        return true;
    }));
    EXPECT_FALSE(late_replaced) << "the automatic save, done later, must not replace it";
    EXPECT_EQ(file.Kept(), at_exit);
}

TEST(DriverCacheFile, AFailedReplaceKeepsTheOldSave) {
    DriverCacheFile file;
    EXPECT_TRUE(file.Commit(4, [] { return true; }));
    EXPECT_FALSE(file.Commit(9, [] { return false; }));
    EXPECT_EQ(file.Kept(), 4u);
    EXPECT_TRUE(file.Commit(9, [] { return true; })) << "the same save may be tried again";
}

TEST(DriverCacheFile, ConcurrentSavesKeepTheLatest) {
    DriverCacheFile file;
    std::atomic<u32> in_replace{};
    std::atomic<bool> overlapped{};
    std::vector<std::thread> threads;
    for (u32 t = 0; t < 4; ++t) {
        threads.emplace_back([&, t] {
            for (u32 i = 1; i <= 200; ++i) {
                file.Commit(i * 4 + t, [&] {
                    overlapped = overlapped || in_replace.fetch_add(1) != 0;
                    in_replace.fetch_sub(1);
                    return true;
                });
            }
        });
    }
    for (auto& thread : threads) {
        thread.join();
    }
    EXPECT_FALSE(overlapped.load()) << "two saves replaced the file at once";
    EXPECT_EQ(file.Kept(), 200u * 4 + 3);
}
