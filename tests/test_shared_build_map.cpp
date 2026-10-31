// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <future>
#include <stdexcept>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/shared_build_map.h"

using Vulkan::SharedBuildMap;

TEST(SharedBuildMap, BuildsOnceThenReuses) {
    SharedBuildMap<int> map;
    int builds = 0;
    const auto build = [&] {
        ++builds;
        return 7;
    };
    EXPECT_EQ(map.GetOrCreate(1, true, build), std::make_pair(7, false));
    EXPECT_EQ(map.GetOrCreate(1, true, build), std::make_pair(7, true));
    EXPECT_EQ(map.Find(1), 7);
    EXPECT_EQ(builds, 1);
}

TEST(SharedBuildMap, WaitersShareTheBuildInProgress) {
    SharedBuildMap<int> map;
    std::promise<void> release;
    std::promise<void> started;
    std::atomic<int> builds{0};
    auto owner = std::async(std::launch::async, [&] {
        return map.GetOrCreate(1, true, [&] {
            ++builds;
            started.set_value();
            release.get_future().wait();
            return 7;
        });
    });
    started.get_future().wait();
    auto waiter = std::async(std::launch::async, [&] {
        return map.GetOrCreate(1, true, [&] {
            ++builds;
            return 8;
        });
    });
    while (map.shared_waits.load() == 0) {
        std::this_thread::yield();
    }
    release.set_value();
    EXPECT_EQ(owner.get(), std::make_pair(7, false));
    EXPECT_EQ(waiter.get(), std::make_pair(7, true));
    EXPECT_EQ(builds.load(), 1);
}

TEST(SharedBuildMap, UrgentCallerBuildsItsOwnCopy) {
    SharedBuildMap<int> map;
    std::promise<void> release;
    std::promise<void> started;
    auto owner = std::async(std::launch::async, [&] {
        return map.GetOrCreate(1, true, [&] {
            started.set_value();
            release.get_future().wait();
            return 7;
        });
    });
    started.get_future().wait();
    EXPECT_EQ(map.GetOrCreate(1, false, [] { return 8; }), std::make_pair(8, false));
    EXPECT_EQ(map.shared_waits.load(), 0u);
    release.set_value();
    EXPECT_EQ(owner.get(), std::make_pair(7, false));
    EXPECT_EQ(map.Find(1), 8);
}

TEST(SharedBuildMap, FailedBuildReachesWaitersAndCanBeRetried) {
    SharedBuildMap<int> map;
    std::promise<void> release;
    std::promise<void> started;
    auto owner = std::async(std::launch::async, [&] {
        return map.GetOrCreate(1, true, [&]() -> int {
            started.set_value();
            release.get_future().wait();
            throw std::runtime_error{"build failed"};
        });
    });
    started.get_future().wait();
    auto waiter =
        std::async(std::launch::async, [&] { return map.GetOrCreate(1, true, [] { return 8; }); });
    while (map.shared_waits.load() == 0) {
        std::this_thread::yield();
    }
    release.set_value();
    EXPECT_THROW(owner.get(), std::runtime_error);
    EXPECT_THROW(waiter.get(), std::runtime_error);
    EXPECT_EQ(map.GetOrCreate(1, true, [] { return 9; }), std::make_pair(9, false));
}
