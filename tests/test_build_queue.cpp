// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/build_queue.h"

using Vulkan::BuildQueue;
using namespace std::chrono_literals;

TEST(BuildQueue, UrgentThenNormal) {
    BuildQueue queue{4};
    std::vector<std::string> order;
    queue.Push([&] { order.push_back("normal1"); }, BuildQueue::Kind::Normal);
    queue.Push([&] { order.push_back("normal2"); }, BuildQueue::Kind::Normal);
    queue.Push([&] { order.push_back("urgent"); }, BuildQueue::Kind::Urgent);
    std::stop_source stop;
    for (int i = 0; i < 3; ++i) {
        BuildQueue::Job job;
        ASSERT_TRUE(queue.Pop(job, stop.get_token()));
        job();
        queue.Finished();
    }
    EXPECT_EQ(order, (std::vector<std::string>{"urgent", "normal1", "normal2"}));
    EXPECT_TRUE(queue.ForegroundIdle());
}

TEST(BuildQueue, RunningJobsAreCapped) {
    BuildQueue queue{1};
    for (int i = 0; i < 3; ++i) {
        queue.Push([] {}, BuildQueue::Kind::Normal);
    }
    std::stop_source stop;
    BuildQueue::Job job;
    ASSERT_TRUE(queue.Pop(job, stop.get_token()));
    // A second worker waits while the first job runs.
    std::atomic<bool> got{};
    std::thread second{[&] {
        BuildQueue::Job next;
        if (queue.Pop(next, stop.get_token())) {
            got = true;
            queue.Finished();
        }
    }};
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(got.load()) << "only one job runs at a time";
    queue.Finished();
    second.join();
    EXPECT_TRUE(got.load());
    EXPECT_EQ(queue.Pending(), 1u);
}

TEST(BuildQueue, StopWakesAWaitingWorker) {
    BuildQueue queue{2};
    std::stop_source stop;
    std::atomic<bool> returned{};
    std::thread worker{[&] {
        BuildQueue::Job job;
        EXPECT_FALSE(queue.Pop(job, stop.get_token()));
        returned = true;
    }};
    std::this_thread::sleep_for(20ms);
    stop.request_stop();
    queue.NotifyAll();
    worker.join();
    EXPECT_TRUE(returned.load());
}

TEST(BuildQueue, MergedOrder) {
    // Urgent (last first), then FIX-GT7-009's screen jobs, PERF-GT7-025's soon ones, PERF-171's
    // tessellated read-ahead ones and the normal ones, each in order.
    BuildQueue queue{4};
    std::vector<std::string> order;
    queue.Push([&] { order.push_back("normal"); }, BuildQueue::Kind::Normal);
    queue.Push([&] { order.push_back("tess1"); }, BuildQueue::Kind::Tessellated);
    queue.Push([&] { order.push_back("soon1"); }, BuildQueue::Kind::Soon);
    queue.Push([&] { order.push_back("screen1"); }, BuildQueue::Kind::Screen);
    queue.Push([&] { order.push_back("urgent1"); }, BuildQueue::Kind::Urgent);
    queue.Push([&] { order.push_back("tess2"); }, BuildQueue::Kind::Tessellated);
    queue.Push([&] { order.push_back("soon2"); }, BuildQueue::Kind::Soon);
    queue.Push([&] { order.push_back("screen2"); }, BuildQueue::Kind::Screen);
    queue.Push([&] { order.push_back("urgent2"); }, BuildQueue::Kind::Urgent);
    EXPECT_EQ(queue.Pending(), 9u);
    std::stop_source stop;
    for (int i = 0; i < 9; ++i) {
        BuildQueue::Job job;
        ASSERT_TRUE(queue.Pop(job, stop.get_token()));
        job();
        queue.Finished();
    }
    EXPECT_EQ(order, (std::vector<std::string>{"urgent2", "urgent1", "screen1", "screen2",
                                               "soon1", "soon2", "tess1", "tess2", "normal"}));
    EXPECT_TRUE(queue.ForegroundIdle());
}

TEST(BuildQueue, ExtraRunningWidens) {
    BuildQueue queue{1};
    queue.Push([] {}, BuildQueue::Kind::Tessellated);
    queue.Push([] {}, BuildQueue::Kind::Normal);
    queue.SetExtraRunning(1);
    std::stop_source stop;
    BuildQueue::Job first, second;
    ASSERT_TRUE(queue.Pop(first, stop.get_token()));
    ASSERT_TRUE(queue.Pop(second, stop.get_token())) << "two may run with one extra";
    queue.Finished();
    queue.Finished();
    EXPECT_TRUE(queue.ForegroundIdle());
}

TEST(BuildQueue, MaxRunningNarrows) {
    BuildQueue queue{2};
    queue.SetMaxRunning(1);
    queue.Push([] {}, BuildQueue::Kind::Normal);
    queue.Push([] {}, BuildQueue::Kind::Normal);
    std::stop_source stop;
    BuildQueue::Job job;
    ASSERT_TRUE(queue.Pop(job, stop.get_token()));
    std::atomic<bool> got{};
    std::thread second{[&] {
        BuildQueue::Job next;
        if (queue.Pop(next, stop.get_token())) {
            got = true;
            queue.Finished();
        }
    }};
    std::this_thread::sleep_for(50ms);
    EXPECT_FALSE(got.load()) << "one job at a time after SetMaxRunning(1)";
    queue.Finished();
    second.join();
    EXPECT_TRUE(got.load());
}
