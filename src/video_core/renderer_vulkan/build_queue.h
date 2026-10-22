// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <stop_token>

#include "common/types.h"

namespace Vulkan {

/// The pipeline build workers' jobs (PERF-019). Urgent jobs (a draw waits) run first, then
/// normal ones (read-ahead) in order, on at most `max_running` workers at once.
class BuildQueue {
public:
    using Job = std::function<void()>;
    enum class Kind { Urgent, Normal };

    explicit BuildQueue(u32 max_running_) : max_running{max_running_} {}

    void Push(Job&& job, Kind kind) {
        {
            std::scoped_lock lk{mutex};
            if (kind == Kind::Urgent) {
                jobs.push_front(std::move(job));
            } else {
                jobs.push_back(std::move(job));
            }
        }
        cv.notify_one();
    }

    /// Waits for the next job. False once stopped. A job must be followed by Finished.
    bool Pop(Job& job, std::stop_token stop) {
        std::unique_lock lk{mutex};
        if (!cv.wait(lk, stop, [&] { return !jobs.empty() && running < max_running; })) {
            return false;
        }
        job = std::move(jobs.front());
        jobs.pop_front();
        ++running;
        return true;
    }

    void Finished() {
        {
            std::scoped_lock lk{mutex};
            --running;
        }
        cv.notify_one();
    }

    /// PERF-101: no draw or read-ahead build is queued or running.
    bool ForegroundIdle() {
        std::scoped_lock lk{mutex};
        return jobs.empty() && running == 0;
    }

    /// PERF-160: jobs waiting for a worker.
    size_t Pending() {
        std::scoped_lock lk{mutex};
        return jobs.size();
    }

    /// Wakes the workers waiting for jobs, so they can see a stop request.
    void NotifyAll() {
        cv.notify_all();
    }

private:
    const u32 max_running;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<Job> jobs;
    u32 running{};
};

} // namespace Vulkan
