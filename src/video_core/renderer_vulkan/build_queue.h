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

/// The pipeline build workers' jobs (PERF-019). Urgent jobs (a draw waits) run first, the last
/// one first; then FIX-GT7-009's screen ones (draws skipped again: each builds the one on screen
/// skipped most) in order; then PERF-GT7-025's soon ones (draws that went on without their
/// pipeline) in draw order; then PERF-171's tessellated read-ahead ones (the cars) in order; then
/// normal ones (read-ahead) in order; on at most `max_running` workers at once.
class BuildQueue {
public:
    using Job = std::function<void()>;
    enum class Kind { Urgent, Screen, Soon, Tessellated, Normal };

    explicit BuildQueue(u32 max_running_) : max_running{max_running_} {}

    void Push(Job&& job, Kind kind) {
        {
            std::scoped_lock lk{mutex};
            if (kind == Kind::Urgent) {
                jobs.push_front(std::move(job));
                ++urgent;
            } else if (kind == Kind::Screen) {
                screen.push_back(std::move(job));
            } else if (kind == Kind::Soon) {
                soon.push_back(std::move(job));
            } else if (kind == Kind::Tessellated) {
                tessellated.push_back(std::move(job));
            } else {
                jobs.push_back(std::move(job));
            }
        }
        cv.notify_one();
    }

    /// Waits for the next job. False once stopped. A job must be followed by Finished.
    bool Pop(Job& job, std::stop_token stop) {
        std::unique_lock lk{mutex};
        if (!cv.wait(lk, stop, [&] {
                return (!jobs.empty() || !screen.empty() || !soon.empty() ||
                        !tessellated.empty()) &&
                       running < max_running + extra_running;
            })) {
            return false;
        }
        if (urgent == 0 && !screen.empty()) {
            job = std::move(screen.front());
            screen.pop_front();
        } else if (urgent == 0 && !soon.empty()) {
            job = std::move(soon.front());
            soon.pop_front();
        } else if (urgent == 0 && !tessellated.empty()) {
            job = std::move(tessellated.front());
            tessellated.pop_front();
        } else {
            job = std::move(jobs.front());
            jobs.pop_front();
            urgent -= urgent != 0 ? 1 : 0;
        }
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
        return jobs.empty() && screen.empty() && soon.empty() && tessellated.empty() &&
               running == 0;
    }

    /// PERF-160: jobs waiting for a worker.
    size_t Pending() {
        std::scoped_lock lk{mutex};
        return jobs.size() + screen.size() + soon.size() + tessellated.size();
    }

    /// Wakes the workers waiting for jobs, so they can see a stop request.
    void NotifyAll() {
        cv.notify_all();
    }

    /// PERF-GT7-034: the limit itself (fewer while the game runs late).
    void SetMaxRunning(u32 limit) {
        {
            std::scoped_lock lk{mutex};
            max_running = limit;
        }
        cv.notify_all();
    }

    /// PERF-GT7-021: this many more jobs may run at once (0: back to the limit).
    void SetExtraRunning(u32 extra) {
        {
            std::scoped_lock lk{mutex};
            extra_running = extra;
        }
        cv.notify_all();
    }

private:
    u32 max_running;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<Job> jobs;        ///< urgent ones at the front, normal ones after
    std::deque<Job> screen;      ///< FIX-GT7-009
    std::deque<Job> soon;        ///< PERF-GT7-025
    std::deque<Job> tessellated; ///< PERF-171
    u32 urgent{};                ///< urgent jobs at the front of `jobs`
    u32 running{};
    u32 extra_running{}; ///< PERF-GT7-021
};

} // namespace Vulkan
