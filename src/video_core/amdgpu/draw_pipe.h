// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <array>
#include <atomic>
#include <vector>
#include <condition_variable>
#include <chrono>
#include <deque>
#include <unordered_map>
#include <functional>
#include <mutex>
#include <thread>

#include "common/types.h"
#include "common/unique_function.h"

namespace AmdGpu {

/// DIAG-GT7-005 (RYGT_NATIVE_AB=1 or RYGT_DIAG_RECORDER_PARTS=1): the recorder thread's time
/// while it runs jobs, by part. Each part counts its own time only (a draw inside a draw job is
/// the draw's); the time waiting for jobs is not counted.
namespace RecorderParts {
enum class Part : u32 {
    Jobs,       ///< jobs outside the parts below (copies, waits, events...)
    DrawJobs,   ///< a draw or dispatch job around the renderer's draw: registers, selection
    Draws,      ///< the native renderer's draws
    Dispatches, ///< and dispatches
    Labels,     ///< label and fence writes
    Dmas,       ///< DMA copies
    Splices,    ///< native lists run in the scheduler's command buffer
    Submits,    ///< scheduler flushes and submits (around the splices)
    Uploads,    ///< image uploads and inits before the work (PrepareImages)
    WriteBacks, ///< pictures written back to guest memory (NAT-112)
    GpuWaits,   ///< waits for the GPU (scheduler Wait)
    Ahead,      ///< PERF-GT7-008: lists begun while the recorder waits for jobs
    Count,
};
inline constexpr u32 PartCount = static_cast<u32>(Part::Count);
bool On();
/// The recorder runs jobs from here to the end of the scope (Jobs until a part starts).
class Running {
public:
    Running() noexcept;
    ~Running();
    Running(const Running&) = delete;
    Running& operator=(const Running&) = delete;

private:
    bool active;
};
/// A part, from here to the end of the scope (nothing outside Running, or on other threads).
class Scope {
public:
    explicit Scope(Part part) noexcept;
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Part previous{};
    bool active;
};
/// This thread's nanoseconds by part since the last call.
std::array<u64, PartCount> Take();
} // namespace RecorderParts

/// PERF-031: the second stage of the GPU command processor. The command thread (stage A) keeps
/// decoding PM4 and the register file; the work that records Vulkan commands and touches the
/// buffer and texture caches (draws, dispatches, copies, fences) runs here, in submission
/// order, on the recorder thread (stage B). Anything else that needs the rasterizer drains the
/// pipe first, so the caches only ever have one user.
///
/// Design after the two-stage pipeline of the Bloodborne PC port (bbport, GPL-2.0-or-later,
/// https://github.com/deadinside28/bloodborne_pc, docs/parallel_gpu.md); written for shadGT.
class DrawPipe {
public:
    using Work = Common::UniqueFunction<void>;
    /// PERF-067: work done for a job on the buffer stage thread before the recorder runs it;
    /// false when the recorder must have run the job before the stage goes on.
    using Prepare = Common::UniqueFunction<bool>;

    /// PERF-067: how a job orders against the buffer stage running ahead of the recorder.
    enum class JobKind : u8 {
        Barrier, ///< Changes what later jobs read (memory, caches): the stage waits for it.
        Neutral, ///< Reads and writes nothing later jobs' buffer work depends on.
    };

    /// `thread_init` runs first on the recorder thread (thread name, thread-local state).
    explicit DrawPipe(std::function<void()> thread_init, size_t max_pending = 4096);

    /// D5: runs on the recorder each time it runs out of work, before it waits. Set once (the
    /// recorder may already run: it calls the hook only once it saw it set).
    void SetIdleHook(std::function<void()>&& hook) {
        {
            std::scoped_lock lk{mutex};
            idle_hook = std::move(hook);
        }
        has_idle_hook.store(true, std::memory_order_release);
    }
    ~DrawPipe();

    DrawPipe(const DrawPipe&) = delete;
    DrawPipe& operator=(const DrawPipe&) = delete;

    /// Queues work behind everything pushed before it. Waits while `max_pending` jobs are queued.
    void Push(Work&& work);
    /// PERF-067: the same, with the job's kind and the work the buffer stage does for it first
    /// (only with the stage on; otherwise `prepare` is dropped and the recorder does it all).
    void Push(Work&& work, JobKind kind, Prepare&& prepare = {}, Work&& order = {});

    /// DIAG-114: the recorder's recent waits for work (longer than 50 us).
    struct Idle {
        std::chrono::steady_clock::time_point from;
        std::chrono::steady_clock::time_point to;
    };
    std::vector<Idle> RecentIdle();

    /// R5: runs first on each stage worker thread, with the worker's index.
    static inline void (*worker_init)(u32 index) = nullptr;

    /// PERF-067: whether the buffer stage thread runs (SHADGT_BUFFER_STAGE=1).
    bool HasStage() const noexcept {
        return stage_enabled;
    }

    /// DIAG-069: what the next barrier job pushed from this thread is (a static string).
    static void SetPushTag(const char* tag) noexcept {
        push_tag = tag;
    }

    /// Returns once every job pushed so far has finished. A no-op on the recorder thread.
    /// PERF-050: the caller (the command thread) then owns the caches until its next Push or
    /// ReleaseExclusive; urgent work waits for it or is run by it (RunUrgentIfExclusive).
    void Drain();

    /// PERF-050: runs `work` as soon as the caches are free: on the recorder between two jobs
    /// (not behind the queued ones), or on the command thread while it owns them after a drain.
    /// Any thread; does not wait.
    void PushUrgent(Work&& work);

    /// Recorder thread, at a point where no job is half done: runs the queued urgent work.
    void RunUrgent();

    /// Command thread: runs the queued urgent work if it owns the caches after a drain.
    void RunUrgentIfExclusive();

    /// Command thread: gives up ownership of the caches taken by Drain (before it goes idle).
    void ReleaseExclusive();

    /// Command thread: whether it owns the caches after a drain.
    bool IsExclusive() {
        std::scoped_lock lk{mutex};
        return exclusive;
    }

    /// PERF-063: jobs pushed so far (command thread) and jobs finished (any thread).
    u64 PushedJobs() const noexcept {
        return pushed_jobs.load(std::memory_order_acquire);
    }
    u64 FinishedJobs() const noexcept {
        return finished_jobs.load(std::memory_order_acquire);
    }

    bool OnRecorderThread() const {
        return std::this_thread::get_id() == recorder_id.load(std::memory_order_acquire);
    }

    std::thread::id RecorderThreadId() const {
        return recorder_id.load(std::memory_order_acquire);
    }

    struct Stats {
        u64 pushed;
        u64 drains;
        u64 drains_that_waited;
        u64 drain_wait_us;
        u64 push_waits;
        u64 recorder_busy_us;
        size_t max_queued;
        u64 stage_prepared; // PERF-067
        u64 stage_syncs;
        u64 stage_barriers;
        // DIAG-084: where the stage's time went (us).
        u64 stage_idle_us;     // waiting for jobs from the command thread
        u64 stage_prepare_us;  // preparing buffers
        u64 stage_lead_us;     // waiting because it was MaxLead jobs ahead
        u64 stage_sync_us;     // waiting for the recorder after a barrier or refused plan
    };
    /// The counters since the previous call (resets them).
    Stats TakeStats();

private:
    void Run(std::stop_token stop);
    // PERF-067
    void Forward(Work&& work);
    void RunStage(std::stop_token stop);
    void WaitFinished(u64 jobs);
    struct StageJob {
        Work run;
        Prepare prepare;
        JobKind kind;
        Work order; // R5: run by the dispatcher in job order before `prepare`
    };
    // R5: SHADGT_STAGE_WORKERS > 1: the stage thread dispatches jobs to worker threads that
    // prepare them in parallel, and forwards them to the recorder in order.
    struct InFlight {
        StageJob job;
        std::atomic<bool> done{};
        bool ok{true};
    };
    void RunDispatcher(std::stop_token stop);
    void RunWorker(std::stop_token stop, u32 index);
    void ForwardReady();
    void RedoWindow();
    u32 worker_count{1};
    std::deque<std::unique_ptr<InFlight>> window; // dispatcher only
    std::mutex work_mutex;
    std::condition_variable_any work_items_cv;
    std::deque<InFlight*> work_items;
    std::vector<std::jthread> workers;
    bool stage_enabled{};
    std::mutex stage_mutex;
    std::condition_variable_any stage_cv;
    std::deque<StageJob> stage_queue;
    u64 forwarded{}; // stage thread only
    std::atomic<u64> stage_syncs{};
    std::atomic<u64> stage_prepared{};
    std::atomic<u64> stage_barriers{};
    std::atomic<u64> stage_idle_us{};
    std::atomic<u64> stage_prepare_us{};
    std::atomic<u64> stage_lead_us{};
    std::atomic<u64> stage_sync_us{};
    static inline thread_local const char* push_tag = nullptr; // DIAG-069
    std::unordered_map<const char*, u64> barrier_tags;         // under stage_mutex
    std::chrono::steady_clock::time_point barrier_tags_since{};
    std::mutex stage_wait_mutex;
    std::condition_variable stage_wait_cv;
    std::atomic<u64> stage_wait_target{};
    std::jthread stage;

    std::mutex mutex;
    std::condition_variable_any work_cv;
    std::condition_variable_any done_cv;
    std::deque<Work> queue;
    // PERF-050
    std::deque<Work> urgent;
    std::atomic<bool> has_urgent{};
    bool exclusive{};
    bool urgent_running{};
    u64 pushed{};
    u64 finished{};
    std::atomic<u64> pushed_jobs{};
    std::atomic<u64> finished_jobs{};
    bool recorder_waiting{};
    const size_t max_pending;
    std::atomic<std::thread::id> recorder_id{};
    std::function<void()> thread_init;
    std::function<void()> idle_hook; // D5
    std::atomic<bool> has_idle_hook{false};
    Stats stats{};
    std::jthread recorder;
    std::mutex idle_mutex; // DIAG-114
    std::array<Idle, 512> idle_ring{};
    u32 idle_next{};
};

} // namespace AmdGpu
