// SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <array>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <future>
#include <mutex>
#include <set>
#include <ranges>
#include <sstream>
#include <chrono>
#include <future>
#include <thread>
#include <string>
#include <unordered_map>

#include "common/elf_info.h"
#include "common/guest_clock.h"
#include "common/nvtx.h"
#include "common/hash.h"
#include "common/io_file.h"
#include "common/path_util.h"
#include "common/perf_monitor.h"
#include "common/recoverable.h"
#include "common/singleton.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "shader_recompiler/backend/spirv/emit_spirv.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/runtime_info.h"
#include "video_core/buffer_cache/direct_memory.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/amdgpu/read_ahead_states.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/build_queue.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/renderer_vulkan/single_sample_barycentric.h"
#include "video_core/renderer_vulkan/spirv_check.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_serialization.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"
#include "ryty_gpu/ryty_diag.h"

namespace Shader::Optimization {
extern thread_local std::span<const u32> g_diag_compiling_code;
extern thread_local u64 g_diag_compiling_hash;
} // namespace Shader::Optimization
namespace Vulkan {

u32 TessLdsEmulation(const AmdGpu::Regs& regs) {
    // FIX-227: hull shaders of draws with several patches a VGT group read their inputs from the
    // LDS emulation scratch buffer (the renderer runs a writer pass first), so input control
    // points past their own patch's are the next patches', as in GCN's LDS (GT7's car hull
    // shaders read such; with FIX-228 that ends the black patches on the car in the race start's
    // close-ups). Slots count from the patch's own first control point (as GCN, the patch id
    // folded out). RYGT_TESS_LDS_EMULATION=0: off; =1: slots from the start of the VGT group.
    static const u32 mode = [] {
        const char* env = std::getenv("RYGT_TESS_LDS_EMULATION");
        return env && *env ? u32(env[0] - '0') : 2u;
    }();
    if (mode == 0 || !regs.stage_enable.hs_en || regs.ls_hs_config.hs_input_control_points == 0 ||
        regs.ls_hs_config.num_patches <= 1) {
        return 0;
    }
    return mode;
}


namespace {
std::atomic<u32> g_shader_translations_running{};
struct TranslationActivity {
    TranslationActivity() { g_shader_translations_running.fetch_add(1, std::memory_order_relaxed); }
    ~TranslationActivity() { g_shader_translations_running.fetch_sub(1, std::memory_order_relaxed); }
    TranslationActivity(const TranslationActivity&) = delete;
    TranslationActivity& operator=(const TranslationActivity&) = delete;
};
} // namespace

bool CompilationInProgress() {
    return PipelineBuildsRunning() != 0 ||
           g_shader_translations_running.load(std::memory_order_relaxed) != 0;
}

// DIAG-GT7-073 (diagnosis, RYGT_DIAG_TRANSLATE_JOBS=1).
static bool TranslateJobsDiag() {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_DIAG_TRANSLATE_JOBS");
        return env && env[0] == '1';
    }();
    return enabled;
}

class PipelineCache::TranslateWorkers {
public:
    TranslateWorkers(const Shader::Profile& profile_, vk::Device device_, u32 count)
        : profile{profile_}, device{device_} {
        for (u32 i = 0; i < count; ++i) {
            threads.emplace_back([this, i](std::stop_token stop) { Run(stop, i); });
        }
    }
    ~TranslateWorkers() {
        for (auto& thread : threads) {
            thread.request_stop();
        }
        cv.notify_all();
    }

    void Queue(std::shared_ptr<TranslateJob> job) {
        {
            std::scoped_lock lk{mutex};
            queue.push_back(std::move(job));
        }
        cv.notify_one();
    }

private:
    void Run(std::stop_token stop, u32 index) {
        Common::SetCurrentThreadName(fmt::format("shadGT:ShaderTranslate{}", index).c_str());
        Common::PreferEfficiencyCores(); // PERF-097
        Shader::Pools pools;
        // FIX-GT7-052: a translation reads the draw's V#s and T#s through its user data when it
        // runs, and by then the game has written that memory again. On GT7's main menu 5100 of 6400
        // translations started 200-500 ms after their draw (2% of those were ever used, against 93%
        // under 10 ms): they described another draw, which queued another, for 25 s of missing
        // buildings at every launch. So the newest job goes first, and one older than a frame is
        // dropped: its draw asks again with what it reads then. RYGT_TRANSLATE_MAX_AGE_MS=0: as before.
        static const auto max_age = [] {
            const char* env = std::getenv("RYGT_TRANSLATE_MAX_AGE_MS");
            return std::chrono::milliseconds{env && *env ? std::strtol(env, nullptr, 0) : 33};
        }();
        while (!stop.stop_requested()) {
            std::shared_ptr<TranslateJob> job;
            std::vector<std::shared_ptr<TranslateJob>> too_old;
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, stop, [this] { return !queue.empty(); });
                if (max_age.count() > 0) {
                    const auto now = std::chrono::steady_clock::now();
                    while (!queue.empty() && now - queue.front()->queued_at > max_age) {
                        too_old.push_back(std::move(queue.front()));
                        queue.pop_front();
                    }
                    if (!queue.empty()) {
                        job = std::move(queue.back());
                        queue.pop_back();
                    }
                } else if (!queue.empty()) {
                    job = std::move(queue.front());
                    queue.pop_front();
                }
            }
            for (const auto& old : too_old) {
                {
                    std::scoped_lock lk{old->mutex};
                    old->too_old = true;
                    old->done = true;
                }
                old->cv.notify_all();
            }
            if (!job) {
                continue;
            }
            Translate(*job, pools);
            {
                std::scoped_lock lk{job->mutex};
                job->done = true;
            }
            job->cv.notify_all();
        }
    }

    void Translate(TranslateJob& job, Shader::Pools& pools) {
        TranslationActivity activity;
        const Shader::ShaderParams params{
            .user_data = std::span<const u32, Shader::ShaderParams::NumShaderUserData>{job.user_data},
            .code = job.code,
            .hash = job.hash,
        };
        auto info = std::make_unique<Shader::Info>(job.hw_stage, job.sw_stage, params);
        info->pinned_fetch_sharps = std::move(job.fetch_sharps); // FIX-GT7-054
        auto runtime_info = job.runtime_info;
        auto binding = job.start;
        const auto start = std::chrono::steady_clock::now();
        Shader::Optimization::g_diag_compiling_code = job.code;
        Shader::Optimization::g_diag_compiling_hash = job.hash;
        try {
            Common::RecoverableScope recoverable;
            const auto program = Shader::TranslateProgram(job.code, pools, *info, runtime_info,
                                                          profile);
            job.spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, program, binding);
        } catch (const Common::RecoverableFailure&) {
            // Translated again on the command thread when needed, which reports the failure.
            return;
        }
        if (!Common::PerfFeatureEnabled(44)) {
            info->resource_usage_known = false;
        }
        job.module = CompileSPV(job.spv, device);
        job.info = std::move(info);
        job.translate_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        // DIAG-GT7-073 (diagnosis, RYGT_DIAG_TRANSLATE_JOBS=1): what each translation made.
        if (TranslateJobsDiag()) {
            LOG_WARNING(Render_Vulkan,
                        "DIAG-GT7-073: translated #{} age {:.0f} ms ahead {} {:#x} stage {} start "
                        "{}/{} {:.1f} ms, {} code "
                        "words, spv {} words {:#x}, runtime {:#x}, user data {:#x}",
                        job.diag_id,
                        std::chrono::duration<double, std::milli>(start - job.queued_at).count(),
                        job.diag_ahead, job.hash, u32(job.hw_stage), job.start.unified, job.start.buffer,
                        job.translate_ms, job.code.size(), job.spv.size(),
                        XXH3_64bits(job.spv.data(), job.spv.size() * sizeof(u32)),
                        XXH3_64bits(&job.runtime_info, sizeof(job.runtime_info)),
                        XXH3_64bits(job.user_data.data(), sizeof(job.user_data)));
        }
    }

    const Shader::Profile& profile;
    vk::Device device;
    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<std::shared_ptr<TranslateJob>> queue;
    std::vector<std::jthread> threads;
};

// PERF-GT7-019: reads a program's stored permutations, then the SPIR-V files they list, so the
// lookup that loads one (LoadStoredModule) finds it read once already.
class PipelineCache::StoreReaders {
public:
    explicit StoreReaders(u32 count) {
        for (u32 i = 0; i < count; ++i) {
            threads.emplace_back([this, i](std::stop_token stop) { Run(stop, i); });
        }
    }
    ~StoreReaders() {
        for (auto& thread : threads) {
            thread.request_stop();
        }
        cv.notify_all();
    }

    std::future<StoredListing> List(u64 pgm_hash) {
        Job job{.pgm_hash = pgm_hash};
        auto listing = job.promise.get_future();
        {
            std::scoped_lock lk{mutex};
            queue.push_back(std::move(job));
        }
        cv.notify_one();
        return listing;
    }

    std::atomic<u64> files_read_ahead{};

private:
    // A program has ~1.4 stored permutations in GT7; more are rare and read when matched.
    static constexpr size_t MaxReadAhead = 8;

    struct Job {
        u64 pgm_hash{};
        std::promise<StoredListing> promise;
    };

    void Run(std::stop_token stop, u32 index) {
        Common::SetCurrentThreadName(fmt::format("shadGT:ShaderStore{}", index).c_str());
        std::vector<u32> words;
        while (!stop.stop_requested()) {
            Job job;
            {
                std::unique_lock lk{mutex};
                cv.wait(lk, stop, [this] { return !queue.empty(); });
                if (queue.empty()) {
                    continue;
                }
                job = std::move(queue.front());
                queue.pop_front();
            }
            StoredListing listing = ReadStoredListing(job.pgm_hash);
            std::vector<std::string> spv_names;
            for (const auto& entry : listing.entries) {
                if (entry.has_spv && spv_names.size() < MaxReadAhead) {
                    spv_names.push_back(fmt::format("{:#018x}_{}", job.pgm_hash, entry.perm_idx));
                }
            }
            job.promise.set_value(std::move(listing));
            for (const auto& name : spv_names) {
                Storage::DataBase::Instance().Load(Storage::BlobType::ShaderBinary, name, words);
                files_read_ahead.fetch_add(1, std::memory_order_relaxed);
            }
        }
    }

    std::mutex mutex;
    std::condition_variable_any cv;
    std::deque<Job> queue;
    std::vector<std::jthread> threads;
};


static bool SaneTessConstants(const Shader::Info& hull, const AmdGpu::Liverpool* liverpool,
                              bool read_ahead); // FIX-058
#ifdef RYTY_GPU
void RytyWatchWrites(std::uint64_t address, std::uint32_t ms, std::string label); // DIAG-067
#endif

// PERF-103: compile slow motion. While draws are skipped because their pipeline is still being
// built, the guest clocks run at a quarter speed: held 3 ms of every 4 (no vblank passes while
// held, so the game's frame rate drops with it), so a long compile looks like slow motion.
static std::atomic<s64> g_last_skip_ns{};

// FIX-GT7-023: the live cache, saved before the window's close ends the process.
static std::atomic<PipelineCache*> g_exit_save_cache{};
static std::mutex g_exit_save_mutex; // FIX-GT7-025

static u32 CompileMode() {
#ifdef RYTY_GPU
    static const u32 mode = static_cast<u32>(std::clamp(EmulatorSettings.GetCompileMode(), 0, 2));
    return mode;
#else
    return 2;
#endif
}

static void NoteSkippedDraw() {
    g_last_skip_ns.store(std::chrono::steady_clock::now().time_since_epoch().count(),
                         std::memory_order_relaxed);
}

#ifdef RYTY_GPU
bool RytyGtsInRace(); // FIX-110, ryty_game_patch.cpp
#endif

bool CompileSlowMotion() {
    const s64 last = g_last_skip_ns.load(std::memory_order_relaxed);
#ifdef RYTY_GPU
    // FIX-113: a race keeps going at full speed while its new pipelines build (objects show a
    // moment later instead of the race slowing down). -DisablePerf 136 slows races down too.
    static const bool race_keeps_going = Common::PerfFeatureEnabled(136);
    if (race_keeps_going && RytyGtsInRace()) {
        return false;
    }
#endif
    return CompileMode() == 1 && last != 0 &&
           std::chrono::steady_clock::now().time_since_epoch().count() - last < 50'000'000;
}

static void SlowMotionThread(std::stop_token stop) {
    Common::SetCurrentThreadName("shadGT:CompileSlowMotion");
    using namespace std::chrono_literals;
    while (!stop.stop_requested()) {
        if (!CompileSlowMotion()) {
            std::this_thread::sleep_for(5ms);
            continue;
        }
        Common::GuestClock::TakeHold();
        std::this_thread::sleep_for(3ms);
        Common::GuestClock::ReleaseHold();
        std::this_thread::sleep_for(1ms);
    }
}

// PERF-019: worker threads that build graphics pipelines ahead of the draws that need them.
class PipelineCache::PipelineBuildWorkers {
public:
    PipelineBuildWorkers() : queue{CompileConcurrency()} {
        // Concurrent driver pipeline compilations stay bounded (the queue's limit): large bursts
        // can overwhelm the driver while the game and its pipeline linker use the GPU.
        for (u32 i = 0; i < AllCores(); ++i) {
            threads.emplace_back([this](std::stop_token stop) { Work(stop); });
        }
    }
    ~PipelineBuildWorkers() {
        for (auto& thread : threads) {
            thread.request_stop();
        }
        queue.NotifyAll();
    }

    void Push(std::function<void()>&& job, BuildQueue::Kind kind) {
        queue.Push(std::move(job), kind);
    }
    bool ForegroundIdle() {
        return queue.ForegroundIdle();
    }
    size_t Pending() {
        return queue.Pending(); // DIAG-GT7-012
    }
    /// PERF-GT7-034: a build is queued or running.
    bool Busy() {
        return !queue.ForegroundIdle();
    }
    /// PERF-GT7-034: at most this many builds at once, unless a draw waits for its own.
    /// FIX-GT7-022: read-ahead builds only.
    void SetPace(u32 limit) {
        std::scoped_lock lk{extra_mutex};
        pace = limit;
        ApplyExtra();
    }
    static u32 UsualWidth() {
        return CompileConcurrency();
    }
    /// PERF-GT7-021: while a draw into a one-off target waits (the game stopped anyway), up to
    /// all but four hardware threads build at once instead of CompileConcurrency (4). Calls nest
    /// (two threads can wait at once).
    void Widen(bool on) {
        {
            std::scoped_lock lk{extra_mutex};
            waiting += on ? 1 : -1;
            ApplyExtra();
        }
        Common::SetCompileCoresWide(on); // PERF-GT7-024: off the compile cores meanwhile
    }
    /// PERF-GT7-029: during a catch-up, 8 at a time on every core; PERF-GT7-032: below normal
    /// priority (RYGT_CATCH_UP_LOW=0: normal, as before).
    void SetCatchUp(bool on) {
        {
            std::scoped_lock lk{extra_mutex};
            catch_up = on;
            ApplyExtra();
        }
        if (CatchUpLow()) {
            Common::SetCompileCatchUp(on);
        } else {
            Common::SetCompileCoresWide(on);
        }
    }
    static bool CatchUpLow() {
        static const bool low = [] {
            const char* env = std::getenv("RYGT_CATCH_UP_LOW");
            return !(env && env[0] == '0');
        }();
        return low;
    }
    /// PERF-GT7-032: while a thread builds a pipeline the GPU waits for, no more builds than
    /// usual start (unless a draw waits for its own).
    void SetForeground(bool on) {
        std::scoped_lock lk{extra_mutex};
        foreground += on ? 1 : -1;
        ApplyExtra();
    }

private:
    void ApplyExtra() {
        const u32 widest = std::clamp(AllCores() > 4 ? AllCores() - 4 : 0u,
                                      CompileConcurrency(), 12u);
        const u32 width = waiting > 0      ? widest
                          : foreground > 0 ? CompileConcurrency()
                          : catch_up       ? std::clamp(8u, CompileConcurrency(), widest)
                                           : CompileConcurrency();
        // FIX-GT7-022: the pace (PERF-GT7-034) holds back the read-ahead builds only, and they
        // leave a worker free for the draws that went on without their pipeline. On a new GT7
        // track (a city), 600 builds of draws on screen waited up to 27 s, one or two at a time,
        // while the game ran at 35-38 ms a frame either way: buildings were missing or black for
        // a while. In the race, a new building's build waited for a worker behind read-ahead
        // ones of up to 1.7 s each.
        u32 background = width;
        if (waiting == 0) {
            background = std::min(std::clamp(pace, 1u, width), std::max(width - 1, 1u));
        }
        queue.SetMaxRunning(std::min(width, CompileConcurrency()));
        queue.SetExtraRunning(width > CompileConcurrency() ? width - CompileConcurrency() : 0);
        queue.SetBackgroundLimit(background);
    }
    std::mutex extra_mutex;
    int waiting = 0;
    int foreground = 0;
    bool catch_up = false;
    u32 pace = 64; ///< PERF-GT7-034: no limit

    static u32 AllCores() {
        return std::max(std::thread::hardware_concurrency(), NumWorkers());
    }

    static u32 NumWorkers() {
        const u32 cores = std::max(std::thread::hardware_concurrency(), 4u);
        // PERF-021: all but two cores (the command thread and the game's own main thread).
        return std::clamp(cores - 2, 2u, 14u);
    }

    static u32 CompileConcurrency() {
        // Keep driver pipeline creation to a small burst even on large CPUs. Four concurrent
        // builds still use the parallel compiler without sending a dozen expensive requests to
        // the graphics driver at once.
        return std::clamp(AllCores() / 3, 2u, 4u);
    }

    void Work(std::stop_token stop) {
        Common::SetCurrentThreadName("shadGT:PipelineBuild");
        // PERF-097, PERF-GT7-024: driver compiles stay off the cores the game and the recorder
        // run on.
        Common::PreferCompileCores();
        bool low_priority = false;
        bool all_cores = false;
        bool backlog = false;
        // PERF-160: when new pipelines pile up (a new menu, the first seconds on a new track), the
        // workers may use every core, at low priority so the game's own threads keep theirs; on
        // Intel hybrid CPUs they otherwise stay on the efficiency cores (PERF-097), where a pile of
        // compiles took several times longer. -DisablePerf 160 keeps them there.
        static const bool backlog_cores = Common::PerfFeatureEnabled(160);
        while (true) {
            BuildQueue::Job job;
            if (!queue.Pop(job, stop)) {
                return;
            }
            if (backlog_cores) {
                const size_t pending = queue.Pending();
                if (pending >= 16) {
                    backlog = true;
                } else if (pending <= 2) {
                    backlog = false;
                }
            }
            // PERF-GT7-024: on a CPU with one kind of core a backlog stays on the compile cores
            // (in a race, compiles on every core slowed the game and the recorder by ~40%);
            // every core only while a draw waits for its pipeline.
            const bool want_all = (backlog && Common::HybridCpu()) || Common::CompileCoresWide();
            if (want_all != all_cores) {
                all_cores = want_all;
                if (all_cores) {
                    Common::AllowAllCores();
                } else {
                    Common::PreferCompileCores();
                }
            }
            // PERF-GT7-032: below normal during a catch-up too, and while a thread builds a
            // pipeline the GPU waits for (from the next job: one takes up to a second).
            const bool want_low = backlog || Common::CompileThreadsLow();
            if (want_low != low_priority) {
                low_priority = want_low;
                Common::SetCurrentThreadPriority(low_priority ? Common::ThreadPriority::Low
                                                              : Common::ThreadPriority::Normal);
            }
            job();
            queue.Finished();
        }
    }

    BuildQueue queue;
    std::vector<std::jthread> threads;
};

// PERF-019: what a worker needs to build one graphics pipeline. The shader infos are copied
// (with their user data) because the command thread keeps changing the cached ones.
struct PipelineCache::PipelineBuild {
    GraphicsPipelineKey key{};
    std::array<Shader::Info, MaxShaderStages> info_copies{};
    std::array<std::vector<u32>, MaxShaderStages> user_data{};
    std::array<const Shader::Info*, MaxShaderStages> infos{};
    std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
    std::optional<Shader::Gcn::FetchShaderData> fetch;
    std::array<vk::ShaderModule, MaxShaderStages> modules{};
    GraphicsPipeline::SerializationSupport sdata{};
    std::unique_ptr<GraphicsPipeline> pipeline;
    std::promise<void> promise;
    std::shared_future<void> done;
    std::chrono::steady_clock::time_point queued;
    bool urgent{};
    // PERF-076: a new pipeline its draws skip until a worker has built it.
    bool async{};
    // Whoever sets this first builds the pipeline: a worker, or the command thread when a
    // draw needs it before a worker got to it.
    std::atomic<bool> started{};
    std::atomic<bool> bumped{}; ///< PERF-GT7-029: queued again at the front
    std::atomic<u32> skips{};   ///< DIAG-GT7-012: draws skipped while it builds
    std::atomic<s64> last_skip_ms{}; ///< FIX-GT7-009: steady clock
    // The cached infos the copies were made from; the pipeline points at them once built.
    std::array<const Shader::Info*, MaxShaderStages> canonical{};
};

using Shader::HwStage;
using Shader::Output;
using Shader::SwStage;

constexpr static auto SpirvVersion1_6 = 0x00010600U;

constexpr static std::array DescriptorHeapSizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 512},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8192},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1024},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1024},
};

static u32 MapOutputs(std::span<Shader::OutputMap, 3> outputs, const AmdGpu::VsOutputControl& ctl) {
    u32 num_outputs = 0;

    if (ctl.vs_out_misc_enable) {
        auto& misc_vec = outputs[num_outputs++];
        misc_vec[0] = ctl.use_vtx_point_size ? Output::PointSize : Output::None;
        misc_vec[1] = ctl.use_vtx_edge_flag
                          ? Output::EdgeFlag
                          : (ctl.use_vtx_gs_cut_flag ? Output::GsCutFlag : Output::None);
        misc_vec[2] =
            ctl.use_vtx_kill_flag
                ? Output::KillFlag
                : (ctl.use_vtx_render_target_idx ? Output::RenderTargetIndex : Output::None);
        misc_vec[3] = ctl.use_vtx_viewport_idx ? Output::ViewportIndex : Output::None;
    }

    if (ctl.vs_out_ccdist0_enable) {
        auto& ccdist0 = outputs[num_outputs++];
        ccdist0[0] = ctl.IsClipDistEnabled(0)
                         ? Output::ClipDist0
                         : (ctl.IsCullDistEnabled(0) ? Output::CullDist0 : Output::None);
        ccdist0[1] = ctl.IsClipDistEnabled(1)
                         ? Output::ClipDist1
                         : (ctl.IsCullDistEnabled(1) ? Output::CullDist1 : Output::None);
        ccdist0[2] = ctl.IsClipDistEnabled(2)
                         ? Output::ClipDist2
                         : (ctl.IsCullDistEnabled(2) ? Output::CullDist2 : Output::None);
        ccdist0[3] = ctl.IsClipDistEnabled(3)
                         ? Output::ClipDist3
                         : (ctl.IsCullDistEnabled(3) ? Output::CullDist3 : Output::None);
    }

    if (ctl.vs_out_ccdist1_enable) {
        auto& ccdist1 = outputs[num_outputs++];
        ccdist1[0] = ctl.IsClipDistEnabled(4)
                         ? Output::ClipDist4
                         : (ctl.IsCullDistEnabled(4) ? Output::CullDist4 : Output::None);
        ccdist1[1] = ctl.IsClipDistEnabled(5)
                         ? Output::ClipDist5
                         : (ctl.IsCullDistEnabled(5) ? Output::CullDist5 : Output::None);
        ccdist1[2] = ctl.IsClipDistEnabled(6)
                         ? Output::ClipDist6
                         : (ctl.IsCullDistEnabled(6) ? Output::CullDist6 : Output::None);
        ccdist1[3] = ctl.IsClipDistEnabled(7)
                         ? Output::ClipDist7
                         : (ctl.IsCullDistEnabled(7) ? Output::CullDist7 : Output::None);
    }

    return num_outputs;
}

/// SHD-004: a color target as far as a pixel shader's translation reads it: the export format,
/// the number class of the output, the conversion, which source channel each channel takes (a
/// constant channel, as formats without alpha have, takes its own, as the translation does) and
/// the blend emulation. A target the draw doesn't bind is translated as a float target with the
/// usual packed export (its writes go nowhere: the pass has no attachment there), so a pixel
/// shader translates the same into passes with one target or two and into float targets of other
/// formats (two in five of GT Sport's pixel shader translations differed only in that).
static Shader::PsColorBuffer TranslationColorBuffer(const Shader::PsColorBuffer& cb,
                                                    bool unorm_fixup) {
    if (unorm_fixup || cb.blend_self_scale || cb.blend_swizzled_alpha ||
        cb.blend_swizzled_factors) {
        return cb; // these read the target's format too
    }
    Shader::PsColorBuffer out{};
    out.data_format = AmdGpu::DataFormat::Format16_16_16_16;
    if (cb.data_format == AmdGpu::DataFormat::FormatInvalid) {
        out.num_format = AmdGpu::NumberFormat::Float;
        out.export_format = AmdGpu::ShaderExportFormat::ABGR_FP16;
        out.swizzle = AmdGpu::IdentityMapping;
        return out;
    }
    switch (AmdGpu::GetNumberClass(cb.num_format)) {
    case AmdGpu::NumberClass::Uint:
        out.num_format = AmdGpu::NumberFormat::Uint;
        break;
    case AmdGpu::NumberClass::Sint:
        out.num_format = AmdGpu::NumberFormat::Sint;
        break;
    default:
        out.num_format = AmdGpu::NumberFormat::Float;
        break;
    }
    out.num_conversion = cb.num_conversion;
    out.export_format = cb.export_format;
    for (u32 i = 0; i < 4; ++i) {
        out.swizzle.array[i] =
            AmdGpu::CompSwizzle(u32(AmdGpu::CompSwizzle::Red) + cb.swizzle.Map(i));
    }
    return out;
}

bool PipelineCache::ReadsVertexValues(const AmdGpu::ShaderProgram& program) {
    // FIX-217: whether a pixel shader has a V_INTERP_MOV_F32 selecting P10 or P20 (a vertex's own
    // attribute value; P0 is the provoking vertex, which flat inputs give anyway). VINTRP is
    // 0b110010 in bits 31:26, the opcode in 17:16 (2: MOV) and the selector in 7:0. Each
    // program's answer is kept by its code address. DIAG-222 (RYGT_DIAG_BARY_COUNT=1) logs the
    // count of such programs.
    const VAddr address = program.Address<VAddr>();
    if (const auto it = vertex_value_programs.find(address); it != vertex_value_programs.end()) {
        return it->second;
    }
    const auto params = AmdGpu::GetParams(program);
    bool found = false;
    for (const u32 word : params.code) {
        if ((word >> 26) == 0b110010 && ((word >> 16) & 3) == 2 && (word & 0xff) < 2) {
            found = true;
            break;
        }
    }
    vertex_value_programs.emplace(address, found);
    static u32 programs = 0, matching = 0;
    ++programs;
    matching += found;
    if (std::getenv("RYGT_DIAG_BARY_COUNT") && (found || programs % 100 == 0)) {
        LOG_WARNING(Render_Vulkan, "DIAG-222: {} of {} pixel shaders read vertex values ({:#x}{})",
                    matching, programs, params.hash, found ? " does" : "");
    }
    return found;
}

const Shader::RuntimeInfo& PipelineCache::BuildRuntimeInfo(HwStage stage, SwStage l_stage) {
    auto& info = S().runtime_infos[u32(l_stage)];
    const auto& regs = Regs();
    const auto BuildCommon = [&](const auto& program) {
        info.props.num_user_data = program.settings.num_user_regs;
        info.props.num_input_vgprs = program.settings.vgpr_comp_cnt;
        info.props.num_allocated_vgprs = program.NumVgprs();
        info.props.fp_denorm_mode32 = program.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = program.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = program.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = program.settings.fp_round_mode64;
    };
    info.Initialize(stage, l_stage);
    switch (stage) {
    case HwStage::Local: {
        BuildCommon(regs.ls_program);
        Shader::TessellationDataConstantBuffer tess_constants{};
        const auto* hull_info = S().infos[u32(SwStage::TessellationControl)];
        if (hull_info) { // FIX-058: no hull info when its draw is skipped
            hull_info->ReadTessConstantBuffer(tess_constants);
        }
        info.hw.ls.ls_stride = tess_constants.ls_stride;
        break;
    }
    case HwStage::Hull:
        BuildCommon(regs.hs_program);
        break;
    case HwStage::Export:
        BuildCommon(regs.es_program);
        info.hw.es.vertex_data_size = regs.vgt_esgs_ring_itemsize;
        break;
    case HwStage::Geometry: {
        BuildCommon(regs.gs_program);
        info.hw.gs.num_outputs = MapOutputs(info.hw.gs.outputs, regs.vs_output_control);
        info.hw.gs.output_vertices = regs.vgt_gs_max_vert_out;
        info.hw.gs.num_invocations =
            regs.vgt_gs_instance_cnt.IsEnabled() ? regs.vgt_gs_instance_cnt.count : 1;
        if (regs.stage_enable.raw == AmdGpu::ShaderStageEnable::LsHsEsGs) {
            info.hw.gs.in_primitive = [&]() {
                switch (regs.tess_config.topology) {
                case AmdGpu::TessellationTopology::Point:
                    return AmdGpu::PrimitiveType::PointList;
                case AmdGpu::TessellationTopology::Line:
                    return AmdGpu::PrimitiveType::LineList;
                case AmdGpu::TessellationTopology::TriangleCw:
                case AmdGpu::TessellationTopology::TriangleCcw:
                    return AmdGpu::PrimitiveType::TriangleList;
                default:
                    UNREACHABLE();
                }
            }();
        } else {
            info.hw.gs.in_primitive = regs.primitive_type;
        }
        for (u32 stream_id = 0; stream_id < Shader::GsMaxOutputStreams; ++stream_id) {
            info.hw.gs.out_primitive[stream_id] =
                regs.vgt_gs_out_prim_type.GetPrimitiveType(stream_id);
        }
        info.hw.gs.in_vertex_data_size = regs.vgt_esgs_ring_itemsize;
        info.hw.gs.out_vertex_data_size = regs.vgt_gs_vert_itemsize[0];
        info.hw.gs.mode = regs.vgt_gs_mode.mode;
        const auto params_vc = AmdGpu::GetParams(regs.vs_program);
        info.hw.gs.vs_copy = params_vc.code;
        info.hw.gs.vs_copy_hash = params_vc.hash;
        DumpShader(info.hw.gs.vs_copy, info.hw.gs.vs_copy_hash, Shader::HwStage::Vertex, 0,
                   "copy.bin");
        break;
    }
    case HwStage::Vertex: {
        BuildCommon(regs.vs_program);
        info.hw.vs.user_clip_plane_mask = regs.clipper_control.user_clip_plane_enable;
        info.hw.vs.num_outputs = MapOutputs(info.hw.vs.outputs, regs.vs_output_control);
        info.hw.vs.emulate_depth_negative_one_to_one =
            !instance.IsDepthClipControlSupported() &&
            regs.clipper_control.clip_space == AmdGpu::ClipSpace::MinusWToW;
        info.hw.vs.clip_disable = regs.IsClipDisabled();
        break;
    }
    case HwStage::Fragment: {
        BuildCommon(regs.ps_program);
        info.hw.fs.en_flags = regs.ps_input_ena;
        info.hw.fs.addr_flags = regs.ps_input_addr;
        info.hw.fs.num_inputs = regs.num_interp;
        static const bool direct_barycentric_requested = SingleSampleBarycentricRequested();
        // FIX-217: only for the fragment shaders 217d4011 is about (a V_INTERP_MOV reading a
        // vertex's own value, P10 or P20); every other shader keeps the hardware interpolation and
        // its 27b translation and cache entries. On for all fragment shaders, it made AMD retranslate
        // and recompile everything with heavier shaders. RYGT_AMD_BARYCENTRIC_ALL=1: all, as before.
        static const bool all_shaders = [] {
            const char* env = std::getenv("RYGT_AMD_BARYCENTRIC_ALL");
            return env && env[0] == '1';
        }();
        static const bool diag_count = std::getenv("RYGT_DIAG_BARY_COUNT") != nullptr;
        const bool candidate = UseSingleSampleBarycentric(
            instance.GetVendorID(), profile.supports_fragment_shader_barycentric,
            S().graphics_key.num_samples, direct_barycentric_requested);
        info.hw.fs.single_sample_barycentric =
            candidate && (all_shaders || ReadsVertexValues(regs.ps_program));
        if (diag_count) {
            ReadsVertexValues(regs.ps_program); // DIAG-222: counted on any GPU
        }
        info.hw.fs.front_face_all_bits = regs.barycentric_control.front_face_all_bits;
        info.hw.fs.quad_sample_mask =
            regs.reg_array[0xa30e] != ~0u || regs.reg_array[0xa30f] != ~0u;
        info.hw.fs.num_samples =
            regs.ps_input_addr.sample_coverage_ena && regs.ps_input_ena.sample_coverage_ena
                ? regs.aa_config.NumSamples()
                : 1;
        info.hw.fs.z_export_format = regs.z_export_format;
        u8 stencil_ref_export_enable = regs.depth_shader_control.stencil_op_val_export_enable |
                                       regs.depth_shader_control.stencil_test_val_export_enable;
        info.hw.fs.mrtz_mask = regs.depth_shader_control.z_export_enable |
                               (stencil_ref_export_enable << 1) |
                               (regs.depth_shader_control.mask_export_enable << 2) |
                               (regs.depth_shader_control.coverage_to_mask_enable << 3);
        const auto& cb0_blend = regs.blend_control[0];
        if (cb0_blend.enable) {
            info.hw.fs.dual_source_blending =
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_dst_factor) ||
                LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.color_src_factor);
            if (cb0_blend.separate_alpha_blend) {
                info.hw.fs.dual_source_blending |=
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_dst_factor) ||
                    LiverpoolToVK::IsDualSourceBlendFactor(cb0_blend.alpha_src_factor);
            }
        } else {
            info.hw.fs.dual_source_blending = false;
        }
        info.hw.fs.dual_source_blending |= S().graphics_key.color_buffers[0].blend_swizzled_alpha ||
                                           S().graphics_key.color_buffers[0].blend_swizzled_factors;
        const auto& ps_inputs = regs.ps_inputs;
        for (u32 i = 0; i < regs.num_interp; i++) {
            info.hw.fs.inputs[i] = {
                .param_index = u8(ps_inputs[i].input_offset),
                .is_default = bool(ps_inputs[i].use_default),
                .is_flat = bool(ps_inputs[i].flat_shade),
                .default_value = u8(ps_inputs[i].default_value),
            };
        }
        for (u32 i = 0; i < Shader::MaxColorBuffers; i++) {
            info.hw.fs.color_buffers[i] = TranslationColorBuffer(
                S().graphics_key.color_buffers[i], profile.needs_unorm_fixup); // SHD-004
        }
        // Lowered user clip planes ride the same emulation path as guest-exported distances, so
        // the fragment side arms whenever the hardware vertex stage lowers them, keeping its input
        // locations in sync with the shifted vertex outputs.
        const bool lowers_user_clip_planes =
            regs.clipper_control.user_clip_plane_enable &&
            !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Geometry));
        info.hw.fs.clip_distance_emulation =
            ((regs.vs_output_control.clip_distance_enable &&
              !regs.stage_enable.IsStageEnabled(static_cast<u32>(HwStage::Local))) ||
             lowers_user_clip_planes) &&
            profile.needs_clip_distance_emulation;
        // AMD-307: the vertex order check takes locations 30/31. The stage feeding the
        // rasterizer exports NumExports() parameters, at most two locations further on (clip
        // distances, layer), so the pixel shader reads the check only when those stay below 30;
        // the vertex side checks its own outputs the same way and then always writes it.
        info.hw.fs.vertex_order_check =
            profile.vertex_order_check &&
            regs.vs_output_config.NumExports() + 2 <= 30; // OrderCheckRefLocation
        break;
    }
    case HwStage::Compute: {
        const auto& cs_pgm = liverpool->DrawCsRegs();
        info.props.num_user_data = cs_pgm.settings.num_user_regs;
        info.props.num_allocated_vgprs = cs_pgm.settings.num_vgprs * 4;
        info.props.fp_denorm_mode32 = cs_pgm.settings.fp_denorm_mode32;
        info.props.fp_denorm_mode16_64 = cs_pgm.settings.fp_denorm_mode64;
        info.props.fp_round_mode32 = cs_pgm.settings.fp_round_mode32;
        info.props.fp_round_mode16_64 = cs_pgm.settings.fp_round_mode64;
        info.hw.cs.workgroup_size = {cs_pgm.num_thread_x.full, cs_pgm.num_thread_y.full,
                                     cs_pgm.num_thread_z.full};
        info.hw.cs.tgid_enable = {cs_pgm.IsTgidEnabled(0), cs_pgm.IsTgidEnabled(1),
                                  cs_pgm.IsTgidEnabled(2)};
        info.hw.cs.shared_memory_size = cs_pgm.SharedMemSize();
        break;
    }
    default:
        break;
    }
    switch (l_stage) {
    case SwStage::Vertex:
        info.sw.vs.step_rate_0 = regs.vgt_instance_step_rate_0;
        info.sw.vs.step_rate_1 = regs.vgt_instance_step_rate_1;
        info.sw.vs.vertex_sgpr_offset = S().draw_indirect_params.vertex_sgpr_offset;
        info.sw.vs.instance_sgpr_offset = S().draw_indirect_params.instance_sgpr_offset;
        info.sw.vs.is_indirect = S().draw_indirect_params.is_indirect; // FIX-192
        info.sw.vs.tess_emulated_primitive =
            regs.primitive_type == AmdGpu::PrimitiveType::RectList ||
            (regs.primitive_type == AmdGpu::PrimitiveType::QuadList && !QuadListAsTriangles());
        break;
    case SwStage::TessellationControl: {
        info.sw.tcs.num_input_control_points = regs.ls_hs_config.hs_input_control_points;
        info.sw.tcs.num_threads = regs.ls_hs_config.hs_output_control_points;
        info.sw.tcs.tess_type = regs.tess_config.type;
        info.sw.tcs.offchip_lds_enable = regs.hs_program.settings.oc_lds_en;
        info.sw.tcs.lds_emulation = u8(TessLdsEmulation(regs)); // FIX-227
        info.sw.tcs.num_patches = u8(info.sw.tcs.lds_emulation ? regs.ls_hs_config.num_patches : 0);
        // DIAG-239 (diagnosis, RYGT_DIAG_TCS_CONFIG=1): the LS/HS configuration, 60 lines at most.
        if (static const bool diag = std::getenv("RYGT_DIAG_TCS_CONFIG") != nullptr; diag) {
            static std::mutex seen_mutex;
            static std::set<u32> seen;
            const u32 combo = u32(regs.ls_hs_config.num_patches) |
                              (u32(regs.ls_hs_config.hs_input_control_points) << 8) |
                              (u32(regs.ls_hs_config.hs_output_control_points) << 16);
            std::scoped_lock lock{seen_mutex};
            if (seen.insert(combo).second) {
                LOG_WARNING(Render_Vulkan, "DIAG-239: hs patches {} input cp {} output threads {} "
                            "offchip {} tess type {}",
                            u32(regs.ls_hs_config.num_patches),
                            u32(regs.ls_hs_config.hs_input_control_points),
                            u32(regs.ls_hs_config.hs_output_control_points),
                            u32(regs.hs_program.settings.oc_lds_en), u32(regs.tess_config.type));
            }
        }
        break;
    }
    case SwStage::TessellationEval: {
        info.sw.tes.tess_type = regs.tess_config.type;
        info.sw.tes.tess_topology = regs.tess_config.topology;
        info.sw.tes.tess_partitioning = regs.tess_config.partitioning;
        break;
    }
    default:
        break;
    }
    return info;
}

// PERF-GT7-037: GT7 uses the same few compute pipelines in every race, some of which take the
// driver ~1 s to build (0x2f3cb32d18925709, 0xe1682ae083e54205), on the command thread while
// the game waits for them: two frames of ~1 s in every race the driver's cache lacks them (after
// a driver update, or a cleared cache). Each compute pipeline is kept once as a recipe, and
// built again in the background at each start, the largest first (quick for those the driver
// has).
// RYGT_WARM_COMPUTE=0: neither.
static bool WarmComputeOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_WARM_COMPUTE");
        return !(env && env[0] == '0');
    }();
    return on;
}

std::filesystem::path PipelineCache::WarmComputeDir() const {
    const auto& serial = Common::Singleton<Common::ElfInfo>::Instance()->GameSerial();
    // FIX-GT7-026: a folder per GPU and driver (the pipeline cache UUID changes with the
    // driver): recipes hold the subgroup size and push descriptors of the device that made them,
    // so another GPU's are never built here. Those of the folder itself (before) are left there.
    const auto props = instance.GetPhysicalDevice().getProperties();
    std::string uuid;
    for (const u8 byte : props.pipelineCacheUUID) {
        uuid += fmt::format("{:02x}", byte);
    }
    return Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
           fmt::format("{}.warm_compute", serial) /
           fmt::format("{:04x}-{:04x}-{}", props.vendorID, props.deviceID, uuid);
}

void PipelineCache::WarmComputePipelines(std::stop_token stop) {
    Common::SetCurrentThreadName("shadGT:WarmCompute");
    Common::PreferCompileCores();
    Common::SetCurrentThreadPriority(Common::ThreadPriority::Low);
    std::error_code ec;
    // The largest SPIR-V first: the ones that take the driver longest (GT7's two of ~1 s are
    // ~95 KB recipes, most others under half that). A race uses ~100 compute pipelines.
    std::vector<std::pair<u64, std::filesystem::path>> sized;
    for (const auto& entry : std::filesystem::directory_iterator(WarmComputeDir(), ec)) {
        if (entry.path().extension() == ".bin" && sized.size() < 1024) {
            std::error_code size_ec;
            sized.emplace_back(entry.file_size(size_ec), entry.path());
        }
    }
    std::ranges::sort(sized, std::greater{}, &std::pair<u64, std::filesystem::path>::first);
    std::vector<std::filesystem::path> files;
    for (auto& [size, path] : sized) {
        files.push_back(std::move(path));
    }
    if (files.empty()) {
        return;
    }
    u32 warmed = 0;
    double total_ms = 0;
    double longest_ms = 0;
    for (const auto& file : files) {
        if (stop.stop_requested()) {
            return;
        }
        const auto start = std::chrono::steady_clock::now();
        if (ComputePipeline::WarmFromRecipe(instance, *pipeline_cache, file)) {
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
            ++warmed;
            total_ms += ms;
            longest_ms = std::max(longest_ms, ms);
        }
    }
    LOG_INFO(Render_Vulkan,
             "PERF-GT7-037: {} of {} compute pipelines built ahead in the background ({:.0f} ms, "
             "the longest {:.0f} ms)",
             warmed, files.size(), total_ms, longest_ms);
}

PipelineCache::PipelineCache(const Instance& instance_, Scheduler& scheduler_,
                             AmdGpu::Liverpool* liverpool_, u32 sparse_page_shift)
    : instance{instance_}, scheduler{scheduler_}, liverpool{liverpool_},
      desc_heap{instance, scheduler.GetWorkSemaphore(), DescriptorHeapSizes} {
    if (liverpool) {
        liverpool->on_command_buffer_start = [this] { ReadAheadAtBufferStart(); };
    }
    const auto& vk12_props = instance.GetVk12Properties();
    profile = Shader::Profile{
        .max_viewport_width = instance.GetMaxViewportWidth(),
        .max_viewport_height = instance.GetMaxViewportHeight(),
        .max_shared_memory_size = instance.MaxComputeSharedMemorySize(),
        .supported_spirv = SpirvVersion1_6,
        .subgroup_size = instance.SubgroupSize(),
        .sparse_page_shift = sparse_page_shift,
        .support_int8 = instance.IsShaderInt8Supported(),
        .support_int16 = instance.IsShaderInt16Supported(),
        .support_int64 = instance.IsShaderInt64Supported(),
        .support_float16 = instance.IsShaderFloat16Supported(),
        .support_float64 = instance.IsShaderFloat64Supported(),
        .supports_denorm_behavior_independence =
            vk12_props.denormBehaviorIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .supports_rounding_mode_independence =
            vk12_props.roundingModeIndependence != vk::ShaderFloatControlsIndependence::eNone,
        .support_fp16_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat16),
        .support_fp16_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat16),
        .support_fp16_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat16),
        .support_fp32_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat32),
        // FIX-223 (RYGT_DISCARD_KILL=0 or 1 overrides): AMD ends discarded pixels at once.
        .discard_kills = [&] {
            const char* env = std::getenv("RYGT_DISCARD_KILL");
            return env && *env ? *env == '1' : instance.GetVendorID() == 0x1002;
        }(),
        .support_fp32_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat32),
        .support_fp32_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat32),
        .support_fp64_denorm_preserve = bool(vk12_props.shaderDenormPreserveFloat64),
        .support_fp64_denorm_flush = bool(vk12_props.shaderDenormFlushToZeroFloat64),
        .support_fp64_round_to_zero = bool(vk12_props.shaderRoundingModeRTZFloat64),
        .support_fp16_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat16),
        .support_fp32_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat32),
        .support_fp64_signed_zero_inf_nan_preserve =
            bool(vk12_props.shaderSignedZeroInfNanPreserveFloat64),
        .supports_image_load_store_lod = instance_.IsImageLoadStoreLodSupported(),
        .supports_native_cube_calc = instance_.IsAmdGcnShaderSupported(),
        .supports_trinary_minmax = instance_.IsAmdShaderTrinaryMinMaxSupported(),
        .supports_buffer_fp32_atomic_min_max =
            instance_.IsShaderAtomicFloatBuffer32MinMaxSupported(),
        .supports_image_fp32_atomic_min_max = instance_.IsShaderAtomicFloatImage32MinMaxSupported(),
        .supports_buffer_int64_atomics = instance_.IsBufferInt64AtomicsSupported(),
        .supports_shared_int64_atomics = instance_.IsSharedInt64AtomicsSupported(),
        .supports_workgroup_explicit_memory_layout =
            instance_.IsWorkgroupMemoryExplicitLayoutSupported(),
        .supports_amd_shader_explicit_vertex_parameter =
            instance_.IsAmdShaderExplicitVertexParameterSupported(),
        .supports_fragment_shader_barycentric = instance_.IsFragmentShaderBarycentricSupported(),
        .supports_shader_subgroup_clock = instance_.IsShaderSubgroupClockSupported(),
        // DIAG-230 (diagnosis): RYGT_DIAG_NO_MANUAL_INTERP turns off per-vertex interpolation.
        .amd_vertex_order = instance_.GetAmdVertexOrder(), // AMD-311
        // AMD-313: FIX-215/216 drew the purple car again on an RX 7600 (fresh store), so RX 7000
        // cards leave them out; SHADGT_AMD_NAN_RULES=0/1 overrides for any GPU.
        .amd_nan_rules = [] {
            if (const char* env = std::getenv("SHADGT_AMD_NAN_RULES"); env && *env) {
                return env[0] != '0';
            }
            return GetStoreGpuFamily() != StoreGpuFamily::AmdRdna3;
        }(),
        .supports_quad_derivatives = instance_.IsQuadDerivativesSupported(), // AMD-306
        .vertex_order_check = instance_.IsVertexOrderCheckEnabled(),        // AMD-307
        .needs_manual_interpolation = instance.IsFragmentShaderBarycentricSupported() &&
                                      instance.GetDriverID() == vk::DriverId::eNvidiaProprietary &&
                                      !std::getenv("RYGT_DIAG_NO_MANUAL_INTERP"),
        // AMD-315: on AMD's explicit vertex path a V_INTERP_MOV makes the whole attribute
        // per-vertex, and the other components' V_INTERP_P2 reads then return vertex 0: flat,
        // single-colour triangles (hills at Fuji on an RX 6600 XT). AMD's older compiler keeps
        // that path (AMD-314), so it does GCN's P1/P2 arithmetic by hand, as 217d4011 does for
        // the KHR path. SHADGT_MANUAL_INTERP=0/1 overrides.
        .needs_manual_interpolation = [&] {
            if (const char* env = std::getenv("SHADGT_MANUAL_INTERP"); env && *env) {
                return env[0] != '0';
            }
            return (instance.IsFragmentShaderBarycentricSupported() &&
                    instance.GetDriverID() == vk::DriverId::eNvidiaProprietary) ||
                   (instance.IsAmdShaderExplicitVertexParameterSupported() &&
                    instance.IsAmdLegacyShaderCompiler());
        }(),
        .needs_lds_barriers = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary ||
                              instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .intel_arc_workarounds = instance.IsIntel(),
        .needs_buffer_offsets = instance.StorageMinAlignment() > 4,
        .needs_unorm_fixup = instance.GetDriverID() == vk::DriverId::eMesaKosmickrisp,
        .needs_clip_distance_emulation = instance.GetDriverID() == vk::DriverId::eNvidiaProprietary,
        .supports_shader_stencil_export = instance_.IsShaderStencilExportSupported(),
    };
    // D2: read-only guest buffers are read in place through the direct-memory page table.
    profile.direct_buffers =
        VideoCore::DirectMemory::Enabled() && instance.IsExternalMemoryHostSupported();
    if (SingleSampleBarycentricRequested() && instance.GetVendorID() == 0x1002) {
        LOG_WARNING(Render_Vulkan, "AMD-BARY: single-sample interpolation on; KHR {}",
                    profile.supports_fragment_shader_barycentric);
    }
    SetVertexOrderCheckCacheEnabled(profile.vertex_order_check); // AMD-307
    SetSingleSampleBarycentricCacheEnabled(UseSingleSampleBarycentric(
        instance.GetVendorID(), profile.supports_fragment_shader_barycentric, 1,
        SingleSampleBarycentricRequested()));
    CreateDriverCache();
    g_exit_save_cache.store(this); // FIX-GT7-023
    if (WarmComputeOn()) { // PERF-GT7-037
        warm_compute_thread =
            std::jthread([this](std::stop_token stop) { WarmComputePipelines(stop); });
    }
    // PERF-032: the shader store: a program's stored translations are read when a draw first
    // uses it, and new translations are stored. -DisablePerf 47 leaves the store closed.
    // FIX-186 (RyGT): at any RAM setting. The launcher's 16 GB setting turns
    // pipeline_cache_enabled off for the driver's cache file, and that closed the store too, so
    // with 16 GB nothing was written to or read from the shaders folder (Lance, 2026-10-21).
#ifdef RYTY_GPU
    const bool store_wanted = true;
#else
    const bool store_wanted = EmulatorSettings.IsPipelineCacheEnabled();
#endif
    // FIX-GT7-029 turned the translation store off: a stored translation's resource description is
    // trusted as stored, and one recorded wrongly (Lance's GT7 shader 0xfb02f94a: the same SPIR-V,
    // other buffer sources) made every later start feed that shader the wrong buffers; GT7 waited
    // on it and the World Map ran at 13-16 fps. FIX-GT7-053: on again. Without it nothing compiled
    // before was there at the next start: the main menu's buildings were missing for seconds at
    // each one (7300 draws skipped on a launch that had to translate, 177 reading the store).
    // RYGT_SHADER_STORE=0: translations are made each start.
    static const bool store_on = [] {
        const char* env = std::getenv("RYGT_SHADER_STORE");
        return !(env && env[0] == '0');
    }();
    if (!store_on) {
        LOG_WARNING(Render_Vulkan, "FIX-GT7-053: shader translations are made each start "
                                   "(RYGT_SHADER_STORE=0)");
    }
    if (store_on && store_wanted && Common::PerfFeatureEnabled(47)) {
        if (EmulatorSettings.IsPipelineCacheArchived()) {
            LOG_WARNING(Render_Vulkan, "PERF-032: the shader store is not used with an archived "
                                       "pipeline cache (pipeline_cache_archived)");
        } else {
            Storage::DataBase::Instance().Open();
            use_stored_shaders = true;
            // D10: the old renderer's translations stand in while the new renderer's are built.
            // SHADGT_LEGACY_FALLBACK=0 turns it off.
            // SHD-002: not with stage descriptor sets: the old translations bind everything
            // in set 0.
            if (profile.direct_buffers && false) {
                const char* fallback_env = std::getenv("SHADGT_LEGACY_FALLBACK");
                if (!fallback_env || fallback_env[0] != '0') {
                    Storage::DataBase::Legacy().OpenLegacy();
                    legacy_fallback = Storage::DataBase::Legacy().IsOpened();
                    legacy_stats.since = std::chrono::steady_clock::now();
                    LOG_WARNING(Render_Vulkan, "D10: old-renderer shader fallback {}",
                                legacy_fallback ? "on" : "off (no old shader store)");
                }
            }
            LOG_WARNING(Render_Vulkan,
                        "PERF-032: shader translations are loaded from the store on first use");
            // PERF-GT7-019 (RYGT_STORE_READERS=0: read on the GPU threads as before).
            const char* readers_env = std::getenv("RYGT_STORE_READERS");
            if (!(readers_env && readers_env[0] == '0')) {
                store_readers = std::make_unique<StoreReaders>(8);
                store_stats.since = std::chrono::steady_clock::now();
            }
        }
    }
    LoadElseScopeFixed(); // after the store's Open: SHD-010's clear removes them with it
    if (CompileMode() == 1 && AsyncPipelines()) {
        static std::jthread slow_motion{SlowMotionThread}; // PERF-103
    }
    // FIX-078: a quarter of the hardware threads (at least 2) translate read-ahead shaders.
    // PERF-GT7-028: half (8 of 16): at a new track's start, 421 shaders took 4.4 s of translation
    // in 3.6 s on 4 workers, and draws waited 0.5-0.85 s a frame for theirs.
    if (Common::PerfFeatureEnabled(102)) {
        const u32 count = std::clamp(std::thread::hardware_concurrency() / 2, 2u, 8u);
        translate_workers =
            std::make_unique<TranslateWorkers>(profile, instance.GetDevice(), count);
        translate_stats.since = std::chrono::steady_clock::now();
    }
}

PipelineCache::~PipelineCache() {
    {
        std::scoped_lock lk{g_exit_save_mutex}; // FIX-GT7-025: after an exit save in progress
        g_exit_save_cache.store(nullptr);       // FIX-GT7-023
    }
    if (warm_compute_thread.joinable()) { // PERF-GT7-037: before the caches go
        warm_compute_thread.request_stop();
        warm_compute_thread.join();
    }
    if (liverpool) {
        liverpool->on_command_buffer_start = nullptr;
    }
    SaveDriverCache(true);
}

// FIX-165 (RyGT): the driver's pipeline cache file is off unless RYGT_DRIVER_CACHE=1. GT Sport's grew to
// 3.7 GB: it was loaded whole at start, and each save (every 30 s while new pipelines were made)
// held the driver for 1.4-4.3 s and then wrote for 11 s. Lance's Lago Maggiore races dropped to
// 10 fps and back again and again, and once the menus froze for a minute. The driver keeps its
// own shader cache, and races ran no slower without the file (2026-10-16). With
// RYGT_DRIVER_CACHE=1 the file is loaded, and saved only when the game ends.
// PERF-164: on again by default (RYGT_DRIVER_CACHE=0 turns it off). GT7's file is about 180 MB,
// read in 68 ms and written in 0.5 s off the GPU thread, and it is saved during play (GT7 never
// reaches the exit-time save): each session adds the pipelines of the cars and tracks it met.
// PERF-167: the driver cache file's size limit, from the launcher's Graphics page
// (RYGT_DRIVER_CACHE_MAX_MB): 0 turns the file off, -1 has no limit; 5 GB by default.
static u64 DriverCacheMaxBytes() {
    static const u64 limit = [] {
        const char* env = std::getenv("RYGT_DRIVER_CACHE_MAX_MB");
        if (!env || !*env) {
            return u64{5} << 30;
        }
        const long long mb = std::strtoll(env, nullptr, 10);
        return mb < 0 ? ~u64{0} : u64(mb) << 20;
    }();
    return limit;
}

static bool DriverCacheFileEnabled() {
#ifdef RYTY_GPU
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_DRIVER_CACHE");
        return !(env && env[0] == '0') && DriverCacheMaxBytes() != 0;
    }();
    return enabled && EmulatorSettings.IsPipelineCacheEnabled();
#else
    return EmulatorSettings.IsPipelineCacheEnabled();
#endif
}

void PipelineCache::CreateDriverCache() {
    std::vector<u8> data;
#ifdef RYTY_GPU
    if (!DriverCacheFileEnabled()) {
        const auto& game_serial = Common::Singleton<Common::ElfInfo>::Instance()->GameSerial();
        const auto unused = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                            fmt::format("{}.vkpipelinecache", game_serial);
        std::error_code ec;
        if (std::filesystem::exists(unused, ec)) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-165: the driver pipeline cache file is not used (RYGT_DRIVER_CACHE=1 "
                        "turns it on); {} ({} MB) can be deleted",
                        unused.string(), std::filesystem::file_size(unused, ec) >> 20);
        }
    }
#endif
    if (DriverCacheFileEnabled()) {
        const auto& serial = Common::Singleton<Common::ElfInfo>::Instance()->GameSerial();
        driver_cache_path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) /
                            fmt::format("{}.vkpipelinecache", serial);
        const Common::FS::IOFile file{driver_cache_path, Common::FS::FileAccessMode::Read};
        if (file.IsOpen()) {
            data.resize(file.GetSize());
            if (file.Read(data) != data.size()) {
                data.clear();
            }
        }
        // Header (Vulkan spec, VkPipelineCacheHeaderVersionOne): size, version, vendor,
        // device, UUID. Data from another GPU or driver starts an empty cache.
        const auto props = instance.GetPhysicalDevice().getProperties();
        struct Header {
            u32 size;
            u32 version;
            u32 vendor_id;
            u32 device_id;
            std::array<u8, VK_UUID_SIZE> uuid;
        } header{};
        if (data.size() < sizeof(Header)) {
            data.clear();
        } else {
            std::memcpy(&header, data.data(), sizeof(Header));
            if (header.version != static_cast<u32>(vk::PipelineCacheHeaderVersion::eOne) ||
                header.vendor_id != props.vendorID || header.device_id != props.deviceID ||
                std::memcmp(header.uuid.data(), props.pipelineCacheUUID.data(), VK_UUID_SIZE)) {
                LOG_INFO(Render_Vulkan, "Driver pipeline cache is from another GPU or driver");
                data.clear();
            }
        }
    }
    auto [cache_result, cache] =
        instance.GetDevice().createPipelineCacheUnique(vk::PipelineCacheCreateInfo{
            .initialDataSize = data.size(),
            .pInitialData = data.data(),
        });
    if (cache_result != vk::Result::eSuccess && !data.empty()) {
        LOG_WARNING(Render_Vulkan, "Driver pipeline cache rejected ({}), starting empty",
                    vk::to_string(cache_result));
        auto [empty_result, empty_cache] = instance.GetDevice().createPipelineCacheUnique({});
        cache_result = empty_result;
        cache = std::move(empty_cache);
    }
    ASSERT_MSG(cache_result == vk::Result::eSuccess, "Failed to create pipeline cache: {}",
               vk::to_string(cache_result));
    pipeline_cache = std::move(cache);
    pipelines_at_driver_save = num_new_pipelines;
    driver_cache_saved_at = std::chrono::steady_clock::now();
    driver_cache_bytes.store(data.size(), std::memory_order_relaxed); // PERF-167
    LOG_INFO(Render_Vulkan, "Driver pipeline cache: loaded {} bytes", data.size());
}

void PipelineCache::SaveDriverCache(bool wait) {
    if (driver_cache_path.empty() || !pipeline_cache) {
        return;
    }
    // PERF-036: GT Sport's driver cache grows to several GB (3.4 GB on October 9). Reading it out
    // with vkGetPipelineCacheData took about 1.1 s on the GPU thread every 30 s, a frozen frame
    // each time. The read and the write now run on their own thread; a save still running is
    // not waited for, the next one comes later. Pipeline caches are internally synchronized, so
    // builds go on meanwhile. -DisablePerf 51 reads it here as before.
    static const bool off_thread = Common::PerfFeatureEnabled(51);
    if (!wait && driver_cache_writing.load()) {
        return;
    }
    if (driver_cache_writer.joinable()) {
        driver_cache_writer.join();
    }
    // FIX-GT7-025: the count this save holds at least (its data is read after).
    const u32 count = num_new_pipelines.load();
    pipelines_at_driver_save = count;
    driver_cache_saved_at = std::chrono::steady_clock::now();
    const auto write = [this, path = driver_cache_path, count](std::vector<u8> data) {
        // PERF-164: a cache over 1 GB is not written (GT Sport's reached 3.7 GB and stalled);
        // the file from before stays.
        if (data.size() > DriverCacheMaxBytes()) {
            LOG_WARNING(Render_Vulkan,
                        "PERF-164: driver pipeline cache not saved: {} MB is over the {} MB limit",
                        data.size() >> 20, DriverCacheMaxBytes() >> 20);
            return;
        }
        // Write beside the file and rename, so an interrupted write never leaves a broken cache.
        auto temp = path;
        temp += ".tmp";
        {
            const Common::FS::IOFile file{temp, Common::FS::FileAccessMode::Create};
            if (!file.IsOpen() || file.Write(data) != data.size()) {
                return;
            }
        }
        // FIX-GT7-025: only over an older save (the exit save may have written a newer one).
        std::error_code ec;
        if (!driver_cache_file.Commit(count, [&] {
                std::filesystem::rename(temp, path, ec);
                return !ec;
            })) {
            std::filesystem::remove(temp, ec);
        }
    };
    const vk::Device device = instance.GetDevice();
    const vk::PipelineCache cache = *pipeline_cache;
    if (off_thread) {
        driver_cache_writing = true;
        driver_cache_writer = std::jthread([this, device, cache, write] {
            Common::SetCurrentThreadName("shadGT:DriverCacheSave");
            const auto start = std::chrono::steady_clock::now();
            auto [result, data] = device.getPipelineCacheData(cache);
            const auto read_end = std::chrono::steady_clock::now();
            if (result == vk::Result::eSuccess && !data.empty()) {
                const size_t size = data.size();
                driver_cache_bytes.store(size, std::memory_order_relaxed); // PERF-167
                write(std::move(data));
                LOG_WARNING(Render_Vulkan,
                            "PERF-036: driver pipeline cache saved off the GPU thread: {} MB, "
                            "read {:.0f} ms, written {:.0f} ms",
                            size >> 20,
                            std::chrono::duration<double, std::milli>(read_end - start).count(),
                            std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - read_end)
                                .count());
            }
            driver_cache_writing = false;
        });
    } else {
        auto [result, data] = device.getPipelineCacheData(cache);
        if (result != vk::Result::eSuccess || data.empty()) {
            return;
        }
        driver_cache_writing = true;
        driver_cache_writer = std::jthread([this, write, data = std::move(data)]() mutable {
            write(std::move(data));
            driver_cache_writing = false;
        });
    }
    if (wait) {
        driver_cache_writer.join();
    }
}

int PipelineCache::SaveDriverCacheNow() {
    // FIX-GT7-023: closing the window ends the game at once (TerminateProcess, ryty_entry.cpp),
    // so the destructor's save never runs, and PERF-167 saves a 1.6 GB cache only every 5 min. A
    // 5-minute session on a new GT7 track (a city) kept none of its pipelines: the next race
    // there built them again, buildings missing or black meanwhile. Read and written on the
    // closing thread, beside a save in progress (another temporary file).
    // FIX-GT7-025: compared with the save the file holds, not the last one asked for: an
    // automatic save still writing when the window closes is cut short by the process's end.
    const u32 count = num_new_pipelines.load();
    if (driver_cache_path.empty() || !pipeline_cache || count <= driver_cache_file.Kept()) {
        return -1;
    }
    auto [result, data] = instance.GetDevice().getPipelineCacheData(*pipeline_cache);
    if (result != vk::Result::eSuccess || data.empty() || data.size() > DriverCacheMaxBytes()) {
        return -1;
    }
    auto temp = driver_cache_path;
    temp += ".exit.tmp";
    {
        const Common::FS::IOFile file{temp, Common::FS::FileAccessMode::Create};
        if (!file.IsOpen() || file.Write(data) != data.size()) {
            return -1;
        }
    }
    std::error_code ec;
    if (!driver_cache_file.Commit(count, [&] {
            std::filesystem::rename(temp, driver_cache_path, ec);
            return !ec;
        })) {
        std::filesystem::remove(temp, ec); // a newer automatic save is in the file already
        return -1;
    }
    return static_cast<int>(data.size() >> 20);
}

int SaveDriverCacheAtExit() {
    // FIX-GT7-025: the cache isn't destroyed during the save (the destructor takes the lock).
    std::scoped_lock lk{g_exit_save_mutex};
    PipelineCache* const cache = g_exit_save_cache.load();
    return cache ? cache->SaveDriverCacheNow() : -1;
}

void PipelineCache::MaybeSaveDriverCache() {
#ifdef RYTY_GPU
    // PERF-164: during play, off the GPU thread, after a batch of at least 32 new pipelines and
    // at most every 45 s (FIX-165 had turned saves during play off for GT Sport's 3.7 GB file).
    // Every 2 minutes missed a race's pipelines: the game was closed before the next save.
    using namespace std::chrono_literals;
    const auto since = std::chrono::steady_clock::now() - driver_cache_saved_at;
    // PERF-167: a big cache is copied out whole each save, so it is saved less often as it
    // grows: every 45 s up to 256 MB, then 45 s more per further 256 MB (2 GB: every 6 min).
    const u64 size_mb = driver_cache_bytes.load(std::memory_order_relaxed) >> 20;
    const auto interval = 45s * std::max<u64>(1, (size_mb + 255) / 256);
    if (num_new_pipelines - pipelines_at_driver_save >= 32 && since >= interval) {
        SaveDriverCache(false);
    }
    return;
#endif
    // New pipelines are compiled during play; save them in batches without blocking the frame.
    using namespace std::chrono_literals;
    if (num_new_pipelines - pipelines_at_driver_save >= 64 &&
        std::chrono::steady_clock::now() - driver_cache_saved_at >= 30s) {
        SaveDriverCache(false);
    }
}

const AmdGpu::Regs& PipelineCache::Regs() const {
    return regs_override ? *regs_override : liverpool->DrawRegs();
}

bool PipelineCache::HasSupportedColorTargets() const {
    if (S().graphics_key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS) {
        return false;
    }
    for (u32 cb = 0; cb < S().graphics_key.num_color_attachments; ++cb) {
        const auto& color_buffer = S().graphics_key.color_buffers[cb];
        if (color_buffer.data_format != AmdGpu::DataFormat::FormatInvalid &&
            Vulkan::LiverpoolToVK::TrySurfaceFormat(
                color_buffer.data_format, color_buffer.num_format) == vk::Format::eUndefined) {
            return false;
        }
    }
    return true;
}

std::shared_ptr<PipelineCache::PipelineBuild> PipelineCache::MakePipelineBuild() {
    auto build = std::make_shared<PipelineBuild>();
    build->key = S().graphics_key;
    for (size_t stage = 0; stage < MaxShaderStages; ++stage) {
        if (!S().infos[stage]) {
            continue;
        }
        auto& copy = build->info_copies[stage];
        copy = *S().infos[stage];
        build->user_data[stage].assign(copy.user_data.begin(), copy.user_data.end());
        copy.user_data = build->user_data[stage];
        build->infos[stage] = &copy;
        build->canonical[stage] = S().infos[stage];
    }
    build->runtime_infos = S().runtime_infos;
    if (S().fetch_shader) {
        build->fetch = *S().fetch_shader;
    }
    build->modules = S().modules;
    build->done = build->promise.get_future().share();
    build->queued = std::chrono::steady_clock::now();
    return build;
}

void PipelineCache::FinishBuild(PipelineBuild& build, bool draw_waits) {
    // FIX-044: the GPU thread waiting for (or doing) a pipeline build holds the guest clocks.
    Common::GuestClock::CompilePause hold;
    if (!build.started.exchange(true)) {
        g_pipeline_build_urgent = draw_waits || build.urgent; // FIX-GT7-030
        build.pipeline = std::make_unique<GraphicsPipeline>(
            instance, scheduler, desc_heap, profile, build.key, *pipeline_cache, build.infos,
            build.runtime_infos, build.fetch ? &*build.fetch : nullptr, build.modules, build.sdata);
        g_pipeline_build_urgent = false;
        // The copies were only needed while building; draws use the cached infos.
        build.pipeline->SetStageInfos(build.canonical);
        build.info_copies = {};
        build.user_data = {};
        build.promise.set_value();
    }
    build.done.wait();
}

void PipelineCache::FinishBuildUnlocked(const std::shared_ptr<PipelineBuild>& build) {
    const auto start = std::chrono::steady_clock::now();
    // PERF-GT7-021: while this draw waits, more workers build at once (one takes this build,
    // urgent, at once), and the read-ahead states blocked by their shaders are evaluated again
    // so the next draws' pipelines start too. Built here only when no worker took it.
    const bool widen = RetryReadAhead() && build_workers != nullptr;
    if (widen) {
        build_workers->Widen(true);
        const auto grace = start + std::chrono::milliseconds{2};
        while (!build->started.load() && std::chrono::steady_clock::now() < grace) {
            std::this_thread::yield();
        }
    }
    const bool here = !build->started.load();
    if (widen && !here) {
        while (build->done.wait_for(std::chrono::milliseconds{5}) != std::future_status::ready) {
            if (lookup_mutex.try_lock()) {
                BuildQueuedStates(S().draw_indirect_params, false, true);
                lookup_mutex.unlock();
            }
        }
    }
    FinishBuild(*build);
    if (widen) {
        build_workers->Widen(false);
    }
    const auto now = std::chrono::steady_clock::now();
    const u64 us =
        u64(std::chrono::duration_cast<std::chrono::microseconds>(now - start).count());
    ++Common::GetWorkCounters().pipeline_waits;
    Common::GetWorkCounters().pipeline_wait_us += us;
    RyDiag::RecordElapsed(RyDiag::Zone::PipelineWait, now - start,
                          std::hash<GraphicsPipelineKey>{}(build->key));
    auto& stats = unlocked_stats;
    stats.waits.fetch_add(1, std::memory_order_relaxed);
    stats.built_here.fetch_add(here ? 1 : 0, std::memory_order_relaxed);
    stats.wait_us.fetch_add(us, std::memory_order_relaxed);
    const s64 now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                           now.time_since_epoch())
                           .count();
    s64 since = stats.since_ms.load(std::memory_order_relaxed);
    if (since == 0) {
        stats.since_ms.store(now_ms, std::memory_order_relaxed);
    } else if (now_ms - since >= 10'000 &&
               stats.since_ms.compare_exchange_strong(since, now_ms)) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-GT7-020: in {:.1f} s, {} draws into one-off targets waited for their "
                    "pipeline without the lookup lock ({:.1f} ms; {} built by the waiting thread); "
                    "PERF-GT7-036: {} for their shaders ({:.1f} ms)",
                    (now_ms - since) / 1000.0, stats.waits.exchange(0), stats.wait_us.exchange(0) / 1000.0,
                    stats.built_here.exchange(0), stats.shader_waits.exchange(0),
                    stats.shader_wait_us.exchange(0) / 1000.0);
    }
}

void PipelineCache::QueueBuild(const std::shared_ptr<PipelineBuild>& build, BuildQueue::Kind kind) {
    if (!build_workers) {
        build_workers = std::make_unique<PipelineBuildWorkers>();
    }
    auto job = [this, build] {
        if (!build->started.load()) {
            FinishBuild(*build);
        }
    };
    // PERF-171: tessellated pipelines (GT7's cars) are built before the other read-ahead
    // ones, so the cars appear first (after the builds of draws waiting, on screen or gone on
    // without their pipeline). RYGT_TESS_FIRST=0 keeps them in order.
    static const bool tess_first = [] {
        const char* env = std::getenv("RYGT_TESS_FIRST");
        return !(env && env[0] == '0');
    }();
    const bool tessellated = build->infos[u32(Shader::SwStage::TessellationControl)] != nullptr;
    if (kind == BuildQueue::Kind::Normal && tess_first && tessellated) {
        kind = BuildQueue::Kind::Tessellated;
    }
    build_workers->Push(std::move(job), kind);
}

std::shared_ptr<PipelineCache::PipelineBuild> PipelineCache::StartPipelineBuild(bool urgent,
                                                                               bool in_order) {
    auto build = MakePipelineBuild();
    build->urgent = urgent;
    QueueBuild(build, !urgent    ? BuildQueue::Kind::Normal
                      : in_order ? BuildQueue::Kind::Soon
                                 : BuildQueue::Kind::Urgent);
    return build;
}

bool PipelineCache::LegacyWaits() const {
    // Keep going: rendering must not wait for a cosmetic pipeline.
    if (CompileMode() == 2) return false;
    // FIX-119: a pipeline of the old renderer's translations (D10) is built at once, as the
    // driver usually has it already. Not in a race: there one the driver did not have stopped
    // the game for up to seconds (19.7 s of such waits in one test race); it builds in the
    // background like the rest. -DisablePerf 141 waits for them in races too.
    if (!S().key_uses_legacy) {
        return false;
    }
#ifdef RYTY_GPU
    static const bool skip_in_races = Common::PerfFeatureEnabled(141);
    if (skip_in_races && RytyGtsInRace()) {
        return false;
    }
#endif
    return true;
}

std::chrono::microseconds PipelineCache::QuickWait() {
    if (CompileMode() == 2) return std::chrono::microseconds{0};
    // FIX-114: a draw whose new pipeline is building waits for it a little, so the pipelines
    // that build quickly (the driver had them, or small ones) are drawn at once and only the
    // slow ones pop in a moment later. At most 4 ms of waiting in any 16 ms, 2 ms per draw.
    // -DisablePerf 137 skips without waiting.
    static const bool enabled = Common::PerfFeatureEnabled(137);
    if (!enabled) {
        return std::chrono::microseconds{0};
    }
    using namespace std::chrono;
    static thread_local steady_clock::time_point window{};
    static thread_local microseconds used{};
    const auto now = steady_clock::now();
    if (now - window > milliseconds{16}) {
        window = now;
        used = microseconds{0};
    }
    const auto wait = std::clamp(microseconds{4000} - used, microseconds{0}, microseconds{2000});
    used += wait; // charged up front: simple, and errs on the side of fewer waits
    return wait;
}

bool PipelineCache::MenusWait() {
    // Keep going: rendering must not wait for a cosmetic pipeline.
    if (CompileMode() == 2) return false;
    // FIX-110: in GT Sport's menus a draw waits for its new pipeline (the guest clocks are held
    // meanwhile, as in the "pause" compile mode); only races skip draws. Some menu scenes draw
    // once into a texture they keep showing: the GR86 slide of the home screen played on black
    // instead of the Fuji pit straight when its scenery draws were skipped while they built.
    // -DisablePerf 134 skips in the menus too.
#ifdef RYTY_GPU
    static const bool enabled = Common::PerfFeatureEnabled(134);
    return enabled && !RytyGtsInRace();
#else
    return false;
#endif
}

bool PipelineCache::CancelBuild(PipelineBuild& build) {
    // A worker may already have taken the job and wait in FinishBuild for whoever claimed the
    // build; release it, the build stays empty.
    if (build.started.exchange(true)) {
        return false;
    }
    build.promise.set_value();
    return true;
}

void PipelineCache::ReadAheadAtBufferStart() {
    using namespace std::chrono_literals;
    if (!build_workers || !Common::PerfFeatureEnabled(20) || !Common::PerfFeatureEnabled(19) ||
        std::chrono::steady_clock::now() - last_pipeline_miss > 2s) {
        return;
    }
    // No draw is being processed, so nothing needs restoring afterwards.
    ReadAhead(S().draw_indirect_params, false);
}

void PipelineCache::ReadAhead(const DrawIndirectParams params, bool restore) {
    const auto start = std::chrono::steady_clock::now();
    u32 draws = 0;
    u32 started = 0;
    u32 evaluated = 0;
    const bool dedupe = Common::PerfFeatureEnabled(21);
    liverpool->ScanAheadForPipelines([&](const AmdGpu::Regs& regs) {
        ++draws;
        // FIX-014: draws the rasterizer filters out (Rasterizer::FilterDraw) never get a
        // pipeline; building one for them can fail (primitive type 'None' asserted on a build
        // worker when selecting a car).
        using OperationMode = AmdGpu::ColorControl::OperationMode;
        const auto mode = regs.color_control.mode;
        if (regs.primitive_type == AmdGpu::PrimitiveType::None ||
            mode == OperationMode::EliminateFastClear || mode == OperationMode::FmaskDecompress ||
            mode == OperationMode::Resolve) {
            return false;
        }
        if (dedupe) {
            // Everything the pipeline key is built from that the commands can change: graphics
            // shader registers (programs and user data), context registers (targets, blending,
            // depth, rasterizer state) and the primitive type.
            const auto& r = regs.reg_array;
            constexpr u32 ShGfxRegs = 0x200;
            constexpr u32 NumContextRegs = 0x400;
            constexpr u32 PrimitiveRegs = 0x100;
            std::array<u32, ShGfxRegs> sh;
            std::memcpy(sh.data(), &r[AmdGpu::Regs::ShRegWordOffset], sizeof(sh));
            if (Common::PerfFeatureEnabled(22)) {
                // PERF-022: user data (per-draw constants and resource pointers) changes on
                // nearly every draw; the programs and the other state decide the pipeline.
                // Each stage's 16 user data registers start 0xC after its program registers.
                for (u32 stage_base = 0; stage_base < ShGfxRegs; stage_base += 0x40) {
                    std::fill_n(sh.begin() + stage_base + 0xC, 16, 0u);
                }
            }
            u64 hash = XXH3_64bits(sh.data(), sizeof(sh));
            hash = XXH3_64bits_withSeed(&r[AmdGpu::Regs::ContextRegWordOffset],
                                        NumContextRegs * sizeof(u32), hash);
            hash = XXH3_64bits_withSeed(&r[AmdGpu::Regs::UconfigRegWordOffset + 0x200],
                                        PrimitiveRegs * sizeof(u32), hash);
            if (!read_ahead_states.insert(hash).second) {
                return false;
            }
            if (read_ahead_states.size() > 1u << 16) {
                read_ahead_states.clear();
            }
        }
        ++evaluated;
        regs_override = &regs;
        S().draw_indirect_params = {};
        bool valid = false;
        try {
            Common::RecoverableScope recoverable; // FIX-GT7-016: as BuildQueuedStates
            valid = RefreshGraphicsKey();
        } catch (const Common::RecoverableFailure&) {
            valid = false;
        }
        regs_override = nullptr;
        if (!valid || !HasSupportedColorTargets() || graphics_pipelines.contains(S().graphics_key)) {
            return false;
        }
        if (pending_builds.contains(S().graphics_key)) {
            return false;
        }
        pending_builds.emplace(S().graphics_key, StartPipelineBuild(false));
        ++started;
        return true;
    });
    if (draws > 0) {
        last_scan_draws = draws;
        draws_since_scan = 0;
        // Back to the draw being processed.
        S().draw_indirect_params = params;
        if (restore && evaluated > 0) {
            RefreshGraphicsKey();
        }
    }
    // One summary every 2 s instead of a line per read.
    auto& stats = read_ahead_stats;
    const auto now = std::chrono::steady_clock::now();
    ++stats.scans;
    stats.draws += draws;
    stats.evaluated += evaluated;
    stats.started += started;
    stats.ms += std::chrono::duration<double, std::milli>(now - start).count();
    if (now - stats.since >= std::chrono::seconds{2}) {
        if (stats.draws > 0) {
            LOG_WARNING(Render_Vulkan,
                        "Pipeline read-ahead in {:.1f} s: {} reads, {} draws read, {} evaluated, "
                        "{} builds started, {:.1f} ms, {} pending",
                        std::chrono::duration<double>(now - stats.since).count(), stats.scans,
                        stats.draws, stats.evaluated, stats.started, stats.ms,
                        pending_builds.size());
        }
        stats = {};
        stats.since = now;
    }
}

bool PipelineCache::RetryReadAhead() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_READ_AHEAD_RETRY");
        return !(env && env[0] == '0');
    }();
    return on;
}

void PipelineCache::BuildQueuedStates(const DrawIndirectParams params, bool restore,
                                      bool waiting) {
    auto* states = liverpool->PipeReadAheadStates();
    if (!states || (states->Empty() && blocked_states.empty())) {
        return;
    }
    const auto start = std::chrono::steady_clock::now();
    if (!queued_state_regs) {
        // A queued state holds every register a key is built from; the others keep these values.
        queued_state_regs = std::make_unique<AmdGpu::Regs>(liverpool->DrawRegs());
    }
    Common::PhaseTimer eval_timer{Common::Phase::ReadAheadEval};
    const auto key = S().graphics_key;
    const bool key_was_legacy = S().key_uses_legacy; // D10: made again the same way below
    u32 evaluated = 0;
    u32 started = 0;
    // Before a draw, 2 ms at most, and while waiting for a build, 64 states between checks on
    // it; the rest comes next time.
    auto budget = start + std::chrono::milliseconds{2};
    // PERF-GT7-026: before draws, also at most 3 ms in any 16 ms. The 2 ms were per draw: in the
    // first seconds of a race on a new track, the frames with many new states spent 300-390 ms
    // here starting up to 523 builds (~0.75 ms each), and the game waited for that frame. Not
    // while a draw waits for its pipeline (the game is stopped then anyway).
    using ReadAheadClock = std::chrono::steady_clock;
    static thread_local ReadAheadClock::time_point window{};
    static thread_local ReadAheadClock::duration window_used{};
    const bool windowed = !restore && !Common::CompileCoresWide();
    if (windowed) {
        if (start - window > std::chrono::milliseconds{16}) {
            window = start;
            window_used = {};
        }
        const auto left = ReadAheadClock::duration{std::chrono::milliseconds{3}} - window_used;
        if (left <= ReadAheadClock::duration::zero()) {
            return;
        }
        budget = std::min(budget, start + left);
    }
    struct ChargeWindow {
        bool on;
        ReadAheadClock::time_point start;
        ~ChargeWindow() {
            if (on) {
                window_used += ReadAheadClock::now() - start;
            }
        }
    } charge_window{windowed, start};
    const auto in_budget = [&] {
        return restore ? evaluated < 64 : std::chrono::steady_clock::now() < budget;
    };
    using Ranges = AmdGpu::ReadAheadStates;
    // The state in queued_state_regs: its build started when new; false when its key waits for
    // shaders (PERF-GT7-021).
    const auto evaluate = [&]() -> bool {
        ++evaluated;
        read_ahead_blocked = false;
        regs_override = queued_state_regs.get();
        S().draw_indirect_params = {};
        // FIX-GT7-016: a state read ahead (or kept, PERF-GT7-021) may point at shader code the
        // game has freed and reused since: after a GT7 race, a kept state's vertex shader read
        // as 0xffffffff, and the fetch-shader lookup's decode assertion ended the process on
        // the command thread. A failed check drops the state instead.
        bool valid = false;
        try {
            Common::RecoverableScope recoverable;
            valid = RefreshGraphicsKey();
        } catch (const Common::RecoverableFailure&) {
            regs_override = nullptr;
            ++blocked_stats.failed;
            return true;
        }
        regs_override = nullptr;
        if (!valid) {
            return !read_ahead_blocked;
        }
        if (!HasSupportedColorTargets() || graphics_pipelines.contains(S().graphics_key) ||
            pending_builds.contains(S().graphics_key)) {
            return true;
        }
        pending_builds.emplace(S().graphics_key, StartPipelineBuild(false));
        ++started;
        return true;
    };
    const auto keep = [&](std::chrono::steady_clock::time_point first) {
        BlockedState blocked{.first = first, .next = std::chrono::steady_clock::now() +
                                                    std::chrono::milliseconds{5}};
        if (!blocked_free.empty()) {
            blocked.ranges = std::move(blocked_free.back());
            blocked_free.pop_back();
        }
        blocked.ranges.resize(Ranges::StateDwords);
        u32 at = 0;
        for (const auto& range : Ranges::Ranges) {
            std::memcpy(blocked.ranges.data() + at, &queued_state_regs->reg_array[range.first],
                        range.count * sizeof(u32));
            at += range.count;
        }
        blocked_states.push_back(std::move(blocked));
        if (blocked_states.size() > 512) {
            blocked_free.push_back(std::move(blocked_states.front().ranges));
            blocked_states.pop_front();
            ++blocked_stats.dropped;
        }
    };
    // PERF-GT7-021: the states kept earlier, once their wait is over.
    // FIX-GT7-015: only while a draw waits for its pipeline (the game stopped, as at a new
    // track's start, which they were for). Evaluated again in a race, up to 3 s after their
    // draw, their user data had moved on: in a GT7 race at 60 fps they started ~500 builds a
    // second of keys no draw ever asked for (1-33 installed in 2 s), 16,000 of them pending
    // and queued, compiled by the workers for minutes (the compile notice never went away).
    // Otherwise they are dropped once 3 s old, as before.
    if (RetryReadAhead() && !(restore || waiting)) {
        const auto now = std::chrono::steady_clock::now();
        while (!blocked_states.empty() &&
               now - blocked_states.front().first >= std::chrono::seconds{3}) {
            blocked_free.push_back(std::move(blocked_states.front().ranges));
            blocked_states.pop_front();
            ++blocked_stats.expired;
        }
    }
    if (RetryReadAhead() && (restore || waiting)) {
        const auto now = std::chrono::steady_clock::now();
        for (size_t left = blocked_states.size(); left != 0 && in_budget(); --left) {
            BlockedState blocked = std::move(blocked_states.front());
            blocked_states.pop_front();
            if (blocked.next > now) {
                blocked_states.push_back(std::move(blocked));
                continue;
            }
            u32 at = 0;
            for (const auto& range : Ranges::Ranges) {
                std::memcpy(&queued_state_regs->reg_array[range.first],
                            blocked.ranges.data() + at, range.count * sizeof(u32));
                at += range.count;
            }
            ++blocked_stats.retried;
            const u32 started_before = started;
            const bool done = evaluate();
            blocked_stats.started += started - started_before;
            if (!done && now - blocked.first < std::chrono::seconds{3}) {
                blocked.next = now + std::chrono::milliseconds{5};
                blocked_states.push_back(std::move(blocked));
                continue;
            }
            blocked_stats.dropped += done ? 0 : 1;
            blocked_free.push_back(std::move(blocked.ranges));
        }
    }
    while (in_budget() && states->Take(queued_state_regs->reg_array)) {
        if (!evaluate() && RetryReadAhead()) {
            // FIX-GT7-005: not a tessellated state. Its hull constants are in memory the game
            // writes again for later draws: read up to 3 s later they were another draw's, or
            // garbage that passes SaneTessConstants (a local stride of 0 crashed a translation).
            using Stages = AmdGpu::ShaderStageEnable::VgtStages;
            const u32 stages = queued_state_regs->stage_enable.raw;
            if (stages == Stages::LsHs || stages == Stages::LsHsEsGs) {
                ++blocked_stats.tessellated;
                continue;
            }
            keep(std::chrono::steady_clock::now());
            ++blocked_stats.kept;
        }
    }
    S().draw_indirect_params = params;
    if (restore && evaluated > 0) {
        const bool was_forced = force_legacy;
        force_legacy = key_was_legacy && legacy_fallback;
        RefreshGraphicsKey();
        force_legacy = was_forced;
        ASSERT_MSG(S().graphics_key == key, "Read-ahead changed the current pipeline key");
    }
    auto& stats = queued_state_stats;
    const auto now = std::chrono::steady_clock::now();
    stats.evaluated += evaluated;
    stats.started += started;
    stats.ms += std::chrono::duration<double, std::milli>(now - start).count();
    if (now - stats.since >= std::chrono::seconds{2}) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-034 pipeline read-ahead in {:.1f} s: {} draw states from the command "
                    "thread evaluated, {} builds started, {:.1f} ms; {} draws waited for a build "
                    "({:.1f} ms); {} pending; PERF-GT7-021: {} states kept for their shaders, {} "
                    "evaluated again, {} builds started from them, {} dropped, {} kept now; "
                    "FIX-GT7-005: {} tessellated not kept; FIX-GT7-015: {} expired unretried, "
                    "{} stale builds cancelled; FIX-GT7-016: {} failed and dropped",
                    std::chrono::duration<double>(now - stats.since).count(), stats.evaluated,
                    stats.started, stats.ms, stats.waits, stats.wait_ms, pending_builds.size(),
                    blocked_stats.kept, blocked_stats.retried, blocked_stats.started,
                    blocked_stats.dropped, blocked_states.size(), blocked_stats.tessellated,
                    blocked_stats.expired, std::exchange(stale_dropped, 0), blocked_stats.failed);
        stats = {};
        stats.since = now;
        blocked_stats = {};
    }
}

void PipelineCache::WaitForBuild(PipelineBuild& build, const DrawIndirectParams params) {
    using namespace std::chrono_literals;
    BuildQueuedStates(params, true);
    // An urgent build is at the front of the workers' queue, and an idle worker takes it at
    // once; with every worker busy, this thread builds it (FinishBuild). So does a read-ahead
    // build no worker has reached yet.
    if (build.urgent) {
        const auto grace = std::chrono::steady_clock::now() + 200us;
        while (!build.started.load() && std::chrono::steady_clock::now() < grace) {
            std::this_thread::yield();
        }
    }
    while (build.started.load() && build.done.wait_for(500us) != std::future_status::ready) {
        BuildQueuedStates(params, true);
    }
#ifdef RYTY_GPU
    // PERF-101: GPU.compile_batch_ms (launcher "Shader compile batching"): with the game paused
    // for a compile anyway, the builds of the upcoming draws the command thread read ahead are
    // finished in the same pause, up to that long, so later draws pause less often.
    static const u32 batch_ms = static_cast<u32>(std::max(0, EmulatorSettings.GetCompileBatchMs()));
    if (batch_ms != 0 && !AsyncPipelines() && build_workers) {
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds{batch_ms};
        BuildQueuedStates(params, true);
        while (!build_workers->ForegroundIdle() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(500us);
            BuildQueuedStates(params, true);
        }
    }
#endif
}

const GraphicsPipeline* PipelineCache::GetGraphicsPipeline(const DrawIndirectParams params) {
    skipped_build.reset(); // PERF-GT7-020
    // PERF-034: builds for the upcoming draws the command thread read ahead start first.
    BuildQueuedStates(params, false);
    ReportAsyncPipelines(std::chrono::steady_clock::now());
    UpdateCatchUp(std::chrono::steady_clock::now()); // PERF-GT7-029
    DropStaleReadAheadBuilds(std::chrono::steady_clock::now()); // FIX-GT7-015
    S().draw_indirect_params = params;
    ++draws_since_scan;
    // FIX-078: missing translations of this draw's stages are queued together first; then the
    // draws after it are read and their shaders queued too, and the key is made again, waiting
    // for the workers' results (the slowest one, not their sum).
    defer_misses = AsyncTranslation();
    deferred_miss = false;
    fallback_miss = false;
    translation_skipped = false; // SHD-005
    bool key_valid = RefreshGraphicsKey();
    // D10: a new translation this draw needs is being made; the old renderer's draw it.
    if (!key_valid && fallback_miss) {
        fallback_nested = true;
        force_legacy = true;
        const bool legacy_valid = RefreshGraphicsKey() && !S().legacy_failed;
        force_legacy = false;
        const GraphicsPipeline* legacy = legacy_valid ? PipelineForCurrentKey(params) : nullptr;
        ++(legacy ? legacy_stats.fallback_ok : legacy_stats.fallback_none);
        ReportLegacy();
        if (legacy) {
            fallback_nested = false;
            return legacy;
        }
        // No old translations: the usual way (the queued translations are taken when ready).
        deferred_miss = false;
        key_valid = RefreshGraphicsKey();
    }
    struct NestedReset {
        bool& flag;
        ~NestedReset() {
            flag = false;
        }
    } nested_reset{fallback_nested};
    // One stage is queued per pass (a stage's start bindings follow from the stages before it);
    // the read-ahead runs once, after the first.
    for (u32 pass = 0; !key_valid && deferred_miss && pass < 6; ++pass) {
        if (pass == 0) {
            liverpool->ReadAheadNow();
            BuildQueuedStates(params, false);
            S().draw_indirect_params = params;
        }
        deferred_miss = false;
        key_valid = RefreshGraphicsKey();
        if (!key_valid && translation_skipped) {
            break; // SHD-005
        }
        if (!key_valid && shaders_unlocked && !skipped_translations.empty()) {
            break; // PERF-GT7-036: the draw waits for them without the lock
        }
    }
    defer_misses = false;
    if (!key_valid && translation_skipped) {
        // SHD-005: skipped while its shaders translate; a later draw takes them.
        deferred_miss = false;
        translation_skipped = false;
        ++async_stats.skipped_translating;
        NoteSkippedDraw();
        return nullptr;
    }
    if (!key_valid && deferred_miss && !(shaders_unlocked && !skipped_translations.empty())) {
        deferred_miss = false;
        key_valid = RefreshGraphicsKey();
    }
    if (!key_valid) {
        return nullptr;
    }
    {
        // DIAG-225 (diagnosis, RYGT_DIAG_AA_MASK=1): draws with no pixel shader while
        // PA_SC_AA_MASK hides pixels (the quad sample mask is applied in the pixel shader only).
        static const bool diag = std::getenv("RYGT_DIAG_AA_MASK") != nullptr;
        const auto& r = Regs();
        if (diag && !S().infos[u32(SwStage::Fragment)] &&
            (r.reg_array[0xa30e] != ~0u || r.reg_array[0xa30f] != ~0u)) {
            static u32 count = 0;
            if (++count <= 30 || count % 500 == 0) {
                LOG_WARNING(Render_Vulkan,
                            "DIAG-225: draw {} with no pixel shader under AA mask {:08x} {:08x}, "
                            "depth write {}, depth test {}",
                            count, r.reg_array[0xa30e], r.reg_array[0xa30f],
                            u32(r.depth_control.depth_write_enable),
                            u32(r.depth_control.depth_enable));
            }
        }
    }
    // D10: a key of new translations whose pipeline is not built yet. It is built in the
    // background; meanwhile the draw uses the key made of the old renderer's translations.
    if (legacy_fallback && !fallback_nested && !S().key_uses_legacy && regs_override == nullptr &&
        !graphics_pipelines.contains(S().graphics_key)) {
        const auto pending = pending_builds.find(S().graphics_key);
        const bool ready = pending != pending_builds.end() &&
                           pending->second->done.wait_for(std::chrono::seconds{0}) ==
                               std::future_status::ready;
        if (!ready && S().graphics_key.num_color_attachments <= AmdGpu::NUM_COLOR_BUFFERS) {
            if (pending == pending_builds.end()) {
                auto build = StartPipelineBuild(false);
                build->async = true;
                pending_builds[S().graphics_key] = build;
            }
            fallback_nested = true;
            force_legacy = true;
            const bool legacy_valid = RefreshGraphicsKey() && !S().legacy_failed;
            force_legacy = false;
            const GraphicsPipeline* legacy =
                legacy_valid ? PipelineForCurrentKey(params) : nullptr;
            fallback_nested = false;
            ++(legacy ? legacy_stats.fallback_ok
                      : legacy_valid ? legacy_stats.fallback_unbuilt : legacy_stats.fallback_none);
            ReportLegacy();
            if (legacy) {
                return legacy;
            }
            // No old translations for every stage: the new key again, the usual way.
            if (!RefreshGraphicsKey()) {
                return nullptr;
            }
        }
    }
    return PipelineForCurrentKey(params);
}

const GraphicsPipeline* PipelineCache::LookupBuiltPipeline(const DrawIndirectParams params) {
    // NAT-031: the key is computed in this thread's own scratch; the cache's own (the threads
    // above's last selection) is not touched.
    static thread_local KeyScratch scratch;
    KeyScratch* const saved = std::exchange(t_scratch, &scratch);
    t_lookup_only = true;
    scratch.draw_indirect_params = params;
    scratch.lookup_miss = nullptr;
    const GraphicsPipeline* found = nullptr;
    if (RefreshGraphicsKey()) {
        if (const auto it = graphics_pipelines.find(scratch.graphics_key);
            it != graphics_pipelines.end()) {
            found = it->second.get();
        }
        if (!found) {
            scratch.lookup_miss = "pipeline not built";
        }
    } else if (!scratch.lookup_miss) {
        scratch.lookup_miss = "no key (unsupported state or hull constants)";
    }
    t_lookup_only = false;
    t_scratch = saved;
    t_job_infos = &scratch.infos;
    // The miss reason is read through LookupMiss() on this thread right after: keep it.
    t_last_miss = scratch.lookup_miss;
    return found;
}

Shader::Info& PipelineCache::MatchInfo(const Shader::Info& cached) {
    struct Copy {
        u64 identity{};
        std::unique_ptr<Shader::Info> info;
    };
    static thread_local std::unordered_map<const Shader::Info*, Copy> copies;
    if (copies.size() > 8192) {
        copies.clear(); // programs come and go: don't keep copies of retired ones forever
    }
    auto& copy = copies[&cached];
    if (!copy.info || copy.identity != cached.identity) {
        copy.info = std::make_unique<Shader::Info>(cached);
        copy.identity = cached.identity;
    }
    return *copy.info;
}

const GraphicsPipeline* PipelineCache::PipelineForCurrentKey(const DrawIndirectParams params) {
    if (S().graphics_key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS) {
        LOG_WARNING(Render_Vulkan, "Skipping draw with {} color attachments (MRT mask={:#x})",
                    S().graphics_key.num_color_attachments, S().graphics_key.mrt_mask);
        return nullptr;
    }
    for (u32 cb = 0; cb < S().graphics_key.num_color_attachments; ++cb) {
        const auto& color_buffer = S().graphics_key.color_buffers[cb];
        if (color_buffer.data_format != AmdGpu::DataFormat::FormatInvalid &&
            Vulkan::LiverpoolToVK::TrySurfaceFormat(
                color_buffer.data_format, color_buffer.num_format) == vk::Format::eUndefined) {
            LOG_WARNING(Render_Vulkan,
                        "Skipping draw with unsupported color target {} format: data={}, number={}",
                        cb, static_cast<u32>(color_buffer.data_format),
                        static_cast<u32>(color_buffer.num_format));
            return nullptr;
        }
    }
    // PERF-GT7-017: a draw whose pipeline a worker is still building (compile mode 2) is skipped
    // at once. It came back through the new-pipeline path below every time: the guest clocks'
    // compile pause (FIX-044), the "Compiling" log line and DIAG-050/051's key bookkeeping, for
    // up to ~20,000 such draws a second in GT7 (a million log lines in one race), which slowed
    // the command processor, held the game's clocks and halved the frame rate for a minute.
    if (CompileMode() == 2 && !force_wait &&
        graphics_pipelines.find(S().graphics_key) == graphics_pipelines.end()) {
        if (const auto pending = pending_builds.find(S().graphics_key);
            pending != pending_builds.end() && pending->second->async &&
            pending->second->done.wait_for(std::chrono::microseconds{0}) !=
                std::future_status::ready &&
            !LegacyWaits() && !WritesMemory(S().infos) && !NeverSkipped(S().infos) &&
            !MenusWait()) {
            ++async_stats.skipped_draws;
            NoteSkippedDraw(); // PERF-103
            NoteAsyncSkip(pending->second, true); // PERF-GT7-029
            skipped_build = pending->second; // PERF-GT7-020
            return nullptr;
        }
    }
    const auto [it, is_new] = graphics_pipelines.try_emplace(S().graphics_key);
    if (is_new) {
        Common::GuestClock::CompilePause hold; // FIX-044
        const auto pipeline_hash = std::hash<GraphicsPipelineKey>{}(S().graphics_key);
        LOG_INFO(Render_Vulkan, "Compiling graphics pipeline {:#x}", pipeline_hash);
        // DIAG-050: the parts of new keys of tessellated pipelines (GT Sport's race summary built
        // hundreds of keys for the same few programs).
        // DIAG-051: which part of a new key differs from the last key with the same stages
        // (GT Sport's race summary asks for thousands of new pipelines).
        {
            static std::unordered_map<u64, GraphicsPipelineKey> last_by_stages;
            static std::unordered_map<std::string, u32> diff_counts;
            static u32 keys_seen = 0;
            const u64 stages_hash = XXH3_64bits(S().graphics_key.stage_hashes.data(),
                                                sizeof(S().graphics_key.stage_hashes));
            auto [prev, first] = last_by_stages.try_emplace(stages_hash, S().graphics_key);
            std::string fields = first ? "new stages" : "";
            if (!first) {
                const auto& a = prev->second;
                const auto& b = S().graphics_key;
                const auto differs = [&](const auto& x, const auto& y) {
                    return std::memcmp(&x, &y, sizeof(x)) != 0;
                };
                const auto add = [&](const char* name) {
                    fields += fields.empty() ? name : std::string(",") + name;
                };
                if (differs(a.vertex_buffer_formats, b.vertex_buffer_formats)) add("vtx");
                if (a.patch_control_points != b.patch_control_points) add("pcp");
                if (a.num_color_attachments != b.num_color_attachments) add("ncb");
                if (differs(a.color_buffers, b.color_buffers)) add("cb");
                if (differs(a.blend_controls, b.blend_controls)) add("blend");
                if (differs(a.write_masks, b.write_masks)) add("mask");
                if (differs(a.cb_shader_mask, b.cb_shader_mask)) add("shmask");
                if (a.logic_op != b.logic_op) add("logic");
                if (a.num_samples != b.num_samples || a.depth_samples != b.depth_samples ||
                    differs(a.color_samples, b.color_samples)) add("samples");
                if (a.mrt_mask != b.mrt_mask) add("mrt");
                if (a.z_format != b.z_format || a.stencil_format != b.stencil_format ||
                    a.depth_clamp_enable != b.depth_clamp_enable) add("depth");
                if (a.prim_type != b.prim_type || a.polygon_mode != b.polygon_mode ||
                    a.clip_space != b.clip_space || a.provoking_vtx_last != b.provoking_vtx_last ||
                    a.depth_clip_enable != b.depth_clip_enable) add("raster");
                if (fields.empty()) {
                    add("padding");
                    static u32 logged = 0;
                    if (logged++ < 10) {
                        const auto* x = reinterpret_cast<const u8*>(&a);
                        const auto* y = reinterpret_cast<const u8*>(&b);
                        std::string offsets;
                        for (size_t i = 0; i < sizeof(GraphicsPipelineKey); ++i) {
                            if (x[i] != y[i]) {
                                offsets += fmt::format(" {}:{:02x}/{:02x}", i, x[i], y[i]);
                            }
                        }
                        LOG_WARNING(Render_Vulkan,
                                    "DIAG-051: keys differ only outside named fields (size {}, "
                                    "color_buffers at {}, mrt_mask at {}):{}",
                                    sizeof(GraphicsPipelineKey),
                                    offsetof(GraphicsPipelineKey, color_buffers),
                                    offsetof(GraphicsPipelineKey, mrt_mask), offsets);
                    }
                }
                prev->second = S().graphics_key;
            }
            ++diff_counts[fields];
            if (++keys_seen % 250 == 0) {
                std::string summary;
                for (const auto& [name, count] : diff_counts) {
                    summary += fmt::format(" {}={}", name, count);
                }
                LOG_WARNING(Render_Vulkan, "DIAG-051: {} new keys; differences from the last "
                            "key with the same stages:{}", keys_seen, summary);
            }
        }
        // SHADGT_LOG_KEYS=1 logs every new key the same way (up to 5000).
        static const bool log_all_keys = [] {
            const char* env = std::getenv("SHADGT_LOG_KEYS");
            return env && env[0] == '1';
        }();
        if (log_all_keys ||
            S().graphics_key.stage_hashes[u32(Shader::SwStage::TessellationControl)] != 0) {
            static std::atomic<u32> logged{};
            if (const u32 n = ++logged; n <= (log_all_keys ? 5000u : 80u)) {
                const auto& k = S().graphics_key;
                const auto part = [](const auto& field) {
                    return XXH3_64bits(&field, sizeof(field)) & 0xffff;
                };
                LOG_WARNING(Render_Vulkan,
                            "DIAG-050: new key {} {:#x}: stages {:#x} {:#x} {:#x} {:#x} {:#x}; "
                            "pcp {} prim {} vtx {:#x} cb {:#x} blend {:#x} mask {:#x} "
                            "shmask {:#x} samples {} {} mrt {:#x} z {} clip {}; all {:#x}",
                            n, pipeline_hash, k.stage_hashes[0], k.stage_hashes[1],
                            k.stage_hashes[2], k.stage_hashes[3], k.stage_hashes[4],
                            k.patch_control_points,
                            u32(k.prim_type), part(k.vertex_buffer_formats), part(k.color_buffers),
                            part(k.blend_controls), part(k.write_masks), part(k.cb_shader_mask),
                            k.num_samples, k.depth_samples, k.mrt_mask, u32(k.z_format),
                            u32(k.depth_clip_enable),
                            XXH3_64bits(&k, sizeof(k)) & 0xffffff);
            }
        }

        GraphicsPipeline::SerializationSupport sdata{};
        // PERF-019: the pipeline may already be building from an earlier read-ahead. If not,
        // it starts building now, and the draws after it are read for more new pipelines to
        // build at the same time on other cores.
        std::shared_ptr<PipelineBuild> build;
        bool predicted = false;
        auto pending = pending_builds.find(S().graphics_key);
        if (pending != pending_builds.end() && pending->second->async) {
            auto stored = pending->second;
            if (stored->done.wait_for(QuickWait()) == std::future_status::ready) {
                // PERF-076: built by a worker since an earlier draw skipped; no miss.
                pending_builds.erase(pending);
                stored->pipeline->SetStageInfos(S().infos);
                sdata = stored->sdata;
                it.value() = std::move(stored->pipeline);
                ++async_stats.installed;
                return it->second.get();
            }
            if (!force_wait && !LegacyWaits() && !WritesMemory(S().infos) &&
                !NeverSkipped(S().infos) && !MenusWait()) {
                // PERF-076: still building; this draw is skipped, the next ones look again.
                graphics_pipelines.erase(it);
                ++async_stats.skipped_draws;
                NoteSkippedDraw(); // PERF-103
                NoteAsyncSkip(stored, true); // PERF-GT7-029
                skipped_build = stored; // PERF-GT7-020
                return nullptr;
            }
            if (CancelBuild(*stored)) {
                // No worker has started it yet: built now the usual way.
                pending_builds.erase(pending);
                pending = pending_builds.end();
            }
        }
        ++Common::GetWorkCounters().pipelines_compiled;
        last_pipeline_miss = std::chrono::steady_clock::now();
        GraphicsPipeline::NotePipelineMiss();
        // PERF-GT7-025: a draw that goes on without its pipeline (PERF-076, below) queues it in
        // draw order, after the builds a thread waits for and before the read-ahead's. Last-first,
        // a new track's start had the workers on the furthest draws' pipelines while the recorder
        // built the next one-off target's itself (25 of 50, one after the other, in a 17 s freeze).
        const bool goes_on = !force_wait && AsyncPipelines() && !LegacyWaits() &&
                             !WritesMemory(S().infos) && !NeverSkipped(S().infos) && !MenusWait();
        auto* const queued_states = liverpool->PipeReadAheadStates();
        if (queued_states) {
            queued_states->NoteMiss();
        }
        if (pending != pending_builds.end()) {
            build = pending->second;
            pending_builds.erase(pending);
            predicted = true;
            // PERF-031: the read-ahead reads the command thread's state; not with the draw pipe.
            if (Common::PerfFeatureEnabled(19) && !liverpool->Pipelined() &&
                draws_since_scan >= last_scan_draws) {
                // Past the last read-ahead: keep reading from where it stopped.
                const auto key = S().graphics_key;
                ReadAhead(params);
                ASSERT_MSG(S().graphics_key == key, "Read-ahead changed the current pipeline key");
            }
        } else if (Common::PerfFeatureEnabled(19) && !liverpool->Pipelined()) {
            const auto key = S().graphics_key;
            build = StartPipelineBuild(true, goes_on);
            ReadAhead(params);
            ASSERT_MSG(S().graphics_key == key, "Read-ahead changed the current pipeline key");
        } else if (queued_states) {
            // PERF-034: a worker builds it while this thread starts the builds the command
            // thread's read-ahead queues meanwhile.
            build = StartPipelineBuild(true, goes_on);
        } else if (CompileMode() == 2 && AsyncPipelines()) {
            // Queue even when read-ahead/draw-pipe helpers are disabled. Retain the job
            // in pending_builds below so a later draw installs its completed pipeline.
            build = StartPipelineBuild(true, goes_on);
        }
        // PERF-076: a draw does not wait for a new pipeline. Its build goes on on a worker, the
        // draws that need it are skipped until it is done (a frame or two of a missing object),
        // and the game does not stall for the 60-140 ms a build takes. SHADGT_ASYNC_PIPELINES=0
        // or -DisablePerf 76 waits as before.
        // PERF-103: in slow-motion mode every new pipeline goes on in the background at once
        // and the game goes on in slow motion until it is built.
        // D10: a pipeline of old translations is built right away (the driver usually has it
        // from the old renderer's sessions, so it takes milliseconds).
        if (build && !force_wait && AsyncPipelines() && !LegacyWaits() &&
            !WritesMemory(S().infos) && !NeverSkipped(S().infos) && !MenusWait() &&
            build->done.wait_for(QuickWait()) != std::future_status::ready) {
            build->async = true;
            pending_builds[S().graphics_key] = build;
            graphics_pipelines.erase(it);
            ++async_stats.deferred;
            ++async_stats.skipped_draws;
            NoteSkippedDraw(); // PERF-103
            NoteAsyncSkip(build, false); // PERF-GT7-029
            skipped_build = build; // PERF-GT7-020
            return nullptr;
        }
        if (build) {
            const auto wait_start = std::chrono::steady_clock::now();
            if (queued_states) {
                Common::Nvtx::Scope nvtx{"wait for pipeline build"}; // DIAG-056
                WaitForBuild(*build, params);
            }
            FinishBuild(*build, true);
            const auto now = std::chrono::steady_clock::now();
            if (queued_states) {
                ++queued_state_stats.waits;
                queued_state_stats.wait_ms +=
                    std::chrono::duration<double, std::milli>(now - wait_start).count();
            }
            ++Common::GetWorkCounters().pipeline_waits;
            Common::GetWorkCounters().pipeline_wait_us += u64(
                std::chrono::duration_cast<std::chrono::microseconds>(now - wait_start).count());
            RyDiag::RecordElapsed(RyDiag::Zone::PipelineWait, now - wait_start, pipeline_hash);
            // DIAG-124: why this draw waited instead of being skipped.
            std::string why;
            if (!AsyncPipelines()) why += " no-async";
            if (S().key_uses_legacy) why += " legacy";
            if (WritesMemory(S().infos)) why += " writes-memory";
            if (NeverSkipped(S().infos)) why += " tessellation";
            if (MenusWait()) why += " menus";
            if (force_wait) why += " one-off target"; // FIX-198
            LOG_WARNING(Render_Vulkan,
                        "Pipeline {:#x} {}: built {:.1f} ms after it was queued, draw waited "
                        "{:.1f} ms (DIAG-124:{})",
                        pipeline_hash, predicted ? "read ahead" : "on demand",
                        std::chrono::duration<double, std::milli>(now - build->queued).count(),
                        std::chrono::duration<double, std::milli>(now - wait_start).count(),
                        why.empty() ? " ready within the quick wait" : why);
            build->pipeline->SetStageInfos(S().infos);
            sdata = build->sdata;
            it.value() = std::move(build->pipeline);
        } else {
            Common::Nvtx::Scope nvtx{"build pipeline here"}; // DIAG-056
            RYDIAG_SCOPE(PipelineBuild, pipeline_hash);
            g_pipeline_build_urgent = true; // FIX-GT7-030
            it.value() = std::make_unique<GraphicsPipeline>(
                instance, scheduler, desc_heap, profile, S().graphics_key, *pipeline_cache, S().infos,
                S().runtime_infos, S().fetch_shader, S().modules, sdata);
            g_pipeline_build_urgent = false;
        }

        if (S().key_uses_legacy) { // D10
            ++legacy_stats.pipelines;
        }
        ++num_new_pipelines;
        MaybeSaveDriverCache();

        if (EmulatorSettings.IsShaderCollect()) {
            for (auto stage = 0; stage < MaxShaderStages; ++stage) {
                if (S().infos[stage]) {
                    auto& m = S().modules[stage];
                    module_related_pipelines[m].emplace_back(S().graphics_key);
                }
            }
        }
    }
    return it->second.get();
}

const ComputePipeline* PipelineCache::GetComputePipeline() {
    fallback_miss = false;
    bool key_valid = RefreshComputeKey();
    if (!key_valid && fallback_miss) {
        // D10: as for graphics, the old translation while the new one is made.
        fallback_nested = true;
        force_legacy = true;
        key_valid = RefreshComputeKey() && !S().legacy_failed;
        force_legacy = false;
        ++(key_valid ? legacy_stats.fallback_ok : legacy_stats.fallback_none);
        if (!key_valid) {
            key_valid = RefreshComputeKey();
        }
        fallback_nested = false;
    }
    if (!key_valid) {
        return nullptr;
    }
    const auto [it, is_new] = compute_pipelines.try_emplace(compute_key);
    if (is_new) {
        Common::GuestClock::CompilePause hold; // FIX-044
        const auto pipeline_hash = std::hash<ComputePipelineKey>{}(compute_key);
        LOG_INFO(Render_Vulkan, "Compiling compute pipeline {:#x}", pipeline_hash);

        ComputePipeline::SerializationSupport sdata{};
        ++Common::GetWorkCounters().pipelines_compiled;
        RYDIAG_SCOPE(PipelineBuild, pipeline_hash);
        // PERF-GT7-032: the GPU's commands wait for this build. Meanwhile the other compiles stay
        // on the compile cores below normal priority and start no more than usual, and this
        // thread runs above normal priority on the other cores. In a GT7 race on a new track two
        // compute pipelines took 1.3-1.6 s each here, beside a catch-up's 8 compiles on every
        // core and the driver's compiler threads. RYGT_FOREGROUND_COMPILE=0: as before.
        static const bool foreground_on = [] {
            const char* env = std::getenv("RYGT_FOREGROUND_COMPILE");
            return !(env && env[0] == '0');
        }();
        struct Foreground {
            PipelineBuildWorkers* workers;
            bool on;
            Foreground(PipelineBuildWorkers* workers_, bool on_) : workers{workers_}, on{on_} {
                if (on) {
                    Common::BeginForegroundCompile();
                    if (workers) {
                        workers->SetForeground(true);
                    }
                }
            }
            ~Foreground() {
                if (on) {
                    if (workers) {
                        workers->SetForeground(false);
                    }
                    Common::EndForegroundCompile();
                }
            }
        };
        const auto build_start = std::chrono::steady_clock::now();
        {
            const Foreground foreground{build_workers.get(), foreground_on};
            it.value() = std::make_unique<ComputePipeline>(
                instance, scheduler, desc_heap, profile, *pipeline_cache, compute_key,
                *S().infos[0], S().modules[0], sdata);
        }
        if (const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - build_start)
                                  .count();
            ms >= 50.0) {
            LOG_INFO(Render_Vulkan, "PERF-GT7-032: compute pipeline {:#x} built in {:.0f} ms{}",
                     pipeline_hash, ms, foreground_on ? ", the other compiles set aside" : "");
        }
        // PERF-GT7-037: kept, to be built ahead at the next start: each compute pipeline once
        // (the driver may already have one that took long the first time). Its SPIR-V is the
        // stored translation of the program's permutation that made the module.
        if (const Shader::Info* info = S().infos[0]; WarmComputeOn() && info) {
            const auto dir = WarmComputeDir();
            const auto file = dir / fmt::format("{:#018x}.bin", pipeline_hash);
            std::error_code ec;
            if (!std::filesystem::exists(file, ec)) {
                std::optional<size_t> perm;
                if (const auto prog = program_cache.find(info->pgm_hash);
                    prog != program_cache.end()) {
                    const auto& modules = prog->second->modules;
                    for (size_t i = 0; i < modules.size(); ++i) {
                        if (modules[i].module == S().modules[0]) {
                            perm = i;
                            break;
                        }
                    }
                }
                std::vector<u32> spirv;
                if (perm) {
                    Storage::DataBase::Instance().Load(
                        Storage::BlobType::ShaderBinary,
                        fmt::format("{:#018x}_{}", info->pgm_hash, *perm), spirv);
                }
                std::filesystem::create_directories(dir, ec);
                if (IsCompleteSpirv(spirv) && it->second->SaveRecipe(file, spirv)) {
                    LOG_INFO(Render_Vulkan,
                             "PERF-GT7-037: compute pipeline {:#x} kept, to be built ahead at "
                             "the next start",
                             pipeline_hash);
                }
            }
        }
        ++num_new_pipelines;
        MaybeSaveDriverCache();

        if (EmulatorSettings.IsShaderCollect()) {
            auto& m = S().modules[0];
            module_related_pipelines[m].emplace_back(compute_key);
        }
    }
    return it->second.get();
}

void PipelineCache::RefreshSwizzledBlend(u32 cb, Shader::PsColorBuffer& color_buffer,
                                         const AmdGpu::BlendControl& bc) {
    const auto& regs = Regs();
    // Dual-source blending needs attachment 0 as the only written color target. Hardware writes
    // another MRT only when it is bound, unmasked, and has a shader export format.
    const auto writes_other_mrt = [&] {
        for (u32 other = 1; other < AmdGpu::NUM_COLOR_BUFFERS; ++other) {
            if (regs.color_buffers[other] && regs.color_target_mask.GetMask(other) != 0 &&
                regs.color_export_format.GetFormat(other) != AmdGpu::ShaderExportFormat::Zero) {
                return true;
            }
        }
        return false;
    };
    const char* rejection = nullptr;
    if (cb != 0) {
        rejection = "not attachment 0";
    } else if (!instance.IsDualSourceBlendSupported()) {
        rejection = "dual-source blending unsupported";
    } else if (writes_other_mrt()) {
        rejection = "other MRT exports";
    } else if ((regs.color_shader_mask.GetMask(cb) & AmdGpu::ColorBufferMask::ComponentA) == 0) {
        rejection = "alpha not exported";
    }
    if (!rejection && LiverpoolToVK::NeedsSwizzledAlphaBlend(color_buffer.swizzle, bc)) {
        color_buffer.blend_swizzled_alpha = 1;
        return;
    }

    using NumberFormat = AmdGpu::NumberFormat;
    const auto num_format = color_buffer.num_format;
    const bool blendable_format =
        num_format == NumberFormat::Unorm || num_format == NumberFormat::Snorm ||
        num_format == NumberFormat::Srgb || num_format == NumberFormat::Float;
    if (!rejection && color_buffer.num_conversion != AmdGpu::NumberConversion::None) {
        rejection = "number conversion";
    } else if (!rejection && !blendable_format) {
        rejection = "number format";
    }
    const auto general = LiverpoolToVK::GetSwizzledFactorBlend(color_buffer.swizzle, bc);
    if (!rejection && general) {
        color_buffer.blend_swizzled_factors = 1;
        color_buffer.swizzled_color_src = general->color_src;
        color_buffer.swizzled_color_dst = general->color_dst;
        color_buffer.swizzled_alpha_src = general->alpha_src;
        color_buffer.swizzled_alpha_dst = general->alpha_dst;
        return;
    }
    if (!rejection) {
        rejection = "unsupported equation";
    }

    // Report each remaining configuration once; it still blends with the wrong lane order.
    const auto& swizzle = color_buffer.swizzle;
    const u64 config =
        u64(std::bit_cast<u32>(bc)) | u64(cb) << 32 | u64(u32(num_format)) << 36 |
        u64(swizzle.r) << 40 | u64(swizzle.g) << 44 | u64(swizzle.b) << 48 | u64(swizzle.a) << 52 |
        u64(writes_other_mrt()) << 56 |
        u64((regs.color_shader_mask.GetMask(cb) & AmdGpu::ColorBufferMask::ComponentA) != 0) << 57;
    static constexpr size_t MaxLoggedSwizzledBlends = 32;
    if (t_lookup_only || // NAT-031: job threads leave the log to the threads above
        logged_swizzled_blends.size() >= MaxLoggedSwizzledBlends ||
        !logged_swizzled_blends.insert(config).second) {
        return;
    }
    const auto blend = LiverpoolToVK::EffectiveBlend(bc);
    LOG_WARNING(Render_Vulkan,
                "Unhandled swizzled blend ({}): cb={} swizzle={},{},{},{} format={}/{} "
                "color={}*src {} {}*dst alpha={}*src {} {}*dst separate={} export_formats={:#x} "
                "shader_mask={:#x} target_mask={:#x}",
                rejection, cb, u32(swizzle.r), u32(swizzle.g), u32(swizzle.b), u32(swizzle.a),
                u32(color_buffer.data_format), u32(num_format), u32(blend.color_src),
                u32(blend.color_func), u32(blend.color_dst), u32(blend.alpha_src),
                u32(blend.alpha_func), u32(blend.alpha_dst), u32(bc.separate_alpha_blend),
                regs.color_export_format.raw, regs.color_shader_mask.raw,
                regs.color_target_mask.raw);
}

bool PipelineCache::RefreshGraphicsKey() {
    std::memset(&S().graphics_key, 0, sizeof(GraphicsPipelineKey));
    const auto& regs = Regs();
    auto& key = S().graphics_key;

    const bool db_enabled = regs.depth_buffer.DepthValid() || regs.depth_buffer.StencilValid();

    key.z_format = regs.depth_buffer.DepthValid() ? regs.depth_buffer.z_info.format
                                                  : AmdGpu::DepthBuffer::ZFormat::Invalid;
    key.stencil_format = regs.depth_buffer.StencilValid()
                             ? regs.depth_buffer.stencil_info.format
                             : AmdGpu::DepthBuffer::StencilFormat::Invalid;
    key.depth_clamp_enable = !regs.depth_render_override.disable_viewport_clamp;
    key.depth_clip_enable = regs.clipper_control.ZclipEnable();
    // FIX-220: DB_ALPHA_TO_MASK. GT Sport draws the track-edge overlays (grass and kerb paint over
    // the asphalt) with alpha to coverage; without it their noise-masked alpha showed as dots.
    key.alpha_to_coverage = regs.db_alpha_to_mask & 1;
    key.clip_space = regs.clipper_control.clip_space;
    key.provoking_vtx_last = regs.polygon_control.provoking_vtx_last;
    key.prim_type =
        QuadListAsTriangles() ? AmdGpu::PrimitiveType::TriangleList : regs.primitive_type;
    key.polygon_mode = regs.polygon_control.PolyMode();
    key.patch_control_points =
        regs.stage_enable.hs_en ? regs.ls_hs_config.hs_input_control_points : 0;
    key.logic_op = regs.color_control.rop3;
    key.depth_samples = db_enabled ? regs.depth_buffer.NumSamples() : 1;
    key.num_samples = key.depth_samples;
    key.cb_shader_mask = regs.color_shader_mask;

    const bool skip_cb_binding =
        regs.color_control.mode == AmdGpu::ColorControl::OperationMode::Disable;

    // First pass to fill render target information needed by shader recompiler
    for (s32 cb = 0; cb < AmdGpu::NUM_COLOR_BUFFERS && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        if (!col_buf || !regs.color_target_mask.GetMask(cb)) {
            // No attachment bound or writing to it is disabled.
            continue;
        }

        const auto data_format = col_buf.GetDataFmt();
        const auto number_format = col_buf.GetNumberFmt();
        if (Vulkan::LiverpoolToVK::TrySurfaceFormat(data_format, number_format) ==
            vk::Format::eUndefined) {
            LOG_WARNING(Render_Vulkan,
                        "Skipping draw with unsupported color target {} format: data={}, number={}",
                        cb, static_cast<u32>(data_format), static_cast<u32>(number_format));
            return false;
        }

        // Fill color target information
        auto& color_buffer = key.color_buffers[cb];
        color_buffer.data_format = data_format;
        color_buffer.num_format = number_format;
        color_buffer.num_conversion = col_buf.GetNumberConversion();
        color_buffer.export_format = regs.color_export_format.GetFormat(cb);
        color_buffer.swizzle = col_buf.Swizzle();

        const auto& bc = regs.blend_control[cb];
        color_buffer.blend_self_scale =
            bc.enable && !col_buf.info.blend_bypass &&
            (bc.color_func == AmdGpu::BlendControl::BlendFunc::Min ||
             bc.color_func == AmdGpu::BlendControl::BlendFunc::Max) &&
            bc.color_src_factor == AmdGpu::BlendControl::BlendFactor::SrcColor &&
            bc.color_dst_factor == AmdGpu::BlendControl::BlendFactor::DstColor;
        if (!col_buf.info.blend_bypass &&
            LiverpoolToVK::IsLaneDependentSwizzledBlend(color_buffer.swizzle, bc)) {
            RefreshSwizzledBlend(cb, color_buffer, bc);
        }
    }

    // Compile and bind shader stages
    if (!RefreshGraphicsStages()) {
        return false;
    }

    // Second pass to mask out render targets not written by shader and fill remaining info
    u8 color_samples = 0;
    bool all_color_samples_same = true;
    for (s32 cb = 0; cb < key.num_color_attachments && !skip_cb_binding; ++cb) {
        const auto& col_buf = regs.color_buffers[cb];
        const u32 target_mask = regs.color_target_mask.GetMask(cb);
        if (!col_buf || !target_mask) {
            continue;
        }
        if ((key.mrt_mask & (1u << cb)) == 0) {
            std::memset(&key.color_buffers[cb], 0, sizeof(Shader::PsColorBuffer));
            continue;
        }

        // Fill color blending information
        if (regs.blend_control[cb].enable && !col_buf.info.blend_bypass) {
            key.blend_controls[cb] = regs.blend_control[cb];
        }

        // Apply swizzle to target mask
        key.write_masks[cb] =
            vk::ColorComponentFlags{key.color_buffers[cb].swizzle.ApplyMask(target_mask)};

        // Fill color samples
        const u8 prev_color_samples = std::exchange(color_samples, col_buf.NumSamples());
        all_color_samples_same &= color_samples == prev_color_samples || prev_color_samples == 0;
        key.color_samples[cb] = color_samples;
        key.num_samples = std::max(key.num_samples, color_samples);
    }

    // Force all color samples to match depth samples to avoid unsupported MSAA configuration
    if (color_samples != 0) {
        const bool depth_mismatch = db_enabled && color_samples != key.depth_samples;
        if (!all_color_samples_same && !instance.IsMixedAnySamplesSupported() ||
            all_color_samples_same && depth_mismatch && !instance.IsMixedDepthSamplesSupported()) {
            key.color_samples.fill(key.depth_samples);
            key.num_samples = key.depth_samples;
        }
    }

    return true;
}

bool PipelineCache::RefreshGraphicsStages() {
    const auto& regs = Regs();
    auto& key = S().graphics_key;
    S().fetch_shader = nullptr;
    S().stage_deferred = false; // FIX-078
    S().key_uses_legacy = false; // D10
    S().legacy_failed = false;

    // FIX-027: a shader program register pointing outside guest memory (GT Sport's dealership had
    // a pixel shader at 0x100) crashed the emulator reading the program. Skip such a draw, like
    // the GPU would fault on it rather than take the system down. Checked again only when an
    // address changes. -DisablePerf 36 reads it anyway.
    static const bool check_programs = Common::PerfFeatureEnabled(36);
    if (check_programs) {
        static std::array<VAddr, 6> checked{};
        for (u32 stage = 0; stage < checked.size(); ++stage) {
            const auto* pgm = regs.ProgramForStage(stage);
            const VAddr address = pgm ? pgm->Address<VAddr>() : 0;
            if (!regs.stage_enable.IsStageEnabled(stage) || address == 0 ||
                address == checked[stage]) {
                continue;
            }
            auto* memory = Core::Memory::Instance();
            if (!memory->IsValidMapping(address, 8) || !memory->IsMappedAddress(address)) {
                static std::atomic<u32> skipped{};
                if (const u32 n = ++skipped; n <= 20 || n % 500 == 0) {
                    LOG_WARNING(Render_Vulkan,
                                "FIX-027: draw {} skipped: hardware stage {} program at {:#x} is "
                                "not in guest memory ({})",
                                n, stage, address, regs_override ? "read-ahead" : "draw");
                }
                return false;
            }
            if (!t_lookup_only) { // NAT-031: job threads only read the list
                checked[stage] = address;
            }
        }
    }

    Shader::Backend::Bindings binding{};
    bool stage_failed = false;
    const auto bind_stage = [&](HwStage stage_in, SwStage stage_out) -> bool {
        const auto stage_in_idx = static_cast<u32>(stage_in);
        const auto stage_out_idx = static_cast<u32>(stage_out);
        if (!regs.stage_enable.IsStageEnabled(stage_in_idx)) {
            key.stage_hashes[stage_out_idx] = 0;
            S().infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto* pgm = regs.ProgramForStage(stage_in_idx);
        if (!pgm || !pgm->Address<u32*>()) {
            key.stage_hashes[stage_out_idx] = 0;
            S().infos[stage_out_idx] = nullptr;
            return false;
        }

        const auto params = AmdGpu::GetParams(*pgm);
        std::tie(S().infos[stage_out_idx], S().modules[stage_out_idx], key.stage_hashes[stage_out_idx]) =
            GetProgram(stage_in, stage_out, params, binding);
        if (!S().modules[stage_out_idx]) {
            stage_failed = true; // FIX-034: the shader could not be translated
        }
        return true;
    };

    S().infos.fill(nullptr);
    S().modules.fill(nullptr);

    bind_stage(HwStage::Fragment, SwStage::Fragment);

    const auto* fs_info = S().infos[static_cast<u32>(SwStage::Fragment)];
    key.mrt_mask = fs_info ? fs_info->mrt_mask : 0u;
    key.num_color_attachments = std::bit_width(key.mrt_mask);
    if (key.num_color_attachments > AmdGpu::NUM_COLOR_BUFFERS) {
        LOG_WARNING(Render_Vulkan, "Skipping draw with invalid MRT mask {:#x}", key.mrt_mask);
        return false;
    }

    switch (regs.stage_enable.raw) {
    case AmdGpu::ShaderStageEnable::VgtStages::EsGs:
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Export, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (const auto* hull = S().infos[u32(SwStage::TessellationControl)];
            !hull || !SaneTessConstants(*hull, liverpool, regs_override != nullptr)) {
            return false; // FIX-058: the local stage reads the hull's constants
        }
        if (!bind_stage(HwStage::Vertex, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::LsHsEsGs:
        if (!instance.IsTessellationSupported()) {
            return false;
        }
        if (!instance.IsGeometryStageSupported()) {
            LOG_WARNING(Render_Vulkan, "Geometry shader stage unsupported, skipping");
            return false;
        }
        if (regs.vgt_gs_mode.onchip || regs.vgt_strmout_config.raw) {
            LOG_WARNING(Render_Vulkan, "Geometry shader features unsupported, skipping");
            return false;
        }
        if (!bind_stage(HwStage::Hull, SwStage::TessellationControl)) {
            return false;
        }
        if (const auto* hull = S().infos[u32(SwStage::TessellationControl)];
            !hull || !SaneTessConstants(*hull, liverpool, regs_override != nullptr)) {
            return false; // FIX-058: the local stage reads the hull's constants
        }
        if (!bind_stage(HwStage::Export, SwStage::TessellationEval)) {
            return false;
        }
        if (!bind_stage(HwStage::Local, SwStage::Vertex)) {
            return false;
        }
        if (!bind_stage(HwStage::Geometry, SwStage::Geometry)) {
            return false;
        }
        break;
    case AmdGpu::ShaderStageEnable::VgtStages::Vs:
        bind_stage(HwStage::Vertex, SwStage::Vertex);
        break;
    default:
        LOG_WARNING(Render_Vulkan, "unimplemented shader stage {}", (u32)regs.stage_enable.raw);
        return false;
    }
    if (stage_failed) {
        return false;
    }

    const auto* vs_info = S().infos[static_cast<u32>(SwStage::Vertex)];
    if (vs_info && S().fetch_shader && !instance.IsVertexInputDynamicState()) {
        // Without vertex input dynamic state, the pipeline needs to specialize on format.
        // Stride will still be handled outside the pipeline using dynamic state.
        u32 vertex_binding = 0;
        for (const auto& attrib : S().fetch_shader->attributes) {
            const auto& buffer = attrib.GetSharp(*vs_info);
            ASSERT_MSG(vertex_binding < MaxVertexBufferCount,
                       "Vertex attribute binding count exceeded limit: {} >= {}", vertex_binding,
                       MaxVertexBufferCount);
            const auto data_format = buffer.GetDataFmt();
            const auto number_format = buffer.GetNumberFmt();
            const auto format = Vulkan::LiverpoolToVK::TrySurfaceFormat(data_format, number_format);
            if (format == vk::Format::eUndefined) {
                LOG_WARNING(Render_Vulkan,
                            "Skipping draw with unsupported vertex attribute {} format: data={}, "
                            "number={}",
                            vertex_binding, static_cast<u32>(data_format),
                            static_cast<u32>(number_format));
                return false;
            }
            key.vertex_buffer_formats[vertex_binding++] = format;
        }
    }

    return true;
}

bool PipelineCache::RefreshComputeKey() {
    S().key_uses_legacy = false; // D10
    S().legacy_failed = false;
    Shader::Backend::Bindings binding{};
    const auto& cs_pgm = liverpool->DrawCsRegs();
    const auto cs_params = AmdGpu::GetParams(cs_pgm);
    std::tie(S().infos[0], S().modules[0], compute_key.value) =
        GetProgram(HwStage::Compute, SwStage::Compute, cs_params, binding);
    return S().modules[0] != nullptr; // FIX-034: false when the shader could not be translated
}

} // namespace Vulkan
namespace Shader::Optimization {
extern thread_local std::span<const u32> g_diag_compiling_code;
extern thread_local u64 g_diag_compiling_hash;
} // namespace Shader::Optimization
namespace Vulkan {

vk::ShaderModule PipelineCache::CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                              const std::span<const u32>& code, size_t perm_idx,
                                              Shader::Backend::Bindings& binding) {
    // FIX-044: translating a shader on the GPU thread holds the guest clocks; a PS4's shaders
    // are compiled ahead of time, so the game never sees this time pass.
    Common::GuestClock::CompilePause hold;
    ++Common::GetWorkCounters().shaders_compiled;
    RYDIAG_SCOPE(ShaderCompile, info.pgm_hash);
    last_spv_hash = 0;
    LOG_INFO(Render_Vulkan, "Compiling {} shader {:#x} {}", info.hw_stage, info.pgm_hash,
             perm_idx != 0 ? "(permutation)" : "");
    DumpShader(code, info.pgm_hash, info.hw_stage, perm_idx, "bin");

    Shader::Optimization::g_diag_compiling_code = code;
    Shader::Optimization::g_diag_compiling_hash = info.pgm_hash;
    const auto translate_start = std::chrono::steady_clock::now();
    std::vector<u32> spv;
    try {
        // FIX-034: a shader the translator cannot handle no longer stops the emulator. Its
        // draws are skipped and it is reported once, so one run lists every such shader.
        Common::RecoverableScope recoverable;
        const auto ir_program = Shader::TranslateProgram(code, pools, info, runtime_info, profile);
        spv = Shader::Backend::SPIRV::EmitSPIRV(profile, runtime_info, ir_program, binding);
    } catch (const Common::RecoverableFailure& e) {
        LOG_ERROR(Render_Vulkan,
                  "SHADER-FAIL: {} shader {:#x} (permutation {}, {} dwords) could not be "
                  "translated: {}; draws that use it are skipped",
                  info.hw_stage, info.pgm_hash, perm_idx, code.size(), e.what());
        DumpFailedShader(code, info.pgm_hash, info.hw_stage);
        return {};
    }
    const auto translate_end = std::chrono::steady_clock::now();
    DumpShader(spv, info.pgm_hash, info.hw_stage, perm_idx, "spv");
    last_spv_hash = XXH3_64bits(spv.data(), spv.size() * sizeof(u32));
    if (!Common::PerfFeatureEnabled(44)) {
        // FIX-043 off (-DisablePerf 44): permutations keyed on every resource property.
        info.resource_usage_known = false;
    }

    vk::ShaderModule module;

    auto patch = GetShaderPatch(info.pgm_hash, info.hw_stage, perm_idx, "spv");
    const bool is_patched = patch && EmulatorSettings.IsPatchShaders();
    if (is_patched) {
        LOG_INFO(Loader, "Loaded patch for {} shader {:#x}", info.hw_stage, info.pgm_hash);
        module = CompileSPV(*patch, instance.GetDevice());
    } else {
        module = CompileSPV(spv, instance.GetDevice());
    }
    // PERF-DIAG-013: where runtime shader compile time goes.
    LOG_WARNING(
        Render_Vulkan, "Shader {} {:#x}: translate {:.1f} ms, module {:.1f} ms, {} words",
        info.hw_stage, info.pgm_hash,
        std::chrono::duration<double, std::milli>(translate_end - translate_start).count(),
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - translate_end)
            .count(),
        spv.size());

    RegisterShaderBinary(std::move(spv), info.pgm_hash, perm_idx);

    const auto name = GetShaderName(info.hw_stage, info.pgm_hash, perm_idx);
    Vulkan::SetObjectName(instance.GetDevice(), module, name);
    if (EmulatorSettings.IsShaderCollect()) {
        DebugState.CollectShader(name, info.sw_stage, module, spv, code,
                                 patch ? *patch : std::span<const u32>{}, is_patched);
    }
    return module;
}

PipelineCache::StoredListing PipelineCache::ReadStoredListing(u64 pgm_hash) {
    return ReadStoredListing(pgm_hash, Storage::DataBase::Instance());
}

PipelineCache::StoredListing PipelineCache::ReadStoredListing(u64 pgm_hash,
                                                              Storage::DataBase& store) {
    // Each permutation is stored as <perm hash>.meta (key and shader info) and
    // <program hash>_<index>.spv; indices are dense from 0 (PERF-013 numbers runtime ones after
    // the stored ones), so the listing stops after a run of missing indices.
    StoredListing listing{.pgm_hash = pgm_hash};
    // DIAG-191 (diagnosis, SHADGT_RETRANSLATE=<hex hash>,...): these programs are translated
    // again (and dumped with GPU.dump_shaders) instead of loaded from the store.
    static const std::vector<u64> retranslate = [] {
        std::vector<u64> hashes;
        if (const char* env = std::getenv("SHADGT_RETRANSLATE"); env && *env) {
            const std::string list{env};
            for (size_t start = 0; start < list.size();) {
                const size_t end = std::min(list.find(',', start), list.size());
                hashes.push_back(std::stoull(list.substr(start, end - start), nullptr, 16));
                start = end + 1;
            }
        }
        return hashes;
    }();
    if (std::ranges::find(retranslate, pgm_hash) != retranslate.end()) {
        return listing;
    }
    u32 missing = 0;
    for (size_t perm_idx = 0; perm_idx < 512 && missing < 8; ++perm_idx) {
        const auto name = fmt::format("{:#018x}", HashCombine(pgm_hash, perm_idx));
        if (!store.Exists(Storage::BlobType::ShaderMeta, name)) {
            ++missing;
            continue;
        }
        missing = 0;
        auto& entry = listing.entries.emplace_back();
        entry.perm_idx = perm_idx;
        store.Load(Storage::BlobType::ShaderMeta, name, entry.meta);
        entry.has_spv = store.Exists(Storage::BlobType::ShaderBinary,
                                     fmt::format("{:#018x}_{}", pgm_hash, perm_idx));
    }
    return listing;
}

void PipelineCache::ApplyStoredListing(StoredListing&& listing, Program& program) {
    static u64 programs = 0;
    static u64 listed = 0;
    static u64 stale = 0;
    const u64 pgm_hash = listing.pgm_hash;
    for (auto& entry : listing.entries) {
        const size_t perm_idx = entry.perm_idx;
        auto info = std::make_unique<Shader::Info>();
        Shader::StageSpecialization spec{};
        spec.info = info.get();
        size_t stored_idx{};
        Serialization::Archive ar{std::move(entry.meta)};
        // Entries of another format version, cut short, or for another program are skipped.
        bool valid = false;
        try {
            Common::RecoverableScope recoverable;
            valid = ar.SizeBytes() >= 24 && LoadShaderMeta(ar, *info, spec, stored_idx);
        } catch (const Common::RecoverableFailure&) {
            valid = false;
        }
        if (!valid || stored_idx != perm_idx || info->pgm_hash != pgm_hash) {
            ++stale;
            continue;
        }
        // D10: stored before D2, the byte uses_direct takes was padding.
        info->uses_direct = std::ranges::any_of(info->buffers, [](const auto& buffer) {
            return buffer.buffer_type == Shader::BufferType::DirectTable;
        });
        auto& end = stored_perm_end[pgm_hash];
        end = std::max(end, perm_idx + 1);
        // FIX-018/020: translated before a translator fix this program needs.
        if (const auto fixed = else_scope_fixed.find(pgm_hash);
            fixed != else_scope_fixed.end() && perm_idx < fixed->second.boundary) {
            continue;
        }
        if (!entry.has_spv) {
            continue;
        }
        if (perm_idx < program.modules.size() && program.modules[perm_idx].info) {
            continue;
        }
        program.InsertPermut({}, std::move(spec), std::move(info), perm_idx);
        program.modules[perm_idx].stored = true;
        ++listed;
    }
    if (++programs % 500 == 0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-032: shader store: {} programs looked up, {} stored permutations "
                    "listed, {} entries of another version skipped",
                    programs, listed, stale);
    }
}

void PipelineCache::ListStoredPermutations(u64 pgm_hash, Program& program) {
    ApplyStoredListing(ReadStoredListing(pgm_hash), program);
}

bool PipelineCache::StoredListingReady(u64 pgm_hash, bool may_skip) {
    auto [pending, queued] = pending_listings.try_emplace(pgm_hash);
    if (queued) {
        pending->second = store_readers->List(pgm_hash);
        ++store_stats.listed;
    }
    bool ready = pending->second.wait_for(std::chrono::seconds{0}) == std::future_status::ready;
    if (!ready && may_skip) {
        ++store_stats.skipped;
    } else if (!ready) {
        const auto start = std::chrono::steady_clock::now();
        pending->second.wait();
        ++store_stats.waited;
        store_stats.wait_ms += std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
        ready = true;
    }
    if (const auto now = std::chrono::steady_clock::now();
        now - store_stats.since >= std::chrono::seconds{10}) {
        if (store_stats.listed != 0) {
            LOG_WARNING(Render_Vulkan,
                        "PERF-GT7-019: shader store in {:.1f} s: {} programs listed on the store "
                        "readers, {} lookups went on without their listing, {} waited for it "
                        "({:.1f} ms); {} SPIR-V files read ahead so far",
                        std::chrono::duration<double>(now - store_stats.since).count(),
                        store_stats.listed, store_stats.skipped, store_stats.waited,
                        store_stats.wait_ms, store_readers->files_read_ahead.load());
        }
        store_stats = {};
        store_stats.since = now;
    }
    return ready;
}

bool PipelineCache::NeverSkipped(const std::array<const Shader::Info*, MaxShaderStages>& infos) {
    // Keep going: rendering must not wait for a cosmetic pipeline.
    if (CompileMode() == 2) return false;
    // FIX-064: tessellated draws wait for their pipeline. On GT Sport's race summary they are
    // the driver figure; skipped while each of its pipelines built, parts of it blinked out
    // frame after frame. -DisablePerf 88 skips them like other draws.
    // FIX-115: not during a race, though: there a tessellated car or road part that builds for
    // 0.3-2 s stopped the race that long. It shows a moment later instead (the quick wait still
    // draws the ones that build fast). -DisablePerf 138 waits in races too.
    static const bool wait_for_tessellation = Common::PerfFeatureEnabled(88);
    static const bool skip_in_races = Common::PerfFeatureEnabled(138);
#ifdef RYTY_GPU
    if (skip_in_races && RytyGtsInRace()) {
        return false;
    }
#endif
    return wait_for_tessellation && infos[u32(Shader::SwStage::TessellationControl)] != nullptr;
}

bool PipelineCache::WritesMemory(const std::array<const Shader::Info*, MaxShaderStages>& infos) {
    // PERF-076: a draw that writes buffers or images is never skipped. GT Sport's results screen
    // writes data later draws read as vertex descriptors; skipping those draws left garbage
    // there and the renderer ran out of memory on it.
    for (const auto* info : infos) {
        if (!info) {
            continue;
        }
        if (info->has_storage_images) {
            return true;
        }
        for (const auto& buffer : info->buffers) {
            if (buffer.is_written) {
                return true;
            }
        }
        for (const auto& image : info->images) {
            if (image.is_written) {
                return true;
            }
        }
    }
    return false;
}

bool PipelineCache::AsyncPipelines() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_ASYNC_PIPELINES");
#ifdef RYTY_GPU
        // PERF-100: GPU.compile_freeze (on by default) pauses the game while a new pipeline
        // compiles instead of skipping its draws.
        // PERF-103: GPU.compile_mode 0 (pause) waits; slow motion and keep going build in
        // the background.
        const bool by_default = CompileMode() != 0;
#else
        const bool by_default = false;
#endif
        const bool on = env ? env[0] != '0' : by_default;
        return on && Common::PerfFeatureEnabled(76);
    }();
    return enabled;
}

static s64 SteadyMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

void PipelineCache::NoteAsyncSkip(const std::shared_ptr<PipelineBuild>& build, bool again) {
    catch_up_skips.fetch_add(1, std::memory_order_relaxed);
    build->skips.fetch_add(1, std::memory_order_relaxed); // DIAG-GT7-012
    build->last_skip_ms.store(SteadyMs(), std::memory_order_relaxed);
    // A draw skipped again comes back frame after frame: it is on screen. Its build, if no worker
    // has started it, goes ahead of the queue (the job already queued finds it started and does
    // nothing).
    // FIX-GT7-009: not to the front itself, last first. On a new track ~1,500 of them were moved
    // so, each in front of the ones before: the draws skipped most (one 111,000 times) waited
    // minutes behind newer ones and were never started, and the catch-up lasted 163 s. Each such
    // draw queues a screen job, which builds the one skipped most lately at the time it runs.
    if (again && build_workers && !build->started.load(std::memory_order_relaxed) &&
        !build->bumped.exchange(true)) {
        {
            std::scoped_lock lk{on_screen_mutex};
            on_screen.push_back(build);
        }
        build_workers->Push([this] { BuildOnScreen(); }, BuildQueue::Kind::Screen);
        catch_up_bumped.fetch_add(1, std::memory_order_relaxed);
    }
}

void PipelineCache::BuildOnScreen() {
    std::shared_ptr<PipelineBuild> best;
    {
        std::scoped_lock lk{on_screen_mutex};
        std::erase_if(on_screen, [](const std::shared_ptr<PipelineBuild>& build) {
            return build->started.load(std::memory_order_relaxed);
        });
        // Skipped in the last second first (still on screen), then the most skipped.
        const s64 now = SteadyMs();
        const auto score = [&](const PipelineBuild& build) {
            const bool lately = now - build.last_skip_ms.load(std::memory_order_relaxed) <= 1000;
            return (u64(lately) << 32) | build.skips.load(std::memory_order_relaxed);
        };
        auto chosen = on_screen.end();
        u64 chosen_score = 0;
        for (auto it = on_screen.begin(); it != on_screen.end(); ++it) {
            if (const u64 s = score(**it); chosen == on_screen.end() || s > chosen_score) {
                chosen = it;
                chosen_score = s;
            }
        }
        if (chosen != on_screen.end()) {
            best = std::move(*chosen);
            on_screen.erase(chosen);
        }
    }
    if (best && !best->started.load()) {
        FinishBuild(*best);
    }
}

void PipelineCache::UpdateCatchUp(std::chrono::steady_clock::time_point now) {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_COMPILE_CATCH_UP");
        return !(env && env[0] == '0');
    }();
    if (!enabled || !build_workers) {
        return;
    }
    std::unique_lock lk{catch_up_mutex, std::try_to_lock};
    if (!lk.owns_lock()) {
        return;
    }
    auto& c = catch_up;
    if (now - c.window_start < std::chrono::milliseconds{250}) {
        return;
    }
    const double seconds = std::chrono::duration<double>(now - c.window_start).count();
    c.window_start = now;
    const u64 skips = catch_up_skips.exchange(0, std::memory_order_relaxed);
    const double rate = double(skips) / seconds;
    // In: 3,000 skipped draws a second (100 a frame at 30 fps; a new track's start skipped up to
    // 25,000 a second for two minutes). Out: under 1,000 a second for a second.
    static const bool paced = [] {
        const char* env = std::getenv("RYGT_CATCH_UP_PACE");
        return !(env && env[0] == '0');
    }();
    if (!c.on) {
        if (rate >= 3000.0) {
            c.on = true;
            c.since = now;
            c.calm = 0;
            c.skipped = skips;
            c.stepped_back = false;
            c.step_backs = 0;
            c.stepped_seconds = 0;
            catch_up_bumped.store(0, std::memory_order_relaxed);
            build_workers->SetCatchUp(true);
            LOG_WARNING(Render_Vulkan,
                        "PERF-GT7-029: {:.0f} draws a second skipped while their pipelines build: "
                        "compiles on every core, 8 at a time{}",
                        rate,
                        PipelineBuildWorkers::CatchUpLow() ? ", below normal priority" : "");
        }
    } else {
        c.skipped += skips;
        c.calm = rate < 1000.0 ? c.calm + 1 : 0;
        if (c.calm >= 4) {
            c.on = false;
            if (c.stepped_back) {
                c.stepped_seconds += std::chrono::duration<double>(now - c.stepped_since).count();
            } else {
                build_workers->SetCatchUp(false);
            }
            LOG_WARNING(Render_Vulkan,
                        "PERF-GT7-029: catch-up over after {:.1f} s ({} draws skipped meanwhile, "
                        "{} builds of draws on screen moved ahead; PERF-GT7-033: stepped back {} "
                        "times, {:.1f} s in all); compiles back on their cores",
                        std::chrono::duration<double>(now - c.since).count(), c.skipped,
                        catch_up_bumped.exchange(0), c.step_backs, c.stepped_seconds);
        }
    }
    // PERF-GT7-033, PERF-GT7-034: the background compiles follow the game's frame rate.
    // Each second the game takes more than 34 ms a frame on average (one frame of 66 ms in a
    // second already makes 34.4), they take one step back: a catch-up goes back to the usual
    // compiles (4 at a time on the compile cores, normal priority), then one fewer at a time,
    // down to one. After three seconds under 33.8 ms they take one step forward again. With 4
    // compiles on the compile cores (the driver's compiler threads ~95% each, the process ~930%
    // of 1,600), the command thread still decoded 9-11 ms a submit instead of 6-7 in GT7's
    // heavy scenes (the pit straight, ~3,000 draws): every 5-10 frames one took 66 ms. A draw
    // waiting for its pipeline still widens the compiles; the draws on screen still build first
    // (FIX-GT7-009). In menus too (they run at 60 fps or wait for their pipelines).
    // FIX-GT7-022: past the catch-up's step back, only the read-ahead builds step back: the
    // builds of draws that went on without their pipeline keep the usual width.
    // RYGT_CATCH_UP_PACE=0: never.
    if (!paced) {
        return;
    }
    if (c.pace_start == std::chrono::steady_clock::time_point{}) {
        c.pace_start = now;
        const auto totals = Common::GetGameFrameTotals();
        c.pace_frames = totals.frames;
        c.pace_us = totals.us;
        c.pace_limit = PipelineBuildWorkers::UsualWidth();
        return;
    }
    if (now - c.pace_start < std::chrono::seconds{1}) {
        return;
    }
    const auto totals = Common::GetGameFrameTotals();
    const u64 frames = totals.frames - c.pace_frames;
    const double frame_ms =
        frames != 0 ? double(totals.us - c.pace_us) / 1000.0 / double(frames)
                    : std::chrono::duration<double, std::milli>(now - c.pace_start).count();
    c.pace_start = now;
    c.pace_frames = totals.frames;
    c.pace_us = totals.us;
    const u32 usual = PipelineBuildWorkers::UsualWidth();
    // FIX-GT7-012: not gated on FIX-110's race flag, which GT7 never sets (its project scripts
    // aren't seen): PERF-GT7-034 never acted. A second over 80 ms a frame (loading, a draw
    // waiting for its pipeline, a stall) is neither late nor on time: compiles are not what
    // holds the game then, and a draw waiting widens them anyway.
    if (frame_ms >= 80.0) {
        c.fast = 0;
        return;
    }
    const auto note = [&](const char* what) {
        if (++c.pace_changes <= 40) {
            LOG_WARNING(Render_Vulkan, "PERF-GT7-034: the game at {:.1f} ms a frame: {}",
                        frame_ms, what);
        }
    };
    if (frame_ms > 34.0) {
        c.fast = 0;
        if (c.on && !c.stepped_back) {
            c.stepped_back = true;
            c.stepped_since = now;
            ++c.step_backs;
            build_workers->SetCatchUp(false);
            note("the catch-up's compiles back to 4 at a time on their cores");
        } else if (c.pace_limit > 1 && build_workers->Busy()) {
            --c.pace_limit;
            build_workers->SetPace(c.pace_limit);
            note(c.pace_limit == 1
                     ? "read-ahead compiles one at a time"
                     : fmt::format("read-ahead compiles {} at a time", c.pace_limit).c_str());
        }
        return;
    }
    c.fast = frame_ms < 33.8 ? c.fast + 1 : 0;
    if (c.fast < 3) {
        return;
    }
    c.fast = 0;
    if (c.pace_limit < usual) {
        ++c.pace_limit;
        build_workers->SetPace(c.pace_limit == usual ? 64 : c.pace_limit);
        note(fmt::format("read-ahead compiles {} at a time", c.pace_limit).c_str());
    } else if (c.on && c.stepped_back) {
        c.stepped_back = false;
        c.stepped_seconds += std::chrono::duration<double>(now - c.stepped_since).count();
        build_workers->SetCatchUp(true);
        note("the catch-up's compiles resume");
    }
}

void PipelineCache::DropStaleReadAheadBuilds(std::chrono::steady_clock::time_point now) {
    if (now - stale_check < std::chrono::seconds{2}) {
        return;
    }
    stale_check = now;
    // A read-ahead build (not async: no draw has asked for it) still queued after 30 s. Its job
    // finds it started and does nothing; a draw that needs the key later builds it again.
    for (auto it = pending_builds.begin(); it != pending_builds.end();) {
        auto& build = *it->second;
        if (build.async || now - build.queued < std::chrono::seconds{30} ||
            build.started.load(std::memory_order_relaxed) || !CancelBuild(build)) {
            ++it;
            continue;
        }
        ++stale_dropped;
        it = pending_builds.erase(it);
    }
}

void PipelineCache::ReportAsyncPipelines(std::chrono::steady_clock::time_point now) {
    if (now - async_stats.since < std::chrono::seconds{2}) {
        return;
    }
    if (async_stats.deferred + async_stats.installed + async_stats.skipped_draws +
            async_stats.skipped_translating !=
        0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-076 background pipelines in {:.1f} s: {} new ones built without a wait, "
                    "{} ready and installed, {} draws skipped meanwhile; SHD-005: {} draws "
                    "skipped while their shaders translated",
                    std::chrono::duration<double>(now - async_stats.since).count(),
                    async_stats.deferred, async_stats.installed, async_stats.skipped_draws,
                    async_stats.skipped_translating);
    }
    if (async_stats.skipped_draws != 0) {
        // DIAG-GT7-012: the builds draws are skipped for. In a GT7 race on a new track ~4,000
        // draws a second were skipped for three minutes (the catch-up never ended) while a few
        // builds finished in 2 s, often none.
        u32 count = 0;
        u32 started = 0;
        const PipelineBuild* oldest = nullptr;
        const PipelineBuild* most = nullptr;
        for (const auto& [key, build] : pending_builds) {
            if (!build->async ||
                build->done.wait_for(std::chrono::microseconds{0}) == std::future_status::ready) {
                continue;
            }
            ++count;
            started += build->started.load(std::memory_order_relaxed) ? 1 : 0;
            if (!oldest || build->queued < oldest->queued) {
                oldest = build.get();
            }
            if (!most || build->skips.load(std::memory_order_relaxed) >
                             most->skips.load(std::memory_order_relaxed)) {
                most = build.get();
            }
        }
        if (count != 0) {
            const auto describe = [&](const PipelineBuild& build) {
                return fmt::format(
                    "{:#x}, queued {:.1f} s ago, {}, {} draws skipped",
                    std::hash<GraphicsPipelineKey>{}(build.key),
                    std::chrono::duration<double>(now - build.queued).count(),
                    build.started.load(std::memory_order_relaxed) ? "building" : "not started",
                    build.skips.load(std::memory_order_relaxed));
            };
            size_t screen = 0;
            {
                std::scoped_lock lk{on_screen_mutex};
                screen = on_screen.size();
            }
            LOG_WARNING(Render_Vulkan,
                        "DIAG-GT7-012: {} pipelines still building for skipped draws ({} started, "
                        "{} jobs queued, {} on screen waiting); the oldest {}; the most skipped {}",
                        count, started, build_workers ? build_workers->Pending() : 0, screen,
                        describe(*oldest), describe(*most));
        }
    }
    async_stats = {};
    async_stats.since = now;
}

vk::ShaderModule PipelineCache::LoadStoredModule(u64 pgm_hash, size_t perm_idx) {
    return LoadStoredModule(pgm_hash, perm_idx, Storage::DataBase::Instance());
}

vk::ShaderModule PipelineCache::LoadStoredModule(u64 pgm_hash, size_t perm_idx,
                                                 Storage::DataBase& store) {
    static u64 loaded = 0;
    std::vector<u32> spv;
    store.Load(Storage::BlobType::ShaderBinary,
                                       fmt::format("{:#018x}_{}", pgm_hash, perm_idx), spv);
    if (!IsCompleteSpirv(spv)) {
        static u32 reports = 0;
        if (reports++ < 20) {
            LOG_WARNING(Render_Vulkan,
                        "PERF-032: stored shader {:#x} permutation {} is incomplete; translating "
                        "it again",
                        pgm_hash, perm_idx);
        }
        return {};
    }
    // FIX-GT7-011: FIX-226 changed how float atomic min/max are made without
    // VK_EXT_shader_atomic_float2, and the stored translations kept their version: one made the
    // other way than RYGT_FLOAT_ATOMIC_EXACT asks is translated again (GT7's occlusion depth
    // pyramid came out wrong from the old way).
    if (const auto emulation = FindFloatAtomicEmulation(spv);
        emulation != FloatAtomicEmulation::None &&
        emulation != (Shader::Backend::SPIRV::FloatAtomicExact()
                          ? FloatAtomicEmulation::OneAtomic
                          : FloatAtomicEmulation::BothAtomics)) {
        static u32 reports = 0;
        if (reports++ < 20) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-GT7-011: stored shader {:#x} permutation {} makes its float atomic "
                        "min/max the other way; translating it again",
                        pgm_hash, perm_idx);
        }
        return {};
    }
    if (++loaded % 500 == 0) {
        LOG_WARNING(Render_Vulkan,
                    "PERF-032: {} shader S().modules loaded from the store instead of translated",
                    loaded);
    }
    return CompileSPV(spv, instance.GetDevice());
}

void PipelineCache::DumpFailedShader(std::span<const u32> code, u64 hash, Shader::HwStage stage) {
    // FIX-034: the guest code of a shader that could not be translated, raw and as a listing,
    // in the log folder (once per program), so it can be studied and tested offline.
    if (!failed_shaders.insert(hash).second) {
        return;
    }
    const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "failed_shaders";
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    const auto name = fmt::format("{}_{:#x}", stage, hash);
    Common::FS::IOFile{dir / (name + ".bin"), Common::FS::FileAccessMode::Create}.WriteSpan(code);
    std::ofstream{dir / (name + ".txt")} << Shader::ListGcnCode(code);
}

bool IsWatchedShader(u64 hash) {
    static const std::vector<u64> watched = [] {
        // The Nurburgring grass: vertex generator, blade generator, and the pass before them.
        // DIAG-039: the dealership shader whose buffer V#s read as float constants.
        std::vector<u64> hashes{0xab6a2d10, 0xaa3822a3, 0x766d0f18, 0xae32f77f};
        if (const char* env = std::getenv("SHADGT_WATCH_SHADERS"); env && *env) {
            hashes.clear();
            const std::string list{env};
            for (size_t start = 0; start < list.size();) {
                const size_t end = std::min(list.find(',', start), list.size());
                hashes.push_back(std::stoull(list.substr(start, end - start), nullptr, 16));
                start = end + 1;
            }
        }
        return hashes;
    }();
    return std::ranges::find(watched, hash) != watched.end();
}

void PipelineCache::LoadElseScopeFixed() {
    else_scope_fixed_path = Storage::RetranslationMarksPath();
    std::ifstream file{else_scope_fixed_path};
    std::string line;
    while (std::getline(file, line)) {
        std::istringstream fields{line};
        u64 hash{};
        size_t boundary{};
        u32 fixes = 1; // Lines written before FIX-020 have no mask.
        if (fields >> std::hex >> hash >> std::dec >> boundary) {
            fields >> fixes;
            else_scope_fixed[hash] = {boundary, fixes};
        }
    }
    if (!else_scope_fixed.empty()) {
        LOG_INFO(Render_Vulkan, "FIX-018: {} programs already translated with the else-scope fix",
                 else_scope_fixed.size());
    }
}

void PipelineCache::SaveElseScopeFixed() const {
    if (else_scope_fixed_path.empty()) {
        return;
    }
    std::ofstream file{else_scope_fixed_path, std::ios::trunc};
    for (const auto& [hash, program] : else_scope_fixed) {
        file << std::hex << hash << ' ' << std::dec << program.boundary << ' ' << program.fixes
             << '\n';
    }
}

// FIX-058: GT Sport's race summary draws tessellated objects whose tessellation constants are
// garbage (strides of hundreds of MB). Hull shaders translated from them declared arrays of 200
// million entries, and the driver ran out of memory compiling their pipelines (the process went
// from 11 GB to 100 GB in seconds; the driver figure flickered while they built). Constants are
// sane when they fit the 64 KB of LDS they describe; draws with others are skipped.
// -DisablePerf 82 draws them.
// DIAG-063: the tessellation-constant V#s (HS user data 7..10) an indirect buffer sets, read
// from a copy of it.
static std::vector<VAddr> TessBasesInIb(const u32* data, u32 dwords) {
    std::vector<u32> ib(dwords);
    Common::CopyGuest(ib.data(), data, ib.size() * sizeof(u32));
    std::vector<VAddr> bases;
    std::array<u32, 16> hs{};
    constexpr u32 HsUserData = 0x2D0C;
    const auto apply = [&](u32 first, std::span<const u32> values) {
        bool touched = false;
        for (u32 i = 0; i < values.size(); ++i) {
            if (first + i >= HsUserData + 7 && first + i < HsUserData + 11) {
                hs[first + i - HsUserData] = values[i];
                touched = true;
            }
        }
        if (touched && hs[9] == 3) {
            bases.push_back((VAddr(hs[8] & 0xffff) << 32) | hs[7]);
        }
    };
    std::span<const u32> rest{ib};
    while (!rest.empty()) {
        const u32 type = rest.front() >> 30;
        if (type == 2) {
            rest = rest.subspan(1);
        } else if (type == 0) {
            const u32 first = rest.front() & 0xffff;
            const u32 count = ((rest.front() >> 16) & 0x3fff) + 1;
            if (count + 1 > rest.size()) {
                break;
            }
            apply(first, rest.subspan(1, count));
            rest = rest.subspan(count + 1);
        } else if (type == 3) {
            const u32 count = ((rest.front() >> 16) & 0x3fff) + 1;
            const u32 opcode = (rest.front() >> 8) & 0xff;
            if (count + 1 > rest.size()) {
                break;
            }
            if (opcode == 0x76 && count >= 1) { // SET_SH_REG
                apply(0x2C00 + (rest[1] & 0xffff), rest.subspan(2, count - 1));
            }
            rest = rest.subspan(count + 1);
        } else {
            break;
        }
    }
    return bases;
}

// DIAG-063: whether the indirect buffer a draw came from changes after it was decoded, and the
// constant V#s it sets before and after.
static void WatchIbAfterDecode(u32 n, bool sane, VAddr base) {
    const auto [ib, dwords] = AmdGpu::Liverpool::CurrentDecodeIb();
    if (!ib || dwords == 0 || dwords > 4'000'000) {
        LOG_WARNING(Render_Vulkan, "DIAG-063: {} draw {}: not inside an indirect buffer",
                    sane ? "sane" : "garbage", n);
        return;
    }
    std::vector<u32> before(dwords);
    Common::CopyGuest(before.data(), ib, before.size() * sizeof(u32));
    const auto bases_before = TessBasesInIb(ib, dwords);
    static std::atomic<u32> dumps{};
    if ((!sane || n == 5000) && dumps.fetch_add(1) < 6) {
        // DIAG-064: the indirect buffer and 64 KB around the constants, for offline study.
        const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
        const auto name = fmt::format("tess_{}_{}", sane ? "sane" : "garbage", n);
        if (std::ofstream f{dir / (name + "_ib.bin"), std::ios::binary}) {
            f.write(reinterpret_cast<const char*>(before.data()), before.size() * sizeof(u32));
        }
        std::vector<u8> around(64_KB);
        const VAddr from = base - 32_KB;
        Common::CopyGuest(around.data(), reinterpret_cast<const void*>(from), around.size());
        if (std::ofstream f{dir / (name + "_mem.bin"), std::ios::binary}) {
            f.write(reinterpret_cast<const char*>(around.data()), around.size());
        }
        LOG_WARNING(Render_Vulkan, "DIAG-064: {} draw {}: indirect buffer {:#x} ({} dwords) and "
                                   "memory from {:#x} saved as {}_*.bin",
                    sane ? "sane" : "garbage", n, reinterpret_cast<VAddr>(ib), dwords, from,
                    name);
    }
    std::thread([n, sane, base, ib, dwords, before = std::move(before), bases_before] {
        std::string timeline;
        for (const u32 ms : {1u, 3u, 10u, 30u}) {
            std::this_thread::sleep_until(std::chrono::steady_clock::now() +
                                          std::chrono::milliseconds(ms));
            std::vector<u32> now(dwords);
            Common::CopyGuest(now.data(), ib, now.size() * sizeof(u32));
            size_t first_diff = dwords;
            for (size_t i = 0; i < dwords; ++i) {
                if (now[i] != before[i]) {
                    first_diff = i;
                    break;
                }
            }
            std::array<u32, 7> c{};
            Common::CopyGuest(c.data(), reinterpret_cast<const void*>(base), sizeof(c));
            const bool constants_sane =
                c[2] != 0 && std::ranges::all_of(c, [](u32 x) { return x <= 64_KB; });
            timeline += fmt::format(" | +{} ms: ib {}, constants {}", ms,
                                    first_diff == dwords
                                        ? std::string("same")
                                        : fmt::format("changed from dword {}", first_diff),
                                    constants_sane ? "sane" : "garbage");
        }
        const auto bases_after = TessBasesInIb(ib, dwords);
        const auto list = [](const std::vector<VAddr>& v) {
            std::string out;
            for (size_t i = 0; i < std::min<size_t>(v.size(), 8); ++i) {
                out += fmt::format(" {:#x}", v[i]);
            }
            return out + (v.size() > 8 ? fmt::format(" (+{})", v.size() - 8) : "");
        };
        LOG_WARNING(Render_Vulkan,
                    "DIAG-063: {} draw {} (constants {:#x}) in indirect buffer {:#x} ({} "
                    "dwords){}; constant V#s in it at decoding:{}; 30 ms later:{}",
                    sane ? "sane" : "garbage", n, base, reinterpret_cast<VAddr>(ib), dwords,
                    timeline, list(bases_before), list(bases_after));
    }).detach();
}

static bool SaneTessConstants(const Shader::Info& hull, const AmdGpu::Liverpool* liverpool,
                              bool read_ahead) {
    static const bool check = Common::PerfFeatureEnabled(82);
    if (!check || hull.tess_consts_dword_offset < 0) {
        return true;
    }
    const auto buf = hull.ReadUdReg<AmdGpu::Buffer>(static_cast<u32>(hull.tess_consts_ptr_base),
                                                    static_cast<u32>(hull.tess_consts_dword_offset));
    const VAddr base = buf.base_address;
    Shader::TessellationDataConstantBuffer c{};
#ifdef RYTY_GPU
    if (base == 0 || !Core::RytyMemory::IsGuestReadable(base, sizeof(c))) {
        return false;
    }
#endif
    constexpr u32 Lds = 64_KB;
    const auto is_sane = [&] {
        return c.ls_stride <= Lds && c.hs_cp_stride <= Lds && c.num_patches != 0 &&
               c.num_patches <= Lds && c.hs_output_base <= Lds && c.patch_const_size <= Lds &&
               c.patch_const_base <= Lds && c.patch_output_size <= Lds;
    };
    Common::CopyGuest(&c, reinterpret_cast<const void*>(base), sizeof(c));
    const bool sane = is_sane();
    {
        // DIAG-060: sane and garbage constants per hull program, every 5 s.
        static std::mutex stats_mutex;
        static std::unordered_map<u64, std::pair<u32, u32>> per_hull;
        static u32 garbage_drawn{};
        static u32 garbage_read_ahead{};
        if (!sane) {
            ++(read_ahead ? garbage_read_ahead : garbage_drawn);
        }
        static auto last = std::chrono::steady_clock::now();
        std::scoped_lock lk{stats_mutex};
        auto& [good, bad] = per_hull[hull.pgm_hash];
        ++(sane ? good : bad);
        if (const auto now = std::chrono::steady_clock::now(); now - last >= std::chrono::seconds{5}) {
            last = now;
            std::string line;
            for (const auto& [hash, counts] : per_hull) {
                if (counts.second != 0) {
                    line += fmt::format(" {:#x}={}/{}", hash, counts.first, counts.second);
                }
            }
            if (!line.empty()) {
                LOG_WARNING(Render_Vulkan,
                            "DIAG-060: hulls with garbage constants (sane/garbage):{}; decoding "
                            "submission {} of {}; garbage in {} draws, {} read-ahead states",
                            line, liverpool ? liverpool->DecodingSubmit() : 0,
                            liverpool ? liverpool->SubmittedCount() : 0, garbage_drawn,
                            garbage_read_ahead);
                garbage_drawn = 0;
                garbage_read_ahead = 0;
            }
            per_hull.clear();
        }
    }
#ifdef RYTY_GPU
    if (sane && !read_ahead) {
        // DIAG-067: who writes these constants next, and when (SHADGT_WATCH_TESS=<ms>).
        static const u32 watch_ms = [] {
            const char* env = std::getenv("SHADGT_WATCH_TESS");
            return env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0u;
        }();
        static auto next_watch = std::chrono::steady_clock::now();
        if (watch_ms != 0 && std::chrono::steady_clock::now() >= next_watch) {
            next_watch = std::chrono::steady_clock::now() + std::chrono::milliseconds{watch_ms + 50};
            RytyWatchWrites(base, watch_ms,
                            fmt::format("tessellation constants of hs {:#x} decoded in submission "
                                        "{}; {}",
                                        hull.pgm_hash, liverpool ? liverpool->DecodingSubmit() : 0,
                                        AmdGpu::Liverpool::DescribeEops()));
        }
    }
#endif
    if (sane) {
        // DIAG-059: what sane constants and their V#s look like, to compare with garbage ones.
        static std::atomic<u32> sane_logged{};
        if (const u32 n = ++sane_logged; n <= 40 || n % 5000 == 0) {
            if (n % 5000 == 0 && std::getenv("SHADGT_TESS_SUBMIT_CHECK")) {
                WatchIbAfterDecode(n, true, base);
            }
            const auto dump = liverpool ? liverpool->FindRecentConstDump(base, sizeof(c))
                                        : std::nullopt;
            const auto rec = AmdGpu::Liverpool::FindTessSubmit(base);
            LOG_WARNING(Render_Vulkan, "DIAG-059: sane constants {}: {}; DIAG-061: {} at "
                                       "submission {}, decoding {} of {}; DIAG-066: {}",
                        n,
                        dump ? fmt::format("CE dump {} of {}", dump->sequence,
                                           liverpool->ConstDumpSequence())
                             : std::string("not from a recent CE dump"),
                        rec ? (rec->sane ? "sane" : "garbage") : "unseen", rec ? rec->submit : 0,
                        liverpool ? liverpool->DecodingSubmit() : 0,
                        liverpool ? liverpool->SubmittedCount() : 0,
                        AmdGpu::Liverpool::DescribeEops());
            LOG_WARNING(Render_Vulkan,
                        "DIAG-059: sane tessellation constants {} of hs {:#x} at {:#x}: ls stride "
                        "{:#x}, cp stride {:#x}, patches {:#x}, output base {:#x}, patch output "
                        "size {:#x}; V# {:08x} {:08x} {:08x} {:08x}",
                        n, hull.pgm_hash, base, c.ls_stride, c.hs_cp_stride, c.num_patches,
                        c.hs_output_base, c.patch_output_size,
                        reinterpret_cast<const u32*>(&buf)[0], reinterpret_cast<const u32*>(&buf)[1],
                        reinterpret_cast<const u32*>(&buf)[2], reinterpret_cast<const u32*>(&buf)[3]);
        }
    }
    if (!sane) {
        static std::atomic<u32> logged{};
        if (const u32 n = ++logged; n <= 60 || n % 100 == 0) {
            if (n <= 40 && std::getenv("SHADGT_TESS_SUBMIT_CHECK")) {
                WatchIbAfterDecode(n, false, base);
            }
            auto* rasterizer = Core::Memory::Instance()->GetRasterizer();
            const bool gpu_written =
                rasterizer && rasterizer->GetBufferCache().IsRegionGpuModified(base, sizeof(c));
            // Where the V# itself was read: a table in memory the user data points to, or the
            // user data registers.
            VAddr table = 0;
            if (u32(hull.tess_consts_ptr_base) != Shader::IR::NumScalarRegs &&
                u32(hull.tess_consts_ptr_base) + 1 < hull.user_data.size()) {
                std::memcpy(&table, &hull.user_data[u32(hull.tess_consts_ptr_base)], sizeof(table));
                table &= 0xFFFFFFFFFFFFULL;
                table += u64(hull.tess_consts_dword_offset) * 4;
            }
            const bool table_gpu_written =
                rasterizer && table != 0 &&
                rasterizer->GetBufferCache().IsRegionGpuModified(table, 16);
            const auto dump = liverpool ? liverpool->FindRecentConstDump(base, sizeof(c))
                                        : std::nullopt;
            LOG_WARNING(Render_Vulkan,
                        "FIX-058: constants of draw {}: {}", n,
                        dump ? fmt::format("CE dump {} of {} (ce {} de {}) at {:#x}+{:#x}",
                                           dump->sequence, liverpool->ConstDumpSequence(),
                                           dump->ce_count, dump->de_count, dump->address,
                                           dump->size)
                             : std::string("not from a recent CE dump"));
            if (liverpool) {
                // DIAG-061: the same constants when the game submitted the draw.
                const s64 now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                                    std::chrono::steady_clock::now().time_since_epoch())
                                    .count();
                const auto cmd = liverpool->FindRecentCmdBuffer(base, sizeof(c));
                const auto rec = AmdGpu::Liverpool::FindTessSubmit(base);
                LOG_WARNING(Render_Vulkan,
                            "DIAG-061: constants of draw {} {}; decoding submission {} of {}; "
                            "inside a recent command buffer: {}; DIAG-062: its indirect buffer "
                            "changed after submission: {}",
                            n,
                            rec ? fmt::format("were {} at submission {} ({:.1f} ms ago, ls stride "
                                              "{:#x}{})",
                                              rec->sane ? "sane" : "garbage", rec->submit,
                                              (now - rec->time_ns) / 1e6, rec->ls_stride,
                                              rec->in_cmd_buffer ? ", in its command buffer" : "")
                                : std::string("were not seen at any submission"),
                            liverpool->DecodingSubmit(), liverpool->SubmittedCount(),
                            cmd ? fmt::format("yes ({} of {})", cmd->sequence,
                                              liverpool->CmdBufferSequence())
                                : std::string("no"),
                            AmdGpu::Liverpool::CurrentIbChangedSinceSubmit());
            }
            // DIAG-065: the renderer's own recent writes over the constants (readbacks, fault
            // flushes, CP writes), newest last.
            LOG_WARNING(Render_Vulkan, "DIAG-065: emulator writes over the constants of draw {}:{}",
                        n, Core::Memory::Instance()->DescribeBackingWrites(base, 0));
            LOG_WARNING(Render_Vulkan, "DIAG-066: draw {} decoded in submission {} of {}; {}", n,
                        liverpool ? liverpool->DecodingSubmit() : 0,
                        liverpool ? liverpool->SubmittedCount() : 0,
                        AmdGpu::Liverpool::DescribeEops());
            LOG_WARNING(Render_Vulkan,
                        "FIX-058: V# of draw {} read from {} {:#x} (GPU-written: {}); user data "
                        "regs {} (ptr base {}, dword {})",
                        n, table ? "memory" : "user data", table, table_gpu_written,
                        hull.user_data.size(), u32(hull.tess_consts_ptr_base),
                        hull.tess_consts_dword_offset);
            LOG_WARNING(Render_Vulkan,
                        "FIX-058: skipping tessellated {} {} of hs {:#x}: constants at {:#x} "
                        "are garbage (ls stride {:#x}, cp stride {:#x}, patches {:#x}, output "
                        "base {:#x}, patch output size {:#x}); GPU-written: {}; V# {:08x} "
                        "{:08x} {:08x} {:08x}",
                        read_ahead ? "read-ahead state" : "draw", n, hull.pgm_hash, base,
                        c.ls_stride, c.hs_cp_stride, c.num_patches,
                        c.hs_output_base, c.patch_output_size, gpu_written,
                        reinterpret_cast<const u32*>(&buf)[0], reinterpret_cast<const u32*>(&buf)[1],
                        reinterpret_cast<const u32*>(&buf)[2], reinterpret_cast<const u32*>(&buf)[3]);
        }
    }
    return sane;
}

bool PipelineCache::SkipTranslationWait(SwStage sw_stage, const Shader::ShaderParams& params) {
    // SHD-005: as a race's draws are skipped while their new pipeline builds (PERF-076), a draw
    // whose new shader still translates on a worker is skipped too, instead of the command thread
    // waiting for it (GT Sport's first race on a track stopped for 100-300 ms whenever a few
    // dozen new shaders came at once). Not compute, and not a shader that writes memory (later
    // work reads what it writes). -DisablePerf 171 waits as before.
    static const bool enabled = Common::PerfFeatureEnabled(171);
    if (!enabled || force_wait || wait_for_shaders || regs_override != nullptr ||
        sw_stage == SwStage::Compute || !AsyncTranslation() || !AsyncPipelines() || MenusWait()) {
        return false;
    }
    auto [it, inserted] = code_writes_memory.try_emplace(params.hash, false);
    if (inserted) {
        it.value() = Shader::HasMemoryWrites(params.code);
    }
    return !it->second;
}

bool PipelineCache::AsyncTranslation() const {
    // FIX-078: -DisablePerf 102 translates on the command thread as before. Dumped, collected or
    // patched shaders need their permutation index while they are translated.
    static const bool enabled = Common::PerfFeatureEnabled(102) &&
                                !EmulatorSettings.IsDumpShaders() &&
                                !EmulatorSettings.IsShaderCollect() &&
                                !EmulatorSettings.IsPatchShaders();
    return enabled && translate_workers != nullptr;
}

// FIX-189: a vertex shader's fetch shader, copied for a translation worker as FIX-160 copies the
// shader's code. The worker read it through the address in the user data, and the game can
// release it first: Brand Central's showroom swaps car models while their shaders wait in the
// queue, and the read crashed the worker (ParseFetchShader). The job's user data then points at
// the copy. False when it is gone already.
static bool CopyFetchShader(std::span<u32, Shader::ShaderParams::NumShaderUserData> user_data,
                            std::span<const u32> code, std::vector<u32>& copy) {
    const auto base = Shader::FetchShaderSgprBase(code);
    if (!base || *base + 1 >= user_data.size()) {
        return true;
    }
    u64 address;
    std::memcpy(&address, &user_data[*base], sizeof(address));
    if (address == 0) {
        return true;
    }
    // The page holding its start, then more pages until its s_setpc_b64 is in the copy; two
    // zero dwords after the copied words keep the decoder's literal reads inside the vector.
    constexpr u64 PageSize = 4096;
    constexpr size_t Padding = 2;
    std::vector<u32> words;
    u64 end = (address & ~(PageSize - 1)) + PageSize;
    for (int pages = 0; pages < 4; ++pages, end += PageSize) {
        const size_t count = (end - address) / sizeof(u32);
        words.resize(count + Padding);
#ifdef RYTY_GPU
        if (!Core::RytyMemory::GuardedCopy(words.data(), reinterpret_cast<const void*>(address),
                                           count * sizeof(u32))) {
            return false;
        }
#else
        std::memcpy(words.data(), reinterpret_cast<const void*>(address), count * sizeof(u32));
#endif
        std::fill(words.begin() + count, words.end(), 0u);
        if (const auto length = Shader::FetchShaderLength({words.data(), count})) {
            words.resize(*length + Padding);
            copy = std::move(words);
            const u64 pointer = reinterpret_cast<u64>(copy.data());
            std::memcpy(&user_data[*base], &pointer, sizeof(pointer));
            return true;
        }
    }
    return false;
}

// FIX-GT7-054: a vertex shader's attribute V#s, as the draw has them when its translation is
// queued. They were read through the user data three times: for the code, for its SPIR-V, and for
// its key when the job was harvested, by when the game had written that memory again. GT7's map
// markers got code made from an empty V# (every attribute 0) under the key of real ones
// (R, G, B, A), so every car's triangle landed on one point; the translation store then kept it for
// every later start. RYGT_PIN_FETCH_SHARPS=0: as before.
static std::vector<Shader::Info::PinnedFetchSharp> PinFetchSharps(
    HwStage hw_stage, SwStage sw_stage, std::span<const u32> code,
    std::span<const u32, Shader::ShaderParams::NumShaderUserData> user_data, u64 hash) {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_PIN_FETCH_SHARPS");
        return !(env && env[0] == '0');
    }();
    const auto base = Shader::FetchShaderSgprBase(code);
    if (!enabled || !base) {
        return {};
    }
    const Shader::ShaderParams params{.user_data = user_data, .code = code, .hash = hash};
    const auto reader = std::make_unique<Shader::Info>(hw_stage, sw_stage, params);
    reader->has_fetch_shader = true;
    reader->fetch_shader_sgpr_base = *base;
    Shader::Gcn::FetchShaderData fetch;
    if (!Shader::Gcn::ParseFetchShader(*reader, fetch)) {
        return {};
    }
    std::vector<Shader::Info::PinnedFetchSharp> sharps;
    sharps.reserve(fetch.attributes.size());
    for (const auto& attrib : fetch.attributes) {
        sharps.push_back({attrib.sgpr_base, attrib.dword_offset,
                          reader->ReadUdReg<AmdGpu::Buffer>(attrib.sgpr_base, attrib.dword_offset)});
    }
    return sharps;
}

bool PipelineCache::QueueTranslation(HwStage hw_stage, SwStage sw_stage,
                                     const Shader::ShaderParams& params,
                                     const Shader::RuntimeInfo& runtime_info,
                                     const Shader::Backend::Bindings& start) {
    auto& jobs = translate_jobs[params.hash];
    for (const auto& job : jobs) {
        if (job->hw_stage == hw_stage && job->runtime_info == runtime_info && job->start == start) {
            ++translate_stats.duplicate;
            return false;
        }
    }
    auto job = std::make_shared<TranslateJob>(runtime_info);
    job->hw_stage = hw_stage;
    job->sw_stage = sw_stage;
    job->hash = params.hash;
    std::ranges::copy(params.user_data, job->user_data.begin());
    // FIX-160: the worker translates its own copy of the code. The game can release that memory
    // before a worker gets to it, and the read then crashed the worker (under Proton). Translation
    // does not use the code's address (pgm_base is set again when the result is taken).
    job->code_copy.resize(params.code.size());
#ifdef RYTY_GPU
    if (!Core::RytyMemory::GuardedCopy(job->code_copy.data(), params.code.data(),
                                       params.code.size_bytes())) {
        return false; // gone already; translated on the command thread if a draw needs it
    }
#else
    std::ranges::copy(params.code, job->code_copy.begin());
#endif
    job->code = job->code_copy;
    if ((hw_stage == HwStage::Vertex || hw_stage == HwStage::Export ||
         hw_stage == HwStage::Local) &&
        !CopyFetchShader(job->user_data, job->code, job->fetch_copy)) {
        return false; // as above
    }
    if (!job->fetch_copy.empty()) {
        job->fetch_sharps = PinFetchSharps(hw_stage, sw_stage, job->code, job->user_data, job->hash);
    }
    job->start = start;
    static std::atomic<u32> diag_next_id{};
    job->diag_id = ++diag_next_id;
    job->diag_ahead = regs_override != nullptr;
    job->queued_at = std::chrono::steady_clock::now();
    jobs.push_back(job);
    ++translate_stats.queued;
    translate_workers->Queue(std::move(job));
    return true;
}

bool PipelineCache::TakeReadyTranslation(HwStage hw_stage, SwStage sw_stage, u64 hash,
                                         Program& program,
                                         const Shader::ShaderParams& params,
                                         const Shader::RuntimeInfo& runtime_info,
                                         const Shader::Backend::Bindings& binding, bool insert) {
    const auto it = ready_translations.find(hash);
    if (it == ready_translations.end()) {
        return false;
    }
    auto& ready = it.value();
    for (auto job_it = ready.begin(); job_it != ready.end(); ++job_it) {
        auto& job = *job_it;
        // Matched the way GetProgram matches a permutation: its specialization for this lookup.
        auto& info = *job->info;
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        { Common::PhaseTimer flat_timer{Common::Phase::SelFlatten}; info.RefreshFlatBuf(); }
        // FIX-GT7-024: the lookup's on the right, as GetProgram compares (its bitsets decide).
        if (job->hw_stage != hw_stage || job->sw_stage != sw_stage ||
            job->spec != Shader::StageSpecialization(info, runtime_info, profile, binding)) {
            if (TranslateJobsDiag()) {
                static std::atomic<u32> logged{};
                if (job->hw_stage == hw_stage && job->sw_stage == sw_stage && ++logged <= 3000) {
                    u32 index{};
                    const auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
                    const char* reason = job->spec.FirstDifference(spec, index);
                    std::string fields;
                    if (reason && std::string_view{reason} == "buffer" && index < spec.buffers.size() &&
                        index < job->spec.buffers.size() && index < info.buffers.size()) {
                        const auto& a = job->spec.buffers[index];
                        const auto& c = spec.buffers[index];
                        const auto& desc = info.buffers[index];
                        fields = fmt::format(
                            " program {:#x} base {:#x} type {} stride used {}: job stride {} fmt {} {}/{} swz {} tid {} coh {} ; this stride {} fmt {} {}/{} swz {} tid {} coh {}",
                            hash, u64(desc.GetSharp(info).base_address), u32(desc.buffer_type), bool(info.buffer_stride_used[index]),
                            u32(a.stride), u32(a.is_formatted), u32(a.data_format), u32(a.num_format), u32(a.swizzle_enable), u32(a.add_tid_enable), u32(a.coherent),
                            u32(c.stride), u32(c.is_formatted), u32(c.data_format), u32(c.num_format), u32(c.swizzle_enable), u32(c.add_tid_enable), u32(c.coherent));
                    }
                    LOG_WARNING(Render_Vulkan,
                                "DIAG-GT7-073: #{} not for this {}: {}[{}]{}", job->diag_id,
                                regs_override != nullptr ? "read-ahead" : "draw",
                                reason ? reason : "none", index, fields);
                }
            }
            continue;
        }
        if (!insert) {
            return true;
        }
        if (TranslateJobsDiag()) {
            LOG_WARNING(Render_Vulkan, "DIAG-GT7-073: #{} taken by a {}", job->diag_id,
                        regs_override != nullptr ? "read-ahead" : "draw");
        }
        LogNewPermutation(program, hw_stage, params, runtime_info, binding); // DIAG-GT7-013
        size_t perm_idx = program.modules.size();
        if (const auto end = stored_perm_end.find(hash); end != stored_perm_end.end()) {
            perm_idx = std::max(perm_idx, end->second);
        }
        const u64 perm_hash = HashCombine(hash, perm_idx);
        RegisterShaderMeta(info, job->spec.fetch_shader_data, job->spec, perm_hash, perm_idx);
        const u64 spv_hash = XXH3_64bits(job->spv.data(), job->spv.size() * sizeof(u32));
        Vulkan::SetObjectName(instance.GetDevice(), job->module,
                              GetShaderName(job->hw_stage, hash, perm_idx));
        program.InsertPermut(job->module, std::move(job->spec), std::move(job->info), perm_idx);
        program.modules[perm_idx].spv_hash = spv_hash;
        RegisterShaderBinary(std::move(job->spv), hash, perm_idx);
        ++Common::GetWorkCounters().shaders_compiled;
        ++translate_stats.used;
        ready.erase(job_it);
        if (ready.empty()) {
            ready_translations.erase(it);
        }
        return true;
    }
    return false;
}

void PipelineCache::NoteUnlockedShaderWait(u64 hash) {
    const auto it = translate_jobs.find(hash);
    if (it == translate_jobs.end()) {
        return;
    }
    for (const auto& job : it.value()) {
        if (!job->done.load() &&
            std::ranges::find(skipped_translations, job) == skipped_translations.end()) {
            skipped_translations.push_back(job);
        }
    }
}

std::function<void()> PipelineCache::TakeShaderWait() {
    if (skipped_translations.empty()) {
        return {};
    }
    return [this, jobs = std::move(skipped_translations)] {
        // As HarvestTranslations: the game never sees a PS4 wait for a shader.
        Common::GuestClock::CompilePause hold;
        const auto start = std::chrono::steady_clock::now();
        for (const auto& job : jobs) {
            RYDIAG_SCOPE(ShaderCompile, job->hash);
            std::unique_lock lk{job->mutex};
            job->cv.wait(lk, [&] { return job->done.load(); });
        }
        unlocked_stats.shader_waits.fetch_add(1, std::memory_order_relaxed);
        unlocked_stats.shader_wait_us.fetch_add(
            u64(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now() - start)
                    .count()),
            std::memory_order_relaxed);
    };
}

void PipelineCache::HarvestTranslations(u64 hash, bool wait) {
    const auto it = translate_jobs.find(hash);
    if (it == translate_jobs.end()) {
        return;
    }
    auto& jobs = it.value();
    if (wait && shaders_unlocked) {
        NoteUnlockedShaderWait(hash); // PERF-GT7-036: waited for without the lookup lock
        wait = false;
    }
    if (wait) {
        const auto wait_start = std::chrono::steady_clock::now();
        bool waited = false;
        for (const auto& job : jobs) {
            if (!job->done) {
                // As CompileModule: the game never sees a PS4 wait for a shader.
                Common::GuestClock::CompilePause hold;
                RYDIAG_SCOPE(ShaderCompile, hash);
                std::unique_lock lk{job->mutex};
                job->cv.wait(lk, [&] { return job->done.load(); });
                waited = true;
            }
        }
        if (waited) {
            ++translate_stats.waited;
            translate_stats.wait_ms += std::chrono::duration<double, std::milli>(
                                           std::chrono::steady_clock::now() - wait_start)
                                           .count();
        }
    }
    auto& ready = ready_translations[hash];
    const auto now = std::chrono::steady_clock::now();
    std::erase_if(jobs, [&](const std::shared_ptr<TranslateJob>& job) {
        if (!job->done) {
            return false;
        }
        if (job->too_old) {
            ++translate_stats.too_old; // FIX-GT7-052
            if (TranslateJobsDiag()) {
                LOG_WARNING(Render_Vulkan, "DIAG-GT7-073: #{} too old", job->diag_id);
            }
            return true;
        }
        if (!job->info || !job->module) {
            ++translate_stats.failed;
            return true;
        }
        translate_stats.translate_ms += job->translate_ms;
        job->spec = Shader::StageSpecialization(*job->info, job->runtime_info, profile, job->start);
        // FIX-GT7-054: the key is made; from here this Info is matched against other draws.
        const bool pinned = !job->info->pinned_fetch_sharps.empty();
        job->info->pinned_fetch_sharps.clear();
        if (TranslateJobsDiag() && pinned &&
            Shader::StageSpecialization(*job->info, job->runtime_info, profile, job->start)
                    .vs_attribs != job->spec.vs_attribs) {
            LOG_WARNING(Render_Vulkan,
                        "DIAG-GT7-073: #{} {:#x}: its attribute V#s changed after its draw, its "
                        "code would have had another key without FIX-GT7-054",
                        job->diag_id, job->hash);
        }
        job->ready_since = now;
        ready.push_back(job);
        return true;
    });
    if (jobs.empty()) {
        translate_jobs.erase(it);
    }
    // Translations no draw took within 30 s came from states that were not drawn.
    std::erase_if(ready, [&](const std::shared_ptr<TranslateJob>& job) {
        if (now - job->ready_since < std::chrono::seconds{30} && ready.size() <= 32) {
            return false;
        }
        if (TranslateJobsDiag()) {
            LOG_WARNING(Render_Vulkan, "DIAG-GT7-073: #{} expired", job->diag_id);
        }
        instance.GetDevice().destroyShaderModule(job->module);
        ++translate_stats.expired;
        return true;
    });
    auto& stats = translate_stats;
    if (now - stats.since >= std::chrono::seconds{2}) {
        if (stats.queued + stats.used != 0) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-078 shader translation in {:.1f} s: {} queued, {} used by draws "
                        "({:.1f} ms of translation done off this thread), {} never drawn, {} "
                        "duplicates, {} failed, FIX-GT7-052: {} too old; {} draws waited for "
                        "one ({:.1f} ms)",
                        std::chrono::duration<double>(now - stats.since).count(), stats.queued,
                        stats.used, stats.translate_ms, stats.expired, stats.duplicate,
                        stats.failed, stats.too_old, stats.waited, stats.wait_ms);
        }
        stats = {};
        stats.since = now;
    }
}

std::optional<PipelineCache::Result> PipelineCache::TryLegacyProgram(
    HwStage hw_stage, const Shader::ShaderParams& params, const Shader::RuntimeInfo& runtime_info,
    Shader::Backend::Bindings& binding) {
    auto& store = Storage::DataBase::Legacy();
    auto [it, is_new] = legacy_programs.try_emplace(params.hash);
    if (is_new) {
        it.value() = std::make_unique<Program>();
        // Like ApplyStoredListing, without its bookkeeping of this store's indices.
        auto listing = ReadStoredListing(params.hash, store);
        for (auto& entry : listing.entries) {
            if (!entry.has_spv) {
                continue;
            }
            auto info = std::make_unique<Shader::Info>();
            Shader::StageSpecialization spec{};
            spec.info = info.get();
            size_t stored_idx{};
            Serialization::Archive ar{std::move(entry.meta)};
            bool valid = false;
            try {
                Common::RecoverableScope recoverable;
                valid = ar.SizeBytes() >= 24 && LoadShaderMeta(ar, *info, spec, stored_idx);
            } catch (const Common::RecoverableFailure&) {
                valid = false;
            }
            if (!valid || stored_idx != entry.perm_idx || info->pgm_hash != params.hash ||
                std::ranges::any_of(info->buffers, [](const auto& buffer) {
                    return buffer.buffer_type == Shader::BufferType::DirectTable;
                })) {
                continue;
            }
            info->uses_direct = false; // the byte was padding in old metas
            it.value()->InsertPermut({}, std::move(spec), std::move(info), entry.perm_idx);
            it.value()->modules[entry.perm_idx].stored = true;
        }
        ++legacy_stats.programs;
    }
    auto& program = *it.value();
    for (size_t perm_idx = 0; perm_idx < program.modules.size(); ++perm_idx) {
        auto& permutation = program.modules[perm_idx];
        if (!permutation.info) {
            continue;
        }
        auto& info = *permutation.info;
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        info.RefreshFlatBuf();
        const auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
        if (permutation.spec != spec) {
            static std::atomic<u32> logged{};
            if (logged++ < 40) {
                u32 index{};
                const char* reason = permutation.spec.FirstDifference(spec, index);
                LOG_WARNING(Render_Vulkan, "D10: old translation {} of {}_{:#x} differs: {}[{}]",
                            perm_idx, hw_stage, params.hash, reason ? reason : "none", index);
            }
            continue;
        }
        if (!info.attribute_flags_known && IsTessEmulatedDraw()) {
            continue; // FIX-012 would store it again: left to the new translation
        }
        if (permutation.stored) {
            permutation.stored = false;
            permutation.module = LoadStoredModule(params.hash, perm_idx, store);
            if (!permutation.module) {
                permutation.info.reset();
                continue;
            }
        }
        info.AddBindings(binding);
        if (auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
            S().fetch_shader = &fetch;
        }
        ++legacy_stats.draws;
        ReportLegacy();
        // Its own pipeline keys: the module hashes of old translations differ from new ones.
        constexpr u64 LegacyTag = 0x4c45474143590000ULL;
        return std::make_tuple(&info, permutation.module,
                               HashCombine(params.hash, perm_idx) ^ LegacyTag);
    }
    return std::nullopt;
}

void PipelineCache::ReportLegacy() {
    const auto now = std::chrono::steady_clock::now();
    if (now - legacy_stats.since < std::chrono::seconds{2}) {
        return;
    }
    LOG_WARNING(Render_Vulkan,
                "D10: old-renderer shaders in 2 s: {} stage lookups used one, {} new translations "
                "queued for them, {} pipelines of old shaders, {} misses without one ({} programs "
                "read from the old store); pipelines not built yet: {} drawn with old shaders, "
                "{} old ones not built either, {} without old shaders",
                legacy_stats.draws, legacy_stats.queued, legacy_stats.pipelines,
                legacy_stats.no_legacy, legacy_stats.programs, legacy_stats.fallback_ok,
                legacy_stats.fallback_unbuilt, legacy_stats.fallback_none);
    const u64 programs = legacy_stats.programs;
    legacy_stats = {};
    legacy_stats.programs = programs;
    legacy_stats.since = now;
}

PipelineCache::Result PipelineCache::GetProgramForJob(HwStage hw_stage,
                                                      const Shader::ShaderParams& params,
                                                      const Shader::RuntimeInfo& runtime_info,
                                                      Shader::Backend::Bindings& binding) {
    // NAT-031: GetProgram for a game job thread (shared lock): only the programs and
    // permutations the threads above made are matched, in this thread's own copies of their
    // infos; nothing is created, loaded, harvested or remembered.
    const auto none = [] { return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0}); };
    const auto it_pgm = program_cache.find(params.hash);
    if (it_pgm == program_cache.end() || !else_scope_checked.contains(params.hash)) {
        S().lookup_miss = "program not used yet";
        return none();
    }
    const Program& program = *it_pgm->second;
    bool tess_checked = false;
    // The permutation the last search found first (R7), then all of them in order.
    const size_t first = program.last_match;
    for (size_t step = 0; step <= program.modules.size(); ++step) {
        const size_t perm_idx = step == 0 ? first : step - 1;
        if (perm_idx >= program.modules.size() || (step != 0 && perm_idx == first)) {
            continue;
        }
        const auto& permutation = program.modules[perm_idx];
        if (!permutation.info) {
            continue;
        }
        auto& info = MatchInfo(*permutation.info);
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        info.RefreshFlatBuf();
        if (hw_stage == HwStage::Hull && !tess_checked) {
            tess_checked = true; // FIX-058, as GetProgram
            if (!SaneTessConstants(info, liverpool, false)) {
                S().lookup_miss = "hull constants";
                return none();
            }
        }
        if (permutation.spec != Shader::StageSpecialization(info, runtime_info, profile, binding)) {
            continue;
        }
        if (permutation.stored) {
            S().lookup_miss = "stored translation not loaded";
            return none();
        }
        if (!info.attribute_flags_known && IsTessEmulatedDraw()) {
            S().lookup_miss = "rect/quad list attribute flags";
            return none();
        }
        info.AddBindings(binding);
        if (const auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
            S().fetch_shader = const_cast<Shader::Gcn::FetchShaderData*>(&fetch);
        }
        return std::make_tuple(&info, permutation.module, HashCombine(params.hash, perm_idx));
    }
    S().lookup_miss = hw_stage == HwStage::Fragment ? "no fragment translation for these inputs"
                      : hw_stage == HwStage::Vertex ? "no vertex translation for these inputs"
                                                     : "no translation for these inputs";
    return none();
}

PipelineCache::Result PipelineCache::GetProgram(HwStage hw_stage, SwStage sw_stage,
                                                const Shader::ShaderParams& params,
                                                Shader::Backend::Bindings& binding) {
    // SHD-002: a stage's bindings start where its own descriptor set does, whatever the stages
    // before it bound.
    binding = Shader::Backend::StageStart(u32(sw_stage));
    auto runtime_info = BuildRuntimeInfo(hw_stage, sw_stage);
    if (t_lookup_only) {
        return GetProgramForJob(hw_stage, params, runtime_info, binding); // NAT-031
    }
    // PERF-GT7-019: a new program's stored permutations are read on a store reader. A read-ahead
    // state, or a draw that may be skipped while its shaders are made (SHD-005, or FIX-078's
    // first pass, which queues all of a draw's stages), goes on without them for now.
    if (store_readers && !program_cache.contains(params.hash)) {
        const bool read_ahead = regs_override != nullptr;
        const bool skip = SkipTranslationWait(sw_stage, params);
        if (!StoredListingReady(params.hash, read_ahead || defer_misses || skip)) {
            if (!read_ahead) {
                deferred_miss = true;
                translation_skipped |= skip;
            } else {
                read_ahead_blocked = true; // PERF-GT7-021
            }
            return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
        }
    }
    auto [it_pgm, new_program] = program_cache.try_emplace(params.hash);
    if (new_program) {
        it_pgm.value() = std::make_unique<Program>();
        if (use_stored_shaders) {
            if (const auto pending = pending_listings.find(params.hash);
                pending != pending_listings.end()) {
                ApplyStoredListing(pending->second.get(), *it_pgm.value()); // PERF-GT7-019
                pending_listings.erase(pending);
            } else {
                ListStoredPermutations(params.hash, *it_pgm.value());
            }
        }
    }
    // FIX-018: stored permutations of a program with an else after an empty if were translated
    // with the else running for every invocation. Translate those programs again, once; the new
    // permutations start after the stored ones, so their pipelines are new as well.
    const bool first_use = else_scope_checked.insert(params.hash).second;
    if (first_use && IsWatchedShader(params.hash)) {
        // DIAG-031: the guest code of watched shaders, raw and as a listing, in the log folder.
        const auto dir = Common::FS::GetUserPath(Common::FS::PathType::LogDir);
        const auto name = fmt::format("gcn_{:#x}", params.hash);
        Common::FS::IOFile{dir / (name + ".bin"), Common::FS::FileAccessMode::Create}.WriteSpan(
            params.code);
        std::ofstream{dir / (name + ".txt")} << Shader::ListGcnCode(params.code);
        LOG_WARNING(Render_Vulkan, "DIAG-031: wrote the code of {}_{:#x} ({} dwords) to {}",
                    hw_stage, params.hash, params.code.size(), (dir / (name + ".txt")).string());
    }
    // FIX-018/FIX-020: stored translations made before a translator fix that changes them.
    // PERF-GT7-027: only a program with stored translations is looked at (each look decodes the
    // whole program up to four times, on the command thread, for hundreds of new programs at a
    // new track's start); a new one is translated by the fixed translator anyway.
    const bool has_stored = !it_pgm.value()->modules.empty() ||
                            stored_perm_end.find(params.hash) != stored_perm_end.end();
    u32 needed_fixes = 0;
    if (first_use && has_stored) {
        needed_fixes |= Shader::HasEmptyScopeBeforeElse(params.code) ? 1u : 0u;
        if (Shader::HasLegacyMad(params.code)) {
            needed_fixes |= 16u; // FIX-101
            LOG_INFO(Render_Vulkan, "FIX-101: {}_{:#x} uses legacy multiply-add", hw_stage,
                     params.hash);
        }
        const auto& wg = runtime_info.hw.cs.workgroup_size;
        if (hw_stage == HwStage::Compute && profile.subgroup_size < 64 &&
            wg[0] * wg[1] * wg[2] > 32 && Common::PerfFeatureEnabled(31) &&
            Shader::HasLaneReadsAndLoop(params.code)) {
            needed_fixes |= 8u | 128u; // Wave64 invocation-guard lowering, including stored loops.
        }
        if (hw_stage == HwStage::Compute && profile.subgroup_size < 64 &&
            wg[0] * wg[1] * wg[2] == 64 && Common::PerfFeatureEnabled(31)) {
            // Wave-aligned guards can contain lane reads with or without a loop.
            needed_fixes |= 128u;
            if (Shader::HasLaneReads(params.code)) {
                needed_fixes |= 1024u; // FIX-GT7-028 (256, 512: its earlier forms)
            }
        }
        if (Shader::HasWholeWaveRegions(params.code)) {
            needed_fixes |= 32u; // FIX-190
            LOG_INFO(Render_Vulkan, "FIX-190: {}_{:#x} turns every lane on inside a branch",
                     hw_stage, params.hash);
        }
    }
    const auto known = else_scope_fixed.find(params.hash);
    const u32 applied_fixes = known != else_scope_fixed.end() ? known->second.fixes : 0u;
    if ((needed_fixes & ~applied_fixes) != 0) {
        auto& boundary = stored_perm_end[params.hash];
        boundary = std::max(boundary, it_pgm.value()->modules.size());
        if (boundary != 0) {
            LOG_WARNING(Render_Vulkan,
                        "FIX-018/020: translating {}_{:#x} again (fixes {:#x}, {} stored "
                        "permutations)",
                        hw_stage, params.hash, needed_fixes, boundary);
            else_scope_fixed[params.hash] = {boundary, applied_fixes | needed_fixes};
            SaveElseScopeFixed();
            if (!it_pgm.value()->modules.empty()) {
                // Pipelines already built keep the old modules alive.
                retired_programs.push_back(std::move(it_pgm.value()));
                it_pgm.value() = std::make_unique<Program>();
            }
        }
    }

    auto& program = it_pgm.value();
    // FIX-078: translations made on the workers become permutations; a draw (not a read-ahead)
    // first waits for any still running, as one of them is likely the permutation it needs.
    // D10: with the fallback a draw does not wait for translations in progress (the old
    // translations draw meanwhile); only when there are none (fallback_nested).
    const bool skip_wait = SkipTranslationWait(sw_stage, params); // SHD-005
    HarvestTranslations(params.hash, regs_override == nullptr && !force_legacy && !skip_wait &&
                                         (!legacy_fallback || fallback_nested));
    // D10: a key is made of new translations only, or (forced) of old ones only.
    if (force_legacy) {
        if (auto legacy = TryLegacyProgram(hw_stage, params, runtime_info, binding)) {
            S().key_uses_legacy = true;
            return *legacy;
        }
        S().legacy_failed = true;
        return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
    }
    std::optional<Common::PhaseTimer> match_timer{std::in_place, Common::Phase::ProgramMatch};
    // Diagnostic for runaway permutations: why each existing permutation was rejected.
    std::string mismatch_reasons;
    // PERF-044: replay the last search when its inputs are the same. -DisablePerf 56.
    static const bool memo_enabled = Common::PerfFeatureEnabled(56);
    auto& memo = program->match_memo;
    const bool memo_stage = memo_enabled && (hw_stage == HwStage::Fragment ||
                                             hw_stage == HwStage::Compute);
    bool memo_usable = memo_stage && memo.valid &&
                       memo.modules_size == program->modules.size() &&
                       memo.runtime_info == runtime_info && memo.start == binding &&
                       std::ranges::equal(memo.user_data, params.user_data);
    const auto start_binding = binding;
    boost::container::small_vector<std::vector<u32>, 4> flats;
    // Leaves the memo's replay: the buffers of the permutations before `end` were the memo's.
    const auto stop_replay = [&](size_t end) {
        memo_usable = false;
        flats.clear();
        for (size_t i = 0; i < end; ++i) {
            flats.push_back(i < memo.flats.size() ? memo.flats[i] : std::vector<u32>{});
        }
    };
    // R7: the permutation the last search found is tried first; a program has up to dozens and
    // each one tried builds a specialization (the command thread's largest cost per draw).
    // -DisablePerf 133 searches from the first every time.
    static const bool last_first = Common::PerfFeatureEnabled(133);
    if (last_first && !memo_usable && hw_stage != HwStage::Hull &&
        program->last_match < program->modules.size()) {
        auto& permutation = program->modules[program->last_match];
        if (permutation.info && !permutation.stored) {
            auto& info = *permutation.info;
            info.pgm_base = params.Base();
            info.user_data = params.user_data;
            { Common::PhaseTimer flat_timer{Common::Phase::SelFlatten}; info.RefreshFlatBuf(); }
            if (permutation.spec == [&] {
                    Common::PhaseTimer spec_timer{Common::Phase::SelSpec};
                    return Shader::StageSpecialization(info, runtime_info, profile, binding);
                }()) {
                if (!info.attribute_flags_known && IsTessEmulatedDraw()) {
                    RecoverAttributeFlags(info, permutation.spec, params, runtime_info,
                                          program->last_match);
                }
                info.AddBindings(binding);
                if (auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
                    S().fetch_shader = &fetch;
                }
                return std::make_tuple(&info, permutation.module,
                                       HashCombine(params.hash, program->last_match));
            }
        }
    }
    bool tess_checked = false;
    for (size_t perm_idx = 0; perm_idx < program->modules.size(); ++perm_idx) {
        auto& permutation = program->modules[perm_idx];
        if (!permutation.info) {
            if (memo_stage && !memo_usable) {
                flats.emplace_back();
            }
            continue;
        }
        auto& info = *permutation.info;
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        { Common::PhaseTimer flat_timer{Common::Phase::SelFlatten}; info.RefreshFlatBuf(); }
        if (hw_stage == HwStage::Hull && !tess_checked) {
            // FIX-058: checked with the first translation's info, before a new permutation
            // could be made from garbage constants.
            tess_checked = true;
            if (!SaneTessConstants(info, liverpool, regs_override != nullptr)) {
                return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
            }
        }
        if (memo_usable && (perm_idx > memo.perm || memo.flats[perm_idx] != info.flattened_ud_buf)) {
            // The inputs differ from here on; the search goes on the usual way.
            stop_replay(perm_idx);
        }
        if (memo_stage && !memo_usable) {
            flats.push_back(info.flattened_ud_buf);
        }
        bool matches;
        if (memo_usable) {
            matches = perm_idx == memo.perm;
        } else {
            std::optional<Common::PhaseTimer> spec_timer{std::in_place, Common::Phase::SelSpec};
            const auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
            spec_timer.reset();
            matches = permutation.spec == spec;
            if (!matches && program->modules.size() >= 4 && mismatch_reasons.size() < 600) {
                u32 index{};
                const char* reason = permutation.spec.FirstDifference(spec, index);
                mismatch_reasons +=
                    fmt::format(" {}:{}[{}]", perm_idx, reason ? reason : "none", index);
            }
        }
        // Newly active resources must match the compiled specialization; inactive ones can reuse
        // it.
        if (!matches) {
            continue;
        }
        if (permutation.stored) {
            // PERF-032: a stored translation matched; its SPIR-V is read now.
            permutation.stored = false;
            permutation.module = LoadStoredModule(params.hash, perm_idx);
            if (!permutation.module) {
                permutation.info.reset();
                if (memo_usable) {
                    stop_replay(perm_idx + 1);
                }
                memo.valid = false;
                continue;
            }
        }
        if (memo_stage && !memo_usable) {
            memo.valid = true;
            memo.perm = perm_idx;
            memo.modules_size = program->modules.size();
            memo.runtime_info = runtime_info;
            memo.start = start_binding;
            memo.user_data.assign(params.user_data.begin(), params.user_data.end());
            memo.flats.assign(flats.begin(), flats.end());
        }
        program->last_match = perm_idx; // R7
        if (!info.attribute_flags_known && IsTessEmulatedDraw()) {
            RecoverAttributeFlags(info, permutation.spec, params, runtime_info, perm_idx);
        }
        info.AddBindings(binding);
        if (auto& fetch = permutation.spec.fetch_shader_data; !fetch.Empty()) {
            S().fetch_shader = &fetch;
        }
        return std::make_tuple(&info, permutation.module, HashCombine(params.hash, perm_idx));
    }
    match_timer.reset();
    if (AsyncTranslation()) {
        // FIX-078: a translation the workers made for these inputs.
        const bool read_ahead = regs_override != nullptr;
        // PERF-GT7-021: a read-ahead state takes a ready translation as a draw would, so its
        // pipeline builds before the draw comes.
        const bool insert = !read_ahead || RetryReadAhead();
        if (TakeReadyTranslation(hw_stage, sw_stage, params.hash, *program, params, runtime_info,
                                 binding, insert)) {
            if (!insert) {
                return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
            }
            return GetProgram(hw_stage, sw_stage, params, binding);
        }
        // D10: the new translation is made on a worker; meanwhile the draw's key is made of the
        // old renderer's translations (GetGraphicsPipeline / GetComputePipeline). Without them,
        // the usual way (fallback_nested).
        if (legacy_fallback && !read_ahead && !fallback_nested) {
            if (QueueTranslation(hw_stage, sw_stage, params, runtime_info, binding)) {
                ++legacy_stats.queued;
            }
            fallback_miss = true;
            return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
        }
        // PERF-GT7-036: a one-off draw's shader is translated on a worker, never here under
        // the lookup lock; the draw waits for it without the lock.
        const bool unlocked = shaders_unlocked && !read_ahead && AsyncTranslation();
        if (read_ahead || defer_misses || skip_wait || unlocked) {
            // A shader an upcoming draw needs (or this draw, which looks again once the next
            // draws' shaders are queued too); translated on a worker while this thread reads
            // on. No key without it: the state is skipped for now. SHD-005: a stage's bindings
            // no longer follow from the stages before it, so a race queues every missing stage
            // of the draw at once and skips it.
            if (!S().stage_deferred || skip_wait || unlocked ||
                (read_ahead && RetryReadAhead())) {
                S().stage_deferred = true;
                QueueTranslation(hw_stage, sw_stage, params, runtime_info, binding);
            }
            if (unlocked) {
                NoteUnlockedShaderWait(params.hash);
            }
            if (!read_ahead) {
                deferred_miss = true;
                translation_skipped |= skip_wait;
            } else {
                read_ahead_blocked = true; // PERF-GT7-021
            }
            return std::make_tuple(nullptr, vk::ShaderModule{}, u64{0});
        }
    }

    // PERF-013: runtime permutations used to take the next free slot of this session, which
    // depends on which stored permutations were preloaded. A slot already used in the stored
    // cache then got a different specialization, overwriting the stored shader and leaving
    // every pipeline that used it unloadable ("stale"; about 1,085 per GT Sport run, compiled
    // again in every race). Start after the stored indices instead.
    size_t perm_idx = program->modules.size();
    if (const auto it = stored_perm_end.find(params.hash); it != stored_perm_end.end()) {
        perm_idx = std::max(perm_idx, it->second);
    }
    const u64 perm_hash = HashCombine(params.hash, perm_idx);
    if (!mismatch_reasons.empty()) {
        static u64 runaway_events = 0;
        if (++runaway_events <= 40 || runaway_events % 500 == 0) {
            LOG_WARNING(Render_Vulkan, "New permutation {} of {}_{:#x} (event {}); rejected:{}",
                        perm_idx, hw_stage, params.hash, runaway_events, mismatch_reasons);
        }
    }
    const auto start = binding;
    auto info = std::make_unique<Shader::Info>(hw_stage, sw_stage, params);
    // Compilation passes may adjust their runtime info (clip-distance emulation adds a fragment
    // input). The stored key must use the unmodified runtime info, or it never matches the next
    // lookup and the same module is recompiled for every draw.
    auto compile_runtime_info = runtime_info;
    const auto module = CompileModule(*info, compile_runtime_info, params.code, perm_idx, binding);
    auto spec = Shader::StageSpecialization(*info, runtime_info, profile, start);
    RegisterShaderMeta(*info, spec.fetch_shader_data, spec, perm_hash, perm_idx);
    const auto* info_ptr = info.get();
    program->InsertPermut(module, std::move(spec), std::move(info), perm_idx);
    program->modules[perm_idx].spv_hash = last_spv_hash;
    LogRepeatedPermutation(*program, perm_idx, hw_stage, params.hash);
    if (auto& fetch = program->modules[perm_idx].spec.fetch_shader_data; !fetch.Empty()) {
        S().fetch_shader = &fetch;
    }
    return std::make_tuple(info_ptr, module, perm_hash);
}

void PipelineCache::LogRepeatedPermutation(const Program& program, size_t perm_idx,
                                           Shader::HwStage hw_stage, u64 pgm_hash) {
    // FIX-043 diagnostic: a new permutation whose SPIR-V equals an existing one's was translated
    // only because their keys differ in a property its code does not read.
    const auto& added = program.modules[perm_idx];
    if (added.spv_hash == 0) {
        return;
    }
    for (size_t i = 0; i < program.modules.size(); ++i) {
        const auto& other = program.modules[i];
        if (i == perm_idx || !other.info || other.spv_hash != added.spv_hash) {
            continue;
        }
        static u64 repeats = 0;
        ++repeats;
        if (repeats <= 40 || repeats % 200 == 0) {
            u32 index{};
            const char* reason = other.spec.FirstDifference(added.spec, index);
            LOG_WARNING(Render_Vulkan,
                        "FIX-043: permutation {} of {}_{:#x} has the same SPIR-V as permutation "
                        "{} (keys differ first in {}[{}]); {} such repeats so far",
                        perm_idx, hw_stage, pgm_hash, i, reason ? reason : "none", index, repeats);
        }
        return;
    }
}

void PipelineCache::LogNewPermutation(const Program& program, Shader::HwStage hw_stage,
                                      const Shader::ShaderParams& params,
                                      const Shader::RuntimeInfo& runtime_info,
                                      const Shader::Backend::Bindings& binding) {
    // DIAG-GT7-013: why a known program gets a new permutation (each one is a pipeline compile,
    // up to a second, with its draws skipped meanwhile), when its translation is taken: for a
    // draw or for a read-ahead state, how the lookup differs from each existing permutation,
    // with the V# of a differing buffer.
    if (std::ranges::none_of(program.modules, [](const auto& m) { return bool(m.info); })) {
        return;
    }
    static u64 events = 0;
    const u64 event = ++events;
    if (event > 80 && event % 100 != 0) {
        return;
    }
    std::string reasons;
    u32 listed = 0;
    for (size_t perm_idx = 0; perm_idx < program.modules.size() && listed < 8; ++perm_idx) {
        const auto& permutation = program.modules[perm_idx];
        if (!permutation.info) {
            continue;
        }
        ++listed;
        auto& info = *permutation.info;
        info.pgm_base = params.Base();
        info.user_data = params.user_data;
        info.RefreshFlatBuf();
        const auto spec = Shader::StageSpecialization(info, runtime_info, profile, binding);
        u32 index{};
        const char* reason = permutation.spec.FirstDifference(spec, index);
        reasons += fmt::format(" {}:{}[{}]", perm_idx, reason ? reason : "none", index);
        if (reason && std::string_view{reason}.starts_with("buffer") &&
            index < info.buffers.size()) {
            const auto sharp = info.buffers[index].GetSharp(info);
            std::array<u32, 4> words{};
            std::memcpy(words.data(), &sharp, sizeof(words));
            reasons += fmt::format("(V# {:08x} {:08x} {:08x} {:08x})", words[0], words[1],
                                   words[2], words[3]);
        }
    }
    LOG_WARNING(Render_Vulkan,
                "DIAG-GT7-013: new permutation {} of {}_{:#x} for {}; existing ones differ in:{}",
                event, hw_stage, params.hash, regs_override ? "a read-ahead state" : "a draw",
                reasons);
}

bool PipelineCache::IsTessEmulatedDraw() const {
    const auto prim = Regs().primitive_type;
    return prim == AmdGpu::PrimitiveType::RectList ||
           (prim == AmdGpu::PrimitiveType::QuadList && !QuadListAsTriangles());
}

bool PipelineCache::QuadListAsTriangles() const {
    // PERF-028: the rasterizer draws quad lists as triangle lists through generated indices
    // (Rasterizer::BindQuadListIndices). Their pipeline is a plain triangle-list pipeline, and
    // the vertex shader writes the fragment shader's inputs directly.
    return Regs().primitive_type == AmdGpu::PrimitiveType::QuadList &&
           !S().draw_indirect_params.tessellate_quads && Common::PerfFeatureEnabled(28);
}

void PipelineCache::RecoverAttributeFlags(Shader::Info& info,
                                          const Shader::StageSpecialization& spec,
                                          const Shader::ShaderParams& params,
                                          const Shader::RuntimeInfo& runtime_info,
                                          size_t perm_idx) {
    // FIX-012: a shader from a cache entry stored without its attribute loads/stores is
    // translated again for them (its stored SPIR-V is kept), because the rect/quad-list helper
    // shaders forward exactly the outputs these flags name. The entry is stored again with them.
    Common::GuestClock::CompilePause hold; // FIX-044
    Shader::Info translated{info.hw_stage, info.sw_stage, params};
    auto translate_runtime_info = runtime_info;
    [[maybe_unused]] const auto program =
        Shader::TranslateProgram(params.code, pools, translated, translate_runtime_info, profile);
    info.loads = translated.loads;
    info.stores = translated.stores;
    info.attribute_flags_known = true;
    RegisterShaderMeta(info, spec.fetch_shader_data, spec, HashCombine(params.hash, perm_idx),
                       perm_idx);
}

std::optional<vk::ShaderModule> PipelineCache::ReplaceShader(vk::ShaderModule module,
                                                             std::span<const u32> spv_code) {
    std::optional<vk::ShaderModule> new_module{};
    for (const auto& [_, program] : program_cache) {
        for (auto& m : program->modules) {
            if (m.module == module) {
                const auto& d = instance.GetDevice();
                d.destroyShaderModule(m.module);
                m.module = CompileSPV(spv_code, d);
                new_module = m.module;
            }
        }
    }
    if (module_related_pipelines.contains(module)) {
        auto& pipeline_keys = module_related_pipelines[module];
        for (auto& key : pipeline_keys) {
            if (std::holds_alternative<GraphicsPipelineKey>(key)) {
                auto& graphics_key = std::get<GraphicsPipelineKey>(key);
                graphics_pipelines.erase(graphics_key);
            } else if (std::holds_alternative<ComputePipelineKey>(key)) {
                auto& compute_key = std::get<ComputePipelineKey>(key);
                compute_pipelines.erase(compute_key);
            }
        }
    }
    return new_module;
}

std::string PipelineCache::GetShaderName(Shader::HwStage stage, u64 hash,
                                         std::optional<size_t> perm) {
    if (perm) {
        return fmt::format("{}_{:#018x}_{}", stage, hash, *perm);
    }
    return fmt::format("{}_{:#018x}", stage, hash);
}

void PipelineCache::DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage,
                               size_t perm_idx, std::string_view ext) {
    if (!EmulatorSettings.IsDumpShaders()) {
        return;
    }

    using namespace Common::FS;
    const auto dump_dir = GetUserPath(PathType::ShaderDir) / "dumps";
    if (!std::filesystem::exists(dump_dir)) {
        std::filesystem::create_directories(dump_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto file = IOFile{dump_dir / filename, FileAccessMode::Create};
    file.WriteSpan(code);
}

std::optional<std::vector<u32>> PipelineCache::GetShaderPatch(u64 hash, Shader::HwStage stage,
                                                              size_t perm_idx,
                                                              std::string_view ext) {

    using namespace Common::FS;
    const auto patch_dir = GetUserPath(PathType::ShaderDir) / "patch";
    if (!std::filesystem::exists(patch_dir)) {
        std::filesystem::create_directories(patch_dir);
    }
    const auto filename = fmt::format("{}.{}", GetShaderName(stage, hash, perm_idx), ext);
    const auto filepath = patch_dir / filename;
    if (!std::filesystem::exists(filepath)) {
        return {};
    }
    const auto file = IOFile{patch_dir / filename, FileAccessMode::Read};
    std::vector<u32> code(file.GetSize() / sizeof(u32));
    file.Read(code);
    return code;
}
} // namespace Vulkan
