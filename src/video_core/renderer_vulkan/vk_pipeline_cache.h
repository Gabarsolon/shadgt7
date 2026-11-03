// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <shared_mutex>
#include <thread>

#include "common/spin_mutex.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <mutex>
#include <optional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <tsl/robin_map.h>
#include "shader_recompiler/profile.h"
#include "shader_recompiler/recompiler.h"
#include "shader_recompiler/specialization.h"
#include "video_core/renderer_vulkan/build_queue.h"
#include "video_core/renderer_vulkan/driver_cache_file.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_resource_pool.h"
#include "video_core/renderer_vulkan/vk_shader_program.h"
#include "vulkan/vulkan.hpp"

template <>
struct std::hash<vk::ShaderModule> {
    std::size_t operator()(const vk::ShaderModule& module) const noexcept {
        return std::hash<size_t>{}(reinterpret_cast<size_t>((VkShaderModule)module));
    }
};

namespace AmdGpu {
struct Liverpool;
union Regs;
} // namespace AmdGpu

namespace Serialization {
struct Archive;
}

namespace Shader {
struct Info;
}

namespace Storage {
class DataBase;
}

namespace Vulkan {

/// FIX-227: the hull shader LDS emulation mode for the current draw (0: off).
u32 TessLdsEmulation(const AmdGpu::Regs& regs);


class Instance;
class Scheduler;
class ShaderCache;

/// DIAG-030: shaders whose inputs are logged (GT Sport's grass compute shaders by default, or
/// the comma-separated hashes in SHADGT_WATCH_SHADERS).
bool IsWatchedShader(u64 hash);

struct DrawIndirectParams {
    u16 vertex_sgpr_offset;
    u32 instance_sgpr_offset;
    /// PERF-028: draw a quad list through the tessellation helpers instead of as triangles
    /// (indirect draws, or indices the command thread cannot read).
    bool tessellate_quads{};
    /// FIX-192: an indirect draw, whose vertex and instance ids the shader unbiases.
    bool is_indirect{};
};

/// PERF-103: the game runs in slow motion while pipelines it needs are compiled.
bool CompileSlowMotion();
/// FIX-104: pipeline creations in progress.
u32 PipelineBuildsRunning();
/// Active shader translation or driver pipeline compilation, independent of guest-clock holds.
bool CompilationInProgress();
/// FIX-GT7-023: saves the driver pipeline cache before the window's close ends the process.
/// The MB written, or -1 if there was nothing new to save (or no cache file).
int SaveDriverCacheAtExit();

/// NAT-031: the pipeline cache's lookup lock. The command and recorder threads take it
/// exclusively (as before); game job threads looking pipelines up take it shared, and step aside
/// while one of the threads above waits for it.
class LookupLock {
public:
    void lock() {
        writers_waiting.fetch_add(1, std::memory_order_acq_rel);
        mutex.lock();
        writers_waiting.fetch_sub(1, std::memory_order_acq_rel);
    }
    bool try_lock() {
        return mutex.try_lock();
    }
    void unlock() {
        mutex.unlock();
    }
    void lock_shared() {
        while (writers_waiting.load(std::memory_order_acquire) != 0) {
            std::this_thread::yield();
        }
        mutex.lock_shared();
    }
    void unlock_shared() {
        mutex.unlock_shared();
    }

private:
    std::shared_mutex mutex;
    std::atomic<u32> writers_waiting{0};
};

class PipelineCache {
public:
    explicit PipelineCache(const Instance& instance, Scheduler& scheduler,
                           AmdGpu::Liverpool* liverpool, u32 sparse_page_shift);
    ~PipelineCache();

    void Sync();

    const GraphicsPipeline* GetGraphicsPipeline(const DrawIndirectParams params = {});
    /// The pipeline of the key RefreshGraphicsKey made (built if needed).
    const GraphicsPipeline* PipelineForCurrentKey(const DrawIndirectParams params);

    /// PERF-047: with pipelines selected on the command thread, both GPU threads look pipelines
    /// up; every GetGraphicsPipeline/GetComputePipeline call holds this lock, and so does any
    /// read of the infos below or copy of a cached info.
    LookupLock& LookupMutex() {
        return lookup_mutex;
    }
    /// The infos the last GetGraphicsPipeline selected, with that draw's user data.
    const std::array<const Shader::Info*, MaxShaderStages>& SelectedInfos() const {
        return scratch_.infos;
    }

    const ComputePipeline* GetComputePipeline();

    /// FIX-198: while set (the lookup lock held), GetGraphicsPipeline waits for a pipeline and
    /// shaders still building instead of skipping the draw.
    void SetForceWait(bool on) {
        force_wait = on;
    }

    struct PipelineBuild;
    /// PERF-GT7-020: while set (the lookup lock held), GetGraphicsPipeline waits for a draw's
    /// shaders (translations, the store's listing) as with SetForceWait, but not for its
    /// pipeline: the draw is skipped, and TakeSkippedBuild gives the build to wait for.
    void SetWaitForShaders(bool on) {
        wait_for_shaders = on;
    }
    /// PERF-GT7-020 (lookup lock held): the build the last GetGraphicsPipeline skipped its draw
    /// for, when it skipped it for a pipeline still building.
    std::shared_ptr<PipelineBuild> TakeSkippedBuild() {
        return std::move(skipped_build);
    }
    /// PERF-GT7-036: with SetWaitForShaders, GetGraphicsPipeline does not wait for the draw's
    /// translations on the workers either: it returns no pipeline, and TakeShaderWait (lookup
    /// lock held) gives what waits for them, to call without the lock. Empty when none.
    void SetShadersUnlocked(bool on) {
        shaders_unlocked = on;
    }
    std::function<void()> TakeShaderWait();
    /// PERF-GT7-020 (without the lookup lock): builds the pipeline here unless a worker started
    /// it, and returns once it is built. PERF-GT7-021: meanwhile more workers build at once, and
    /// the read-ahead states blocked by their shaders are evaluated again (the lock taken when
    /// free).
    void FinishBuildUnlocked(const std::shared_ptr<PipelineBuild>& build);

    /// NAT-030: the pipeline already built for the draw a game job thread is reading (its
    /// registers installed with Liverpool::SetThreadDrawState), or nullptr. Nothing is
    /// translated, loaded, built or waited for, and the selection made by the threads above is
    /// left as it was. Hold LookupMutex.
    const GraphicsPipeline* LookupBuiltPipeline(const DrawIndirectParams params);
    /// NAT-030: why the last LookupBuiltPipeline found nothing.
    const char* LookupMiss() const {
        return t_last_miss;
    }
    /// NAT-032: the stage infos of this thread's last LookupBuiltPipeline (its own copies, with
    /// the draw's user data), valid until its next lookup.
    static const std::array<const Shader::Info*, MaxShaderStages>& JobInfos() {
        return *t_job_infos;
    }

    using Result = std::tuple<const Shader::Info*, vk::ShaderModule, u64>;
    Result GetProgram(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                      const Shader::ShaderParams& params, Shader::Backend::Bindings& binding);

    std::optional<vk::ShaderModule> ReplaceShader(vk::ShaderModule module,
                                                  std::span<const u32> spv_code);

    static std::string GetShaderName(Shader::HwStage stage, u64 hash,
                                     std::optional<size_t> perm = {});

    auto& GetProfile() const {
        return profile;
    }

    /// FIX-GT7-023: see SaveDriverCacheAtExit.
    int SaveDriverCacheNow();

private:
    bool RefreshGraphicsKey();
    /// Registers the pipeline keys are built from: the current ones, or during read-ahead the
    /// registers at an upcoming draw (PERF-019).
    const AmdGpu::Regs& Regs() const;
    bool HasSupportedColorTargets() const;
    class PipelineBuildWorkers;
    /// PERF-019: builds the pipeline for the current key on a worker thread. PERF-GT7-025:
    /// in_order: an urgent build in draw order (a draw that goes on without it), not first.
    std::shared_ptr<PipelineBuild> StartPipelineBuild(bool urgent, bool in_order = false);
    /// A build of the current key, infos and modules.
    std::shared_ptr<PipelineBuild> MakePipelineBuild();
    void QueueBuild(const std::shared_ptr<PipelineBuild>& build, BuildQueue::Kind kind);
    /// Builds the pipeline unless a worker already started; returns when it is built.
    void FinishBuild(PipelineBuild& build, bool draw_waits = false);
    /// PERF-019: starts builds for the new pipelines of the draws after the current one.
    void ReadAhead(const DrawIndirectParams params, bool restore = true);
    /// PERF-020: while new pipelines keep appearing, reads each command buffer ahead as soon as
    /// it starts, so its first new pipelines are built before their draws.
    void ReadAheadAtBufferStart();
    /// PERF-034: with the draw pipe, starts builds for the upcoming draw states the command
    /// thread queued. During a draw (`restore`), its key is computed again afterwards.
    /// FIX-GT7-015: `waiting`: a draw waits for its pipeline (as with `restore`).
    void BuildQueuedStates(const DrawIndirectParams params, bool restore, bool waiting = false);
    /// FIX-GT7-015: read-ahead builds no draw asked for that no worker started in 30 s are
    /// cancelled (every 2 s).
    void DropStaleReadAheadBuilds(std::chrono::steady_clock::time_point now);
    std::chrono::steady_clock::time_point stale_check{};
    u64 stale_dropped = 0;
    /// Claims a build nobody started, so it is never built; false if one started.
    static bool CancelBuild(PipelineBuild& build);
    /// FIX-110: in the menus draws wait for new pipelines instead of being skipped.
    static bool MenusWait();
    /// FIX-114: how long a draw may wait for its building pipeline before it is skipped.
    static std::chrono::microseconds QuickWait();
    /// FIX-119: whether this draw's old-renderer (D10) pipeline is waited for.
    bool LegacyWaits() const;
    /// PERF-076: logs the background pipeline counts every 2 s.
    void ReportAsyncPipelines(std::chrono::steady_clock::time_point now);
    /// PERF-GT7-029: a draw skipped for a pipeline still building (counted for the catch-up; one
    /// skipped again moves its build to the front once, `again`).
    void NoteAsyncSkip(const std::shared_ptr<PipelineBuild>& build, bool again);
    /// FIX-GT7-009: a worker's screen job: builds the not yet started build of a draw on screen
    /// skipped most lately.
    void BuildOnScreen();
    /// PERF-GT7-029: while many draws are skipped for their builds, compiles use every core, 8 at
    /// a time; back on the compile cores once few are. RYGT_COMPILE_CATCH_UP=0 turns it off.
    void UpdateCatchUp(std::chrono::steady_clock::time_point now);
    /// PERF-034: waits while a worker builds a pipeline, starting builds for the states the
    /// command thread queues meanwhile. Returns at once if no worker took the build, which
    /// FinishBuild then does on this thread.
    void WaitForBuild(PipelineBuild& build, const DrawIndirectParams params);
    /// Selects swizzled-alpha blend emulation for a lane-dependent blend, or reports it once.
    void RefreshSwizzledBlend(u32 cb, Shader::PsColorBuffer& color_buffer,
                              const AmdGpu::BlendControl& bc);
    bool RefreshGraphicsStages();
    bool RefreshComputeKey();

    void DumpFailedShader(std::span<const u32> code, u64 hash, Shader::HwStage stage);
    void LogRepeatedPermutation(const Program& program, size_t perm_idx, Shader::HwStage hw_stage,
                                u64 pgm_hash);
    void LogNewPermutation(const Program& program, Shader::HwStage hw_stage,
                           const Shader::ShaderParams& params,
                           const Shader::RuntimeInfo& runtime_info,
                           const Shader::Backend::Bindings& binding); // DIAG-GT7-013
    std::unordered_set<u64> failed_shaders;
    /// FIX-043 diagnostic: hash of the SPIR-V that CompileModule produced last.
    u64 last_spv_hash{};
    void DumpShader(std::span<const u32> code, u64 hash, Shader::HwStage stage, size_t perm_idx,
                    std::string_view ext);
    std::optional<std::vector<u32>> GetShaderPatch(u64 hash, Shader::HwStage stage, size_t perm_idx,
                                                   std::string_view ext);
    vk::ShaderModule CompileModule(Shader::Info& info, Shader::RuntimeInfo& runtime_info,
                                   const std::span<const u32>& code, size_t perm_idx,
                                   Shader::Backend::Bindings& binding);
    const Shader::RuntimeInfo& BuildRuntimeInfo(Shader::HwStage stage, Shader::SwStage l_stage);
    bool IsTessEmulatedDraw() const;
    /// PERF-028: the current draw is a quad list drawn as a triangle list.
    bool QuadListAsTriangles() const;
    void RecoverAttributeFlags(Shader::Info& info, const Shader::StageSpecialization& spec,
                               const Shader::ShaderParams& params,
                               const Shader::RuntimeInfo& runtime_info, size_t perm_idx);

    [[nodiscard]] bool IsPipelineCacheDirty() const {
        return num_new_pipelines > 0;
    }

private:
    const Instance& instance;
    Scheduler& scheduler;
    AmdGpu::Liverpool* liverpool;
    DescriptorHeap desc_heap;
    vk::UniquePipelineCache pipeline_cache;
    vk::UniquePipelineLayout pipeline_layout;
    Shader::Profile profile{};
    Shader::Pools pools;
    tsl::robin_map<size_t, std::unique_ptr<Program>> program_cache;
    /// PERF-013: per program, one past the highest permutation index in the stored cache, so a
    /// permutation created at runtime never reuses (and overwrites) a stored index.
    tsl::robin_map<size_t, size_t> stored_perm_end;
    /// FIX-018: programs translated again with the else-scope fix, and the first permutation
    /// index made with it; stored permutations below it are not preloaded.
    struct RetranslatedProgram {
        size_t boundary;
        u32 fixes; ///< 1: FIX-018 else scope, 2/4: FIX-020 wave64 lane reads (v1/v2)
    };
    std::unordered_map<u64, RetranslatedProgram> else_scope_fixed;
    std::unordered_set<u64> else_scope_checked;
    std::vector<std::unique_ptr<Program>> retired_programs;
    std::filesystem::path else_scope_fixed_path;
    void LoadElseScopeFixed();
    void SaveElseScopeFixed() const;
    /// PERF-032: stored translations are loaded when their program is first used, and new ones
    /// are stored.
    bool use_stored_shaders{};
    void ListStoredPermutations(u64 pgm_hash, Program& program);
    /// A program's stored permutations as read from the store, and their insertion into the
    /// program.
    struct StoredListing {
        u64 pgm_hash{};
        struct Entry {
            size_t perm_idx;
            std::vector<u8> meta;
            bool has_spv;
        };
        std::vector<Entry> entries;
    };
    static StoredListing ReadStoredListing(u64 pgm_hash);
    static StoredListing ReadStoredListing(u64 pgm_hash, Storage::DataBase& store);
    void ApplyStoredListing(StoredListing&& listing, Program& program);
    vk::ShaderModule LoadStoredModule(u64 pgm_hash, size_t perm_idx);
    vk::ShaderModule LoadStoredModule(u64 pgm_hash, size_t perm_idx, Storage::DataBase& store);

    // D10: while the new renderer's own translations of a draw's shaders, or its pipeline, are
    // not built yet, the draw uses the old renderer's stored translations (its own store, read
    // only). The new ones are built in the background meanwhile, so the game neither waits nor
    // skips draws for them.
    bool LegacyFallback() const {
        return legacy_fallback;
    }
    std::optional<Result> TryLegacyProgram(Shader::HwStage hw_stage, const Shader::ShaderParams& params,
                                           const Shader::RuntimeInfo& runtime_info,
                                           Shader::Backend::Bindings& binding);
    tsl::robin_map<size_t, std::unique_ptr<Program>> legacy_programs;
    bool legacy_fallback{};
    bool force_legacy{};      ///< making the key with the old translations only
    bool fallback_nested{};
    bool fallback_miss{};     ///< a new translation was queued for the current key
    struct LegacyStats {
        u64 programs{};
        u64 draws{};
        u64 queued{};
        u64 pipelines{};
        u64 no_legacy{};
        u64 fallback_ok{};
        u64 fallback_unbuilt{};
        u64 fallback_none{};
        std::chrono::steady_clock::time_point since{};
    } legacy_stats;
    void ReportLegacy();
    tsl::robin_map<ComputePipelineKey, std::unique_ptr<ComputePipeline>> compute_pipelines;
    tsl::robin_map<GraphicsPipelineKey, std::unique_ptr<GraphicsPipeline>> graphics_pipelines;
    /// NAT-031: what a key is computed in. The command and recorder threads use the cache's own
    /// (scratch_, under the exclusive lock); a game job thread looking a pipeline up uses its
    /// own (under the shared lock), so job threads look pipelines up at the same time.
    struct KeyScratch {
        std::array<Shader::RuntimeInfo, MaxShaderStages> runtime_infos{};
        std::array<const Shader::Info*, MaxShaderStages> infos{};
        std::array<vk::ShaderModule, MaxShaderStages> modules{};
        Shader::Gcn::FetchShaderData* fetch_shader{};
        GraphicsPipelineKey graphics_key{};
        DrawIndirectParams draw_indirect_params{};
        bool stage_deferred{};
        bool legacy_failed{};   ///< a stage had none
        bool key_uses_legacy{}; ///< the current key has a stage with an old translation
        const char* lookup_miss{}; ///< NAT-030: why a job thread's lookup found nothing
    };
    KeyScratch scratch_;
    static inline thread_local KeyScratch* t_scratch = nullptr;
    /// NAT-030: this thread is a game job thread looking a pipeline up (LookupBuiltPipeline).
    static inline thread_local bool t_lookup_only = false;
    static inline thread_local const char* t_last_miss = nullptr; ///< of this thread's last lookup
    static inline thread_local const std::array<const Shader::Info*, MaxShaderStages>* t_job_infos =
        nullptr;
    KeyScratch& S() {
        return t_scratch ? *t_scratch : scratch_;
    }
    const KeyScratch& S() const {
        return t_scratch ? *t_scratch : scratch_;
    }
    /// NAT-031: the info a job thread's lookup matches in: its own copy of a cached one (the
    /// threads above write the draw's user data into the cached ones).
    Shader::Info& MatchInfo(const Shader::Info& cached);
    /// NAT-031: GetProgram for a game job thread's lookup (read-only).
    Result GetProgramForJob(Shader::HwStage hw_stage, const Shader::ShaderParams& params,
                            const Shader::RuntimeInfo& runtime_info,
                            Shader::Backend::Bindings& binding);
    ComputePipelineKey compute_key{};
    // New pipelines added to the cache since the game start (FIX-GT7-025: atomic, the window's
    // closing thread reads it).
    std::atomic<u32> num_new_pipelines{};

    // The driver's VkPipelineCache, kept on disk so cached pipelines are not recompiled by the
    // driver on every launch.
    void CreateDriverCache();
    void SaveDriverCache(bool wait);
    void MaybeSaveDriverCache();
    bool DriverCacheOutgrown(vk::Device device, vk::PipelineCache cache, u32 count);
    std::atomic<bool> driver_cache_full{}; // FIX-GT7-050: this session saves no more
    std::filesystem::path driver_cache_path;
    /// PERF-GT7-037: the recipes of compute pipelines that took the driver long to build, and
    /// the thread that builds them again in the background at start.
    std::filesystem::path WarmComputeDir() const;
    void WarmComputePipelines(std::stop_token stop);
    std::jthread warm_compute_thread;
    u32 pipelines_at_driver_save{};
    std::chrono::steady_clock::time_point driver_cache_saved_at{};
    std::atomic<bool> driver_cache_writing{}; // PERF-036
    std::jthread driver_cache_writer;
    DriverCacheFile driver_cache_file; ///< FIX-GT7-025: the save the file holds
    /// FIX-217: pixel shaders (by code address) with a V_INTERP_MOV of a vertex's own value.
    bool ReadsVertexValues(const AmdGpu::ShaderProgram& program);
    std::unordered_map<VAddr, bool> vertex_value_programs;
    std::atomic<u64> driver_cache_bytes{}; // PERF-167: size of the last save

    std::unordered_set<u64> logged_swizzled_blends;

    // Only if Config::collectShadersForDebug()
    tsl::robin_map<vk::ShaderModule,
                   std::vector<std::variant<GraphicsPipelineKey, ComputePipelineKey>>>
        module_related_pipelines;

    // PERF-019: pipelines being built ahead of their draws.
    const AmdGpu::Regs* regs_override{};
    u32 draws_since_scan{};
    std::chrono::steady_clock::time_point last_pipeline_miss{};
    /// PERF-021: register states already evaluated by the read-ahead, so reading the same
    /// commands again (every command buffer start re-reads what follows) costs only a hash.
    std::unordered_set<u64> read_ahead_states;
    struct ReadAheadStats {
        u32 scans{};
        u32 draws{};
        u32 evaluated{};
        u32 started{};
        double ms{};
        std::chrono::steady_clock::time_point since{};
    } read_ahead_stats;
    u32 last_scan_draws{};
    // PERF-034: the registers of the queued draw state being evaluated.
    std::unique_ptr<AmdGpu::Regs> queued_state_regs;
    struct QueuedStateStats {
        u32 evaluated{};
        u32 started{};
        u32 waits{};
        double wait_ms{};
        double ms{};
        std::chrono::steady_clock::time_point since{};
    } queued_state_stats;
    tsl::robin_map<GraphicsPipelineKey, std::shared_ptr<PipelineBuild>> pending_builds;
    /// PERF-076: new graphics pipelines build in the background while their draws are skipped.
    static bool AsyncPipelines();
    static bool WritesMemory(const std::array<const Shader::Info*, MaxShaderStages>& infos);
    static bool NeverSkipped(const std::array<const Shader::Info*, MaxShaderStages>& infos);
    struct AsyncStats {
        u64 deferred{};
        u64 installed{};
        u64 skipped_draws{};
        u64 skipped_translating{}; ///< SHD-005
        std::chrono::steady_clock::time_point since{};
    } async_stats;
    /// SHD-005: a draw of a race skipped while its new shader translates on a worker, instead of
    /// the command thread waiting for the translation.
    bool SkipTranslationWait(Shader::SwStage sw_stage, const Shader::ShaderParams& params);
    tsl::robin_map<u64, bool> code_writes_memory; ///< by program hash
    bool translation_skipped{};
    LookupLock lookup_mutex; // PERF-047, PERF-060; NAT-031: shared for job threads
    bool force_wait = false; ///< FIX-198
    bool wait_for_shaders = false;                ///< PERF-GT7-020
    bool shaders_unlocked = false; ///< PERF-GT7-036
    /// PERF-GT7-036: translations a draw would have waited for under the lookup lock.
    struct TranslateJob;
    std::vector<std::shared_ptr<TranslateJob>> skipped_translations;
    void NoteUnlockedShaderWait(u64 hash);
    /// PERF-GT7-021: read-ahead states whose key waited for shaders (a translation queued or not
    /// taken yet, the store's listing), evaluated again until they have one (3 s at most). They
    /// were dropped: the first frame of a new track decoded every draw while its shaders were
    /// made, and the recorder then built each one-off target's pipeline itself, one at a time.
    static bool RetryReadAhead();
    struct BlockedState {
        std::vector<u32> ranges; ///< ReadAheadStates::Ranges, in order
        std::chrono::steady_clock::time_point first;
        std::chrono::steady_clock::time_point next;
    };
    std::deque<BlockedState> blocked_states;
    std::vector<std::vector<u32>> blocked_free;
    bool read_ahead_blocked = false; ///< set by GetProgram for the state being evaluated
    struct {
        u64 kept{};
        u64 retried{};
        u64 started{};
        u64 dropped{};
        u64 tessellated{}; ///< FIX-GT7-005: not kept
        u64 expired{};     ///< FIX-GT7-015: dropped unretried, no draw waiting
        u64 failed{};      ///< FIX-GT7-016: their key failed a check (stale code)
    } blocked_stats;
    std::shared_ptr<PipelineBuild> skipped_build; ///< PERF-GT7-020
    std::atomic<u64> catch_up_skips{}; ///< PERF-GT7-029: since the last look
    std::atomic<u64> catch_up_bumped{};
    std::mutex catch_up_mutex;
    struct {
        bool on = false;
        std::chrono::steady_clock::time_point window_start{};
        std::chrono::steady_clock::time_point since{};
        u32 calm = 0;
        u64 skipped = 0;
        // PERF-GT7-033: stepped back while the game runs slower than 30 fps.
        bool stepped_back = false;
        std::chrono::steady_clock::time_point pace_start{};
        std::chrono::steady_clock::time_point stepped_since{};
        u64 pace_frames = 0;
        u64 pace_us = 0;
        u32 fast = 0;
        u32 step_backs = 0;
        double stepped_seconds = 0;
        u32 pace_limit = 0;   ///< PERF-GT7-034: builds at once (the usual width: no limit)
        u32 pace_changes = 0; ///< PERF-GT7-034: logged ones
    } catch_up;
    std::mutex on_screen_mutex; ///< FIX-GT7-009
    std::vector<std::shared_ptr<PipelineBuild>> on_screen;
    struct {
        std::atomic<u64> waits{};
        std::atomic<u64> built_here{};
        std::atomic<u64> wait_us{};
        std::atomic<s64> since_ms{};
        std::atomic<u64> shader_waits{};   ///< PERF-GT7-036
        std::atomic<u64> shader_wait_us{}; ///< PERF-GT7-036
    } unlocked_stats; ///< PERF-GT7-020

    /// FIX-078: a shader the read-ahead found with no matching translation is translated on a
    /// worker thread; the draw that needs it takes the result (or waits for it) instead of
    /// translating it on the command thread, one shader after another, under the lookup lock.
    struct TranslateJob {
        explicit TranslateJob(const Shader::RuntimeInfo& runtime_info_)
            : runtime_info{runtime_info_} {}
        Shader::HwStage hw_stage;
        Shader::SwStage sw_stage;
        u64 hash;
        std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data;
        std::span<const u32> code; // FIX-160: over code_copy, not guest memory
        std::vector<u32> code_copy;
        std::vector<u32> fetch_copy; // FIX-189: the fetch shader user_data points at
        std::vector<Shader::Info::PinnedFetchSharp> fetch_sharps; // FIX-GT7-054
        Shader::RuntimeInfo runtime_info;
        Shader::Backend::Bindings start;
        std::unique_ptr<Shader::Info> info;
        std::vector<u32> spv;
        vk::ShaderModule module{};
        double translate_ms{};
        u32 diag_id{};        ///< DIAG-GT7-073
        bool diag_ahead{};    ///< DIAG-GT7-073: queued by a read-ahead state
        std::chrono::steady_clock::time_point queued_at{};
        bool too_old{}; ///< FIX-GT7-052: dropped before a worker translated it
        /// Set when harvested: its specialization, and when it became ready.
        Shader::StageSpecialization spec;
        std::chrono::steady_clock::time_point ready_since{};
        std::atomic<bool> done{};
        std::mutex mutex;
        std::condition_variable cv;
    };
    class TranslateWorkers;
    bool AsyncTranslation() const;
    /// FIX-078 (lookup lock held): moves the finished translations of a program to its ready
    /// list, first waiting for its unfinished ones when `wait`.
    void HarvestTranslations(u64 hash, bool wait);
    /// FIX-078 (lookup lock held): a ready translation made for the inputs of this lookup. A draw
    /// (`insert`) adds it as a permutation; the read-ahead only learns it exists. Translations
    /// no draw matches (made from read-ahead states that were never drawn) are dropped later.
    bool TakeReadyTranslation(Shader::HwStage hw_stage, Shader::SwStage sw_stage, u64 hash,
                              Program& program, const Shader::ShaderParams& params,
                              const Shader::RuntimeInfo& runtime_info,
                              const Shader::Backend::Bindings& binding, bool insert);
    /// FIX-078 (lookup lock held): queues a translation; false when one with the same inputs is
    /// already queued.
    bool QueueTranslation(Shader::HwStage hw_stage, Shader::SwStage sw_stage,
                          const Shader::ShaderParams& params,
                          const Shader::RuntimeInfo& runtime_info,
                          const Shader::Backend::Bindings& start);
    tsl::robin_map<u64, std::vector<std::shared_ptr<TranslateJob>>> translate_jobs;
    tsl::robin_map<u64, std::vector<std::shared_ptr<TranslateJob>>> ready_translations;
    /// FIX-078: while set, a draw's own missing translations are queued (all its stages at
    /// once) instead of made here one by one; `deferred_miss` tells that one was.
    bool defer_misses{};
    bool deferred_miss{};
    struct {
        u64 queued{};
        u64 used{};
        u64 expired{};
        u64 duplicate{};
        u64 failed{};
        u64 too_old{}; ///< FIX-GT7-052
        u64 waited{};
        double wait_ms{};
        double translate_ms{};
        std::chrono::steady_clock::time_point since{};
    } translate_stats;
    std::unique_ptr<TranslateWorkers> translate_workers;

    /// PERF-GT7-019: the shader store's reads (a new program's stored permutations, then their
    /// SPIR-V files) on threads of their own. The first open of a stored file took ~5 ms (it is
    /// scanned on open; 0.1 ms after), and a new track's first race read ~2,000 of them one at
    /// a time on the GPU threads: a 6 s stop. Eight readers open them at once.
    class StoreReaders;
    /// PERF-GT7-019 (lookup lock held): whether the program's stored permutations were read (by a
    /// store reader). When `may_skip`, false while they are read; otherwise waits for them.
    bool StoredListingReady(u64 pgm_hash, bool may_skip);
    std::unordered_map<u64, std::future<StoredListing>> pending_listings;
    struct {
        u64 listed{};
        u64 skipped{};
        u64 waited{};
        double wait_ms{};
        std::chrono::steady_clock::time_point since{};
    } store_stats;
    std::unique_ptr<StoreReaders> store_readers;

    // Last member, so the workers stop before anything they use is destroyed.
    std::unique_ptr<PipelineBuildWorkers> build_workers;
};

} // namespace Vulkan
