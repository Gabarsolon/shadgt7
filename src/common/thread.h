// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <chrono>
#include "common/types.h"

namespace Common {

enum class ThreadPriority : u32 {
    Low = 0,
    Normal = 1,
    High = 2,
    VeryHigh = 3,
    Critical = 4,
};

void SetCurrentThreadRealtime(std::chrono::nanoseconds period_ns);

void SetCurrentThreadPriority(ThreadPriority new_priority);

void SetCurrentThreadName(const char* name);

/// PERF-088: on a CPU with fast and slow cores (Intel 12th gen and later), keeps the calling
/// thread on the fast ones (SHADGT_FAST_CORES=0 turns it off). No effect on other CPUs.
void PreferPerformanceCores();

/// PERF-088: 1 when the calling thread runs on a fast core of such a CPU, 0 on a slow one,
/// -1 when the CPU has only one kind.
int CurrentCoreIsFast();

/// PERF-097: on hybrid CPUs, keeps the calling thread on the efficiency cores (shader and
/// pipeline compiles, so a burst of them leaves the fast cores to the game and the renderer).
/// SHADGT_BUILD_ON_SLOW_CORES=0 turns it off.
void PreferEfficiencyCores();

/// PERF-GT7-024: pipeline builds: on hybrid CPUs as PreferEfficiencyCores; on CPUs with one kind
/// of core, the compile cores (the last third of the cores, both hardware threads of each).
void PreferCompileCores();

/// Undoes PreferPerformanceCores/PreferEfficiencyCores for the calling thread.
void UpdateCoreOwner(u32 slot, bool wanted); ///< PERF-GT7-048

void AllowAllCores();

/// PERF-088: the CPU has fast and slow cores.
bool HybridCpu();

/// PERF-GT7-024: on a CPU with one kind of core, the threads of the driver's shader compiler are
/// kept on the compile cores too: unnamed threads busy since the last call are sampled, and one
/// found running in nvgpucomp64.dll is moved. Called every 2 s. RYGT_COMPILE_CORES=0 turns the
/// compile cores off.
void ConfineDriverCompileThreads();

/// PERF-GT7-024: while a draw waits for its pipeline (the game is stopped anyway), compile work
/// may use every core; calls nest (on, then off once per on).
void SetCompileCoresWide(bool wide);
/// PERF-GT7-024, PERF-GT7-032: compile work uses every core now.
bool CompileCoresWide();

/// PERF-GT7-032: during a catch-up (PERF-GT7-029: many draws skipped while their pipelines
/// build), compile work uses every core below normal priority, taking the time the game's and the
/// renderer's threads leave. Not while a draw waits for its pipeline. Calls nest.
void SetCompileCatchUp(bool on);
/// PERF-GT7-032: compile work runs below normal priority now.
bool CompileThreadsLow();

/// PERF-GT7-032: around a build the calling thread does while the GPU's commands wait for it (a
/// new compute pipeline on the command thread). Meanwhile the other compile work stays on the
/// compile cores below normal priority (unless a draw waits for its pipeline), and the calling
/// thread runs above normal priority on the other cores. Calls nest; End once per Begin, on the
/// same thread.
void BeginForegroundCompile();
void EndForegroundCompile();

void SetThreadName(void* thread, const char* name);

bool AccurateSleep(std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   bool interruptible);

class AccurateTimer {
    std::chrono::nanoseconds target_interval{};
    std::chrono::nanoseconds total_wait{};

    std::chrono::high_resolution_clock::time_point start_time;

public:
    explicit AccurateTimer(std::chrono::nanoseconds target_interval);

    void Start();

    void End();

    std::chrono::nanoseconds GetTotalWait() const {
        return total_wait;
    }
};

std::string GetCurrentThreadName();

} // namespace Common
