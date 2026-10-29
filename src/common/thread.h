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

/// Undoes PreferPerformanceCores/PreferEfficiencyCores for the calling thread.
void AllowAllCores();

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
