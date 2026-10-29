// SPDX-FileCopyrightText: 2013 Dolphin Emulator Project
// SPDX-FileCopyrightText: 2014 Citra Emulator Project
// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <ctime>
#include <vector>
#include <string>
#include <thread>
#include <fmt/format.h>

#include "core/libraries/fiber/fiber.h"
#include "core/libraries/kernel/threads/pthread.h"

#include "common/error.h"
#include "common/logging/log.h"
#include "common/thread.h"
#include "ntapi.h"
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_time.h>
#include <pthread.h>
#elif defined(_WIN32)
#include <windows.h>
#include "common/string_util.h"
#else
#if defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
#include <pthread_np.h>
#else
#include <pthread.h>
#endif
#include <sched.h>
#endif
#ifndef _WIN32
#include <unistd.h>
#endif

#ifdef __FreeBSD__
#define cpu_set_t cpuset_t
#endif

namespace Common {

#ifdef __APPLE__

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // CPU time to grant.
    const std::chrono::nanoseconds computation_ns = period_ns / 2;

    // Determine the timebase for converting time to ticks.
    struct mach_timebase_info timebase{};
    mach_timebase_info(&timebase);
    const auto ticks_per_ns =
        static_cast<double>(timebase.denom) / static_cast<double>(timebase.numer);

    const auto period_ticks =
        static_cast<u32>(static_cast<double>(period_ns.count()) * ticks_per_ns);
    const auto computation_ticks =
        static_cast<u32>(static_cast<double>(computation_ns.count()) * ticks_per_ns);

    thread_time_constraint_policy policy = {
        .period = period_ticks,
        .computation = computation_ticks,
        // Should not matter since preemptible is false, but needs to be >= computation regardless.
        .constraint = computation_ticks,
        .preemptible = false,
    };

    int ret = thread_policy_set(
        pthread_mach_thread_np(pthread_self()), THREAD_TIME_CONSTRAINT_POLICY,
        reinterpret_cast<thread_policy_t>(&policy), THREAD_TIME_CONSTRAINT_POLICY_COUNT);
    if (ret != KERN_SUCCESS) {
        LOG_ERROR(Common, "Could not set thread to real-time with period {} ns: {}",
                  period_ns.count(), ret);
    }
}

#else

void SetCurrentThreadRealtime(const std::chrono::nanoseconds period_ns) {
    // Not implemented
}

#endif

#ifdef _WIN32

namespace {
// PERF-088: the CPU sets and their efficiency classes (the highest class is the fast cores).
struct CoreClasses {
    std::vector<ULONG> fast_ids;
    std::vector<ULONG> slow_ids;
    std::vector<s8> fast_by_index; // by group * 64 + logical processor index: 1 fast, 0 slow
    bool hybrid = false;
};
const CoreClasses& GetCoreClasses() {
    static const CoreClasses classes = [] {
        CoreClasses result;
        ULONG length = 0;
        GetSystemCpuSetInformation(nullptr, 0, &length, GetCurrentProcess(), 0);
        std::vector<u8> buffer(length);
        auto* info = reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data());
        if (length == 0 ||
            !GetSystemCpuSetInformation(info, length, &length, GetCurrentProcess(), 0)) {
            return result;
        }
        BYTE lowest = 255, highest = 0;
        for (ULONG offset = 0; offset < length;) {
            const auto& entry = *reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
            if (entry.Type == CpuSetInformation) {
                lowest = std::min(lowest, entry.CpuSet.EfficiencyClass);
                highest = std::max(highest, entry.CpuSet.EfficiencyClass);
            }
            offset += entry.Size;
        }
        result.hybrid = lowest != highest;
        for (ULONG offset = 0; offset < length;) {
            const auto& entry = *reinterpret_cast<SYSTEM_CPU_SET_INFORMATION*>(buffer.data() + offset);
            if (entry.Type == CpuSetInformation) {
                const bool fast = entry.CpuSet.EfficiencyClass == highest;
                if (fast) {
                    result.fast_ids.push_back(entry.CpuSet.Id);
                } else {
                    result.slow_ids.push_back(entry.CpuSet.Id);
                }
                const size_t index = size_t(entry.CpuSet.Group) * 64 +
                                     entry.CpuSet.LogicalProcessorIndex;
                if (result.fast_by_index.size() <= index) {
                    result.fast_by_index.resize(index + 1, -1);
                }
                result.fast_by_index[index] = fast ? 1 : 0;
            }
            offset += entry.Size;
        }
        LOG_INFO(Common, "PERF-088: {} CPU, {} fast logical processors",
                 result.hybrid ? "hybrid" : "uniform", result.fast_ids.size());
        return result;
    }();
    return classes;
}
} // namespace

void PreferPerformanceCores() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_FAST_CORES");
        return !(env && env[0] == '0');
    }();
    const auto& classes = GetCoreClasses();
    if (!enabled || !classes.hybrid || classes.fast_ids.empty()) {
        return;
    }
    // Not throttled as a background thread either (EcoQoS sends those to the slow cores).
    THREAD_POWER_THROTTLING_STATE throttling{};
    throttling.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
    throttling.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
    throttling.StateMask = 0;
    SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &throttling,
                         sizeof(throttling));
    SetThreadSelectedCpuSets(GetCurrentThread(), classes.fast_ids.data(),
                             static_cast<ULONG>(classes.fast_ids.size()));
}

void PreferEfficiencyCores() {
    static const bool enabled = [] {
        const char* env = std::getenv("SHADGT_BUILD_ON_SLOW_CORES");
        return !(env && env[0] == '0');
    }();
    const auto& classes = GetCoreClasses();
    if (!enabled || !classes.hybrid || classes.slow_ids.empty()) {
        return;
    }
    SetThreadSelectedCpuSets(GetCurrentThread(), classes.slow_ids.data(),
                             static_cast<ULONG>(classes.slow_ids.size()));
}

void AllowAllCores() {
    SetThreadSelectedCpuSets(GetCurrentThread(), nullptr, 0);
}

int CurrentCoreIsFast() {
    const auto& classes = GetCoreClasses();
    if (!classes.hybrid) {
        return -1;
    }
    PROCESSOR_NUMBER number;
    GetCurrentProcessorNumberEx(&number);
    const size_t index = size_t(number.Group) * 64 + number.Number;
    return index < classes.fast_by_index.size() ? classes.fast_by_index[index] : -1;
}

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    auto handle = GetCurrentThread();
    int windows_priority = 0;
    switch (new_priority) {
    case ThreadPriority::Low:
        windows_priority = THREAD_PRIORITY_BELOW_NORMAL;
        break;
    case ThreadPriority::Normal:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    case ThreadPriority::High:
        windows_priority = THREAD_PRIORITY_ABOVE_NORMAL;
        break;
    case ThreadPriority::VeryHigh:
        windows_priority = THREAD_PRIORITY_HIGHEST;
        break;
    case ThreadPriority::Critical:
        windows_priority = THREAD_PRIORITY_TIME_CRITICAL;
        break;
    default:
        windows_priority = THREAD_PRIORITY_NORMAL;
        break;
    }
    SetThreadPriority(handle, windows_priority);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();

    LARGE_INTEGER interval{
        .QuadPart = -1 * (duration.count() / 100u),
    };
    HANDLE timer = ::CreateWaitableTimer(NULL, TRUE, NULL);
    SetWaitableTimer(timer, &interval, 0, NULL, NULL, 0);
    const auto ret = WaitForSingleObjectEx(timer, INFINITE, interruptible);
    ::CloseHandle(timer);

    if (remaining) {
        const auto end_sleep = std::chrono::high_resolution_clock::now();
        const auto sleep_time = end_sleep - begin_sleep;
        *remaining = duration > sleep_time ? duration - sleep_time : std::chrono::nanoseconds(0);
    }
    return ret == WAIT_OBJECT_0;
}

#else

void PreferPerformanceCores() {}

void PreferEfficiencyCores() {}

void AllowAllCores() {}

int CurrentCoreIsFast() {
    return -1;
}

void SetCurrentThreadPriority(ThreadPriority new_priority) {
    pthread_t this_thread = pthread_self();

    const auto scheduling_type = SCHED_OTHER;
    s32 max_prio = sched_get_priority_max(scheduling_type);
    s32 min_prio = sched_get_priority_min(scheduling_type);
    u32 level = std::max(static_cast<u32>(new_priority) + 1, 4U);

    struct sched_param params;
    if (max_prio > min_prio) {
        params.sched_priority = min_prio + ((max_prio - min_prio) * level) / 4;
    } else {
        params.sched_priority = min_prio - ((min_prio - max_prio) * level) / 4;
    }

    pthread_setschedparam(this_thread, scheduling_type, &params);
}

bool AccurateSleep(const std::chrono::nanoseconds duration, std::chrono::nanoseconds* remaining,
                   const bool interruptible) {
    timespec request = {
        .tv_sec = duration.count() / 1'000'000'000,
        .tv_nsec = duration.count() % 1'000'000'000,
    };
    timespec remain;
    int ret;
    while ((ret = nanosleep(&request, &remain)) < 0 && errno == EINTR) {
        if (interruptible) {
            break;
        }
        request = remain;
    }
    if (remaining) {
        *remaining = std::chrono::nanoseconds(remain.tv_sec * 1'000'000'000 + remain.tv_nsec);
    }
    return ret == 0 || errno != EINTR;
}

#endif

#ifdef _WIN32

// Sets the debugger-visible name of the current thread.
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    SetThreadDescription(GetCurrentThread(), UTF8ToUTF16W(name).data());
}

void SetThreadName(void* thread, const char* name) {
    SetThreadDescription(thread, UTF8ToUTF16W(name).data());
}

#else // !_WIN32, so must be POSIX threads

// MinGW with the POSIX threading model does not support pthread_setname_np
#if !defined(_WIN32) || defined(_MSC_VER)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
#ifdef __APPLE__
    pthread_setname_np(name);
#elif defined(__Bitrig__) || defined(__DragonFly__) || defined(__FreeBSD__) || defined(__OpenBSD__)
    pthread_set_name_np(pthread_self(), name);
#elif defined(__NetBSD__)
    pthread_setname_np(pthread_self(), "%s", (void*)name);
#elif defined(__linux__)
    // Linux limits thread names to 15 characters and will outright reject any
    // attempt to set a longer name with ERANGE.
    std::string truncated(name, std::min(strlen(name), static_cast<std::size_t>(15)));
    if (int e = pthread_setname_np(pthread_self(), truncated.c_str())) {
        errno = e;
        LOG_ERROR(Common, "Failed to set thread name to '{}': {}", truncated, GetLastErrorMsg());
    }
#else
    pthread_setname_np(pthread_self(), name);
#endif
}

void SetThreadName(void* thread, const char* name) {
    // TODO
}
#endif

#if defined(_WIN32)
void SetCurrentThreadName(const char* name) {
    if (Libraries::Kernel::g_curthread) {
        Libraries::Kernel::g_curthread->name = name;
    }
    // Do Nothing on MinGW
}

void SetThreadName(void* thread, const char* name) {
    // Do Nothing on MinGW
}
#endif

#endif

AccurateTimer::AccurateTimer(std::chrono::nanoseconds target_interval)
    : target_interval(target_interval) {}

void AccurateTimer::Start() {
    const auto begin_sleep = std::chrono::high_resolution_clock::now();
    if (total_wait.count() > 0) {
        AccurateSleep(total_wait, nullptr, false);
    }
    start_time = std::chrono::high_resolution_clock::now();
    total_wait -= std::chrono::duration_cast<std::chrono::nanoseconds>(start_time - begin_sleep);
}

void AccurateTimer::End() {
    auto now = std::chrono::high_resolution_clock::now();
    total_wait +=
        target_interval - std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time);
}

std::string GetCurrentThreadName() {
    using namespace Libraries::Kernel;
    if (g_curthread && !g_curthread->name.empty()) {
        if (g_curthread->tcb->tcb_fiber) {
            return fmt::format("{}@@{}", g_curthread->name,
                               g_curthread->tcb->tcb_fiber->current_fiber->name);
        }
        return g_curthread->name;
    }
#ifdef _WIN32
    PWSTR name{};
    if (FAILED(GetThreadDescription(GetCurrentThread(), &name)) || name == nullptr) {
        return "<unknown name>";
    }
    const auto result = Common::UTF16ToUTF8(name);
    LocalFree(name);
    return result;
#else
    char name[256];
    if (pthread_getname_np(pthread_self(), name, sizeof(name)) != 0) {
        return "<unknown name>";
    }
    return std::string{name};
#endif
}

} // namespace Common
