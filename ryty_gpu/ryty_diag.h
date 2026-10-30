// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// ryGT stutter diagnostics (RYGT_DIAG=1). Timed zones go into per-thread rings (no locks, no
// allocation once a thread's ring exists); each frame the screen clock is checked against the
// recent median, and a spike writes what every thread did during it (zones and stack samples)
// to <exe dir>\diag\<time>\. Ryty's libraries record their zones through the exported
// RyGtDiagRecord. Off by default: one predictable branch per zone.

#pragma once

#include <chrono>
#include <intrin.h>
#include "common/types.h"

namespace RyDiag {

enum class Zone : u32 {
    None = 0,
    // Frame clocks (marks).
    GameFlip = 1,     // the game submitted a frame (flip request)
    FrameRecorded = 2, // the renderer prepared the frame for presentation
    FrameShown = 3,   // the frame was presented
    // Renderer.
    ShaderCompile = 10,
    PipelineBuild = 11,
    PipelineWait = 12,
    WriteFault = 20,
    FaultFlush = 21,
    ReadbackWait = 22,
    Protect = 23,
    Submit = 30,
    GpuWait = 31,
    CommandWait = 32,
    PresentWait = 40, // waiting for a free presentation frame
    Present = 41,     // swapchain present
    // Game threads (recorded by Ryty's libkernel).
    EqueueWait = 50,
    SemaWait = 51,
    EventFlagWait = 52,
    MutexWait = 53,
    CondWait = 54,
    Sleep = 55,
    FileRead = 56,
    CondWoken = 57,   // mark: arg = signalling thread id << 32 | its call site in the executable
    EqueueWoken = 58, // mark: arg = first event's filter << 48 | its ident
    GameSubmit = 59,   // mark: the game submitted command buffers (arg = count)
    GpuEventSent = 60, // mark: the renderer triggered a game event (arg = filter << 48 | ident)
    Count = 64,
};

#ifdef RYTY_DIAG
bool Enabled();
#else
// Other builds of these sources (shadGT itself, Ryty's other libraries) record nothing.
constexpr bool Enabled() {
    return false;
}
#endif

inline u64 Now() {
    return __rdtsc();
}

/// Records a finished zone on this thread.
void Record(Zone zone, u64 start, u64 end, u64 arg = 0);

/// Records a zone that ended now and lasted `elapsed` (for sites that already time themselves).
void RecordElapsed(Zone zone, std::chrono::steady_clock::duration elapsed, u64 arg = 0);

/// Records a mark (a zone with no duration).
inline void Mark(Zone zone, u64 arg = 0) {
    if (Enabled()) {
        const u64 now = Now();
        Record(zone, now, now, arg);
    }
}

/// Called once per presented game frame (the screen clock).
void FrameShown();

/// DIAG-GT7-002: tags the stack samples taken from now on (the renderer's A/B window << 8 | mode).
void SetProfileTag(u32 tag);

/// DIAG-GT7-007: a line of timeline.csv (kind, four times or values, an integer), for each
/// frame's way from the game's submit to the screen. Times are seconds on frames.csv's clock.
void Timeline(char kind, double t0, double t1 = 0.0, double t2 = 0.0, double t3 = 0.0, u64 arg = 0);
/// A steady_clock time point on that clock.
double Seconds(std::chrono::steady_clock::time_point tp);

#ifndef RYTY_DIAG
inline void Record(Zone, u64, u64, u64) {}
inline void RecordElapsed(Zone, std::chrono::steady_clock::duration, u64) {}
inline void FrameShown() {}
inline void SetProfileTag(u32) {}
inline void Timeline(char, double, double, double, double, u64) {}
inline double Seconds(std::chrono::steady_clock::time_point) {
    return 0.0;
}
#endif

class Scope {
public:
    explicit Scope(Zone zone_, u64 arg_ = 0) : zone{zone_}, arg{arg_} {
        if (Enabled()) {
            start = Now();
        }
    }
    ~Scope() {
        if (start) {
            Record(zone, start, Now(), arg);
        }
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

private:
    Zone zone;
    u64 arg;
    u64 start{};
};

} // namespace RyDiag

#define RYDIAG_CONCAT2(x, y) x##y
#define RYDIAG_CONCAT(x, y) RYDIAG_CONCAT2(x, y)
#define RYDIAG_SCOPE(zone, ...)                                                                    \
    ::RyDiag::Scope RYDIAG_CONCAT(rydiag_scope_, __LINE__) {                                       \
        ::RyDiag::Zone::zone __VA_OPT__(, ) __VA_ARGS__                                            \
    }
