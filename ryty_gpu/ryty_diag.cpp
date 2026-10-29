// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fmt/format.h>

#include "common/perf_monitor.h"
#include "ryty_gpu/ryty_diag.h"
#include "video_core/renderdoc.h"

namespace RyDiag {

std::atomic<u32> g_profile_tag{0}; // DIAG-GT7-002

void SetProfileTag(u32 tag) {
    g_profile_tag.store(tag, std::memory_order_relaxed);
}

namespace {

constexpr const char* ZoneName(Zone zone) {
    switch (zone) {
    case Zone::GameFlip: return "game flip";
    case Zone::FrameRecorded: return "frame recorded";
    case Zone::FrameShown: return "frame shown";
    case Zone::ShaderCompile: return "shader compile";
    case Zone::PipelineBuild: return "pipeline build";
    case Zone::PipelineWait: return "draw waits for pipeline";
    case Zone::WriteFault: return "guest fault on GPU memory";
    case Zone::FaultFlush: return "fault flush (GPU->CPU)";
    case Zone::ReadbackWait: return "wait for fault writebacks";
    case Zone::Protect: return "VirtualProtect";
    case Zone::Submit: return "Vulkan submit";
    case Zone::GpuWait: return "CPU waits for GPU";
    case Zone::CommandWait: return "command processor waits";
    case Zone::PresentWait: return "wait for free present frame";
    case Zone::Present: return "swapchain present";
    case Zone::EqueueWait: return "game: event queue wait";
    case Zone::SemaWait: return "game: semaphore wait";
    case Zone::EventFlagWait: return "game: event flag wait";
    case Zone::MutexWait: return "game: mutex wait (contended)";
    case Zone::CondWait: return "game: condition wait";
    case Zone::Sleep: return "game: sleep";
    case Zone::FileRead: return "game: file read";
    case Zone::CondWoken: return "game: woken by a signal";
    case Zone::EqueueWoken: return "game: event received";
    case Zone::GameSubmit: return "game submits GPU work";
    case Zone::GpuEventSent: return "renderer sends an event";
    default: return "?";
    }
}

// Renderer zones: time on the emulator's own path to a frame.
constexpr std::array RendererZones = {
    Zone::ShaderCompile, Zone::PipelineBuild, Zone::PipelineWait, Zone::WriteFault,
    Zone::FaultFlush,    Zone::ReadbackWait,  Zone::Protect,      Zone::Submit,
    Zone::GpuWait,       Zone::CommandWait,   Zone::PresentWait,  Zone::Present,
};
// Game waits: idle worker threads wait all the time, so these only mean something on a thread
// the frame depends on (the flip thread, or a thread that was busy).
constexpr std::array GameZones = {
    Zone::EqueueWait, Zone::SemaWait, Zone::EventFlagWait, Zone::MutexWait,
    Zone::CondWait,   Zone::Sleep,    Zone::FileRead,
};

bool IsGameWait(u32 zone) {
    return zone >= u32(Zone::EqueueWait) && zone <= u32(Zone::Sleep);
}

struct Event {
    u64 start;
    u64 end;
    u64 arg;
    u32 zone;
    u32 pad;
};

constexpr u32 RingSize = 1u << 14;

struct Ring {
    u32 tid{};
    std::atomic<u64> head{};
    std::array<Event, RingSize> events{};
};

// A thread that ran during one sampler tick: its CPU time in the tick and where it was.
struct Sample {
    u64 tsc;
    u64 cycles;
    u64 rip;
    std::array<u64, 5> callers;
    u32 tid;
    u32 pad;
};

constexpr u32 SampleRingSize = 1u << 20;
constexpr auto SamplePeriod = std::chrono::microseconds{2000};

struct Accum {
    std::atomic<u64> ticks{};
    std::atomic<u64> count{};
};

struct ModuleRange {
    u64 base;
    u64 end;
};

struct ThreadTotals {
    u64 cycles{};
    u64 first_tsc{};
    u64 last_tsc{};
};

struct CauseStats {
    u32 count{};
    u32 game_late{};
    double excess_ms{};
    std::vector<u32> examples;
    std::map<std::string, u32> leads;
};

struct State {
    double ticks_per_ms{1.0};
    u64 start_tsc{};
    std::filesystem::path dir;
    std::mutex rings_mutex;
    std::vector<Ring*> rings;
    std::array<Accum, u32(Zone::Count)> accum{};
    // Samples (one writer: the sampler thread).
    std::unique_ptr<Sample[]> samples;
    std::atomic<u64> sample_head{};
    std::mutex threads_mutex;
    std::unordered_map<u32, std::string> thread_names;
    std::unordered_map<u32, ThreadTotals> thread_totals;
    // Frames.
    FILE* frames_csv{};
    std::mutex frames_mutex;
    std::vector<float> frame_ms;
    std::array<double, 120> recent{};
    u32 recent_count{};
    u64 frame_count{};
    u64 last_frame_tsc{};
    u64 prev_frame_tsc{};
    std::atomic<u64> last_flip_tsc{};
    std::array<u64, u32(Zone::Count)> last_accum{};
    std::array<double, u32(Zone::Count)> normal_sum{};
    std::array<double, u32(Zone::Count)> spike_sum{};
    u64 normal_frames{};
    u64 spike_frames{};
    // Spike writer.
    std::mutex spike_mutex;
    std::condition_variable spike_cv;
    struct SpikeRequest {
        u64 frame;
        u64 window_start;
        u64 window_end;
        double ms;
        double median;
    };
    std::deque<SpikeRequest> spike_queue;
    u32 spikes_written{};
    std::map<std::string, CauseStats> causes;
    std::map<std::string, double> thread_excess;
    std::map<std::string, std::pair<u32, double>> fault_sites;
    std::chrono::steady_clock::time_point last_screenshot{};
    FILE* spikes_txt{};
};

State& GetState() {
    static State state;
    return state;
}

thread_local Ring* t_ring{};

double CalibrateTicksPerMs() {
    LARGE_INTEGER freq, a, b;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&a);
    const u64 t0 = __rdtsc();
    Sleep(50);
    QueryPerformanceCounter(&b);
    const u64 t1 = __rdtsc();
    const double ms = double(b.QuadPart - a.QuadPart) * 1000.0 / double(freq.QuadPart);
    return double(t1 - t0) / ms;
}

double TicksToMs(u64 ticks) {
    return double(ticks) / GetState().ticks_per_ms;
}

std::string ReadThreadName(HANDLE h, u32 tid) {
    std::string name = fmt::format("tid {}", tid);
    PWSTR desc = nullptr;
    if (SUCCEEDED(GetThreadDescription(h, &desc)) && desc && *desc) {
        std::string utf8;
        for (const wchar_t* c = desc; *c; ++c) {
            utf8.push_back(*c < 0x80 ? char(*c) : '?');
        }
        name = utf8;
    }
    if (desc) {
        LocalFree(desc);
    }
    return name;
}

std::string ThreadName(u32 tid) {
    auto& s = GetState();
    {
        std::scoped_lock lk{s.threads_mutex};
        if (const auto it = s.thread_names.find(tid); it != s.thread_names.end()) {
            return it->second;
        }
    }
    std::string name = fmt::format("tid {}", tid);
    if (HANDLE h = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid)) {
        name = ReadThreadName(h, tid);
        CloseHandle(h);
    }
    return name;
}

std::string Symbol(u64 address) {
    HMODULE module = nullptr;
    if (address > 0x10000 &&
        GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(address), &module)) {
        wchar_t path[MAX_PATH] = {};
        GetModuleFileNameW(module, path, MAX_PATH);
        std::wstring name{path};
        name = name.substr(name.find_last_of(L"\\/") + 1);
        std::string narrow;
        for (const wchar_t c : name) {
            narrow.push_back(c < 0x80 ? char(c) : '?');
        }
        return fmt::format("{}+{:#x}", narrow, address - reinterpret_cast<u64>(module));
    }
    return fmt::format("{:#x}", address);
}

bool IsSystemFrame(const std::string& symbol) {
    return symbol.starts_with("ntdll") || symbol.starts_with("KERNELBASE") ||
           symbol.starts_with("KERNEL32") || symbol.starts_with("ucrtbase") ||
           symbol.starts_with("libwinpthread");
}

// Reads a stack slot of a suspended thread; its stack can end at a guard page.
bool ReadStack(u64 address, u64& value) {
    __try {
        value = *reinterpret_cast<const volatile u64*>(address);
        return true;
    } __except (EXCEPTION_EXECUTE_HANDLER) {
        return false;
    }
}

// While a thread is suspended nothing here may take a lock it could hold (the loader lock, the
// heap): code addresses are checked against a module table built beforehand.
bool InModules(const std::vector<ModuleRange>& modules, u64 address) {
    auto it = std::upper_bound(modules.begin(), modules.end(), address,
                               [](u64 a, const ModuleRange& m) { return a < m.base; });
    if (it == modules.begin()) {
        return false;
    }
    --it;
    return address < it->end;
}

// The top of a thread's stack (NT_TIB::StackBase in its TEB), so a scan never reads past it:
// Ryty's crash handler sees access violations before any __except here does.
u64 StackBase(HANDLE thread) {
    struct BasicInformation {
        LONG exit_status;
        PVOID teb;
        PVOID client_id[2];
        ULONG_PTR affinity;
        LONG priority;
        LONG base_priority;
    };
    using QueryFn = LONG(NTAPI*)(HANDLE, int, PVOID, ULONG, PULONG);
    static const auto query = reinterpret_cast<QueryFn>(reinterpret_cast<void*>(
        GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationThread")));
    BasicInformation info{};
    if (!query || query(thread, 0, &info, sizeof(info), nullptr) < 0 || !info.teb) {
        return 0;
    }
    return reinterpret_cast<const NT_TIB*>(info.teb)->StackBase
               ? reinterpret_cast<u64>(reinterpret_cast<const NT_TIB*>(info.teb)->StackBase)
               : 0;
}

void SamplerThread() {
    SetThreadDescription(GetCurrentThread(), L"ryGT:DiagSampler");
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    auto& s = GetState();
    const DWORD self = GetCurrentThreadId();
    const DWORD pid = GetCurrentProcessId();
    struct Target {
        DWORD tid;
        HANDLE handle;
        ULONG64 last_cycles;
        u64 stack_base;
        bool skip;
    };
    std::vector<Target> targets;
    std::vector<ModuleRange> modules;
    auto next_refresh = std::chrono::steady_clock::now();
    u64 tick = 0;
    // A thread counts as running in a tick when it used at least 0.1 ms of it.
    const u64 min_cycles = u64(s.ticks_per_ms * 0.1);
    while (true) {
        const auto now = std::chrono::steady_clock::now();
        if (now >= next_refresh) {
            next_refresh = now + std::chrono::seconds{2};
            std::vector<DWORD> live;
            HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
            THREADENTRY32 te{.dwSize = sizeof(te)};
            if (snap != INVALID_HANDLE_VALUE && Thread32First(snap, &te)) {
                do {
                    if (te.th32OwnerProcessID == pid && te.th32ThreadID != self) {
                        live.push_back(te.th32ThreadID);
                    }
                } while (Thread32Next(snap, &te));
            }
            if (snap != INVALID_HANDLE_VALUE) {
                CloseHandle(snap);
            }
            std::erase_if(targets, [&](const Target& t) {
                if (std::find(live.begin(), live.end(), t.tid) != live.end()) {
                    return false;
                }
                CloseHandle(t.handle);
                return true;
            });
            for (const DWORD tid : live) {
                auto it = std::find_if(targets.begin(), targets.end(),
                                       [&](const Target& t) { return t.tid == tid; });
                if (it == targets.end()) {
                    HANDLE h = OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT |
                                              THREAD_QUERY_INFORMATION,
                                          FALSE, tid);
                    if (!h) {
                        continue;
                    }
                    ULONG64 cycles = 0;
                    QueryThreadCycleTime(h, &cycles);
                    targets.push_back({tid, h, cycles, StackBase(h), false});
                    it = targets.end() - 1;
                }
                // Names change after creation (Ryty names guest threads when they start).
                const std::string name = ReadThreadName(it->handle, tid);
                it->skip = name.starts_with("ryGT:Diag");
                std::scoped_lock lk{s.threads_mutex};
                s.thread_names[tid] = name;
            }
            modules.clear();
            snap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE, pid);
            MODULEENTRY32W me{.dwSize = sizeof(me)};
            if (snap != INVALID_HANDLE_VALUE && Module32FirstW(snap, &me)) {
                do {
                    const u64 base = reinterpret_cast<u64>(me.modBaseAddr);
                    modules.push_back({base, base + me.modBaseSize});
                } while (Module32NextW(snap, &me));
            }
            if (snap != INVALID_HANDLE_VALUE) {
                CloseHandle(snap);
            }
            std::sort(modules.begin(), modules.end(),
                      [](const ModuleRange& a, const ModuleRange& b) { return a.base < b.base; });
        }
        const u64 tick_tsc = __rdtsc();
        // CPU time is read every tick; stacks are taken every 4th (stopping a thread costs it).
        const bool stack_tick = (++tick & 3) == 0;
        for (Target& t : targets) {
            ULONG64 cycles = 0;
            if (t.skip || !QueryThreadCycleTime(t.handle, &cycles)) {
                continue;
            }
            const u64 used = cycles - t.last_cycles;
            t.last_cycles = cycles;
            {
                std::scoped_lock lk{s.threads_mutex};
                auto& totals = s.thread_totals[t.tid];
                if (totals.first_tsc == 0) {
                    totals.first_tsc = tick_tsc;
                }
                totals.cycles += used;
                totals.last_tsc = tick_tsc;
            }
            // Only threads that ran since the last tick are stopped for a stack.
            if (used < min_cycles) {
                continue;
            }
            Sample sample{};
            sample.cycles = used;
            sample.tid = t.tid;
            sample.pad = g_profile_tag.load(std::memory_order_relaxed); // DIAG-GT7-002
            const bool stop = stack_tick && t.stack_base && SuspendThread(t.handle) != DWORD(-1);
            CONTEXT ctx{};
            ctx.ContextFlags = CONTEXT_CONTROL | CONTEXT_INTEGER;
            if (stop && GetThreadContext(t.handle, &ctx)) {
                sample.rip = ctx.Rip;
                u32 found = 0;
                const u64 scan_end = std::min<u64>(ctx.Rsp + 0x1000, t.stack_base);
                for (u64 slot = ctx.Rsp; slot + 8 <= scan_end && found < sample.callers.size();
                     slot += 8) {
                    u64 value = 0;
                    if (!ReadStack(slot, value)) {
                        break;
                    }
                    if (value > 0x10000 && InModules(modules, value)) {
                        sample.callers[found++] = value;
                    }
                }
            }
            if (stop) {
                ResumeThread(t.handle);
            }
            {
                sample.tsc = __rdtsc();
                const u64 index = s.sample_head.load(std::memory_order_relaxed);
                s.samples[index % SampleRingSize] = sample;
                s.sample_head.store(index + 1, std::memory_order_release);
            }
        }
        std::this_thread::sleep_until(now + SamplePeriod);
    }
}

// The most telling frame of a sample: the first one outside the C runtime and Windows.
std::string StackKey(const Sample& sm, size_t depth) {
    std::string key = Symbol(sm.rip);
    size_t shown = 0;
    for (const u64 caller : sm.callers) {
        if (!caller || shown >= depth) {
            break;
        }
        key += " < " + Symbol(caller);
        ++shown;
    }
    return key;
}

std::string LeadFrame(const Sample& sm) {
    std::string top = Symbol(sm.rip);
    if (!IsSystemFrame(top)) {
        return top;
    }
    for (const u64 caller : sm.callers) {
        if (!caller) {
            break;
        }
        const std::string sym = Symbol(caller);
        if (!IsSystemFrame(sym)) {
            return top.substr(0, top.find('+')) + " from " + sym;
        }
    }
    return top;
}

void WriteSpike(const State::SpikeRequest& req) {
    auto& s = GetState();
    const u64 ws = req.window_start, we = req.window_end;
    const double window_ms = TicksToMs(we - ws);
    const double excess = req.ms - req.median;
    const u32 number = ++s.spikes_written;

    // Zones of every thread that overlap the window, and the game's flips around it.
    std::map<u32, std::vector<Event>> by_thread;
    // Up to 2 s before the window too, to follow a long wait back to what ended it.
    std::map<u32, std::vector<Event>> wide;
    const u64 wide_start = ws - std::min<u64>(ws, u64(s.ticks_per_ms * 2000));
    std::vector<u64> flips;
    u32 flip_tid = 0;
    {
        std::scoped_lock lk{s.rings_mutex};
        for (const Ring* ring : s.rings) {
            const u64 head = ring->head.load(std::memory_order_acquire);
            const u64 count = std::min<u64>(head, RingSize - 64);
            for (u64 i = head - count; i < head; ++i) {
                const Event e = ring->events[i % RingSize];
                if (e.zone == u32(Zone::GameFlip) && e.end + u64(s.ticks_per_ms * 200) >= ws &&
                    e.end <= we) {
                    flips.push_back(e.end);
                    flip_tid = ring->tid;
                }
                if (e.end >= ws && e.start <= we) {
                    by_thread[ring->tid].push_back(e);
                }
                if (e.end >= wide_start && e.start <= we) {
                    wide[ring->tid].push_back(e);
                }
            }
        }
    }
    std::sort(flips.begin(), flips.end());
    double flip_gap = 0;
    for (size_t i = 1; i < flips.size(); ++i) {
        flip_gap = std::max(flip_gap, TicksToMs(flips[i] - flips[i - 1]));
    }
    if (!flips.empty()) {
        flip_gap = std::max(flip_gap, TicksToMs(we - flips.back()));
    }

    // CPU time per thread in the window, against its usual share.
    struct Busy {
        double ms{};
        std::map<std::string, u32> stacks;
        std::map<std::string, u32> leads;
        u32 samples{};
    };
    std::map<u32, Busy> busy;
    std::vector<const Sample*> window_samples;
    const u64 sample_head = s.sample_head.load(std::memory_order_acquire);
    const u64 sample_count = std::min<u64>(sample_head, SampleRingSize - 1024);
    for (u64 i = sample_head - sample_count; i < sample_head; ++i) {
        const Sample& sm = s.samples[i % SampleRingSize];
        if (sm.tsc < ws || sm.tsc > we) {
            continue;
        }
        auto& b = busy[sm.tid];
        b.ms += TicksToMs(sm.cycles);
        window_samples.push_back(&sm);
        if (sm.rip) {
            ++b.stacks[StackKey(sm, 4)];
            ++b.leads[LeadFrame(sm)];
            ++b.samples;
        }
    }
    struct Excess {
        u32 tid;
        double busy;
        double normal;
    };
    std::vector<Excess> excesses;
    {
        std::scoped_lock lk{s.threads_mutex};
        for (const auto& [tid, b] : busy) {
            double normal = 0;
            if (const auto it = s.thread_totals.find(tid); it != s.thread_totals.end() &&
                                                           it->second.last_tsc > it->second.first_tsc) {
                normal = double(it->second.cycles) /
                         double(it->second.last_tsc - it->second.first_tsc) * window_ms;
            }
            excesses.push_back({tid, b.ms, normal});
        }
    }
    std::sort(excesses.begin(), excesses.end(), [](const Excess& a, const Excess& b) {
        return a.busy - a.normal > b.busy - b.normal;
    });

    // Renderer zones in the window (all threads).
    std::array<double, u32(Zone::Count)> totals{};
    std::array<u32, u32(Zone::Count)> counts{};
    for (const auto& [tid, events] : by_thread) {
        for (const Event& e : events) {
            const u64 a = std::max(e.start, ws), b = std::min(e.end, we);
            totals[e.zone] += b > a ? TicksToMs(b - a) : 0.0;
            ++counts[e.zone];
        }
    }

    // The flip thread's longest wait, and (for a condition wait) the thread that ended it, what
    // that thread did meanwhile, and one step further back.
    const auto longest_wait = [&](u32 tid, u64 from, u64 to) -> const Event* {
        const Event* longest = nullptr;
        if (!wide.contains(tid)) {
            return nullptr;
        }
        for (const Event& e : wide[tid]) {
            if (IsGameWait(e.zone) && e.end >= from && e.start <= to &&
                (!longest || e.end - e.start > longest->end - longest->start)) {
                longest = &e;
            }
        }
        return longest;
    };
    const auto waker_of = [&](u32 tid, const Event& wait) -> u64 {
        u64 waker = 0;
        if (wait.zone != u32(Zone::CondWait) || !wide.contains(tid)) {
            return 0;
        }
        const u64 slack = u64(s.ticks_per_ms * 1.0);
        for (const Event& e : wide[tid]) {
            if (e.zone == u32(Zone::CondWoken) && e.end + slack >= wait.end &&
                e.end <= wait.end + slack && e.end + slack >= wait.start) {
                waker = e.arg;
            }
        }
        return waker;
    };
    std::string waker_text;
    std::string waker_key;
    const Event* flip_wait = flip_tid ? longest_wait(flip_tid, ws, we) : nullptr;
    if (flip_wait) {
        u32 tid = flip_tid;
        const Event* wait = flip_wait;
        for (u32 hop = 0; hop < 3 && wait; ++hop) {
            const u64 waker = waker_of(tid, *wait);
            if (!waker) {
                break;
            }
            const u32 waker_tid = u32(waker >> 32);
            const u64 site = waker & 0xFFFFFFFFull;
            const double ran = busy.contains(waker_tid) ? busy[waker_tid].ms : 0.0;
            waker_text += fmt::format(
                "  {}woken by [{}] (signal from GTSport.exe+{:#x}), which ran {:.1f} ms of {:.1f}",
                hop ? "  that was " : "", ThreadName(waker_tid), site, ran, window_ms);
            if (hop == 0) {
                waker_key = fmt::format(", woken by [{}] from GTSport.exe+{:#x}",
                                        ThreadName(waker_tid), site);
            }
            // What the signalling thread waited on before it signalled.
            const Event* before = longest_wait(waker_tid, wait->start, wait->end);
            if (before) {
                waker_text += fmt::format("; it waited {:.1f} ms ({} at GTSport.exe+{:#x})",
                                          TicksToMs(before->end - before->start),
                                          ZoneName(Zone(before->zone)), before->arg);
                if (before->zone == u32(Zone::EqueueWait) && wide.contains(waker_tid)) {
                    for (const Event& e : wide[waker_tid]) {
                        if (e.zone == u32(Zone::EqueueWoken) && e.end >= before->start &&
                            e.end <= before->end + u64(s.ticks_per_ms)) {
                            waker_text += fmt::format(" for event filter {} ident {:#x}",
                                                      s16(e.arg >> 48), e.arg & 0xFFFFFFFFFFFFull);
                        }
                    }
                }
            }
            waker_text += "\n";
            if (busy.contains(waker_tid)) {
                std::vector<std::pair<std::string, u32>> sorted(busy[waker_tid].stacks.begin(),
                                                                busy[waker_tid].stacks.end());
                std::sort(sorted.begin(), sorted.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
                for (size_t i = 0; i < std::min<size_t>(2, sorted.size()); ++i) {
                    waker_text += fmt::format("      {}/{} samples: {}\n", sorted[i].second,
                                              busy[waker_tid].samples, sorted[i].first);
                }
            }
            tid = waker_tid;
            wait = before;
        }
    }

    // The cause: a renderer zone that covers much of the extra time, else the thread that ran
    // the most beyond its usual share, else the flip thread's longest wait.
    std::string cause;
    std::map<std::string, u32> cause_leads;
    Zone top_zone = Zone::None;
    for (const Zone z : RendererZones) {
        if (z != Zone::Protect && totals[u32(z)] > (top_zone == Zone::None ? 0 : totals[u32(top_zone)])) {
            top_zone = z;
        }
    }
    if (top_zone != Zone::None && totals[u32(top_zone)] >= 0.4 * excess) {
        cause = fmt::format("renderer: {}", ZoneName(top_zone));
    } else if (!excesses.empty() && excesses[0].busy - excesses[0].normal >= 0.4 * excess) {
        cause = fmt::format("busy thread [{}]", ThreadName(excesses[0].tid));
        for (const auto& [lead, n] : busy[excesses[0].tid].leads) {
            cause_leads[lead] += n;
        }
    } else if (flip_tid && by_thread.contains(flip_tid)) {
        const Event* longest = flip_wait;
        cause = longest && TicksToMs(longest->end - longest->start) >= 0.4 * excess
                    ? fmt::format("flip thread waits: {} at GTSport.exe+{:#x}{}",
                                  ZoneName(Zone(longest->zone)), longest->arg, waker_key)
                    : "unexplained (no CPU work or wait stands out: GPU-bound or Windows)";
    } else {
        cause = "unexplained (no CPU work or wait stands out: GPU-bound or Windows)";
    }
    // Game code that faulted on GPU memory in the window.
    std::map<std::string, std::pair<u32, double>> fault_sites;
    for (const auto& [tid, events] : by_thread) {
        for (const Event& e : events) {
            if (e.zone == u32(Zone::WriteFault)) {
                auto& site = fault_sites[fmt::format("{} {}", e.arg >> 63 ? "write" : "read",
                                                     Symbol(e.arg & ~(1ull << 63)))];
                ++site.first;
                site.second += TicksToMs(e.end - e.start);
            }
        }
    }
    auto& stats = s.causes[cause];
    ++stats.count;
    stats.excess_ms += excess;
    stats.game_late += flip_gap >= req.median + 0.5 * excess ? 1 : 0;
    for (const auto& [lead, n] : cause_leads) {
        stats.leads[lead] += n;
    }
    for (const auto& [site, v] : fault_sites) {
        s.fault_sites[site].first += v.first;
        s.fault_sites[site].second += v.second;
    }
    if (stats.examples.size() < 6) {
        stats.examples.push_back(number);
    }
    for (size_t i = 0; i < std::min<size_t>(3, excesses.size()); ++i) {
        const double extra = excesses[i].busy - excesses[i].normal;
        if (extra > 1.0) {
            s.thread_excess[ThreadName(excesses[i].tid)] += extra;
        }
    }

    std::string text = fmt::format(
        "=== spike {}: frame {} took {:.1f} ms (median {:.1f}, +{:.1f}) at {:.2f} s\n"
        "  cause: {}\n"
        "  game flips: {} in the window, longest gap {:.1f} ms ({})\n",
        number, req.frame, req.ms, req.median, excess, TicksToMs(we - s.start_tsc) / 1000.0, cause,
        flips.size(), flip_gap,
        flip_gap >= req.median + 0.5 * excess ? "the game itself was late"
                                              : "the game was on time: the delay is after the flip");
    // DIAG-GT7-006: past 400 spikes only their first lines are written, except for stalls of a
    // second and more (a 6 s freeze late in a mission kept nothing) and every 10th spike (the
    // race after 20 minutes of other ones had no detailed spike), and each entry reaches the file
    // at once (the short ones stayed in the buffer and were lost when the game closed).
    if (number > 400 && req.ms < 1000.0 && number % 10 != 0) {
        std::fputs(text.c_str(), s.spikes_txt);
        std::fflush(s.spikes_txt);
        return;
    }
    std::string zones;
    for (const Zone z : RendererZones) {
        if (totals[u32(z)] >= 0.3) {
            zones += fmt::format(" {} {:.1f} ms x{};", ZoneName(z), totals[u32(z)], counts[u32(z)]);
        }
    }
    text += fmt::format("  renderer zones:{}\n", zones.empty() ? " none over 0.3 ms" : zones);
    if (!fault_sites.empty()) {
        std::vector<std::pair<std::string, std::pair<u32, double>>> sites(fault_sites.begin(),
                                                                          fault_sites.end());
        std::sort(sites.begin(), sites.end(),
                  [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
        text += "  GPU-memory faults by game instruction:";
        for (size_t i = 0; i < std::min<size_t>(4, sites.size()); ++i) {
            text += fmt::format(" {} x{} {:.1f} ms;", sites[i].first, sites[i].second.first,
                                sites[i].second.second);
        }
        text += "\n";
    }
    const auto thread_line = [&](u32 tid, const char* label) {
        std::string line = fmt::format("  {}[{}] ran {:.1f} ms of {:.1f}", label, ThreadName(tid),
                                       busy.contains(tid) ? busy[tid].ms : 0.0, window_ms);
        if (const auto it = std::find_if(excesses.begin(), excesses.end(),
                                         [&](const Excess& e) { return e.tid == tid; });
            it != excesses.end()) {
            line += fmt::format(" (usually {:.1f})", it->normal);
        }
        std::vector<Event> waits;
        if (by_thread.contains(tid)) {
            for (const Event& e : by_thread[tid]) {
                if (IsGameWait(e.zone) || e.zone == u32(Zone::FileRead) ||
                    std::find(RendererZones.begin(), RendererZones.end(), Zone(e.zone)) !=
                        RendererZones.end()) {
                    if (TicksToMs(e.end - e.start) >= 1.0) {
                        waits.push_back(e);
                    }
                }
            }
        }
        std::sort(waits.begin(), waits.end(), [](const Event& a, const Event& b) {
            return a.end - a.start > b.end - b.start;
        });
        for (size_t i = 0; i < std::min<size_t>(3, waits.size()); ++i) {
            line += fmt::format("; {} {:.1f} ms", ZoneName(Zone(waits[i].zone)),
                                TicksToMs(waits[i].end - waits[i].start));
            if (IsGameWait(waits[i].zone) || waits[i].zone == u32(Zone::FileRead)) {
                line += fmt::format(" at GTSport.exe+{:#x}", waits[i].arg);
            }
        }
        text += line + "\n";
        if (busy.contains(tid)) {
            std::vector<std::pair<std::string, u32>> sorted(busy[tid].stacks.begin(),
                                                            busy[tid].stacks.end());
            std::sort(sorted.begin(), sorted.end(),
                      [](const auto& a, const auto& b) { return a.second > b.second; });
            for (size_t i = 0; i < std::min<size_t>(3, sorted.size()); ++i) {
                text += fmt::format("      {}/{} samples: {}\n", sorted[i].second,
                                    busy[tid].samples, sorted[i].first);
            }
        }
    };
    if (flip_tid) {
        thread_line(flip_tid, "flip thread ");
        text += waker_text;
    }
    {
        // When the renderer told the game its GPU work was done (graphics events), against when
        // the game submitted work: a long gap after a submit is the renderer (or GPU) being late.
        std::vector<u64> events;
        std::vector<u64> submits;
        for (const auto& [tid, list] : wide) {
            for (const Event& e : list) {
                if (e.end < ws || e.end > we) {
                    continue;
                }
                if (e.zone == u32(Zone::GpuEventSent) && s16(e.arg >> 48) == -14) {
                    events.push_back(e.end);
                } else if (e.zone == u32(Zone::GameSubmit)) {
                    submits.push_back(e.end);
                }
            }
        }
        std::sort(events.begin(), events.end());
        std::sort(submits.begin(), submits.end());
        u64 gap_from = ws, gap_to = events.empty() ? we : events.front();
        for (size_t i = 1; i < events.size(); ++i) {
            if (events[i] - events[i - 1] > gap_to - gap_from) {
                gap_from = events[i - 1];
                gap_to = events[i];
            }
        }
        if (!events.empty() && we - events.back() > gap_to - gap_from) {
            gap_from = events.back();
            gap_to = we;
        }
        const auto first_submit = std::find_if(submits.begin(), submits.end(),
                                               [&](u64 t) { return t >= gap_from; });
        // Every event the renderer sent (graphics and video out), with its longest gap.
        std::map<u64, std::vector<u64>> by_event;
        for (const auto& [tid, list] : wide) {
            for (const Event& e : list) {
                if (e.zone == u32(Zone::GpuEventSent) && e.end >= ws && e.end <= we) {
                    by_event[e.arg].push_back(e.end);
                }
            }
        }
        std::string sent;
        for (auto& [key, times] : by_event) {
            std::sort(times.begin(), times.end());
            u64 gap = times.front() - ws;
            for (size_t i = 1; i < times.size(); ++i) {
                gap = std::max(gap, times[i] - times[i - 1]);
            }
            gap = std::max(gap, we - times.back());
            sent += fmt::format(" {}/{:#x} x{} (gap {:.0f} ms);", s16(key >> 48),
                                key & 0xFFFFFFFFFFFFull, times.size(), TicksToMs(gap));
        }
        text += fmt::format("  renderer events sent (filter/ident):{}\n",
                            sent.empty() ? " none" : sent);
        text += fmt::format("  GPU-done events to the game: {}, longest gap {:.1f} ms ({:.1f} to "
                            "{:.1f} ms into the window); game submits: {}",
                            events.size(), TicksToMs(gap_to - gap_from), TicksToMs(gap_from - ws),
                            TicksToMs(gap_to - ws), submits.size());
        if (first_submit != submits.end() && *first_submit < gap_to) {
            text += fmt::format(", first one in the gap at {:.1f} ms, {:.1f} ms before the next "
                                "GPU-done event",
                                TicksToMs(*first_submit - ws), TicksToMs(gap_to - *first_submit));
        }
        text += "\n";
    }
    u32 shown = 0;
    for (const Excess& e : excesses) {
        if (e.tid == flip_tid || shown >= 5 || e.busy - e.normal < 1.0) {
            continue;
        }
        thread_line(e.tid, "");
        ++shown;
    }
    std::fputs(text.c_str(), s.spikes_txt);
    std::fflush(s.spikes_txt);

    // A timeline for ui.perfetto.dev (or chrome://tracing) for the first spikes.
    if (number > 80 || window_ms > 2000) {
        return;
    }
    const auto path = s.dir / fmt::format("spike_{:03}.json", number);
    FILE* f = _wfopen(path.c_str(), L"w");
    if (!f) {
        return;
    }
    std::fputs("{\"traceEvents\":[\n", f);
    std::fprintf(f, "{\"name\":\"spike %u: %.1f ms (cause: %s)\",\"ph\":\"i\",\"s\":\"g\",\"pid\":1,"
                    "\"tid\":0,\"ts\":0}",
                 number, req.ms, cause.c_str());
    std::map<u32, bool> named;
    const auto name_thread = [&](u32 tid) {
        if (!named[tid]) {
            named[tid] = true;
            std::string name = ThreadName(tid);
            std::erase(name, '"');
            std::fprintf(f, ",\n{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":%u,"
                            "\"args\":{\"name\":\"%s\"}}",
                         tid, name.c_str());
        }
    };
    u32 written = 0;
    for (const auto& [tid, events] : by_thread) {
        for (const Event& e : events) {
            const bool mark = e.end == e.start;
            if ((!mark && TicksToMs(e.end - e.start) < 0.05) || ++written > 30000) {
                continue;
            }
            name_thread(tid);
            const double ts = (double(e.start) - double(ws)) / s.ticks_per_ms * 1000.0;
            std::fprintf(f,
                         ",\n{\"name\":\"%s\",\"ph\":\"%s\",\"pid\":1,\"tid\":%u,\"ts\":%.1f,"
                         "\"dur\":%.1f,\"args\":{\"arg\":\"0x%llx\"}}",
                         ZoneName(Zone(e.zone)), mark ? "i" : "X", tid, ts,
                         TicksToMs(e.end - e.start) * 1000.0,
                         static_cast<unsigned long long>(e.arg));
        }
    }
    for (const Sample* sm : window_samples) {
        name_thread(sm->tid);
        std::string where = StackKey(*sm, 3);
        std::erase(where, '"');
        const double dur = TicksToMs(sm->cycles) * 1000.0;
        const double ts = (double(sm->tsc) - double(ws)) / s.ticks_per_ms * 1000.0 - dur;
        std::fprintf(f,
                     ",\n{\"name\":\"running\",\"cat\":\"cpu\",\"ph\":\"X\",\"pid\":2,\"tid\":%u,"
                     "\"ts\":%.1f,\"dur\":%.1f,\"args\":{\"at\":\"%s\"}}",
                     sm->tid, ts, dur, where.c_str());
    }
    std::fputs("\n]}\n", f);
    std::fclose(f);
}

// DIAG-GT7-002: where the GPU recorder thread is, from every stack sample since the start, by the
// mode the renderer set (A/B/C: 1/2/3), as module+offset (system frames with their first caller in
// the renderer library). Symbolize offline with the build's PDB.
void WriteRecorderProfile() {
    auto& s = GetState();
    static u64 read_upto = 0;
    static std::map<std::pair<u32, std::string>, u64> counts;
    static std::map<u32, u64> totals;
    u32 recorder = 0;
    {
        std::scoped_lock lk{s.threads_mutex};
        for (const auto& [tid, name] : s.thread_names) {
            if (name == "shadGT:GpuRecorder") {
                recorder = tid;
            }
        }
    }
    const u64 head = s.sample_head.load(std::memory_order_acquire);
    if (head - read_upto > SampleRingSize) {
        read_upto = head - SampleRingSize;
    }
    // Every sample as taken too (window, mode, the frame and the callers found on the stack), for
    // inclusive costs and per-window cuts offline; the renderer's tag is window << 8 | mode.
    static FILE* raw = nullptr;
    if (!raw && recorder != 0) {
        raw = _wfopen((s.dir / "recorder_samples.csv").c_str(), L"w");
        if (raw) {
            std::fputs("window,mode,frames\n", raw);
        }
    }
    for (; read_upto < head; ++read_upto) {
        const Sample sm = s.samples[read_upto % SampleRingSize];
        if (recorder == 0 || sm.tid != recorder || sm.rip == 0) {
            continue;
        }
        const u32 mode = sm.pad & 0xff;
        std::string key = Symbol(sm.rip);
        std::string frames = key;
        bool keyed = key.starts_with("libSceVideoOut");
        for (const u64 caller : sm.callers) {
            if (!caller) {
                break;
            }
            const std::string sym = Symbol(caller);
            frames += " " + sym;
            if (!keyed && sym.starts_with("libSceVideoOut")) {
                key += " < " + sym;
                keyed = true;
            }
        }
        if (raw) {
            std::fprintf(raw, "%u,%u,%s\n", sm.pad >> 8, mode, frames.c_str());
        }
        ++counts[{mode, key}];
        ++totals[mode];
    }
    if (raw) {
        std::fflush(raw);
    }
    std::string text = "ryGT recorder profile (DIAG-GT7-002): stack samples of shadGT:GpuRecorder "
                       "by tag (renderer A/B mode)\n";
    for (const auto& [tag, total] : totals) {
        text += fmt::format("\n== tag {}: {} samples\n", tag, total);
        std::vector<std::pair<u64, std::string>> top;
        for (const auto& [key, n] : counts) {
            if (key.first == tag) {
                top.emplace_back(n, key.second);
            }
        }
        std::sort(top.begin(), top.end(), std::greater{});
        for (size_t i = 0; i < std::min<size_t>(top.size(), 400); ++i) {
            text += fmt::format("{:8} {:6.2f}%  {}\n", top[i].first,
                                100.0 * top[i].first / total, top[i].second);
        }
    }
    if (FILE* f = _wfopen((s.dir / "recorder_profile.txt").c_str(), L"w")) {
        std::fputs(text.c_str(), f);
        std::fclose(f);
    }
}

void WriteSummary() {
    auto& s = GetState();
    WriteRecorderProfile(); // DIAG-GT7-002
    std::vector<float> sorted;
    {
        std::scoped_lock lk{s.frames_mutex};
        sorted = s.frame_ms;
    }
    if (sorted.empty()) {
        return;
    }
    std::sort(sorted.begin(), sorted.end());
    const auto pct = [&](double p) {
        return sorted[std::min<size_t>(sorted.size() - 1, size_t(p * sorted.size()))];
    };
    double total = 0;
    for (const float f : sorted) {
        total += f;
    }
    double lost = 0;
    for (const auto& [cause, stats] : s.causes) {
        lost += stats.excess_ms;
    }
    std::string text = fmt::format(
        "ryGT stutter diagnostics\n"
        "{} frames over {:.1f} s, avg {:.2f} ms ({:.1f} fps)\n"
        "frame time: p50 {:.1f} ms, p90 {:.1f}, p99 {:.1f}, p99.9 {:.1f}, max {:.1f}\n"
        "spikes (over 1.5x the recent median and 8 ms longer): {} ({:.1f} per minute), "
        "{:.1f} s lost to them ({:.0f}% of the run)\n\n",
        sorted.size(), total / 1000.0, total / sorted.size(), 1000.0 * sorted.size() / total,
        pct(0.5), pct(0.9), pct(0.99), pct(0.999), sorted.back(), s.spikes_written,
        s.spikes_written * 60000.0 / total, lost / 1000.0, 100.0 * lost / total);

    text += "causes, by time lost (spike numbers refer to spikes.txt):\n";
    std::vector<std::pair<std::string, CauseStats>> causes(s.causes.begin(), s.causes.end());
    std::sort(causes.begin(), causes.end(),
              [](const auto& a, const auto& b) { return a.second.excess_ms > b.second.excess_ms; });
    for (const auto& [cause, stats] : causes) {
        std::string examples;
        for (const u32 n : stats.examples) {
            examples += fmt::format(" {}", n);
        }
        text += fmt::format("  {:7.0f} ms in {:4} spikes  {}  (game late in {}; e.g.{})\n",
                            stats.excess_ms, stats.count, cause, stats.game_late, examples);
        std::vector<std::pair<std::string, u32>> leads(stats.leads.begin(), stats.leads.end());
        std::sort(leads.begin(), leads.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        for (size_t i = 0; i < std::min<size_t>(3, leads.size()); ++i) {
            text += fmt::format("            {} samples in {}\n", leads[i].second, leads[i].first);
        }
    }

    text += "\nGPU-memory faults during spikes, by game instruction (count, ms in the handler):\n";
    std::vector<std::pair<std::string, std::pair<u32, double>>> sites(s.fault_sites.begin(),
                                                                      s.fault_sites.end());
    std::sort(sites.begin(), sites.end(),
              [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    for (size_t i = 0; i < std::min<size_t>(12, sites.size()); ++i) {
        text += fmt::format("  {:7.0f} ms x{:6}  {}\n", sites[i].second.second,
                            sites[i].second.first, sites[i].first);
    }

    text += "\nthreads that ran longer than usual during spikes (extra CPU ms, summed):\n";
    std::vector<std::pair<std::string, double>> threads(s.thread_excess.begin(),
                                                        s.thread_excess.end());
    std::sort(threads.begin(), threads.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    for (size_t i = 0; i < std::min<size_t>(12, threads.size()); ++i) {
        text += fmt::format("  {:7.0f} ms  {}\n", threads[i].second, threads[i].first);
    }

    text += "\nrenderer time per frame (ms, all threads): normal frames vs spike frames\n";
    for (const Zone z : RendererZones) {
        const double normal = s.normal_frames ? s.normal_sum[u32(z)] / s.normal_frames : 0;
        const double spike = s.spike_frames ? s.spike_sum[u32(z)] / s.spike_frames : 0;
        text += fmt::format("  {:<32} normal {:7.2f}   spike {:7.2f}\n", ZoneName(z), normal, spike);
    }
    text += "\nDetails: spikes.txt (each spike: cause, flip timing, busy threads with stacks, "
            "long waits), spike_NNN.json (timeline, open in ui.perfetto.dev), frames.csv (one row "
            "per frame), spike_frame_N.png (screenshots).\n"
            "Addresses are module+offset; GTSport.exe offsets are the game's own code.\n";
    if (FILE* f = _wfopen((s.dir / "summary.txt").c_str(), L"w")) {
        std::fputs(text.c_str(), f);
        std::fclose(f);
    }
}

void WriterThread() {
    SetThreadDescription(GetCurrentThread(), L"ryGT:DiagWriter");
    auto& s = GetState();
    auto next_summary = std::chrono::steady_clock::now() + std::chrono::seconds{10};
    while (true) {
        std::unique_lock lk{s.spike_mutex};
        s.spike_cv.wait_for(lk, std::chrono::seconds{1});
        while (!s.spike_queue.empty()) {
            const auto req = s.spike_queue.front();
            s.spike_queue.pop_front();
            lk.unlock();
            // Let the window's last zones and samples land: 100 ms after it ended (DIAG-GT7-006:
            // not 100 ms for every queued spike, which left the writer a minute behind in bursts).
            const u64 now_tsc = __rdtsc();
            const double since_ms =
                now_tsc > req.window_end ? TicksToMs(now_tsc - req.window_end) : 0.0;
            if (since_ms < 100.0) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds{static_cast<s64>((100.0 - since_ms) * 1000.0)});
            }
            WriteSpike(req);
            lk.lock();
        }
        lk.unlock();
        if (std::chrono::steady_clock::now() >= next_summary) {
            next_summary = std::chrono::steady_clock::now() + std::chrono::seconds{10};
            WriteSummary();
            if (s.frames_csv) {
                std::fflush(s.frames_csv);
            }
        }
    }
}

void Init(State& s) {
    s.ticks_per_ms = CalibrateTicksPerMs();
    s.start_tsc = __rdtsc();
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::filesystem::path dir = std::filesystem::path{exe}.parent_path() / L"diag";
    SYSTEMTIME t;
    GetLocalTime(&t);
    dir /= fmt::format("{:04}{:02}{:02}-{:02}{:02}{:02}", t.wYear, t.wMonth, t.wDay, t.wHour,
                       t.wMinute, t.wSecond);
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    s.dir = dir;
    s.frames_csv = _wfopen((dir / "frames.csv").c_str(), L"w");
    s.spikes_txt = _wfopen((dir / "spikes.txt").c_str(), L"w");
    if (s.frames_csv) {
        std::string header = "frame,time_s,frame_ms,since_game_flip_ms,spike";
        for (const Zone z : RendererZones) {
            header += fmt::format(",{} ms", ZoneName(z));
        }
        for (const Zone z : GameZones) {
            header += fmt::format(",{} ms", ZoneName(z));
        }
        header += ",draws,dispatches,pipelines,shaders,uploads,upload_kb,protects\n";
        std::fputs(header.c_str(), s.frames_csv);
    }
    s.samples = std::make_unique<Sample[]>(SampleRingSize);
    std::thread{SamplerThread}.detach();
    std::thread{WriterThread}.detach();
}

} // namespace

bool Enabled() {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_DIAG");
        if (!env || env[0] != '1') {
            return false;
        }
        Init(GetState());
        return true;
    }();
    return enabled;
}

void Record(Zone zone, u64 start, u64 end, u64 arg) {
    auto& s = GetState();
    const u32 z = u32(zone) < u32(Zone::Count) ? u32(zone) : 0;
    s.accum[z].ticks.fetch_add(end - start, std::memory_order_relaxed);
    s.accum[z].count.fetch_add(1, std::memory_order_relaxed);
    if (zone == Zone::GameFlip) {
        s.last_flip_tsc.store(end, std::memory_order_relaxed);
    }
    if (zone == Zone::Protect) {
        return; // thousands per second: only counted per frame
    }
    Ring* ring = t_ring;
    if (!ring) {
        ring = new Ring{};
        ring->tid = GetCurrentThreadId();
        {
            std::scoped_lock lk{s.rings_mutex};
            s.rings.push_back(ring);
        }
        t_ring = ring;
    }
    const u64 index = ring->head.load(std::memory_order_relaxed);
    ring->events[index % RingSize] = Event{start, end, arg, z, 0};
    ring->head.store(index + 1, std::memory_order_release);
}

void RecordElapsed(Zone zone, std::chrono::steady_clock::duration elapsed, u64 arg) {
    if (!Enabled()) {
        return;
    }
    const u64 end = __rdtsc();
    const double ms = std::chrono::duration<double, std::milli>(elapsed).count();
    Record(zone, end - u64(ms * GetState().ticks_per_ms), end, arg);
}

void FrameShown() {
    if (!Enabled()) {
        return;
    }
    auto& s = GetState();
    const u64 now = __rdtsc();
    Record(Zone::FrameShown, now, now);
    if (s.last_frame_tsc == 0) {
        s.last_frame_tsc = now;
        s.prev_frame_tsc = now;
        return;
    }
    const double ms = TicksToMs(now - s.last_frame_tsc);
    const u32 n = std::min<u32>(s.recent_count, u32(s.recent.size()));
    std::array<double, 120> sorted = s.recent;
    std::sort(sorted.begin(), sorted.begin() + n);
    const double median = n ? sorted[n / 2] : ms;
    s.recent[s.recent_count++ % s.recent.size()] = ms;
    {
        std::scoped_lock lk{s.frames_mutex};
        s.frame_ms.push_back(float(ms));
    }
    const u64 frame = ++s.frame_count;
    const bool spike = n >= 30 && ms > 1.5 * median && ms - median >= 8.0;

    std::array<double, u32(Zone::Count)> delta{};
    for (u32 z = 0; z < u32(Zone::Count); ++z) {
        const u64 t = s.accum[z].ticks.load(std::memory_order_relaxed);
        delta[z] = TicksToMs(t - s.last_accum[z]);
        s.last_accum[z] = t;
    }
    for (const Zone z : RendererZones) {
        (spike ? s.spike_sum : s.normal_sum)[u32(z)] += delta[u32(z)];
    }
    ++(spike ? s.spike_frames : s.normal_frames);

    static Common::WorkCounters* counters = &Common::GetWorkCounters();
    static u64 last_draws, last_dispatches, last_pipelines, last_shaders, last_uploads,
        last_upload_bytes, last_protects;
    const u64 draws = counters->draws, dispatches = counters->dispatches,
              pipelines = counters->pipelines_compiled, shaders = counters->shaders_compiled,
              uploads = counters->uploads, upload_bytes = counters->upload_bytes,
              protects = counters->protects;
    if (s.frames_csv) {
        const u64 flip = s.last_flip_tsc.load(std::memory_order_relaxed);
        std::string row = fmt::format("{},{:.3f},{:.2f},{:.2f},{}", frame,
                                      TicksToMs(now - s.start_tsc) / 1000.0, ms,
                                      flip ? TicksToMs(now - flip) : -1.0, spike ? 1 : 0);
        for (const Zone z : RendererZones) {
            row += fmt::format(",{:.2f}", delta[u32(z)]);
        }
        for (const Zone z : GameZones) {
            row += fmt::format(",{:.2f}", delta[u32(z)]);
        }
        row += fmt::format(",{},{},{},{},{},{},{}\n", draws - last_draws,
                           dispatches - last_dispatches, pipelines - last_pipelines,
                           shaders - last_shaders, uploads - last_uploads,
                           (upload_bytes - last_upload_bytes) / 1024, protects - last_protects);
        std::fputs(row.c_str(), s.frames_csv);
    }
    last_draws = draws, last_dispatches = dispatches, last_pipelines = pipelines,
    last_shaders = shaders, last_uploads = uploads, last_upload_bytes = upload_bytes,
    last_protects = protects;

    if (spike) {
        {
            std::scoped_lock lk{s.spike_mutex};
            // The window is this frame's interval.
            s.spike_queue.push_back({frame, s.last_frame_tsc, now, ms, median});
        }
        s.spike_cv.notify_one();
        const auto t = std::chrono::steady_clock::now();
        // Opt-in (RYGT_DIAG_SHOTS=1): the PNG is encoded on the GPU's completion thread, which
        // delayed the fault downloads waiting there by up to 100 ms (spikes of their own).
        static const bool shots_on = [] {
            const char* env = std::getenv("RYGT_DIAG_SHOTS");
            return env && env[0] == '1';
        }();
        static u32 shots = 0;
        if (shots_on && shots < 25 && t - s.last_screenshot > std::chrono::seconds{3}) {
            ++shots;
            s.last_screenshot = t;
            VideoCore::RequestScreenshotToPath(s.dir / fmt::format("spike_frame_{}.png", frame));
        }
    }
    s.prev_frame_tsc = s.last_frame_tsc;
    s.last_frame_tsc = now;

    // RYGT_DIAG_TEST_HITCH=ms: busy work on this thread every 600 frames, to check that a spike
    // is caught and blamed on it.
    static const u32 test_hitch = [] {
        const char* env = std::getenv("RYGT_DIAG_TEST_HITCH");
        return env ? u32(std::atoi(env)) : 0u;
    }();
    if (test_hitch && frame % 600 == 0) {
        const u64 until = __rdtsc() + u64(test_hitch * s.ticks_per_ms);
        while (__rdtsc() < until) {
            _mm_pause();
        }
    }
}

} // namespace RyDiag

// For Ryty's libraries (resolved with GetProcAddress on libSceVideoOut.prx).
extern "C" __declspec(dllexport) int RyGtDiagEnabled() {
    return RyDiag::Enabled() ? 1 : 0;
}

extern "C" __declspec(dllexport) void RyGtDiagRecord(u32 zone, u64 start, u64 end, u64 arg) {
    if (RyDiag::Enabled()) {
        RyDiag::Record(RyDiag::Zone(zone), start, end, arg);
    }
}

// The same for zones timed elsewhere in nanoseconds (Ryty's libkernel waits); arg is the guest
// call site as an offset into the game executable.
extern "C" __declspec(dllexport) void RyGtDiagRecordNs(u32 zone, u64 nanos, u64 arg) {
    if (RyDiag::Enabled()) {
        RyDiag::RecordElapsed(RyDiag::Zone(zone), std::chrono::nanoseconds{nanos}, arg);
    }
}
