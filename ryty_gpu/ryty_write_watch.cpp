// DIAG-067: which code writes a guest address, caught with a hardware data breakpoint. GT Sport's
// race summary finds its tessellation constants replaced by vertex data; this watches the
// constants of a draw after it was decoded and lists every write to them (time, thread, code
// address in GTSport.exe, value) for a few milliseconds.

#include <windows.h>
#include <tlhelp32.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <string>
#include <thread>

#include <fmt/format.h>
#include "common/logging/log.h"

namespace {

struct Hit {
    std::int64_t ns;
    DWORD tid;
    std::uint64_t rip;
    std::uint64_t value;
    std::uint64_t rcx, rdx, rbx, rsi, rdi, r11, rsp, r12, below_rsp_16, below_rsp_8, r14, r15;
    std::uint64_t rax, r8, r9, r10, r13;
};

std::atomic<bool> g_active{false};
std::atomic<bool> g_execute{false}; // an instruction breakpoint instead of a write watch
std::atomic<bool> g_read_write{false}; // DIAG-097: reads trap too
std::atomic<std::uint64_t> g_address{0};
std::array<Hit, 2048> g_hits{};
std::atomic<std::uint32_t> g_num_hits{0};
std::atomic<std::int64_t> g_start_ns{0};
std::atomic<bool> g_ring{false}; // DIAG-193
PVOID g_handler = nullptr;

std::int64_t NowNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

LONG CALLBACK OnException(EXCEPTION_POINTERS* pointers) {
    if (pointers->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    auto* context = pointers->ContextRecord;
    if ((context->Dr6 & 1) == 0) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    context->Dr6 &= ~0xFull;
    if (g_active.load(std::memory_order_acquire)) {
        const std::uint64_t address = g_address.load(std::memory_order_relaxed);
        // DIAG-193: with a ring watch, the newest hits are kept.
        const std::uint32_t count = g_num_hits.fetch_add(1, std::memory_order_relaxed);
        const std::uint32_t index =
            g_ring.load(std::memory_order_relaxed) ? count % g_hits.size() : count;
        if (index < g_hits.size()) {
            // Reading the watched address here would trap again in read/write mode.
            const std::uint64_t value =
                g_read_write.load(std::memory_order_relaxed)
                    ? 0
                    : *reinterpret_cast<const volatile std::uint64_t*>(address);
            g_hits[index] = {NowNs(),      GetCurrentThreadId(),
                             context->Rip, value,
                             context->Rcx, context->Rdx,
                             context->Rbx, context->Rsi,
                             context->Rdi, context->R11,
                             context->Rsp, context->R12,
                             *reinterpret_cast<const volatile std::uint64_t*>(context->Rsp - 16),
                             *reinterpret_cast<const volatile std::uint64_t*>(context->Rsp - 8),
                             context->R14, context->R15,
                             context->Rax, context->R8,
                             context->R9,  context->R10,
                             context->R13};
        }
    }
    if (g_execute.load(std::memory_order_relaxed)) {
        context->EFlags |= 0x10000; // RF: resume past the instruction breakpoint
    }
    return EXCEPTION_CONTINUE_EXECUTION;
}

// Sets (or clears, with address 0) data breakpoint 0 on every other thread of the process.
void SetBreakpoint(std::uint64_t address) {
    const DWORD self = GetCurrentThreadId();
    const DWORD pid = GetCurrentProcessId();
    HANDLE snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    if (snapshot == INVALID_HANDLE_VALUE) {
        return;
    }
    THREADENTRY32 entry{};
    entry.dwSize = sizeof(entry);
    for (BOOL more = Thread32First(snapshot, &entry); more; more = Thread32Next(snapshot, &entry)) {
        if (entry.th32OwnerProcessID != pid || entry.th32ThreadID == self) {
            continue;
        }
        HANDLE thread = OpenThread(THREAD_GET_CONTEXT | THREAD_SET_CONTEXT | THREAD_SUSPEND_RESUME,
                                   FALSE, entry.th32ThreadID);
        if (!thread) {
            continue;
        }
        if (SuspendThread(thread) != static_cast<DWORD>(-1)) {
            CONTEXT context{};
            context.ContextFlags = CONTEXT_DEBUG_REGISTERS;
            if (GetThreadContext(thread, &context)) {
                // Breakpoint 0: enabled locally, on writes, 8 bytes.
                context.Dr7 &= ~((0xFull << 16) | 3ull);
                if (address != 0) {
                    context.Dr0 = address;
                    context.Dr7 |= g_execute      ? 1ull
                                   : g_read_write ? (1ull | (3ull << 16) | (3ull << 18))
                                                  : (1ull | (1ull << 16) | (3ull << 18));
                } else {
                    context.Dr0 = 0;
                }
                SetThreadContext(thread, &context);
            }
            ResumeThread(thread);
        }
        CloseHandle(thread);
    }
    CloseHandle(snapshot);
}

std::string Describe(std::uint64_t rip) {
    HMODULE module = nullptr;
    char name[MAX_PATH] = "?";
    if (GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                               GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCSTR>(rip), &module) &&
        module) {
        GetModuleFileNameA(module, name, sizeof(name));
        std::string path{name};
        if (const auto slash = path.find_last_of("\\/"); slash != std::string::npos) {
            path = path.substr(slash + 1);
        }
        return fmt::format("{}+{:#x}", path, rip - reinterpret_cast<std::uint64_t>(module));
    }
    return fmt::format("{:#x}", rip);
}

std::string ThreadName(DWORD tid) {
    std::string out = std::to_string(tid);
    HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid);
    if (!thread) {
        return out;
    }
    PWSTR description = nullptr;
    if (SUCCEEDED(GetThreadDescription(thread, &description)) && description && *description) {
        char narrow[128] = {};
        WideCharToMultiByte(CP_UTF8, 0, description, -1, narrow, sizeof(narrow) - 1, nullptr,
                            nullptr);
        out += fmt::format(" ({})", narrow);
    }
    if (description) {
        LocalFree(description);
    }
    CloseHandle(thread);
    return out;
}

} // namespace

namespace Vulkan {
// DIAG-193 (diagnosis): the newest hits of the running watch, oldest first, into a file.
void RytyWatchDump(std::string label) {
    const std::uint32_t total = g_num_hits.load();
    const std::uint32_t kept = std::min<std::uint32_t>(total, g_hits.size());
    static int dump_index = 0;
    const std::string path = fmt::format("watch_dump_{}.txt", dump_index++);
    if (FILE* file = std::fopen(path.c_str(), "w")) {
        for (std::uint32_t i = total - kept; i < total; ++i) {
            const auto& hit = g_hits[i % g_hits.size()];
            std::fprintf(file,
                         "+%.2f ms thread %s at %s wrote %016llx (rcx %llx rdx %llx rsi %llx "
                         "rdi %llx r8 %llx)\n",
                         (hit.ns - g_start_ns.load()) / 1e6, ThreadName(hit.tid).c_str(),
                         Describe(hit.rip).c_str(), (unsigned long long)hit.value,
                         (unsigned long long)hit.rcx, (unsigned long long)hit.rdx,
                         (unsigned long long)hit.rsi, (unsigned long long)hit.rdi,
                         (unsigned long long)hit.r8);
        }
        std::fclose(file);
    }
    LOG_WARNING(Render_Vulkan, "DIAG-193: {}: {} writes to {:#x} so far, the newest {} in {}", label,
                total, g_address.load(), kept, path);
}

// Watches address for ms milliseconds from now and logs the writes, unless a watch is running.
void RytyWatchWrites(std::uint64_t address, std::uint32_t ms, std::string label) {
    bool expected = false;
    if (!g_active.compare_exchange_strong(expected, true)) {
        return;
    }
    if (!g_handler) {
        g_handler = AddVectoredExceptionHandler(1, OnException);
    }
    // A label starting with "exec" makes it an instruction breakpoint at address (registers are
    // captured each time that instruction is about to run).
    g_execute.store(label.starts_with("exec"));
    // A label starting with "rw" makes it a read/write watch.
    g_read_write.store(label.starts_with("rw"));
    if (!g_execute) {
        address &= ~7ull;
    }
    g_address.store(address);
    g_ring.store(label.starts_with("ring")); // DIAG-193
    g_num_hits.store(0);
    g_start_ns.store(NowNs());
    const std::uint64_t value = *reinterpret_cast<const volatile std::uint64_t*>(address);
    std::thread([address, ms, label = std::move(label), value] {
        SetBreakpoint(address);
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
        SetBreakpoint(0);
        const std::uint32_t count = std::min<std::uint32_t>(g_num_hits.load(), g_hits.size());
        std::string line;
        for (std::uint32_t i = 0; i < count; ++i) {
            const auto& hit = g_hits[i];
            const auto has = [&](std::uint32_t word) {
                return std::uint32_t(hit.value) == word || std::uint32_t(hit.value >> 32) == word;
            };
            // The vertex stream found over GT Sport's frame data: (x, y, z, 0.932, 1.0105, 0.932).
            const bool vertex_like = has(0x3f815729) || has(0x3f6e90f7) || has(0x3f6ea9b5);
            line += fmt::format(" | +{:.2f} ms thread {} at {} wrote {:016x}{} (rcx {:x} rdx {:x} "
                                "rbx {:x} rsi {:x} rdi {:x} r11 {:x} rsp {:x} r12 {:x} "
                                "[rsp-16] {:016x} [rsp-8] {:016x} r14 {:x} r15 {:x} rax {:x} r8 {:x} "
                                "r9 {:x} r10 {:x} r13 {:x})",
                                (hit.ns - g_start_ns.load()) / 1e6, ThreadName(hit.tid),
                                Describe(hit.rip), hit.value, vertex_like ? " VERTEX-LIKE" : "",
                                hit.rcx, hit.rdx, hit.rbx, hit.rsi, hit.rdi, hit.r11, hit.rsp,
                                hit.r12, hit.below_rsp_16, hit.below_rsp_8, hit.r14, hit.r15,
                                hit.rax, hit.r8, hit.r9, hit.r10, hit.r13);
        }
        if (count > 64) {
            // Long lists go to a file, one hit per line.
            static int file_index = 0;
            const std::string path = fmt::format("watch_hits_{}.txt", file_index++);
            if (FILE* file = std::fopen(path.c_str(), "w")) {
                std::size_t start = 0;
                while (start < line.size()) {
                    const std::size_t next = line.find(" | ", start + 1);
                    const std::size_t end = next == std::string::npos ? line.size() : next;
                    std::fprintf(file, "%s\n", line.substr(start, end - start).c_str());
                    start = end;
                }
                std::fclose(file);
            }
            line = fmt::format(" (written to {})", path);
        }
        LOG_WARNING(Render_Vulkan, "DIAG-067: {} ({:#x}, was {:016x}): {} writes in {} ms{}",
                    label, address, value, g_num_hits.load(), ms, line);
        g_active.store(false, std::memory_order_release);
    }).detach();
}
} // namespace Vulkan
