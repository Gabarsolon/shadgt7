// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <mutex>
#include <utility>
#include "common/adaptive_mutex.h"
#include "common/assert.h"
#include "common/debug.h"
#include "common/decoder.h"
#include "common/div_ceil.h"
#include "common/error.h"
#include "common/multi_level_page_table.h"
#include "common/perf_monitor.h"
#include "common/sampling_profiler.h"
#include "common/signal_context.h"
#include "common/thread.h"
#include "core/emulator_settings.h"
#include "core/memory.h"
#include "core/signals.h"
#include "video_core/page_manager.h"
#include "video_core/buffer_cache/proto_import.h"
#include "video_core/renderer_vulkan/vk_rasterizer.h"

#ifndef _WIN64
#include <sys/mman.h>
#include "common/adaptive_mutex.h"
#else
#include <windows.h>
#endif

#ifdef __linux__
#include <thread>
#include <fcntl.h>
#include <linux/userfaultfd.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#else
#include "common/spin_lock.h"
#include "ryty_gpu/ryty_diag.h"
#endif

namespace VideoCore {

// DIAG-196 (diagnosis): what protecting pages costs, reported with DIAG-190.
struct ProtectCost {
    std::atomic<u64> calls{}, lock_ns{}, imported_ns{}, rest_ns{};
};
static ProtectCost g_protect_cost;

/// DIAG-190/196 (diagnosis, RYGT_DIAG_FAULT_COST=1): fault and protection costs are timed.
static bool FaultCostDiag() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_DIAG_FAULT_COST");
        return env && env[0] == '1';
    }();
    return on;
}

struct PageManager::Impl {
    struct PageState {
        u8 num_write_watchers;
        u8 num_read_watchers;
        // PERF-070: guards of GPU writes decoded but not recorded yet (no CPU access at all).
        u8 num_guards;
        // PERF-105: the native renderer watches the page (one of its write watchers).
        u8 native_watch;

        Core::MemoryPermission WritePerm() const noexcept {
            return num_write_watchers == 0 ? Core::MemoryPermission::Write
                                           : Core::MemoryPermission::None;
        }

        Core::MemoryPermission ReadPerm() const noexcept {
            return num_read_watchers == 0 ? Core::MemoryPermission::Read
                                          : Core::MemoryPermission::None;
        }

        Core::MemoryPermission Perms() const noexcept {
            if (num_guards != 0) {
                return Core::MemoryPermission::None;
            }
            return ReadPerm() | WritePerm();
        }

        template <bool is_read>
        u8 GetPage() const {
            if constexpr (is_read) {
                return num_read_watchers;
            } else {
                return num_write_watchers;
            }
        }

        constexpr Core::MemoryPermission Update(PageOp write_op, bool update_write = true,
                                                PageOp read_op = PageOp::None,
                                                bool update_read = false) {
            if (update_read) {
                if (read_op == PageOp::Track) {
                    ASSERT_MSG(num_read_watchers == 0, "Too many watchers");
                } else if (read_op == PageOp::Untrack) {
                    ASSERT_MSG(num_read_watchers > 0, "Not enough watchers");
                }
                num_read_watchers += std::to_underlying(read_op);
            }
            if (update_write) {
                if (write_op == PageOp::Track) {
                    ASSERT_MSG(num_write_watchers < 255, "Too many watchers");
                } else if (write_op == PageOp::Untrack) {
                    ASSERT_MSG(num_write_watchers > 0, "Not enough watchers");
                }
                num_write_watchers += std::to_underlying(write_op);
            }
            return Perms();
        }
    };

    static constexpr size_t ADDRESS_BITS = 40;
    static constexpr size_t NUM_ADDRESS_PAGES = 1ULL << (40 - PM_PAGE_BITS);
    static constexpr size_t NUM_ADDRESS_LOCKS = NUM_ADDRESS_PAGES / NUM_REGION_PAGES;
    inline static Vulkan::Rasterizer* rasterizer;

    Impl() = default;
    virtual ~Impl() = default;

    virtual void OnMap(VAddr address, size_t size) {
        // No-op
        EnsurePages(address, address + size);
    }

    virtual void OnUnmap(VAddr address, size_t size) {
        // No-op
    }

    virtual void Protect(VAddr address, size_t size, Core::MemoryPermission perms) = 0;

    void EnsurePages(VAddr begin, VAddr end) {
        end = std::min(end, VAddr{1} << ADDRESS_BITS) - 1;
        const size_t start_page = begin >> PM_PAGE_BITS;
        const size_t end_page = end >> PM_PAGE_BITS;
        cached_pages.reserve(start_page, end_page);
        locks.reserve(start_page, end_page);
    }

    /// PERF-070: adds (or removes) a guard on each tracked page of the range and applies the
    /// resulting protection. Returns the number of pages guarded.
    u64 Guard(VAddr addr, u64 size, bool add) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);
        u64 guarded = 0;
        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                continue;
            }
            std::scoped_lock lk{locks[page]};
            const auto old_perms = state->Perms();
            if (add) {
                if (state->num_guards == 255) {
                    continue;
                }
                ++state->num_guards;
            } else {
                if (state->num_guards == 0) {
                    continue;
                }
                --state->num_guards;
            }
            ++guarded;
            if (const auto new_perms = state->Perms(); new_perms != old_perms) {
                Protect(page << PM_PAGE_BITS, PM_PAGE_SIZE, new_perms);
            }
        }
        return guarded;
    }

#ifdef RYTY_GPU
    /// Ryty: the game changed the protection of the range (sceKernelMprotect), which replaced
    /// the protections tracking set; tracked pages get them back.
    void ReapplyProtection(VAddr addr, u64 size) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);
        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                continue;
            }
            std::scoped_lock lk{locks[page]};
            if (const auto perms = state->Perms(); perms != Core::MemoryPermission::ReadWrite) {
                Protect(page << PM_PAGE_BITS, PM_PAGE_SIZE, perms);
            }
        }
    }
#endif

    /// PERF-070: every guard on the page goes (an emulator thread must access it).
    void DropGuards(VAddr page_addr) {
        const u64 page = page_addr >> PM_PAGE_BITS;
        PageState* state = cached_pages.find(page);
        if (!state) {
            return;
        }
        std::scoped_lock lk{locks[page]};
        if (state->num_guards == 0) {
            return;
        }
        const auto old_perms = state->Perms();
        state->num_guards = 0;
        if (const auto new_perms = state->Perms(); new_perms != old_perms) {
            Protect(page << PM_PAGE_BITS, PM_PAGE_SIZE, new_perms);
        }
    }

    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);

        Core::MemoryPermission perms{};
        u64 range_begin = page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        // Iterate requested pages
        const u64 aligned_addr = page_start << PM_PAGE_BITS;
        const u64 aligned_end = page_end << PM_PAGE_BITS;
#ifdef RYTY_GPU
        EnsurePages(aligned_addr, aligned_end);
#else
        if (!rasterizer->IsMapped(aligned_addr, aligned_end - aligned_addr)) {
            LOG_WARNING(Render,
                        "Tracking memory region {:#x} - {:#x} which is not fully GPU mapped.",
                        aligned_addr, aligned_end);
            EnsurePages(aligned_addr, aligned_end);
        }
#endif

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                continue;
            }

            locks[page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const auto new_perms = state->Update(write_op);
            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = page;
                    potential_pages = 1;
                }
                // Extend current range up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(page)) {
                lock->unlock();
            }
        }
    }

    /// PERF-105: the native renderer's watch of each page of the range, at most one a page (the
    /// flag in the page's state, under its lock): watching a watched page or unwatching one
    /// that isn't changes nothing. A run of pages is protected at once, its pages locked until
    /// then; no lock is held across ranges, so a write fault on one page never waits for the
    /// renderer to protect a whole region.
    void UpdateNativeWatch(VAddr addr, u64 size, bool watch) {
        const u64 page_start = addr >> PM_PAGE_BITS;
        const u64 page_end = Common::DivCeil(addr + size, PM_PAGE_SIZE);
        EnsurePages(page_start << PM_PAGE_BITS, page_end << PM_PAGE_BITS);
        constexpr u64 MaxRunPages = 1024;
        u64 run_begin = 0;
        u64 run_pages = 0;
        Core::MemoryPermission run_perms{};
        const auto flush = [&] {
            if (run_pages == 0) {
                return;
            }
            Protect(run_begin << PM_PAGE_BITS, run_pages << PM_PAGE_BITS, run_perms);
            for (u64 page = run_begin; page != run_begin + run_pages; ++page) {
                locks.find(page)->unlock();
            }
            run_pages = 0;
        };
        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(page);
            if (!state) {
                flush();
                continue;
            }
            auto& lock = locks[page];
            lock.lock();
            if ((state->native_watch != 0) == watch) {
                lock.unlock();
                flush();
                continue;
            }
            const auto old_perms = state->Perms();
            state->native_watch = watch ? 1 : 0;
            const auto new_perms = state->Update(watch ? PageOp::Track : PageOp::Untrack);
            if (new_perms == old_perms) {
                lock.unlock();
                flush();
                continue;
            }
            if (run_pages != 0 && (new_perms != run_perms || run_begin + run_pages != page ||
                                   run_pages >= MaxRunPages)) {
                flush();
            }
            if (run_pages == 0) {
                run_begin = page;
                run_perms = new_perms;
            }
            ++run_pages;
        }
        flush();
    }

    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) {
        const u64 base_page = base_addr >> PM_PAGE_BITS;
        const u64 page_start = bounds.start_word * PAGES_PER_WORD + bounds.start_page;
        const u64 page_end = bounds.end_word * PAGES_PER_WORD + bounds.end_page + 1;
#ifdef RYTY_GPU
        // Under Ryty, guest memory is mapped without telling the renderer beforehand, so a
        // region may be tracked before its pages exist here. Tracking skipped such pages while a
        // later untrack found them, and the watcher count went below zero.
        EnsurePages((base_page + page_start) << PM_PAGE_BITS, (base_page + page_end) << PM_PAGE_BITS);
#endif

        Core::MemoryPermission perms{};
        u64 range_begin = base_page + page_start;
        u64 range_pages = 0;
        u64 potential_pages = 0;

        const auto release_pending = [&] {
            if (range_pages > 0) {
                Protect(range_begin << PM_PAGE_BITS, range_pages << PM_PAGE_BITS, perms);
                range_pages = 0;
                potential_pages = 0;
            }
        };

        for (u64 page = page_start; page != page_end; ++page) {
            PageState* state = cached_pages.find(base_page + page);
            if (!state) {
                continue;
            }

            locks[base_page + page].lock();

            const auto old_perms = state->Perms();
            if (page == page_start) {
                perms = old_perms;
            }

            // Apply the change to the page state
            const bool update_write = write_op != PageOp::None && write_mask.GetPage(page);
            const bool update_read = read_op != PageOp::None && read_mask.GetPage(page);
            const auto new_perms = state->Update(write_op, update_write, read_op, update_read);

            if (new_perms != perms) [[unlikely]] {
                // If the protection changed add pending (un)protect action
                release_pending();
                perms = new_perms;
            } else if (range_pages != 0) {
                // If the protection did not change, extend the potential range
                ++potential_pages;
            }

            // If the page must be (un)protected
            if (new_perms != old_perms) {
                if (range_pages == 0) {
                    // Start a new potential range
                    range_begin = base_page + page;
                    potential_pages = 1;
                }
                // Extend current rango up to potential range
                range_pages = potential_pages;
            }
        }

        // Add pending (un)protect action
        release_pending();

        for (u64 page = page_start; page != page_end; ++page) {
            if (auto* lock = locks.find(base_page + page)) {
                lock->unlock();
            }
        }
    }

    struct PageTraits {
        using Entry = PageState;
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<PageTraits> cached_pages;
    struct MutexTraits {
#ifdef PTHREAD_ADAPTIVE_MUTEX_INITIALIZER_NP
        using Entry = Common::AdaptiveMutex;
#else
        using Entry = std::mutex;
#endif
        static constexpr size_t ADDRESS_SPACE_BITS = ADDRESS_BITS;
        static constexpr size_t L1_BITS = 16;
        static constexpr size_t PAGE_BITS = PM_PAGE_BITS;
        static constexpr bool NULL_CHECK = false;
    };
    Common::MultiLevelPageTable<MutexTraits> locks;
};

#ifdef __linux__
struct UffdImpl : public PageManager::Impl {
private:
    std::jthread ufd_thread;
    int uffd;

public:
    UffdImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;
        uffd = syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY);
        if (uffd == -1) {
            LOG_ERROR(Common_Memory,
                      "userfaultfd syscall failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("userfaultfd");
        }

        // Request uffdio features from kernel.
        uffdio_api api;
        api.api = UFFD_API;
        api.features = UFFD_FEATURE_THREAD_ID;
        const int ret = ioctl(uffd, UFFDIO_API, &api);
        if (ret != 0) {
            LOG_ERROR(Common_Memory,
                      "uffdio_api call failed: {}, falling back to signal implementation",
                      Common::GetLastErrorMsg());
            throw std::runtime_error("uffdio_api");
        }

        // Create uffd handler thread
        ufd_thread = std::jthread([&](std::stop_token token) { UffdHandler(token); });
    }

    ~UffdImpl() = default;

    void OnMap(VAddr address, size_t size) override {
        PageManager::Impl::OnMap(address, size);
        uffdio_register reg;
        reg.range.start = address;
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_REGISTER, &reg);
        ASSERT_MSG(ret != -1, "Uffdio register failed with error: {}", Common::GetLastErrorMsg());
    }

    void OnUnmap(VAddr address, size_t size) override {
        uffdio_range range;
        range.start = address;
        range.len = size;
        const int ret = ioctl(uffd, UFFDIO_UNREGISTER, &range);
        ASSERT_MSG(ret != -1, "Uffdio unregister failed with error: {}", Common::GetLastErrorMsg());
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        bool allow_write = True(perms & Core::MemoryPermission::Write);
        uffdio_writeprotect wp;
        wp.range.start = address;
        wp.range.len = size;
        wp.mode = allow_write ? UFFDIO_WRITEPROTECT_MODE_DONTWAKE : UFFDIO_WRITEPROTECT_MODE_WP;
        const int ret = ioctl(uffd, UFFDIO_WRITEPROTECT, &wp);
        ASSERT_MSG(ret != -1, "Uffdio writeprotect failed with error: {}",
                   Common::GetLastErrorMsg());
    }

    void UffdHandler(std::stop_token token) {
        Common::SetCurrentThreadName("shadGT:Uffd");

        auto regions = Core::Memory::Instance()->GetAddressSpace().GetUsableRegions();
        for (auto& region : regions) {
            OnMap(region.lower(), region.upper());
        }
        LOG_INFO(Common_Memory, "registered reserved memory with userfaultfd");

        while (!token.stop_requested()) {
            pollfd pollfd;
            pollfd.fd = uffd;
            pollfd.events = POLLIN;

            // Block until the descriptor is ready for data reads.
            const int pollres = poll(&pollfd, 1, -1);
            switch (pollres) {
            case -1:
                perror("Poll userfaultfd");
                continue;
                break;
            case 0:
                continue;
            case 1:
                break;
            default:
                UNREACHABLE_MSG("Unexpected number of descriptors {} out of poll", pollres);
            }

            // We don't want an error condition to have occured.
            ASSERT_MSG(!(pollfd.revents & POLLERR), "POLLERR on userfaultfd");

            // We waited until there is data to read, we don't care about anything else.
            if (!(pollfd.revents & POLLIN)) {
                continue;
            }

            // Read message from kernel.
            uffd_msg msg;
            const int readret = read(uffd, &msg, sizeof(msg));
            if (readret == -1) {
                if (errno == EAGAIN || errno == EWOULDBLOCK) {
                    continue;
                }
                LOG_ERROR(Common_Memory, "Unexpected result of uffd read: {}",
                          Common::GetLastErrorMsg());
                break;
            }
            ASSERT_MSG(readret == sizeof(msg), "Unexpected short read, exiting");
            ASSERT(msg.arg.pagefault.flags & UFFD_PAGEFAULT_FLAG_WP);

            // Notify rasterizer about the fault.
            const VAddr addr = msg.arg.pagefault.address;
            const auto ptid = msg.arg.pagefault.feat.ptid;
            rasterizer->InvalidateMemory(addr, 1, rasterizer->IsGpuThreadTid(ptid));

            // Some calls to InvalidateMemory never reach the UFFDIO_WRITEPROTECT ioctl in
            // ::Protect, therefore we use MODE_DONTWAKE and wake the thread with UFFDIO_WAKE here
            uffdio_range wake;
            wake.start = msg.arg.pagefault.address;
            wake.len = PageManager::PM_PAGE_SIZE;
            const int ret = ioctl(uffd, UFFDIO_WAKE, &wake);
            ASSERT_MSG(ret != -1, "Waking thread {} failed with: {}", ptid,
                       Common::GetLastErrorMsg());
        }
    }
};
#endif // __linux__

struct SignalImpl : public PageManager::Impl {
    SignalImpl(Vulkan::Rasterizer* rasterizer_) : Impl() {
        rasterizer = rasterizer_;

        // Should be called first.
        constexpr auto priority = std::numeric_limits<u32>::min();
        Core::Signals::Instance()->RegisterAccessViolationHandler(GuestFaultSignalHandler,
                                                                  priority);
#ifdef _WIN32
        // FIX-079: the trap after a write-through store; ahead of every other handler.
        g_write_through_rasterizer = rasterizer_;
        AddVectoredExceptionHandler(1, WriteThroughStep);
#endif
    }

    void Protect(VAddr address, size_t size, Core::MemoryPermission perms) override {
        RENDERER_TRACE;
        // DIAG-196: the protections' cost (lock wait, the imported-view check, the rest).
        const bool timed = FaultCostDiag();
        const auto protect_start =
            timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        if (timed) {
            ++g_protect_cost.calls;
        }
#ifdef _WIN32
        // FIX-079: not while a write-through store has a page open (its protection is put back
        // as it was when the store has run). PERF-106: there are none in native mode, and the
        // lock had every write fault of the game's threads wait for the others' protections.
        std::unique_lock<std::recursive_mutex> wt_lock{WriteThroughLock(), std::defer_lock};
        if (rasterizer->WritesThrough()) {
            wt_lock.lock();
        }
#endif
        auto lap = timed ? std::chrono::steady_clock::now() : protect_start;
        if (timed) {
            g_protect_cost.lock_ns += u64((lap - protect_start).count());
        }
        struct Rest {
            bool timed;
            std::chrono::steady_clock::time_point* lap;
            ~Rest() {
                if (timed) {
                    g_protect_cost.rest_ns +=
                        u64((std::chrono::steady_clock::now() - *lap).count());
                }
            }
        } rest{timed, &lap};
        ++Common::GetWorkCounters().protects;
        Common::GetWorkCounters().protect_bytes += size;
        RYDIAG_SCOPE(Protect, address);
        // P2: imported guest views stay writable (the GPU faults on protected imported pages).
        if (VideoCore::g_imported_count.load(std::memory_order_acquire) != 0) {
            bool any_imported = false;
            VideoCore::ForEachNotImported(address, size, [&](VAddr start, u64 length) {
                any_imported |= start != address || length != size;
            });
            if (any_imported) {
                boost::container::small_vector<std::pair<VAddr, u64>, 4> parts;
                VideoCore::ForEachNotImported(address, size, [&](VAddr start, u64 length) {
                    parts.emplace_back(start, length);
                });
                for (const auto& [start, length] : parts) {
                    Protect(start, length, perms);
                }
                return;
            }
        }
        if (timed) {
            const auto now = std::chrono::steady_clock::now();
            g_protect_cost.imported_ns += u64((now - lap).count());
            lap = now;
        }
        auto* memory = Core::Memory::Instance();
        auto& impl = memory->GetAddressSpace();
        ASSERT_MSG(perms != Core::MemoryPermission::Write,
                   "Attempted to protect region as write-only which is not a valid permission");
        // FIX-028: keep the execute permission of guest code. A GPU buffer over module code
        // (GT Sport's dealership bound one over libc) turned its pages non-executable; every
        // instruction fetch then faulted as a "read" of GPU-mapped memory, was reported handled,
        // and the game threads looped on the fault forever (the dealership freeze). On x86 an
        // executable page stays readable, so only CPU writes to code are tracked there.
        // -DisablePerf 37 protects without it.
        static const bool keep_exec = Common::PerfFeatureEnabled(37);
        if (!keep_exec) {
            impl.Protect(address, size, perms);
            return;
        }
        memory->ForEachExecRange(address, size, [&](VAddr start, u64 length, bool exec) {
            if (exec) {
                static std::atomic<u32> logged{};
                if (logged++ < 10) {
                    LOG_WARNING(Render,
                                "FIX-028: GPU tracking of guest code at {:#x}+{:#x} keeps it "
                                "executable",
                                start, length);
                }
            }
            impl.Protect(start, length, exec ? perms | Core::MemoryPermission::Execute : perms);
        });
    }

    /// PERF-011b: bytes written by the faulting instruction when they are known exactly: one
    /// explicit memory destination, no REP prefix or string operation, and starting at the fault
    /// address (an access cannot have begun on the previous page). 0 when unknown.
    static u64 ExactWriteSize(void* context, VAddr addr) {
        ZydisDecodedInstruction instruction;
        ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
        const auto status = Common::Decoder::Instance()->decodeInstruction(instruction, operands,
                                                                           Common::GetRip(context));
        if (!ZYAN_SUCCESS(status) ||
            (instruction.attributes &
             (ZYDIS_ATTRIB_HAS_REP | ZYDIS_ATTRIB_HAS_REPE | ZYDIS_ATTRIB_HAS_REPNE)) != 0) {
            return 0;
        }
        switch (instruction.mnemonic) {
        case ZYDIS_MNEMONIC_MOVSB:
        case ZYDIS_MNEMONIC_MOVSW:
        case ZYDIS_MNEMONIC_MOVSD:
        case ZYDIS_MNEMONIC_MOVSQ:
        case ZYDIS_MNEMONIC_STOSB:
        case ZYDIS_MNEMONIC_STOSW:
        case ZYDIS_MNEMONIC_STOSD:
        case ZYDIS_MNEMONIC_STOSQ:
            return 0;
        default:
            break;
        }
        u64 size = 0;
        for (u32 i = 0; i < instruction.operand_count_visible; ++i) {
            const auto& operand = operands[i];
            if (operand.type != ZYDIS_OPERAND_TYPE_MEMORY ||
                (operand.actions & ZYDIS_OPERAND_ACTION_MASK_WRITE) == 0) {
                continue;
            }
            if (size != 0 || operand.size == 0 || operand.size % 8 != 0) {
                return 0;
            }
            size = operand.size / 8;
        }
        if (size == 0 || (addr & 0xFFFULL) < size) {
            return 0;
        }
        return size;
    }

#ifdef _WIN32
    /// FIX-079: a write fault whose store touches no GPU-written byte of the page runs with the
    /// page open for that one instruction (trap flag), then the page is protected again and the
    /// bytes it changed go to the GPU copy. Nothing waits for the GPU.
    struct WriteThroughState {
        bool active{};
        VAddr page{};
        DWORD old_protect{};
        VAddr store{};
        u64 store_size{};
        std::array<u8, 4096> before{};
    };
    static WriteThroughState& WriteThrough() {
        static thread_local WriteThroughState state;
        return state;
    }

    static bool TryWriteThrough(void* context, VAddr addr) {
        auto& state = WriteThrough();
        if (state.active || !rasterizer->WritesThrough()) {
            return false; // PERF-106: no instruction decode when it can't go through anyway
        }
        // One explicit memory operand, the destination: no second access can fault meanwhile.
        ZydisDecodedInstruction instruction;
        ZydisDecodedOperand operands[ZYDIS_MAX_OPERAND_COUNT];
        if (!ZYAN_SUCCESS(Common::Decoder::Instance()->decodeInstruction(
                instruction, operands, Common::GetRip(context)))) {
            return false;
        }
        u32 memory_operands = 0;
        for (u32 i = 0; i < instruction.operand_count; ++i) {
            memory_operands += operands[i].type == ZYDIS_OPERAND_TYPE_MEMORY ? 1 : 0;
        }
        const u64 size = ExactWriteSize(context, addr);
        if (memory_operands != 1 || size == 0 || !rasterizer->CanWriteThrough(addr, size)) {
            return false;
        }
        const VAddr page = PageManager::GetPageAddr(addr);
        auto& lock = WriteThroughLock();
        lock.lock();
        MEMORY_BASIC_INFORMATION mbi{};
        const bool readable =
            VirtualQuery(reinterpret_cast<void*>(page), &mbi, sizeof(mbi)) == sizeof(mbi) &&
            (mbi.Protect == PAGE_READONLY || mbi.Protect == PAGE_EXECUTE_READ);
        DWORD old_protect = 0;
        if (!readable ||
            !VirtualProtect(reinterpret_cast<void*>(page), 4096,
                            mbi.Protect == PAGE_READONLY ? PAGE_READWRITE : PAGE_EXECUTE_READWRITE,
                            &old_protect)) {
            lock.unlock();
            return false;
        }
        std::memcpy(state.before.data(), reinterpret_cast<const void*>(page), 4096);
        state.active = true;
        state.page = page;
        state.old_protect = old_protect;
        state.store = addr;
        state.store_size = size;
        // (The signal context is the EXCEPTION_POINTERS.)
        static_cast<EXCEPTION_POINTERS*>(context)->ContextRecord->EFlags |= 0x100; // trap after the store
        return true;
    }

    static LONG CALLBACK WriteThroughStep(EXCEPTION_POINTERS* pointers) {
        if (pointers->ExceptionRecord->ExceptionCode != EXCEPTION_SINGLE_STEP) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        auto& state = WriteThrough();
        if (!state.active) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        DWORD unused = 0;
        VirtualProtect(reinterpret_cast<void*>(state.page), 4096, state.old_protect, &unused);
        pointers->ContextRecord->EFlags &= ~0x100u;
        state.active = false;
        WriteThroughLock().unlock();
        if (g_write_through_rasterizer) {
            g_write_through_rasterizer->FinishWriteThrough(state.page, state.before, state.store,
                                                           state.store_size);
        }
        return EXCEPTION_CONTINUE_EXECUTION;
    }

    static std::recursive_mutex& WriteThroughLock() {
        static std::recursive_mutex lock;
        return lock;
    }
    static inline Vulkan::Rasterizer* g_write_through_rasterizer{};
#endif

    // DIAG-190: the faults' cost on the threads that take them (every 10 s).
    struct FaultCost {
        std::atomic<u64> faults{}, gpu_faults{}, reads{}, total_ns{}, fence_ns{}, guard_ns{},
            invalidate_ns{};
        std::atomic<s64> since{};
    };
    static FaultCost& Cost() {
        static FaultCost cost;
        return cost;
    }
    static u64 NsSince(std::chrono::steady_clock::time_point start) {
        return u64(std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - start)
                       .count());
    }
    // DIAG-190: faults and handler time by thread (the busiest few are reported).
    struct ThreadFaults {
        std::atomic<u32> tid{};
        std::atomic<u64> faults{}, ns{};
    };
    static std::array<ThreadFaults, 64>& FaultThreads() {
        static std::array<ThreadFaults, 64> threads;
        return threads;
    }
    static void NoteThreadFault(u64 ns) {
#ifdef _WIN32
        const u32 tid = GetCurrentThreadId();
#else
        const u32 tid = 0;
#endif
        auto& threads = FaultThreads();
        for (auto& slot : threads) {
            u32 seen = slot.tid.load(std::memory_order_relaxed);
            if (seen == 0 && slot.tid.compare_exchange_strong(seen, tid)) {
                seen = tid;
            }
            if (seen == tid) {
                ++slot.faults;
                slot.ns += ns;
                return;
            }
        }
    }
    static std::string BusiestFaultThreads() {
        std::vector<std::tuple<u64, u64, u32>> list;
        for (auto& slot : FaultThreads()) {
            if (const u32 tid = slot.tid.load(); tid != 0) {
                list.emplace_back(slot.ns.exchange(0), slot.faults.exchange(0), tid);
            }
        }
        std::ranges::sort(list, std::greater{});
        std::string out;
        for (size_t i = 0; i < std::min<size_t>(list.size(), 5); ++i) {
            const auto& [ns, faults, tid] = list[i];
            std::string name = std::to_string(tid);
#ifdef _WIN32
            if (HANDLE thread = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, FALSE, tid)) {
                PWSTR description = nullptr;
                if (SUCCEEDED(GetThreadDescription(thread, &description)) && description &&
                    *description) {
                    char narrow[64] = {};
                    WideCharToMultiByte(CP_UTF8, 0, description, -1, narrow, sizeof(narrow) - 1,
                                        nullptr, nullptr);
                    name = narrow;
                }
                if (description) {
                    LocalFree(description);
                }
                CloseHandle(thread);
            }
#endif
            out += fmt::format(" {} {} faults {:.1f} ms;", name, faults, ns / 1e6);
        }
        return out;
    }
    static void ReportFaultCost() {
        auto& cost = Cost();
        const s64 now = std::chrono::steady_clock::now().time_since_epoch().count();
        s64 since = cost.since.load(std::memory_order_relaxed);
        if (since == 0) {
            cost.since.compare_exchange_strong(since, now);
            return;
        }
        if (now - since < 10'000'000'000 || !cost.since.compare_exchange_strong(since, now)) {
            return;
        }
        LOG_INFO(Render, "DIAG-190: faults in {:.1f} s: {} ({} on GPU threads, {} reads); in the "
                 "handler {:.1f} ms (early fences {:.1f} ms, guards {:.1f} ms, invalidation "
                 "{:.1f} ms)", (now - since) / 1e9, cost.faults.exchange(0),
                 cost.gpu_faults.exchange(0), cost.reads.exchange(0),
                 cost.total_ns.exchange(0) / 1e6, cost.fence_ns.exchange(0) / 1e6,
                 cost.guard_ns.exchange(0) / 1e6, cost.invalidate_ns.exchange(0) / 1e6);
        LOG_INFO(Render, "DIAG-196: {} protections (all threads): {:.1f} ms waiting for the "
                 "write-through lock, {:.1f} ms checking imported views, {:.1f} ms protecting",
                 g_protect_cost.calls.exchange(0), g_protect_cost.lock_ns.exchange(0) / 1e6,
                 g_protect_cost.imported_ns.exchange(0) / 1e6,
                 g_protect_cost.rest_ns.exchange(0) / 1e6);
        LOG_INFO(Render, "DIAG-190: faults by thread:{}", BusiestFaultThreads());
    }

    static bool GuestFaultSignalHandler(void* context, void* fault_address) {
        const bool timed = FaultCostDiag();
        const auto handler_start =
            timed ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
        struct Total {
            bool timed;
            std::chrono::steady_clock::time_point start;
            ~Total() {
                if (timed) {
                    const u64 ns = NsSince(start);
                    Cost().total_ns += ns;
                    NoteThreadFault(ns);
                    ReportFaultCost();
                }
            }
        } total{timed, handler_start};
        if (timed) {
            ++Cost().faults;
        }
        const auto addr = reinterpret_cast<VAddr>(fault_address);
        const auto size = std::min<u64>(8, PageManager::GetNextPageAddr(addr) - addr);
        // PERF-031: the draw pipe's recorder thread is a GPU thread too; a fault on the command
        // thread lets the recorder finish first, since the handling uses the caches.
        const auto is_gpu_thread = rasterizer->IsGpuThread();
        if (is_gpu_thread) {
            // DIAG-049: which GPU-thread code touched protected guest memory. A fault on the
            // command thread drains the recorder and may wait for the GPU.
            // Per thread, so the recorder's faults do not hide the command thread's.
            static thread_local u32 logged = 0;
            if (logged++ < 48) {
                LOG_WARNING(Render, "DIAG-049 GPU-thread {} fault at {:#x} from {}; stack:{}",
                            Common::IsWriteError(context) ? "write" : "read", addr,
                            Common::SamplingProfiler::DescribeCode(
                                reinterpret_cast<u64>(Common::GetRip(context))),
                            Common::SamplingProfiler::DescribeStack());
            }
        }
        const VAddr fault_page = PageManager::GetPageAddr(addr);
#ifdef _WIN32
        // DIAG-198 (diagnosis, SHADGT_PROFILE=2): game threads are sampled from their first fault
        // on (Ryty makes them, so they are not registered where shadGT makes its own).
        static thread_local bool profiled = false;
        if (!profiled && !is_gpu_thread) {
            profiled = true;
            std::string name = "game";
            PWSTR description = nullptr;
            if (SUCCEEDED(GetThreadDescription(GetCurrentThread(), &description)) && description &&
                *description) {
                char narrow[64] = {};
                WideCharToMultiByte(CP_UTF8, 0, description, -1, narrow, sizeof(narrow) - 1,
                                    nullptr, nullptr);
                name = narrow;
            }
            if (description) {
                LocalFree(description);
            }
            Common::SamplingProfiler::RegisterGuestThread(name.c_str(), nullptr, nullptr);
        }
#endif
        if (timed) {
            Cost().gpu_faults += is_gpu_thread ? 1 : 0;
            Cost().reads += Common::IsWriteError(context) ? 0 : 1;
        }
        if (!is_gpu_thread) {
            // PERF-063: the game was told work is done (an early fence) that the recorder has
            // not recorded yet; if it reads this page, the access waits for it.
            auto start = timed ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
            rasterizer->WaitForEarlyFences(fault_page);
            if (timed) {
                Cost().fence_ns += NsSince(start);
                start = std::chrono::steady_clock::now();
            }
            // PERF-070: a page a decoded GPU write guards opens once the write is recorded.
            rasterizer->WaitForGuard(fault_page);
            if (timed) {
                Cost().guard_ns += NsSince(start);
            }
        } else if (rasterizer->GetPageManager().IsGuarded(fault_page)) {
            // PERF-070: an emulator thread reads or writes a guarded page itself: open it.
            rasterizer->GetPageManager().DropGuards(fault_page);
            rasterizer->NoteGuardDropped();
            return true;
        }
        // PERF-054: the command thread's fault is flushed as urgent work on the recorder,
        // which then holds the caches; otherwise this thread drains the recorder and uses them.
        const bool assume_locks = is_gpu_thread && !rasterizer->CommandThreadFaultIsUrgent();
        if (assume_locks) {
            rasterizer->OnGpuThreadFault();
        }
#ifdef _WIN32
        if (!is_gpu_thread && Common::IsWriteError(context) && TryWriteThrough(context, addr)) {
            return true; // FIX-079
        }
#endif
        // The faulting instruction, with bit 63 set for a write.
        RYDIAG_SCOPE(WriteFault, reinterpret_cast<u64>(Common::GetRip(context)) |
                                     (u64(Common::IsWriteError(context)) << 63));
        if (Common::IsWriteError(context)) {
            const auto start = timed ? std::chrono::steady_clock::now()
                                     : std::chrono::steady_clock::time_point{};
            const bool handled = rasterizer->InvalidateMemory(addr, size, assume_locks,
                                                              ExactWriteSize(context, addr));
            if (timed) {
                Cost().invalidate_ns += NsSince(start);
            }
            return handled;
        } else {
            return rasterizer->ReadMemory(addr, size, assume_locks);
        }
        return false;
    }
};

PageManager::PageManager(Vulkan::Rasterizer* rasterizer_) {
#ifdef __linux__
    if (EmulatorSettings.IsUserfaultfdTracking()) {
        try {
            impl = std::make_unique<UffdImpl>(rasterizer_);
            LOG_INFO(Config, "Memory tracking method: userfaultfd");
            return;
        } catch (const std::runtime_error& e) {
            // if uffd is unsupported, falls back to SignalImpl
        }
    }
    LOG_INFO(Config, "Memory tracking method: signals");
#endif
    impl = std::make_unique<SignalImpl>(rasterizer_);
}

PageManager::~PageManager() = default;

u64 PageManager::Guard(VAddr address, u64 size) {
    return impl->Guard(address, size, true);
}

void PageManager::Unguard(VAddr address, u64 size) {
    impl->Guard(address, size, false);
}

bool PageManager::IsGuarded(VAddr page) const {
    // Host/invalid addresses are outside the guest page table. Fault dispatch also sees them.
    if (page >= (VAddr{1} << Impl::ADDRESS_BITS)) {
        return false;
    }
    const auto* state = impl->cached_pages.find(page >> PM_PAGE_BITS);
    return state && state->num_guards != 0;
}

bool PageManager::IsUnwatched(VAddr page) const {
    const auto* state = impl->cached_pages.find(page >> PM_PAGE_BITS);
    return !state || (state->num_write_watchers == 0 && state->num_read_watchers == 0 &&
                      state->num_guards == 0);
}

#ifdef RYTY_GPU
void PageManager::ReapplyProtection(VAddr address, u64 size) {
    impl->ReapplyProtection(address, size);
}
#endif

void PageManager::DropGuards(VAddr page) {
    impl->DropGuards(page);
}

void PageManager::OnGpuMap(VAddr address, size_t size) {
    impl->OnMap(address, size);
}

void PageManager::OnGpuUnmap(VAddr address, size_t size) {
    impl->OnUnmap(address, size);
}

void PageManager::UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const {
    impl->UpdatePageWatchers(addr, size, write_op);
}

void PageManager::UpdateNativeWatch(VAddr addr, u64 size, bool watch) const {
    impl->UpdateNativeWatch(addr, size, watch);
}

void PageManager::UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                              const RegionBits& write_mask,
                                              const RegionBits& read_mask, PageOp write_op,
                                              PageOp read_op) const {
    impl->UpdatePageWatchersForRegion(base_addr, bounds, write_mask, read_mask, write_op, read_op);
}

} // namespace VideoCore
