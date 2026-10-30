// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <memory>
#include "common/alignment.h"
#include "common/types.h"
#include "core/address_space.h"
#include "video_core/buffer_cache/region_definitions.h"

namespace Vulkan {
class Rasterizer;
}

namespace VideoCore {

/// FIX-233: the protection a page gets on the host. x86 has no write-only pages: a page whose
/// reads are watched (GPU-written data, precise readbacks) and whose writes are not becomes
/// no-access; its CPU write faults too and is handled as a read first (PageManager).
constexpr Core::MemoryPermission HostPermission(bool read_watched, bool write_watched) {
    if (read_watched) {
        return Core::MemoryPermission::None;
    }
    return write_watched ? Core::MemoryPermission::Read : Core::MemoryPermission::ReadWrite;
}

struct UffdImpl;
struct SignalImpl;

enum class PageOp : s8 {
    None = 0,
    Track = 1,
    Untrack = -1,
};

class PageManager {
    static constexpr size_t PM_PAGE_BITS = 12;
    static constexpr size_t PM_PAGE_SIZE = 1ULL << PM_PAGE_BITS;

public:
    explicit PageManager(Vulkan::Rasterizer* rasterizer);
    ~PageManager();

    /// Register a range of mapped gpu memory.
    void OnGpuMap(VAddr address, size_t size);

    /// Unregister a range of gpu memory that was unmapped.
    void OnGpuUnmap(VAddr address, size_t size);

    /// Updates watches in the pages touching the specified region.
    void UpdatePageWatchers(VAddr addr, u64 size, PageOp write_op) const;

    /// PERF-105: the native renderer's own write watch of the pages (at most one a page).
    void UpdateNativeWatch(VAddr addr, u64 size, bool watch) const;

    /// Updates watches in the pages touching the inclusive bounds using a mask.
    void UpdatePageWatchersForRegion(VAddr base_addr, const Bounds& bounds,
                                     const RegionBits& write_mask, const RegionBits& read_mask,
                                     PageOp write_op, PageOp read_op) const;

    /// PERF-070: no CPU access to the tracked pages of the range until Unguard (GPU writes
    /// decoded ahead of the recorder). Returns how many pages were guarded.
    u64 Guard(VAddr address, u64 size);
    void Unguard(VAddr address, u64 size);
    bool IsGuarded(VAddr page) const;
    void DropGuards(VAddr page);
#ifdef RYTY_GPU
    /// Ryty: puts back the protections of tracked pages after the game changed the range's.
    void ReapplyProtection(VAddr address, u64 size);
#endif

    /// PERF-063: whether a CPU write to the page goes unnoticed (no write or read watcher, so
    /// the page is neither write- nor read-protected). Any thread; a racy but word-sized read.
    bool IsUnwatched(VAddr page) const;

    /// PERF-GT7-030: whether the native renderer watches the page. Any thread; a racy but
    /// word-sized read.
    bool IsNativeWatched(VAddr page) const;

    /// Returns page aligned address.
    static constexpr VAddr GetPageAddr(VAddr addr) {
        return Common::AlignDown(addr, PM_PAGE_SIZE);
    }

    /// Returns address of the next page.
    static constexpr VAddr GetNextPageAddr(VAddr addr) {
        return Common::AlignUp(addr + 1, PM_PAGE_SIZE);
    }

private:
    friend struct UffdImpl;
    friend struct SignalImpl;
    struct Impl;
    std::unique_ptr<Impl> impl;
};

} // namespace VideoCore
