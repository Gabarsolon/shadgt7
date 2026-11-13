// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once
#include <cstdint>

namespace Vulkan::RytyDlssG::Budget {
using Bytes = std::uint64_t;
inline constexpr Bytes MiB = 1ull << 20;

struct Decision {
    bool allowed;
    const char* reason;
};
// The game's startup cache target is a reclamation target, not a permanent
// driver limit. The real allocation also uses
// VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT.
constexpr Decision Check(Bytes owned, Bytes ui, Bytes extra, Bytes usage, Bytes driver_budget) {
    if (owned > 512 * MiB)
        return {false, "temporal-cap"};
    if (ui > 1024 * MiB || owned > 1024 * MiB - ui)
        return {false, "ui-plus-temporal-cap"};
    if (driver_budget && (usage >= driver_budget || extra > driver_budget - usage))
        return {false, "driver-headroom"};
    return {true, "ready"};
}
// Time-based backoff is temporary, never an SDK/session disable latch.
constexpr bool Retry(std::int64_t now, std::int64_t next) { return now >= next; }
// The SDK viewport must be idle in real time and past its last submitted GPU
// use.
constexpr bool CanRelease(bool live, Bytes last_tick, Bytes completed_tick, std::int64_t idle_ms) {
    return live && last_tick && completed_tick >= last_tick && idle_ms >= 2000;
}
} // namespace Vulkan::RytyDlssG::Budget
