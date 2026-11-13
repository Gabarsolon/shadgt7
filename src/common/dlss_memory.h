// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace Common::DlssMemory {
inline std::atomic<std::int64_t> pressure_until{};
inline std::int64_t Milliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
inline void RequestTrim() { pressure_until.store(Milliseconds() + 10000); }
inline bool Pressure() { return Milliseconds() < pressure_until.load(); }
} // namespace Common::DlssMemory
