// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/types.h"

namespace Common {

// Every PERF switch at its default, as a run without SHADGT_DISABLE_PERF (common/perf_monitor.cpp).
bool PerfFeatureEnabled(u32) {
    return true;
}

} // namespace Common
