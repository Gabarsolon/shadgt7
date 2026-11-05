// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// Diagnostics in the build (CMake's SHADGT_DIAGNOSTICS: on in Debug and RelWithDebInfo, off in
// Release unless asked for). Without them, a diagnostic environment switch reads as unset, so the
// code it guards is dropped at compile time and Release behaves like a run with none of them set.
// Switches that change what the game shows or how it runs keep std::getenv in every build.

#include <cstdlib>

namespace Common {

#ifdef SHADGT_DIAGNOSTICS
inline constexpr bool DiagnosticsBuilt = true;

/// std::getenv for a diagnostic switch (RYGT_DIAG_*, RYGT_AB, the F1 dumps, ...).
inline const char* DiagEnv(const char* name) {
    return std::getenv(name);
}
#else
inline constexpr bool DiagnosticsBuilt = false;

constexpr const char* DiagEnv(const char*) {
    return nullptr;
}
#endif

} // namespace Common
