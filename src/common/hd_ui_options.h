// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdlib>
#include <optional>
#include <string>

#include "common/ryty_config.h"

namespace Common {

/**
 * GPU.hd_ui in the launcher's settings (user\custom_configs\CUSA03220.json), or RYGT_HD_UI=1/0:
 * the game's UI drawn at the window's resolution, with the game's 4K images and sharper
 * pavilion icons. Off by default. Read once; the settings are not watched while the game runs.
 */
inline bool HdUiRequested() {
    static const bool requested = [] {
        if (const char* env = std::getenv("RYGT_HD_UI")) {
            return env[0] == '1';
        }
        const std::optional<std::string> value = RytyGameConfigValue("hd_ui");
        return value && value->starts_with("true");
    }();
    return requested;
}

} // namespace Common
