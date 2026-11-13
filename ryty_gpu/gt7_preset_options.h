// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <string_view>
#include <utility>

// NEO-GT7-014: the five options of GT7's PS4 Pro presets RYGT_NEO_EXTRAS turns on one by one
// (ryty_game_patch.cpp hooks the copy of the preset the game picked).
namespace Gt7PresetOptions {

enum : uint32_t { Blur = 1, Replays60 = 2, ReplayBlur = 4, Garage60 = 8, Lod = 16, All = 31 };

constexpr std::pair<std::string_view, uint32_t> kNames[] = {
    {"blur", Blur},         {"replays60", Replays60}, {"replayblur", ReplayBlur},
    {"garage60", Garage60}, {"lod", Lod},
};

// Names separated by commas, spaces around each ignored. "all" turns the five on; "none" and empty
// items add nothing, they don't reset ("blur,none" is blur). An unknown name goes to on_unknown
// and is ignored. Returns the options on.
template <typename OnUnknown>
uint32_t Parse(std::string_view list, OnUnknown&& on_unknown) {
    uint32_t on = 0;
    size_t start = 0;
    while (start <= list.size()) {
        const size_t end = std::min(list.find(',', start), list.size());
        std::string_view item = list.substr(start, end - start);
        while (!item.empty() && item.front() == ' ') {
            item.remove_prefix(1);
        }
        while (!item.empty() && item.back() == ' ') {
            item.remove_suffix(1);
        }
        bool known = item.empty() || item == "none";
        if (item == "all") {
            on = All;
            known = true;
        }
        for (const auto& [name, bit] : kNames) {
            if (item == name) {
                on |= bit;
                known = true;
            }
        }
        if (!known) {
            on_unknown(item);
        }
        start = end + 1;
    }
    return on;
}

} // namespace Gt7PresetOptions
