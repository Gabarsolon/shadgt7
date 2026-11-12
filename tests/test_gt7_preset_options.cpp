// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "ryty_gpu/gt7_preset_options.h"

namespace {

using namespace Gt7PresetOptions;

struct Parsed {
    uint32_t on;
    std::vector<std::string> unknown;
};

Parsed ParseList(std::string_view list) {
    Parsed parsed{};
    parsed.on =
        Parse(list, [&](std::string_view item) { parsed.unknown.emplace_back(item); });
    return parsed;
}

TEST(Gt7PresetOptions, EachNameTurnsItsOptionOn) {
    EXPECT_EQ(ParseList("blur").on, uint32_t{Blur});
    EXPECT_EQ(ParseList("replays60").on, uint32_t{Replays60});
    EXPECT_EQ(ParseList("replayblur").on, uint32_t{ReplayBlur});
    EXPECT_EQ(ParseList("garage60").on, uint32_t{Garage60});
    EXPECT_EQ(ParseList("lod").on, uint32_t{Lod});
    EXPECT_EQ(ParseList("blur,replays60,replayblur,garage60,lod").on, uint32_t{All});
}

TEST(Gt7PresetOptions, AllAndNone) {
    EXPECT_EQ(ParseList("all").on, uint32_t{All});
    EXPECT_EQ(ParseList("none").on, 0u);
    EXPECT_EQ(ParseList("").on, 0u);
    // "none" adds nothing; it does not reset what came before.
    EXPECT_EQ(ParseList("blur,none").on, uint32_t{Blur});
    EXPECT_TRUE(ParseList("all,none").unknown.empty());
}

TEST(Gt7PresetOptions, SpacesAndEmptyItemsAreIgnored) {
    const Parsed parsed = ParseList(" blur , lod ,,");
    EXPECT_EQ(parsed.on, uint32_t{Blur | Lod});
    EXPECT_TRUE(parsed.unknown.empty());
}

TEST(Gt7PresetOptions, UnknownNamesAreReportedAndIgnored) {
    const Parsed parsed = ParseList("blur,fog,LOD");
    EXPECT_EQ(parsed.on, uint32_t{Blur});
    ASSERT_EQ(parsed.unknown.size(), 2u);
    EXPECT_EQ(parsed.unknown[0], "fog");
    EXPECT_EQ(parsed.unknown[1], "LOD");
}

} // namespace
