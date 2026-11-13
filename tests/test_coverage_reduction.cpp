// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <gtest/gtest.h>

#include <array>
#include <cstdint>

#include "video_core/renderer_vulkan/coverage_reduction.h"

namespace {

using namespace Vulkan::CoverageReduction;

// NEO-GT7-016: the PS4 Pro mode's checkerboard draws (2 raster and depth samples, 1 colour).
constexpr std::array<uint32_t, 1> kOneColour{1};

TEST(CoverageReduction, FewerColourSamplesOnlyWhenAColourTargetHasFewer) {
    EXPECT_TRUE(HasFewerColourSamples(2, kOneColour));
    EXPECT_FALSE(HasFewerColourSamples(1, kOneColour));
    EXPECT_FALSE(HasFewerColourSamples(2, std::array<uint32_t, 2>{2, 2}));
    EXPECT_TRUE(HasFewerColourSamples(4, std::array<uint32_t, 2>{4, 1}));
    // An unused slot (0) is not a colour target with fewer samples.
    EXPECT_FALSE(HasFewerColourSamples(2, std::array<uint32_t, 2>{0, 2}));
    EXPECT_FALSE(HasFewerColourSamples(2, std::span<const uint32_t>{}));
}

TEST(CoverageReduction, ListedNeedsTheSameRasterDepthAndColour) {
    const std::array<Combination, 2> truncate{{{2, 2, 1}, {4, 4, 1 | 2}}};
    EXPECT_TRUE(Listed(truncate, 2, 2, kOneColour));
    EXPECT_FALSE(Listed(truncate, 4, 2, kOneColour)); // raster
    EXPECT_FALSE(Listed(truncate, 2, 1, kOneColour)); // depth
    EXPECT_FALSE(Listed(truncate, 2, 2, std::array<uint32_t, 1>{2})); // colour
    // Masks: each colour target's count must be in the combination's colour mask.
    EXPECT_TRUE(Listed(truncate, 4, 4, std::array<uint32_t, 2>{1, 2}));
    EXPECT_FALSE(Listed(truncate, 4, 4, std::array<uint32_t, 2>{1, 4}));
    // Unused colour slots are skipped.
    EXPECT_TRUE(Listed(truncate, 2, 2, std::array<uint32_t, 2>{0, 1}));
}

TEST(CoverageReduction, RtxTable) {
    // The 20 TRUNCATE combinations an RTX 4070 lists (driver 9b438000): one bit per mask, and
    // only mixed ones (colour fewer than raster).
    constexpr std::array<Combination, 20> rtx{{
        {2, 0, 1},   {2, 2, 1},   {4, 0, 1},   {4, 0, 2},   {4, 4, 1},   {4, 4, 2},   {8, 0, 1},
        {8, 0, 2},   {8, 0, 4},   {8, 8, 1},   {8, 8, 2},   {8, 8, 4},   {16, 0, 1},  {16, 0, 2},
        {16, 0, 4},  {16, 0, 8},  {16, 16, 1}, {16, 16, 2}, {16, 16, 4}, {16, 16, 8},
    }};
    // The checkerboard, one or two colour targets of 1 sample.
    EXPECT_TRUE(Listed(rtx, 2, 2, kOneColour));
    EXPECT_TRUE(Listed(rtx, 2, 2, std::array<uint32_t, 2>{1, 1}));
    // Targets of different counts: no combination describes them, MERGE is kept.
    EXPECT_FALSE(Listed(rtx, 2, 2, std::array<uint32_t, 2>{2, 1}));
    EXPECT_TRUE(Listed(rtx, 2, 0, kOneColour)); // without depth
    // Pipelines that are not mixed never ask (they keep the implicit MERGE).
    EXPECT_FALSE(HasFewerColourSamples(1, kOneColour));
    EXPECT_FALSE(HasFewerColourSamples(2, std::array<uint32_t, 1>{2}));
}

TEST(CoverageReduction, NoDepthOrNoColourMustBeListedAsSuch) {
    const std::array<Combination, 2> truncate{{{2, 0, 1}, {2, 2, 0}}};
    EXPECT_TRUE(Listed(truncate, 2, 0, kOneColour));  // no depth attachment
    EXPECT_FALSE(Listed(truncate, 2, 1, kOneColour)); // a 1-sample depth is not "none"
    EXPECT_TRUE(Listed(truncate, 2, 2, std::array<uint32_t, 1>{0})); // no colour attachment
    EXPECT_TRUE(Listed(truncate, 2, 2, std::span<const uint32_t>{}));
    EXPECT_FALSE(Listed(std::span<const Combination>{}, 2, 2, kOneColour));
}

} // namespace
