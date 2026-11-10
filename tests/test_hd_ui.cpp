// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// GPU.hd_ui: which stored fragment shaders are translated again, and how the output's density
// reaches the pavilion icon caches.

#include <vector>

#include <gtest/gtest.h>

#include "ryty_gpu/ryty_ui_cache_hires.h"
#include "video_core/renderer_vulkan/spirv_check.h"

namespace {

constexpr u32 OpDecorate = 71;
constexpr u32 OpMemberDecorate = 72;

u32 Word(u32 op, u32 count) {
    return (count << 16) | op;
}

// A module header, then the instructions the check looks at.
std::vector<u32> Module(bool frag_coord, bool scale_member) {
    std::vector<u32> words{0x07230203, 0x00010600, 0, 100, 0};
    if (frag_coord) {
        // OpDecorate %10 BuiltIn FragCoord
        words.insert(words.end(), {Word(OpDecorate, 4), 10, 11, 15});
    }
    if (scale_member) {
        // OpMemberDecorate %20 11 Offset 120
        words.insert(words.end(), {Word(OpMemberDecorate, 5), 20, 11, 35, 120});
    }
    // OpMemberDecorate %20 10 Offset 112, the member before it
    words.insert(words.end(), {Word(OpMemberDecorate, 5), 20, 10, 35, 112});
    return words;
}

} // namespace

TEST(HdUiSpirv, FragCoordWithoutScaleIsTranslatedAgain) {
    EXPECT_TRUE(Vulkan::NeedsNativeFragCoordScale(Module(true, false)));
}

TEST(HdUiSpirv, FragCoordWithScaleIsKept) {
    EXPECT_FALSE(Vulkan::NeedsNativeFragCoordScale(Module(true, true)));
}

TEST(HdUiSpirv, ShaderWithoutFragCoordIsKept) {
    EXPECT_FALSE(Vulkan::NeedsNativeFragCoordScale(Module(false, false)));
}

TEST(HdUiSpirv, BrokenWordCountIsKept) {
    auto words = Module(true, false);
    words.push_back(Word(OpDecorate, 9)); // runs past the end
    EXPECT_FALSE(Vulkan::NeedsNativeFragCoordScale(words));
}

TEST(HdUiCache, DensityNeedsTheSetting) {
    Ryty::UiCache::enabled = false;
    Ryty::UiCache::output_scale = 1.f;
    Ryty::UiCache::Output(1920, 1080, 2560, 1440);
    EXPECT_FLOAT_EQ(Ryty::UiCache::output_scale.load(), 1.f);
}

TEST(HdUiCache, DensityIsUniformAndFromOneToTwo) {
    Ryty::UiCache::enabled = true;
    Ryty::UiCache::Output(1920, 1080, 2560, 1440);
    EXPECT_NEAR(Ryty::UiCache::output_scale.load(), 4.f / 3.f, 1e-5f);
    Ryty::UiCache::Output(1920, 1080, 3840, 2160);
    EXPECT_FLOAT_EQ(Ryty::UiCache::output_scale.load(), 2.f);
    // Past 2x it stays at 2.
    Ryty::UiCache::Output(1920, 1080, 7680, 4320);
    EXPECT_FLOAT_EQ(Ryty::UiCache::output_scale.load(), 2.f);
    // A window that stretches the picture is not a scale, and a smaller one is 1.
    Ryty::UiCache::Output(1920, 1080, 2560, 1080);
    EXPECT_FLOAT_EQ(Ryty::UiCache::output_scale.load(), 2.f);
    Ryty::UiCache::Output(1920, 1080, 1280, 720);
    EXPECT_FLOAT_EQ(Ryty::UiCache::output_scale.load(), 1.f);
    Ryty::UiCache::enabled = false;
}
