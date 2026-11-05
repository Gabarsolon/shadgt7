// SPDX-FileCopyrightText: Copyright 2026 RyGT Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <gtest/gtest.h>

#include "ryty_gpu/native/same_texel.h"

using Native::SameTexelImage;
using Native::SameTexelKey;
using Native::SameTexelRun;

namespace {
constexpr VAddr Probe = 0x1081270000;     // the probe cube the run stores (DIAG-GT7-090)
constexpr u64 ProbeSize = 0x400000;
constexpr VAddr Pool = 0x1002800000;      // a probe pool the code samples, never written
constexpr u32 Fetch = 3, Store = 4;       // cs_img3 (texelFetch), cs_img4 (imageStore)

// GT7's probe filter as bound: three sampled pools, the stored cube read and written.
std::array<SameTexelImage, 5> ProbeFilter() {
    std::array<SameTexelImage, 5> images{};
    for (u32 i = 0; i < 3; ++i) {
        images[i] = {.address = Pool + i * 0x1000000, .size = 0x100000};
    }
    const SameTexelImage cube{.address = Probe,
                              .size = ProbeSize,
                              .base_level = 0,
                              .levels = 7,
                              .base_layer = 0,
                              .layers = 2052,
                              .image_mips = 7,
                              .image_layers = 2052};
    images[Fetch] = cube;
    images[Store] = cube;
    images[Store].written = true;
    images[Store].level_per_binding = true; // NVIDIA: one storage view per level
    return images;
}
} // namespace

TEST(SameTexel, KeysTheLevelAndLayerTheViewsReachInThePicture) {
    const auto images = ProbeFilter();
    EXPECT_EQ(SameTexelKey(images, Fetch, Store, 0, 2), u64{2});
    EXPECT_EQ(SameTexelKey(images, Fetch, Store, 3, 2048 + 3), (u64{3} << 32) | (2048 + 3));
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 3, 2052)); // past the view: no write there
    // Other faces and mips: other keys, so the run lets them overlap.
    SameTexelRun run;
    run.Note(Probe, SameTexelKey(images, Fetch, Store, 1, 0), true);
    EXPECT_TRUE(run.IsNew(Probe, ProbeSize, *SameTexelKey(images, Fetch, Store, 1, 1)));
    EXPECT_TRUE(run.IsNew(Probe, ProbeSize, *SameTexelKey(images, Fetch, Store, 2, 0)));
    EXPECT_FALSE(run.IsNew(Probe, ProbeSize, *SameTexelKey(images, Fetch, Store, 1, 0)));
}

// Review point 1: base 0 at mip 1 and base 1 at mip 0 are the same level of the picture.
TEST(SameTexel, TwoViewsOfOneLevelGiveOneKeySoTheSecondDispatchWaits) {
    auto first = ProbeFilter();
    auto second = ProbeFilter();
    second[Fetch].base_level = 1;
    second[Fetch].levels = 6;
    second[Store].base_level = 1;
    second[Store].levels = 6;
    const auto a = SameTexelKey(first, Fetch, Store, 1, 0);
    const auto b = SameTexelKey(second, Fetch, Store, 0, 0);
    ASSERT_TRUE(a && b);
    EXPECT_EQ(*a, *b);
    SameTexelRun run;
    run.Note(Probe, a, true);
    EXPECT_FALSE(run.IsNew(Probe, ProbeSize, *b)); // a barrier
    // The same with layers: base layer 6 at face 0 is base 0 at layer 6.
    auto third = ProbeFilter();
    third[Fetch].base_layer = 6;
    third[Store].base_layer = 6;
    EXPECT_EQ(SameTexelKey(third, Fetch, Store, 1, 0), SameTexelKey(first, Fetch, Store, 1, 6));
}

// Review point 2: a sampled input other than the fetched one meets the stored picture.
TEST(SameTexel, AnotherSampledInputOnTheStoredPictureKeepsTheBarrier) {
    auto images = ProbeFilter();
    images[0] = images[Fetch]; // cs_img0, read with textureLod elsewhere, on the destination
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 1, 0));
    // Or a part of it.
    images = ProbeFilter();
    images[2] = {.address = Probe + 0x10000, .size = 0x1000};
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 1, 0));
    // The fetched descriptor is another picture: the stored one is not read at its texel.
    images = ProbeFilter();
    images[Fetch].address = Pool + 0x4000000;
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 1, 0));
    // Nothing bound to the fetched descriptor.
    images = ProbeFilter();
    images[Fetch].size = 0;
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 1, 0));
}

TEST(SameTexel, ViewsThatReachOtherLevelsOrLayersKeepTheBarrier) {
    auto images = ProbeFilter();
    images[Fetch].base_level = 1;
    images[Fetch].levels = 6;
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 1, 0)); // reads mip 2, stores mip 1
    images = ProbeFilter();
    images[Fetch].base_layer = 6;
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 1, 0));
}

TEST(SameTexel, LevelsPastThePictureAreBoundedAsTheViewsAre) {
    auto images = ProbeFilter();
    images[Store].levels = 8; // the T# names 8 levels, the picture has 7
    images[Fetch].levels = 8;
    // The 8th storage view is the 7th level; the read's view ends there.
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 7, 0));
    EXPECT_EQ(SameTexelKey(images, Fetch, Store, 6, 0), u64{6} << 32);
    // One storage view of all levels (AMD's image load/store with a level).
    images = ProbeFilter();
    images[Store].level_per_binding = false;
    EXPECT_EQ(SameTexelKey(images, Fetch, Store, 6, 1), (u64{6} << 32) | 1);
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 7, 1));
    // A base layer past the picture is its last layer, as View has it.
    images = ProbeFilter();
    images[Fetch].base_layer = images[Store].base_layer = 5000;
    EXPECT_EQ(SameTexelKey(images, Fetch, Store, 0, 0), u64{2051});
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 0, 1));
}

TEST(SameTexel, OneStoredPictureOnly) {
    auto images = ProbeFilter();
    images[1].written = true;
    EXPECT_FALSE(SameTexelKey(images, Fetch, Store, 1, 0));
    images = ProbeFilter();
    EXPECT_FALSE(SameTexelKey(images, Store, Fetch, 1, 0));
    EXPECT_FALSE(SameTexelKey(images, Fetch, 9, 1, 0));
}

// Review point 3: a dispatch whose stored texels aren't known, then one with a key.
TEST(SameTexel, AStoreWithoutAKeyEndsTheRunsExemptions) {
    const auto images = ProbeFilter();
    const auto key = SameTexelKey(images, Fetch, Store, 1, 0);
    ASSERT_TRUE(key);
    SameTexelRun run;
    EXPECT_TRUE(run.IsNew(Probe, ProbeSize, *key));
    run.Note(Probe, std::nullopt, true); // views with other bases: no key, a barrier before it
    EXPECT_FALSE(run.IsNew(Probe, ProbeSize, *key));
    EXPECT_FALSE(run.IsNew(Probe, ProbeSize, *SameTexelKey(images, Fetch, Store, 2, 3)));
    run.Clear(); // the next global barrier
    EXPECT_TRUE(run.IsNew(Probe, ProbeSize, *key));
    // A dispatch that stores nothing leaves the run as it was.
    run.Note(Probe, std::nullopt, false);
    EXPECT_TRUE(run.IsNew(Probe, ProbeSize, *key));
}

// Review 7dc6d3b..fafde1d point 2: the first dispatch stores A and samples B through cs_img0; the
// second stores B. Each has a key, but the second would write B while the first still samples it.
TEST(SameTexel, ADestinationTheRunSampledElsewhereIsNotNew) {
    constexpr VAddr A = Probe;
    constexpr VAddr B = Probe + 0x1000000;
    auto first = ProbeFilter();
    first[0] = {.address = B, .size = ProbeSize};
    auto second = ProbeFilter();
    for (const u32 i : {Fetch, Store}) {
        second[i].address = B;
    }
    second[0] = {.address = Pool + 0x8000000, .size = 0x100000};
    const auto a = SameTexelKey(first, Fetch, Store, 1, 0);
    const auto b = SameTexelKey(second, Fetch, Store, 1, 0);
    ASSERT_TRUE(a && b);
    SameTexelRun run;
    run.Note(A, a, true);
    for (u32 i = 0; i < 3; ++i) {
        run.NoteRead(first[i].address, first[i].size);
    }
    EXPECT_FALSE(run.IsNew(B, ProbeSize, *b)); // a barrier
    EXPECT_FALSE(run.IsNew(B + 0x10000, 0x1000, *b)); // or a part of it
    // The run's own destination, the pools read again: other texels are still new.
    EXPECT_TRUE(run.IsNew(A, ProbeSize, *SameTexelKey(first, Fetch, Store, 2, 0)));
    // Read the same way by every dispatch of a run: kept once.
    for (u32 i = 0; i < 3; ++i) {
        run.NoteRead(first[i].address, first[i].size);
    }
    EXPECT_EQ(run.other_reads.size(), 3u);
    run.Clear();
    EXPECT_TRUE(run.IsNew(B, ProbeSize, *b));
}

// A dispatch of the run that stores nothing (its destination bound empty) still sampled B.
TEST(SameTexel, ReadsOfADispatchWithoutAKeyCountToo) {
    constexpr VAddr B = Probe + 0x1000000;
    auto second = ProbeFilter();
    for (const u32 i : {Fetch, Store}) {
        second[i].address = B;
    }
    const auto b = SameTexelKey(second, Fetch, Store, 0, 0);
    ASSERT_TRUE(b);
    SameTexelRun run;
    run.Note(0, std::nullopt, false);
    run.NoteRead(B, ProbeSize);
    EXPECT_FALSE(run.IsNew(B, ProbeSize, *b));
}
