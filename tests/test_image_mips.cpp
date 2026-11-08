// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FIX-039: render targets for the micro-tiled mips of a macro-tiled image. The values are the
// ones GT Sport programs for the environment map of the car purchase thumbnail (decoded from a
// diagnostic bundle's PM4 capture, tools/mcp/pm4_draws.py): a 2048x1024 B10G11R11 texture at
// 0x1009bc0000 with 12 mips and tile index 14, whose mips are rendered as separate targets, mips
// 1-4 with tile index 14 and mips 5-11 with tile index 13.

#include <gtest/gtest.h>
#include "core/libraries/kernel/process.h"
#include "video_core/amdgpu/tiling.h"
#include "video_core/renderer_vulkan/liverpool_to_vk.h"
#include "video_core/texture_cache/image_info.h"

// image_info.cpp also holds constructors from guest descriptors that this test does not use.
namespace Libraries::Kernel {
s32 PS4_SYSV_ABI sceKernelIsNeoMode() {
    return 0;
}
} // namespace Libraries::Kernel
namespace Common::Log {
void VLog(Class, Level, const char*, int, const char*, fmt::string_view, fmt::format_args) {}
} // namespace Common::Log
namespace Vulkan::LiverpoolToVK {
const std::array<vk::Format, surface_format_table_size> surface_format_table{};
vk::Format DepthFormat(AmdGpu::DepthBuffer::ZFormat, AmdGpu::DepthBuffer::StencilFormat) {
    return vk::Format::eUndefined;
}
} // namespace Vulkan::LiverpoolToVK
void assert_fail_impl() {
    std::abort();
}
[[noreturn]] void unreachable_impl() {
    std::abort();
}

namespace {
using VideoCore::ImageInfo;

ImageInfo Color2D(VAddr address, u32 width, u32 height, u32 levels, AmdGpu::TileMode tile_mode,
                  vk::Format format = vk::Format::eB10G11R11UfloatPack32, u32 bits = 32) {
    ImageInfo info{};
    info.guest_address = address;
    info.type = AmdGpu::ImageType::Color2D;
    info.resources = {static_cast<u16>(levels), 1};
    info.props.is_block = false;
    info.props.is_pow2 = false;
    info.props.is_tiled = true;
    info.pixel_format = format;
    info.num_bits = bits;
    info.num_samples = 1;
    info.size = {width, height, 1};
    info.pitch = std::max(width, 8u);
    info.tile_mode = tile_mode;
    info.array_mode = AmdGpu::GetArrayMode(tile_mode);
    info.UpdateSize();
    return info;
}

constexpr VAddr EnvMap = 0x1009bc0000;
constexpr auto Macro = AmdGpu::TileMode::Thin2DThin; // tile index 14
constexpr auto Micro = AmdGpu::TileMode::Thin1DThin; // tile index 13

// The render targets the game binds for mips 1-11, as (address, width, height, tile mode).
struct MipTarget {
    VAddr address;
    u32 width;
    u32 height;
    AmdGpu::TileMode tile_mode;
};
constexpr MipTarget Targets[] = {
    {0x100a3c0000, 1024, 512, Macro}, {0x100a5c0000, 512, 256, Macro},
    {0x100a640000, 256, 128, Macro},  {0x100a660000, 128, 64, Macro},
    {0x100a668000, 64, 32, Micro},    {0x100a66a000, 32, 16, Micro},
    {0x100a66a800, 16, 8, Micro},     {0x100a66aa00, 8, 4, Micro},
    {0x100a66ab00, 4, 2, Micro},      {0x100a66ac00, 2, 1, Micro},
    {0x100a66ad00, 1, 1, Micro},
};
} // namespace

TEST(ImageMips, LayoutMatchesTheGameMipAddresses) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    for (u32 mip = 1; mip < 12; ++mip) {
        EXPECT_EQ(texture.guest_address + texture.mips_layout[mip].offset, Targets[mip - 1].address)
            << "mip " << mip;
    }
    // Mips 5 and smaller are micro-tiled, matching the tile index the game uses for them.
    EXPECT_EQ(texture.micro_tiled_mips & 0x1E, 0u);
    EXPECT_EQ(texture.micro_tiled_mips & 0xFE0, 0xFE0u);
}

TEST(ImageMips, RenderTargetsResolveToTheTextureMips) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    for (u32 mip = 1; mip < 12; ++mip) {
        const auto& t = Targets[mip - 1];
        const auto target = Color2D(t.address, t.width, t.height, 1, t.tile_mode);
        EXPECT_EQ(target.MipOf(texture), static_cast<s32>(mip)) << "mip " << mip;
        EXPECT_EQ(target.SliceOf(texture, mip), 0) << "mip " << mip;
    }
}

TEST(ImageMips, MacroTiledLevelsStillNeedTheSameArrayMode) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    // Mip 1 is macro-tiled: a 1D target at its address is a different surface.
    const auto wrong = Color2D(0x100a3c0000, 1024, 512, 1, Micro);
    EXPECT_EQ(wrong.MipOf(texture), -1);
    // A depth (different micro tile mode) target at a micro-tiled mip is not that mip either.
    const auto depth = Color2D(0x100a668000, 64, 32, 1, AmdGpu::TileMode::Depth1DThin);
    EXPECT_EQ(depth.MipOf(texture), -1);
}

// FIX-040: in the bad car thumbnail capture (bundle 20261009-130850), the environment memory at
// 0x1009bc0000 was cached as a 1-level R8G8B8A8 image (uid 8693) when the game rendered and
// sampled it as B10G11R11; in the good one (20261009-130627) it was cached as B10G11R11 (uid
// 1985). Sampling it as the 12-level texture must expand the cached image in both cases.
TEST(ImageMips, FullChainExtendsAnImageCachedUnderAnotherFormat) {
    const auto texture = Color2D(EnvMap, 2048, 1024, 12, Macro);
    const auto cached_rgba8 =
        Color2D(EnvMap, 2048, 1024, 1, Macro, vk::Format::eR8G8B8A8Unorm, 32);
    EXPECT_TRUE(texture.ExtendsAcrossFormat(cached_rgba8));

    // Same format: the ordinary expansion path handles it.
    const auto cached_same = Color2D(EnvMap, 2048, 1024, 1, Macro);
    EXPECT_FALSE(texture.ExtendsAcrossFormat(cached_same));
    // A single level of the other format stays a view (no more resources to add).
    const auto level0 = Color2D(EnvMap, 2048, 1024, 1, Macro);
    EXPECT_FALSE(level0.ExtendsAcrossFormat(cached_rgba8));
    // Different texel size, size or tiling: a different surface, never copied bit for bit.
    const auto cached_rgba16 =
        Color2D(EnvMap, 2048, 1024, 1, Macro, vk::Format::eR16G16B16A16Sfloat, 64);
    EXPECT_FALSE(texture.ExtendsAcrossFormat(cached_rgba16));
    const auto cached_small = Color2D(EnvMap, 512, 512, 1, Macro, vk::Format::eR8G8B8A8Unorm, 32);
    EXPECT_FALSE(texture.ExtendsAcrossFormat(cached_small));
    const auto cached_linear =
        Color2D(EnvMap, 2048, 1024, 1, Micro, vk::Format::eR8G8B8A8Unorm, 32);
    EXPECT_FALSE(texture.ExtendsAcrossFormat(cached_linear));
}

// FIX-GT7-127: GT7's Scapes compress their HDR photo to BC6H on the GPU. The values come from a
// Scapes session (run-20261108-083523, NAT-041 and DIAG-197 lines, frame dump T#s): the scene
// samples a 6024x4016 BC6H texture at 0x1036e00000 (5 levels, pitch 8192, tile index 13, 0x2aa0000
// bytes), and the encoder wrote its levels through R32G32B32A32Uint storage images: 2048x1024 at
// the texture's address, 1024x512 at 0x1038e00000.
namespace {
constexpr VAddr ScapesPhoto = 0x1036e00000;

ImageInfo BlockTexture(VAddr address, u32 width, u32 height, u32 pitch, u32 levels, u32 bits,
                       vk::Format format, bool pow2) {
    ImageInfo info{};
    info.guest_address = address;
    info.type = AmdGpu::ImageType::Color2D;
    info.resources = {static_cast<u16>(levels), 1};
    info.props.is_block = true;
    info.props.is_pow2 = pow2;
    info.props.is_tiled = true;
    info.pixel_format = format;
    info.num_bits = bits;
    info.num_samples = 1;
    info.size = {width, height, 1};
    info.pitch = pitch;
    info.tile_mode = Micro;
    info.array_mode = AmdGpu::GetArrayMode(Micro);
    info.UpdateSize();
    return info;
}

ImageInfo ScapesBc6h() {
    return BlockTexture(ScapesPhoto, 6024, 4016, 8192, 5, 128, vk::Format::eBc6HUfloatBlock, true);
}

ImageInfo BlockTarget(VAddr address, u32 width, u32 height, u32 pitch,
                      vk::Format format = vk::Format::eR32G32B32A32Uint, u32 bits = 128,
                      AmdGpu::TileMode tile_mode = Micro) {
    auto info = Color2D(address, width, height, 1, tile_mode, format, bits);
    info.pitch = pitch;
    info.UpdateSize();
    return info;
}
} // namespace

TEST(ImageMips, ScapesBc6hLayoutMatchesTheGame) {
    const auto texture = ScapesBc6h();
    EXPECT_EQ(texture.guest_size, 0x2aa0000u);
    EXPECT_EQ(texture.mips_layout[1].offset, 0x2000000u);
}

TEST(ImageMips, EncoderTargetsAreTheBlockTextureLevels) {
    const auto texture = ScapesBc6h();
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 2048, 1024, 2048)), 0);
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto + 0x2000000, 1024, 512, 1024)), 1);
    // Only the level's blocks (1506x1004), with the level's pitch: the same surface.
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 1506, 1004, 2048)), 0);
    // BC1 (64-bit blocks) through a 64-bit target.
    const auto bc1 = BlockTexture(0x1000000000, 256, 256, 256, 1, 64, vk::Format::eBc1RgbaUnormBlock,
                                  false);
    EXPECT_EQ(bc1.BlockLevelOf(BlockTarget(0x1000000000, 64, 64, 64, vk::Format::eR32G32Uint, 64)),
              0);
}

TEST(ImageMips, OtherTargetsOverABlockTextureAreNotItsLevels) {
    const auto texture = ScapesBc6h();
    // Not at a level's address.
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto + 0x1000000, 2048, 1024, 2048)), -1);
    // Another row pitch, fewer blocks than the level, another element size, another tiling.
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 1506, 1004, 1506)), -1);
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 1024, 1024, 2048)), -1);
    EXPECT_EQ(texture.BlockLevelOf(
                  BlockTarget(ScapesPhoto, 2048, 1024, 2048, vk::Format::eR32G32Uint, 64)),
              -1);
    EXPECT_EQ(texture.BlockLevelOf(BlockTarget(ScapesPhoto, 2048, 1024, 2048,
                                               vk::Format::eR32G32B32A32Uint, 128, Macro)),
              -1);
    // An uncompressed texture has no block levels.
    const auto plain = Color2D(ScapesPhoto, 2048, 1024, 1, Micro, vk::Format::eR32G32B32A32Uint,
                               128);
    EXPECT_EQ(plain.BlockLevelOf(BlockTarget(ScapesPhoto, 2048, 1024, 2048)), -1);
}

// FIX-GT7-134: GT7's depth of field (Scapes' background blurring, photo 21:15) samples an 8-level
// 690x493 B10G11R11 pyramid (pow2-padded, pitch 1024) whose levels it draws as targets sized to
// their own extent: 690x496, 345x248, 172x124, ... (the log's NAT-041 lines), a little larger than
// the texture's levels (345x246, 172x123, ...).
namespace {
constexpr VAddr DofPyramid = 0x100a230000;
struct DofTarget {
    VAddr address;
    u32 width;
    u32 height;
    u32 pitch;
    AmdGpu::TileMode tile_mode;
};
constexpr DofTarget DofTargets[] = {
    {0x100a230000, 690, 496, 1024, Macro}, {0x100a430000, 345, 248, 512, Macro},
    {0x100a4b0000, 172, 124, 256, Macro},  {0x100a4d0000, 86, 62, 128, Macro},
    {0x100a4d8000, 43, 31, 64, Micro},     {0x100a4da000, 21, 15, 32, Micro},
    {0x100a4da800, 10, 7, 16, Micro},      {0x100a4daa00, 5, 3, 8, Micro},
};
ImageInfo DofTexture(u32 height) {
    auto info = Color2D(DofPyramid, 690, height, 8, Macro);
    info.props.is_pow2 = true;
    info.pitch = 1024;
    info.UpdateSize();
    return info;
}
ImageInfo DofTargetInfo(const DofTarget& t) {
    auto info = Color2D(t.address, t.width, t.height, 1, t.tile_mode);
    info.pitch = t.pitch;
    info.UpdateSize();
    return info;
}
} // namespace

TEST(ImageMips, DepthOfFieldPyramidLevelsAreAtTheGameAddresses) {
    const auto texture = DofTexture(493);
    for (u32 mip = 0; mip < 8; ++mip) {
        EXPECT_EQ(texture.guest_address + texture.mips_layout[mip].offset, DofTargets[mip].address)
            << "mip " << mip;
    }
}

TEST(ImageMips, PaddedTargetsAreTheDepthOfFieldLevels) {
    for (const u32 height : {493u, 479u, 496u}) {
        const auto texture = DofTexture(height);
        for (u32 mip = 0; mip < 8; ++mip) {
            const auto target = DofTargetInfo(DofTargets[mip]);
            EXPECT_EQ(target.MipOf(texture, true), static_cast<s32>(mip))
                << "height " << height << " mip " << mip;
            EXPECT_EQ(target.SliceOf(texture, mip, true), 0)
                << "height " << height << " mip " << mip;
        }
    }
    // Without the padding rule, the larger level targets are not levels (as before).
    const auto texture = DofTexture(493);
    EXPECT_EQ(DofTargetInfo(DofTargets[1]).MipOf(texture), -1);
}

// FIX-GT7-135: the Scapes salt flat's backdrop, a 6024x4016 RGBA16F texture of 5 levels (pitch
// 8192, padded to powers of two), and the targets the game draws its levels 1-4 into at the
// location's load, as the game binds them (address, extent, pitch).
constexpr VAddr SaltFlat = 0x102c600000;
struct SaltFlatTarget {
    VAddr address;
    u32 width, height, pitch;
};
constexpr SaltFlatTarget SaltFlatTargets[] = {
    {0x103c600000, 3012, 2008, 4096},
    {0x1040600000, 1506, 1004, 2048},
    {0x1041600000, 753, 502, 1024},
    {0x1041a00000, 376, 251, 512},
};
ImageInfo SaltFlatTexture() {
    auto info = Color2D(SaltFlat, 6024, 4016, 5, Macro, vk::Format::eR16G16B16A16Sfloat, 64);
    info.props.is_pow2 = true;
    info.pitch = 8192;
    info.UpdateSize();
    return info;
}

TEST(ImageMips, SaltFlatLevelTargetsAreTheBackdropLevels) {
    const auto texture = SaltFlatTexture();
    EXPECT_EQ(texture.guest_size, 0x15500000u);
    for (u32 i = 0; i < 4; ++i) {
        const auto& t = SaltFlatTargets[i];
        auto target = Color2D(t.address, t.width, t.height, 1, Macro,
                              vk::Format::eR16G16B16A16Sfloat, 64);
        target.pitch = t.pitch;
        target.UpdateSize();
        EXPECT_EQ(texture.guest_address + texture.mips_layout[i + 1].offset, t.address)
            << "level " << i + 1;
        EXPECT_EQ(target.MipOf(texture), static_cast<s32>(i + 1)) << "level " << i + 1;
    }
}

TEST(ImageMips, PaddingDoesNotMakeAnyTargetALevel) {
    const auto texture = DofTexture(493);
    // Twice the level's height is past its padding.
    auto tall = DofTargets[1];
    tall.height = 496;
    EXPECT_EQ(DofTargetInfo(tall).MipOf(texture, true), -1);
    // Another pitch at the level's address is another surface.
    auto other_pitch = DofTargets[1];
    other_pitch.pitch = 384;
    EXPECT_EQ(DofTargetInfo(other_pitch).MipOf(texture, true), -1);
    // Smaller than the level (345x246) is not the level either.
    auto narrow = DofTargets[1];
    narrow.width = 344;
    narrow.height = 246;
    EXPECT_EQ(DofTargetInfo(narrow).MipOf(texture, true), -1);
    auto short_target = DofTargets[1];
    short_target.width = 345;
    short_target.height = 245;
    EXPECT_EQ(DofTargetInfo(short_target).MipOf(texture, true), -1);
    // Wider than the level's pitch is past its padding.
    auto wide = DofTargets[1];
    wide.width = 513;
    EXPECT_EQ(DofTargetInfo(wide).MipOf(texture, true), -1);
}

TEST(ImageMips, PaddingIsForSingleLayerLevelsOnly) {
    // The same pyramid with two layers: a padded target at level 1's address is not its slice
    // (PlaceOf takes a level only through SliceOf).
    auto texture = DofTexture(493);
    texture.resources.layers = 2;
    texture.UpdateSize();
    auto t = DofTargets[1];
    t.address = texture.guest_address + texture.mips_layout[1].offset;
    EXPECT_EQ(DofTargetInfo(t).SliceOf(texture, 1, true), -1);
}

