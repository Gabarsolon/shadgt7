// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <bit>
#include <limits>
#include <magic_enum/magic_enum.hpp>
#include "video_core/amdgpu/pixel_format.h"
#include "video_core/amdgpu/resource.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace VideoCore {

enum class ImageDescriptorGeometryError {
    None,
    InvalidType,
    UnrepresentableLayers,
    DeviceLayerLimit,
    DeviceExtentLimit,
    MipLevels,
    ViewMipLevels,
    SampleCount,
    GuestSizeLimit,
    ArrayMode,
    CubeNotSquare,
};

inline ImageDescriptorGeometryError CheckImageDescriptorGeometry(
    const AmdGpu::Image& image, const vk::PhysicalDeviceLimits& limits) {
    if (!image.Valid()) {
        return ImageDescriptorGeometryError::InvalidType;
    }
    // FIX-059: ImageInfo asserts on the linear-general array mode (no real texture uses it); a
    // garbage T# of GT Sport's grass compute shader 0xaa3822a3 had it and stopped the game in
    // a race.
    // GetArrayMode itself stops the emulator on a tile mode it does not know.
    if (const auto tile_mode = image.GetTileMode();
        !magic_enum::enum_contains(tile_mode) ||
        tile_mode == AmdGpu::TileMode::DisplayLinearGeneral) {
        return ImageDescriptorGeometryError::ArrayMode;
    }
    const u32 layers = image.NumLayers();
    if (layers == 0 || layers > std::numeric_limits<u16>::max()) {
        // Check before ImageInfo narrows the count to SubresourceExtent::layers.
        return ImageDescriptorGeometryError::UnrepresentableLayers;
    }
    if (layers > limits.maxImageArrayLayers) {
        return ImageDescriptorGeometryError::DeviceLayerLimit;
    }
    const u32 width = image.width + 1;
    const u32 height = image.height + 1;
    // FIX-080: a cube's faces are square (Vulkan requires it of cube views). A garbage T# of the
    // grass compute shader 0xaa3822a3 (a 1x15873 cube at 0x3f80000000, the bits of 1.0f) passed
    // every check above, and the GPU driver reset on the draw ("device lost") in a race.
    if (image.GetType() == AmdGpu::ImageType::Cube && width != height) {
        return ImageDescriptorGeometryError::CubeNotSquare;
    }
    const bool volume = image.GetBaseType() == AmdGpu::ImageType::Color3D;
    const u32 depth = volume ? image.depth + 1 : 1;
    if (volume) {
        if (std::max({width, height, depth}) > limits.maxImageDimension3D) {
            return ImageDescriptorGeometryError::DeviceExtentLimit;
        }
    } else {
        // The renderer represents guest 1D images using a Vulkan 2D backing too.
        if (std::max(width, height) > limits.maxImageDimension2D ||
            (image.GetBaseType() == AmdGpu::ImageType::Color1D &&
             width > limits.maxImageDimension1D)) {
            return ImageDescriptorGeometryError::DeviceExtentLimit;
        }
    }
    // A descriptor read from stale memory can describe gigabytes of guest memory; uploading or
    // detiling it would run far out of bounds. No real texture comes near 4 GiB.
    static constexpr u64 MaxGuestSize = 4ULL << 30;
    const auto data_format = image.GetDataFmt();
    const u32 block = AmdGpu::IsBlockCoded(data_format) ? 4 : 1;
    const u64 row_blocks = (u64(image.Pitch()) + block - 1) / block;
    const u64 rows = (u64(height) + block - 1) / block;
    const u64 slices = volume ? depth : layers;
    const u64 block_bytes = std::max(AmdGpu::NumBitsPerBlock(data_format) / 8, 1U);
    if (row_blocks * rows * slices * block_bytes > MaxGuestSize) {
        return ImageDescriptorGeometryError::GuestSizeLimit;
    }
    if (image.NumLevels() > std::bit_width(std::max({width, height, depth}))) {
        return ImageDescriptorGeometryError::MipLevels;
    }
    const bool multisampled = image.GetType() == AmdGpu::ImageType::Color2DMsaa ||
                              image.GetType() == AmdGpu::ImageType::Color2DMsaaArray;
    if (!multisampled && image.base_level > image.last_level) {
        return ImageDescriptorGeometryError::ViewMipLevels;
    }
    // RawNumSamples accepts the guest's 1/2/4/8/16 sample counts and deliberately
    // lowers them to the physical format's supported count later.
    if (image.NumSamples() > 16) {
        return ImageDescriptorGeometryError::SampleCount;
    }
    // FIX-GT7-076: the macro tile tables (GetMacroTileExtents) stop at 8 samples, and no render
    // target stores more than 8 fragments. A macro-tiled MSAA T# with last_level 4 (16 samples)
    // passed here and stopped GT7 on tile.h's assertion in BindTextures (2026-11-04, after a
    // race, right after the replay was saved). The tile mode is known to be valid from above.
    if (image.NumSamples() > 8 && AmdGpu::IsMacroTiled(AmdGpu::GetArrayMode(image.GetTileMode()))) {
        return ImageDescriptorGeometryError::SampleCount;
    }
    return ImageDescriptorGeometryError::None;
}

} // namespace VideoCore
