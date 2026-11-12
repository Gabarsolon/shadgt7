// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>

// NEO-GT7-016: which pipelines take the TRUNCATE coverage reduction of
// VK_NV_coverage_reduction_mode (the PS4 Pro mode's checkerboard on NVIDIA's mixed samples).
// Sample counts are Vulkan's sample count bits (1, 2, 4, ...); 0 means no such attachment.
namespace Vulkan::CoverageReduction {

/// A combination vkGetPhysicalDeviceSupportedFramebufferMixedSamplesCombinationsNV lists for
/// VK_COVERAGE_REDUCTION_MODE_TRUNCATE_NV. depth and color are sample count masks (0: the
/// combination has no such attachment).
struct Combination {
    uint32_t raster;
    uint32_t depth;
    uint32_t color;
};

/// True when a colour attachment has fewer samples than the rasterization: there NVIDIA's
/// default (MERGE) writes a pixel's colour when any of its samples is covered.
constexpr bool HasFewerColourSamples(uint32_t raster, std::span<const uint32_t> colors) {
    return std::ranges::any_of(colors, [&](uint32_t color) { return color != 0 && color < raster; });
}

/// True when TRUNCATE is listed for this rasterization, depth (0: none) and colour attachments
/// (0: unused slot; every other one must be in the combination's colour mask).
constexpr bool Listed(std::span<const Combination> truncate, uint32_t raster, uint32_t depth,
                      std::span<const uint32_t> colors) {
    const bool any_color = std::ranges::any_of(colors, [](uint32_t color) { return color != 0; });
    return std::ranges::any_of(truncate, [&](const Combination& listed) {
        if (listed.raster != raster) {
            return false;
        }
        if (depth == 0 ? listed.depth != 0 : (listed.depth & depth) == 0) {
            return false;
        }
        if (!any_color) {
            return listed.color == 0;
        }
        return std::ranges::all_of(colors, [&](uint32_t color) {
            return color == 0 || (listed.color & color) != 0;
        });
    });
}

} // namespace Vulkan::CoverageReduction
