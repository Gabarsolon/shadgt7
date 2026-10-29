// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "video_core/renderer_vulkan/vk_common.h"

// ryGT: the size DLSS and FSR 3.1 scale the game picture to (RYGT_UPSCALE_QUALITY, set by the
// launcher's Upscaling quality list). The game always draws at its own size (1920x1080 for GT7),
// so a quality mode cannot lower the render resolution: it picks the output instead, the picture
// times the mode's scale (Native 1x, Quality 1.5x, Balanced 1.72x, Performance 2x, Ultra
// Performance 3x), and the post-process pass then fits that output to the window. "auto" (the
// default) scales to the window size, as before.
namespace Vulkan::RytyUpscale {

enum class Quality { Auto, Native, Quality, Balanced, Performance, UltraPerformance };

inline Quality GetQuality() {
    static const Quality quality = [] {
        const char* env = std::getenv("RYGT_UPSCALE_QUALITY");
        if (!env) {
            return Quality::Auto;
        }
        if (std::strcmp(env, "native") == 0) {
            return Quality::Native;
        }
        if (std::strcmp(env, "quality") == 0) {
            return Quality::Quality;
        }
        if (std::strcmp(env, "balanced") == 0) {
            return Quality::Balanced;
        }
        if (std::strcmp(env, "performance") == 0) {
            return Quality::Performance;
        }
        if (std::strcmp(env, "ultra_performance") == 0) {
            return Quality::UltraPerformance;
        }
        return Quality::Auto;
    }();
    return quality;
}

inline double Scale(Quality quality) {
    switch (quality) {
    case Quality::Native:
        return 1.0;
    case Quality::Quality:
        return 1.5;
    case Quality::Balanced:
        return 1.0 / 0.58;
    case Quality::Performance:
        return 2.0;
    case Quality::UltraPerformance:
        return 3.0;
    default:
        return 0.0;
    }
}

/// The upscaler's output size for the game picture input shown in window.
inline vk::Extent2D Target(vk::Extent2D input, vk::Extent2D window) {
    const double scale = Scale(GetQuality());
    if (scale == 0.0) {
        return window;
    }
    return {static_cast<u32>(std::lround(input.width * scale)),
            static_cast<u32>(std::lround(input.height * scale))};
}

/// Whether the upscaler should run at the picture's own size (DLAA / FSR native AA).
inline bool NativeAA() {
    return GetQuality() == Quality::Native;
}

} // namespace Vulkan::RytyUpscale
