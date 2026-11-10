// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <optional>
#include "ryty_gpu/ryty_gt7_dlss_policy.h"

namespace Vulkan::Gt7Temporal {
struct Settings {
    bool enabled{};
    float jitter_sign_x = -1, jitter_sign_y = -1, mvec_scale = 1;
    // GT7 run3: raw stationary velocity is -C20/C21 (negative raster jitter in pixels).
    float motion_jitter = 1;
};
const Settings& Config();
struct Camera {
    Matrix projection{}, view{};
    std::array<float, 2> normalized_jitter{};
    uint64_t identity{}, sequence{};
    Projection lens{};
};
void InstallCameraHook(); // called before guest execution; exact prologue + vtable guards
std::optional<Camera> MatchCamera(const std::array<float, 29>& constants, uint64_t after);
} // namespace Vulkan::Gt7Temporal
