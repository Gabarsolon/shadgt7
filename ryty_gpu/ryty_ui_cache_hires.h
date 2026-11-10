// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>

namespace Ryty::UiCache {

/// GPU.hd_ui: the pavilion icon caches are made at the output's density (see
/// ryty_ui_cache_hires.inl). Set before the guest's threads start.
inline bool enabled = false;
/// Output pixels per display buffer pixel, from 1 to 2.
inline std::atomic<float> output_scale{1.f};

/// The sizes of the display buffer and the output, from each flip. Only a uniform scale counts.
inline void Output(unsigned native_width, unsigned native_height, unsigned output_width,
                   unsigned output_height) {
    if (!enabled || !native_width || !native_height || !output_width || !output_height) {
        return;
    }
    const float x = float(output_width) / native_width;
    const float y = float(output_height) / native_height;
    if (std::abs(x - y) < .002f) {
        output_scale.store(std::clamp(x, 1.f, 2.f), std::memory_order_relaxed);
    }
}

} // namespace Ryty::UiCache
