// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

namespace Vulkan::Gt7Spatial {
// Without current temporal inputs, a post-processed menu uses the native filter
// instead of resetting a DLSS history with dummy depth and motion every frame.
constexpr bool PreferNativeFilter(bool temporal_path, bool scene, bool reconstructed) {
    return temporal_path && scene && !reconstructed;
}
} // namespace Vulkan::Gt7Spatial
