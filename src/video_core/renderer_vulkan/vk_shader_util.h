// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan {

/**
 * @brief Creates a vulkan shader module from SPIR-V bytecode.
 * @param code The SPIR-V bytecode data.
 * @param device The vulkan device handle
 */
vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device);

/// SO-001: the SPIR-V a module was made from (empty unless RYGT_SHADER_OBJECT=1).
std::vector<u32> ModuleCode(vk::ShaderModule module);

} // namespace Vulkan
