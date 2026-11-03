// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/assert.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {
// SO-001: shader objects are made from SPIR-V code, not modules; the code of each module is kept
// while RYGT_SHADER_OBJECT=1.
bool KeepCode() {
    static const bool keep = [] {
        const char* env = std::getenv("RYGT_SHADER_OBJECT");
        return env && env[0] == '1';
    }();
    return keep;
}
std::mutex code_mutex;
std::unordered_map<VkShaderModule, std::vector<u32>> module_code;
} // namespace

std::vector<u32> ModuleCode(vk::ShaderModule module) {
    std::scoped_lock lock{code_mutex};
    const auto it = module_code.find(static_cast<VkShaderModule>(module));
    return it != module_code.end() ? it->second : std::vector<u32>{};
}

vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device) {
    const vk::ShaderModuleCreateInfo shader_info = {
        .codeSize = code.size() * sizeof(u32),
        .pCode = code.data(),
    };

    auto [module_result, module] = device.createShaderModule(shader_info);
    ASSERT_MSG(module_result == vk::Result::eSuccess, "Failed to compile SPIR-V shader: {}",
               vk::to_string(module_result));
    if (KeepCode()) {
        std::scoped_lock lock{code_mutex};
        module_code[static_cast<VkShaderModule>(module)].assign(code.begin(), code.end());
    }
    return module;
}

} // namespace Vulkan
