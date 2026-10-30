// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <fstream>
#include <boost/container/small_vector.hpp>

#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info_, vk::ShaderModule module,
                                 SerializationSupport& sdata)
    : Pipeline{instance, scheduler, desc_heap, profile, pipeline_cache, true},
      compute_key{compute_key_} {
    auto& info = stages[int(Shader::SwStage::Compute)];
    info = &info_;
    canonical_stages[int(Shader::SwStage::Compute)] = &info_;
    const auto debug_str = GetDebugString();

    const u32 required_subgroup = instance.IsSubgroupSize64Supported() ? 64u : 32u;
    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_size_ci = {
        .requiredSubgroupSize = required_subgroup,
    };
    // FIX-188: Intel Arc (#65) runs compute in 32-wide subgroups when it can't do 64.
    const bool use_subgroup_control =
        instance.IsSubgroupSize64Supported() ||
        (instance.IsIntel() && instance.IsSubgroupSizeControlSupported() &&
         instance.SubgroupSize() == 32u);
    const vk::PipelineShaderStageCreateInfo shader_ci = {
        .pNext = use_subgroup_control ? &subgroup_size_ci : nullptr,
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = module,
        .pName = "main",
    };

    u32 binding{};
    boost::container::small_vector<vk::DescriptorSetLayoutBinding, 32> bindings;
    for (const auto& buffer : info->buffers) {
        const auto sharp = buffer.GetSharp(*info);
        bindings.push_back({
            .binding = binding++,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }
    for (const auto& image : info->images) {
        const u32 num_bindings = image.NumBindings(*info);
        bindings.push_back({
            .binding = binding,
            .descriptorType = image.is_written ? vk::DescriptorType::eStorageImage
                                               : vk::DescriptorType::eSampledImage,
            .descriptorCount = num_bindings,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
        binding += num_bindings;
    }
    for (const auto& sampler : info->samplers) {
        bindings.push_back({
            .binding = binding++,
            .descriptorType = vk::DescriptorType::eSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }

    const vk::PushConstantRange push_constants = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };

    const bool push = binding < instance.MaxPushDescriptors();
    push_set = push ? 0u : NoPushSet;
    recipe_bindings.assign(bindings.begin(), bindings.end()); // PERF-GT7-037
    recipe_push = push;
    recipe_subgroup_control = use_subgroup_control;
    recipe_subgroup = required_subgroup;
    const auto flags = push ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                            : vk::DescriptorSetLayoutCreateFlagBits{};
    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci = {
        .flags = flags,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };
    const auto device = instance.GetDevice();
    auto [descriptor_set_result, descriptor_set] =
        device.createDescriptorSetLayoutUnique(desc_layout_ci);
    ASSERT_MSG(descriptor_set_result == vk::Result::eSuccess,
               "Failed to create compute descriptor set layout: {}",
               vk::to_string(descriptor_set_result));
    desc_layouts[0] = std::move(descriptor_set);
    set_layouts[0] = *desc_layouts[0];

    const vk::DescriptorSetLayout set_layout = set_layouts[0];
    const vk::PipelineLayoutCreateInfo layout_info = {
        .setLayoutCount = 1U,
        .pSetLayouts = &set_layout,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_constants,
    };
    auto [layout_result, layout] = instance.GetDevice().createPipelineLayoutUnique(layout_info);
    ASSERT_MSG(layout_result == vk::Result::eSuccess,
               "Failed to create compute pipeline layout: {}", vk::to_string(layout_result));
    pipeline_layout = std::move(layout);
    SetObjectName(device, *pipeline_layout, "Compute PipelineLayout {}", debug_str);

    const vk::ComputePipelineCreateInfo compute_pipeline_ci = {
        .stage = shader_ci,
        .layout = *pipeline_layout,
    };
    auto [pipeline_result, pipe] =
        instance.GetDevice().createComputePipelineUnique(pipeline_cache, compute_pipeline_ci);
    ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create compute pipeline: {}",
               vk::to_string(pipeline_result));
    pipeline = std::move(pipe);
    SetObjectName(device, *pipeline, "Compute Pipeline {}", debug_str);
}

ComputePipeline::~ComputePipeline() = default;

namespace {
constexpr u32 RecipeMagic = 0x43574752; // "RGWC"
constexpr u32 RecipeVersion = 1;
} // namespace

bool ComputePipeline::SaveRecipe(const std::filesystem::path& file,
                                 std::span<const u32> spirv) const {
    std::vector<u32> words{RecipeMagic,
                           RecipeVersion,
                           u32(recipe_push),
                           u32(recipe_subgroup_control),
                           recipe_subgroup,
                           u32(recipe_bindings.size())};
    for (const auto& b : recipe_bindings) {
        words.push_back(b.binding);
        words.push_back(u32(b.descriptorType));
        words.push_back(b.descriptorCount);
    }
    words.push_back(u32(spirv.size()));
    words.insert(words.end(), spirv.begin(), spirv.end());
    // Beside the file and renamed, so a cut write never leaves half a recipe.
    auto temp = file;
    temp += ".tmp";
    {
        std::ofstream out{temp, std::ios::binary | std::ios::trunc};
        out.write(reinterpret_cast<const char*>(words.data()),
                  std::streamsize(words.size() * sizeof(u32)));
        if (!out) {
            return false;
        }
    }
    std::error_code ec;
    std::filesystem::rename(temp, file, ec);
    return !ec;
}

bool ComputePipeline::WarmFromRecipe(const Instance& instance, vk::PipelineCache cache,
                                     const std::filesystem::path& file) {
    std::ifstream in{file, std::ios::binary};
    std::vector<u32> words;
    {
        std::error_code ec;
        const auto size = std::filesystem::file_size(file, ec);
        if (ec || size < 7 * sizeof(u32) || size % sizeof(u32) != 0 || size > (16u << 20)) {
            return false;
        }
        words.resize(size / sizeof(u32));
        in.read(reinterpret_cast<char*>(words.data()), std::streamsize(size));
        if (!in) {
            return false;
        }
    }
    if (words[0] != RecipeMagic || words[1] != RecipeVersion) {
        return false;
    }
    const bool push = words[2] != 0;
    const bool subgroup_control = words[3] != 0;
    const u32 subgroup = words[4];
    const u32 count = words[5];
    size_t at = 6;
    if (count > 256 || words.size() < at + size_t(count) * 3 + 1) {
        return false;
    }
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
    for (u32 i = 0; i < count; ++i, at += 3) {
        bindings.push_back({
            .binding = words[at],
            .descriptorType = vk::DescriptorType(words[at + 1]),
            .descriptorCount = words[at + 2],
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
        });
    }
    const u32 spirv_words = words[at++];
    if (spirv_words == 0 || words.size() - at != spirv_words) {
        return false;
    }
    const auto device = instance.GetDevice();
    auto [module_result, module] = device.createShaderModuleUnique({
        .codeSize = spirv_words * sizeof(u32),
        .pCode = words.data() + at,
    });
    if (module_result != vk::Result::eSuccess) {
        return false;
    }
    const vk::DescriptorSetLayoutCreateInfo set_ci = {
        .flags = push ? vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR
                      : vk::DescriptorSetLayoutCreateFlagBits{},
        .bindingCount = u32(bindings.size()),
        .pBindings = bindings.data(),
    };
    auto [set_result, set_layout] = device.createDescriptorSetLayoutUnique(set_ci);
    if (set_result != vk::Result::eSuccess) {
        return false;
    }
    const vk::PushConstantRange push_constants = {
        .stageFlags = vk::ShaderStageFlagBits::eCompute,
        .offset = 0,
        .size = sizeof(Shader::PushData),
    };
    const vk::DescriptorSetLayout set = *set_layout;
    auto [layout_result, layout] = device.createPipelineLayoutUnique({
        .setLayoutCount = 1U,
        .pSetLayouts = &set,
        .pushConstantRangeCount = 1U,
        .pPushConstantRanges = &push_constants,
    });
    if (layout_result != vk::Result::eSuccess) {
        return false;
    }
    const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_ci = {
        .requiredSubgroupSize = subgroup,
    };
    const vk::ComputePipelineCreateInfo pipeline_ci = {
        .stage =
            {
                .pNext = subgroup_control ? &subgroup_ci : nullptr,
                .stage = vk::ShaderStageFlagBits::eCompute,
                .module = *module,
                .pName = "main",
            },
        .layout = *layout,
    };
    auto [pipeline_result, pipeline] = device.createComputePipelineUnique(cache, pipeline_ci);
    return pipeline_result == vk::Result::eSuccess;
}

} // namespace Vulkan
