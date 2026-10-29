// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <chrono>
#include <thread>
#include <boost/container/small_vector.hpp>

#include "common/logging/log.h"
#include "shader_recompiler/info.h"
#include "video_core/renderer_vulkan/vk_compute_pipeline.h"
#include "video_core/renderer_vulkan/vk_graphics_pipeline.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"

namespace Vulkan {

ComputePipeline::ComputePipeline(const Instance& instance, Scheduler& scheduler,
                                 DescriptorHeap& desc_heap, const Shader::Profile& profile,
                                 vk::PipelineCache pipeline_cache, ComputePipelineKey compute_key_,
                                 const Shader::Info& info_, vk::ShaderModule module,
                                 SerializationSupport& sdata, bool fast_first)
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
        .flags = fast_first ? vk::PipelineCreateFlagBits::eDisableOptimization
                            : vk::PipelineCreateFlags{},
        .stage = shader_ci,
        .layout = *pipeline_layout,
    };
    const auto create_start = std::chrono::steady_clock::now();
    auto [pipeline_result, pipe] =
        instance.GetDevice().createComputePipelineUnique(pipeline_cache, compute_pipeline_ci);
    ASSERT_MSG(pipeline_result == vk::Result::eSuccess, "Failed to create compute pipeline: {}",
               vk::to_string(pipeline_result));
    pipeline = std::move(pipe);
    SetObjectName(device, *pipeline, "Compute Pipeline {}", debug_str);
    if (!fast_first) {
        return;
    }
    // PERF-GT7-035: the optimized pipeline, a second later, in the background. The fast one
    // stays alive (command buffers may still use it) until this pipeline goes.
    const double fast_ms = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - create_start)
                               .count();
    optimize_cancelled = std::make_shared<std::atomic<bool>>(false);
    auto done = std::make_shared<std::promise<void>>();
    optimize_done = done->get_future();
    const vk::PipelineLayout layout_handle = *pipeline_layout;
    QueueBackgroundPipelineBuild([this, device, layout_handle, module, cache = pipeline_cache,
                                  cancelled = optimize_cancelled, done, use_subgroup_control,
                                  required_subgroup, debug_str, fast_ms] {
        for (int i = 0; i < 10 && !cancelled->load(); ++i) {
            std::this_thread::sleep_for(std::chrono::milliseconds{100});
        }
        if (!cancelled->load()) {
            const vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_ci = {
                .requiredSubgroupSize = required_subgroup,
            };
            const vk::ComputePipelineCreateInfo optimized_ci = {
                .stage =
                    {
                        .pNext = use_subgroup_control ? &subgroup_ci : nullptr,
                        .stage = vk::ShaderStageFlagBits::eCompute,
                        .module = module,
                        .pName = "main",
                    },
                .layout = layout_handle,
            };
            const auto start = std::chrono::steady_clock::now();
            auto [result, optimized] = device.createComputePipeline(cache, optimized_ci);
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
            if (result == vk::Result::eSuccess) {
                optimized_pipeline.store(optimized, std::memory_order_release);
            }
            if (fast_ms >= 50.0 || ms >= 50.0) {
                LOG_INFO(Render_Vulkan,
                         "PERF-GT7-035: compute pipeline {}: unoptimized in {:.0f} ms, optimized "
                         "in {:.0f} ms in the background{}",
                         debug_str, fast_ms, ms,
                         result == vk::Result::eSuccess ? "" : " (failed; the fast one stays)");
            }
        }
        done->set_value();
    });
}

ComputePipeline::~ComputePipeline() {
    if (optimize_cancelled) {
        optimize_cancelled->store(true);
        // Bounded: a job still queued when the link workers stop at exit never runs.
        optimize_done.wait_for(std::chrono::seconds{10});
    }
    if (const VkPipeline optimized = optimized_pipeline.exchange(VK_NULL_HANDLE)) {
        instance.GetDevice().destroyPipeline(optimized);
    }
}

} // namespace Vulkan
