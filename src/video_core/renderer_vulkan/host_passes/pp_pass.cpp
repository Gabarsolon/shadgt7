//  SPDX-FileCopyrightText: Copyright 2024-2026 shadPS4 Emulator Project
//  SPDX-License-Identifier: GPL-2.0-or-later

#include "video_core/renderer_vulkan/host_passes/pp_pass.h"

#include "common/assert.h"
#include "common/sharp_options.h"
#include "core/emulator_settings.h"
#include "video_core/host_shaders/fs_tri_vert.h"
#include "video_core/host_shaders/post_process_frag.h"
#include "video_core/renderer_vulkan/vk_platform.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

#include <boost/container/static_vector.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vk_mem_alloc.h>

#include "video_core/renderer_vulkan/vk_ui_hires.h"

namespace Vulkan::HostPasses {

namespace {
struct PostProcessingConstants {
    PostProcessingPass::Settings settings;
    float dlss_sharpening = 0.0f; // the RCAS gain of the scene pass; 0 = off
    u32 scene_output = 0; // the scene pass: RCAS only, stored as it was read
    u32 scene_srgb = 0;   // the scene was read through an sRGB view (linear samples)
};
static_assert(sizeof(PostProcessingPass::Settings) == 24);
static_assert(offsetof(PostProcessingConstants, dlss_sharpening) == 24);
static_assert(offsetof(PostProcessingConstants, scene_output) == 28);
static_assert(offsetof(PostProcessingConstants, scene_srgb) == 32);
static_assert(sizeof(PostProcessingConstants) == 36);
} // namespace

namespace {
// A number in the config, or a string holding one; the fallback for anything else.
double ConfigNumber(const nlohmann::json& object, const char* key, double fallback) {
    const auto it = object.find(key);
    if (it == object.end()) {
        return fallback;
    }
    double value = fallback;
    if (it->is_number()) {
        value = it->get<double>();
    } else if (it->is_string()) {
        const auto& text = it->get_ref<const std::string&>();
        char* end{};
        value = std::strtod(text.c_str(), &end);
        if (end == text.c_str() || *end != 0) {
            return fallback;
        }
    } else {
        return fallback;
    }
    return std::isfinite(value) ? value : fallback;
}
} // namespace

float ReadSharpeningGain() {
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    static float gain = 0.0f;
    static auto next_read = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (now < next_read) {
        return gain;
    }
    next_read = now + std::chrono::seconds(2);
    double percent = 0.0;
#if defined(RYTY_GPU) && defined(_WIN32)
    const auto document = nlohmann::json::parse(Common::RytyReadGameConfig(), nullptr, false);
    if (document.is_object() && document.contains("GPU") && document["GPU"].is_object()) {
        percent = ConfigNumber(document["GPU"], "dlss_sharpening", 0.0);
    }
#endif
    if (const char* env = std::getenv("RYGT_DLSS_SHARPENING"); env && *env) {
        char* end{};
        const double value = std::strtod(env, &end);
        if (end != env && *end == 0 && std::isfinite(value)) {
            percent = value;
        }
    }
    percent = std::clamp(percent, 0.0, 100.0);
    gain = static_cast<float>(percent / 100.0);
    Common::Sharp::sharpen_percent.store(static_cast<float>(percent), std::memory_order_relaxed);
    return gain;
}

void PostProcessingPass::Create(vk::Device device, const vk::Format surface_format) {
    boost::container::static_vector<vk::DescriptorSetLayoutBinding, 4> bindings{
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        { // GT7-UW: the HUD over black
            .binding = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        { // GT7-UW: the HUD over white
            .binding = 2,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
        { // GT7-UW: the HUD layout
            .binding = 3,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eFragment,
        },
    };

    const vk::DescriptorSetLayoutCreateInfo desc_layout_ci{
        .flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptorKHR,
        .bindingCount = static_cast<u32>(bindings.size()),
        .pBindings = bindings.data(),
    };

    desc_set_layout = Check<"create pp descriptor set layout">(
        device.createDescriptorSetLayoutUnique(desc_layout_ci));

    const vk::PushConstantRange push_constants{
        .stageFlags = vk::ShaderStageFlagBits::eFragment,
        .offset = 0,
        .size = sizeof(PostProcessingConstants),
    };

    const auto& vs_module = CompileSPV(FS_TRI_VERT, device);
    ASSERT(vs_module);
    SetObjectName(device, vs_module, "fs_tri.vert");

    const auto& fs_module = CompileSPV(POST_PROCESS_FRAG, device);
    ASSERT(fs_module);
    SetObjectName(device, fs_module, "post_process.frag");

    const std::array shaders_ci{
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eVertex,
            .module = vs_module,
            .pName = "main",
        },
        vk::PipelineShaderStageCreateInfo{
            .stage = vk::ShaderStageFlagBits::eFragment,
            .module = fs_module,
            .pName = "main",
        },
    };

    const vk::PipelineLayoutCreateInfo layout_info{
        .setLayoutCount = 1U,
        .pSetLayouts = &*desc_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constants,
    };

    pipeline_layout =
        Check<"create pp pipeline layout">(device.createPipelineLayoutUnique(layout_info));

    const std::array pp_color_formats{
        surface_format,
    };
    const vk::PipelineRenderingCreateInfo pipeline_rendering_ci{
        .colorAttachmentCount = pp_color_formats.size(),
        .pColorAttachmentFormats = pp_color_formats.data(),
    };

    const vk::PipelineVertexInputStateCreateInfo vertex_input_info{
        .vertexBindingDescriptionCount = 0u,
        .vertexAttributeDescriptionCount = 0u,
    };

    const vk::PipelineInputAssemblyStateCreateInfo input_assembly{
        .topology = vk::PrimitiveTopology::eTriangleList,
    };

    const vk::Viewport viewport{
        .x = 0.0f,
        .y = 0.0f,
        .width = 1.0f,
        .height = 1.0f,
        .minDepth = 0.0f,
        .maxDepth = 1.0f,
    };

    const vk::Rect2D scissor = {
        .offset = {0, 0},
        .extent = {1, 1},
    };

    const vk::PipelineViewportStateCreateInfo viewport_info{
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    const vk::PipelineRasterizationStateCreateInfo raster_state{
        .depthClampEnable = false,
        .rasterizerDiscardEnable = false,
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eClockwise,
        .depthBiasEnable = false,
        .lineWidth = 1.0f,
    };

    const vk::PipelineMultisampleStateCreateInfo multisampling{
        .rasterizationSamples = vk::SampleCountFlagBits::e1,
    };

    const std::array attachments{
        vk::PipelineColorBlendAttachmentState{
            .blendEnable = false,
            .colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                              vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
        },
    };

    const vk::PipelineColorBlendStateCreateInfo color_blending{
        .logicOpEnable = false,
        .logicOp = vk::LogicOp::eCopy,
        .attachmentCount = attachments.size(),
        .pAttachments = attachments.data(),
        .blendConstants = std::array{1.0f, 1.0f, 1.0f, 1.0f},
    };

    const std::array dynamic_states{
        vk::DynamicState::eViewport,
        vk::DynamicState::eScissor,
    };

    const vk::PipelineDynamicStateCreateInfo dynamic_info{
        .dynamicStateCount = dynamic_states.size(),
        .pDynamicStates = dynamic_states.data(),
    };

    const vk::GraphicsPipelineCreateInfo pipeline_info{
        .pNext = &pipeline_rendering_ci,
        .stageCount = shaders_ci.size(),
        .pStages = shaders_ci.data(),
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_info,
        .pRasterizationState = &raster_state,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &color_blending,
        .pDynamicState = &dynamic_info,
        .layout = *pipeline_layout,
    };

    pipeline = Check<"create post process pipeline">(device.createGraphicsPipelineUnique(
        /*pipeline_cache*/ {}, pipeline_info));

    // FIX-GT7-145: the same pipeline for the frames of an HDR10 output.
    const std::array hdr_color_formats{
        HdrFrameFormat,
    };
    const vk::PipelineRenderingCreateInfo hdr_rendering_ci{
        .colorAttachmentCount = hdr_color_formats.size(),
        .pColorAttachmentFormats = hdr_color_formats.data(),
    };
    auto hdr_pipeline_info = pipeline_info;
    hdr_pipeline_info.pNext = &hdr_rendering_ci;
    hdr_pipeline = Check<"create HDR post process pipeline">(
        device.createGraphicsPipelineUnique(/*pipeline_cache*/ {}, hdr_pipeline_info));

    // Once pipeline is compiled, we don't need the shader module anymore
    device.destroyShaderModule(vs_module);
    device.destroyShaderModule(fs_module);

    // Create sampler resource
    const vk::SamplerCreateInfo sampler_ci{
        .magFilter = vk::Filter::eLinear,
        .minFilter = vk::Filter::eLinear,
        .mipmapMode = vk::SamplerMipmapMode::eNearest,
        .addressModeU = vk::SamplerAddressMode::eClampToEdge,
        .addressModeV = vk::SamplerAddressMode::eClampToEdge,
    };
    sampler = Check<"create pp sampler">(device.createSamplerUnique(sampler_ci));
}

void PostProcessingPass::CreateHudBuffer(VmaAllocator allocator) {
    hud_allocator = allocator;
    const VkBufferCreateInfo buffer_ci{
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = HudSlots * HudSlotSize,
        .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
    };
    const VmaAllocationCreateInfo alloc_ci{
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                 VMA_ALLOCATION_CREATE_MAPPED_BIT,
        .usage = VMA_MEMORY_USAGE_AUTO,
    };
    VkBuffer buffer{};
    VmaAllocationInfo info{};
    if (vmaCreateBuffer(allocator, &buffer_ci, &alloc_ci, &buffer, &hud_allocation, &info) ==
        VK_SUCCESS) {
        hud_buffer = buffer;
        hud_mapped = static_cast<u8*>(info.pMappedData);
        std::memset(hud_mapped, 0, HudSlots * HudSlotSize);
    }
}

void PostProcessingPass::Render(vk::CommandBuffer cmdbuf, vk::ImageView input,
                                vk::Extent2D input_size, Frame& frame, Settings settings,
                                vk::ImageView hud_black, vk::ImageView hud_white,
                                std::span<const HudWidget> hud_widgets, float scene_gain) {
    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.beginDebugUtilsLabelEXT(vk::DebugUtilsLabelEXT{
            .pLabelName = "Host/Post processing",
        });
    }

    constexpr vk::ImageSubresourceRange simple_subresource = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const std::array<vk::RenderingAttachmentInfo, 1> attachments{{
        {
            .imageView = frame.image_view,
            .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
        },
    }};
    const vk::RenderingInfo rendering_info{
        .renderArea{
            .extent{
                .width = frame.width,
                .height = frame.height,
            },
        },
        .layerCount = 1,
        .colorAttachmentCount = attachments.size(),
        .pColorAttachments = attachments.data(),
    };

    vk::DescriptorImageInfo image_info{
        .sampler = *sampler,
        .imageView = input,
        .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    // GT7-UW: the HUD pictures stay in the general layout native mode keeps its pictures in;
    // without a HUD the input stands in (the shader does not read it then).
    const bool hud = hud_black && hud_white;
    settings.hud = hud ? 1u : 0u;
    const vk::DescriptorImageInfo hud_black_info{
        .sampler = *sampler,
        .imageView = hud ? hud_black : input,
        .imageLayout = hud ? vk::ImageLayout::eGeneral : vk::ImageLayout::eShaderReadOnlyOptimal,
    };
    const vk::DescriptorImageInfo hud_white_info{
        .sampler = *sampler,
        .imageView = hud ? hud_white : input,
        .imageLayout = hud ? vk::ImageLayout::eGeneral : vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    // GT7-UW: the layout in this frame's slot of the buffer (a count, then the pieces).
    const u32 slot = hud_slot++ % HudSlots;
    if (hud_mapped) {
        u8* data = hud_mapped + slot * HudSlotSize;
        const u32 count = static_cast<u32>(std::min<size_t>(hud_widgets.size(), MaxHudWidgets));
        const std::array<u32, 4> header{hud ? count : 0u, 0, 0, 0};
        std::memcpy(data, header.data(), sizeof(header));
        if (count != 0) {
            std::memcpy(data + sizeof(header), hud_widgets.data(), count * sizeof(HudWidget));
        }
        vmaFlushAllocation(hud_allocator, hud_allocation, slot * HudSlotSize, HudSlotSize);
    }
    const vk::DescriptorBufferInfo hud_layout_info{
        .buffer = hud_buffer,
        .offset = slot * HudSlotSize,
        .range = 16 + MaxHudWidgets * sizeof(HudWidget),
    };

    const std::array set_writes{
        vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &image_info,
        },
        vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &hud_black_info,
        },
        vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 2,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eCombinedImageSampler,
            .pImageInfo = &hud_white_info,
        },
        vk::WriteDescriptorSet{
            .dstSet = VK_NULL_HANDLE,
            .dstBinding = 3,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eUniformBuffer,
            .pBufferInfo = &hud_layout_info,
        },
    };

    // FIX-GT7-145: by the frame's own format, not the pass's setting: the two change at
    // different moments around an HDR switch.
    cmdbuf.bindPipeline(vk::PipelineBindPoint::eGraphics,
                        frame.is_hdr ? *hdr_pipeline : *pipeline);

    const std::array viewports = {
        vk::Viewport{
            .width = static_cast<float>(frame.width),
            .height = static_cast<float>(frame.height),
            .minDepth = 0.0f,
            .maxDepth = 1.0f,
        },
    };

    cmdbuf.setViewport(0, viewports);
    cmdbuf.setScissor(0, vk::Rect2D{
                             .extent{
                                 .width = frame.width,
                                 .height = frame.height,
                             },
                         });

    cmdbuf.pushDescriptorSetKHR(vk::PipelineBindPoint::eGraphics, *pipeline_layout, 0, set_writes);
    const bool scene = scene_gain >= 0.0f;
    const PostProcessingConstants constants{settings, scene ? scene_gain : 0.0f, scene ? 1u : 0u,
                                            scene && settings.srgb_input ? 1u : 0u};
    cmdbuf.pushConstants(*pipeline_layout, vk::ShaderStageFlagBits::eFragment, 0, sizeof(constants),
                         &constants);

    cmdbuf.beginRendering(rendering_info);
    cmdbuf.draw(3, 1, 0, 0);
    cmdbuf.endRendering();

    const auto post_barrier = vk::ImageMemoryBarrier2{
        .srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput,
        .srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eFragmentShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eColorAttachmentOptimal,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = frame.image,
        .subresourceRange = simple_subresource,
    };

    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &post_barrier,
    });

    if (EmulatorSettings.IsVkHostMarkersEnabled()) {
        cmdbuf.endDebugUtilsLabelEXT();
    }
}

void PostProcessingPass::DestroyHudBuffer() {
    if (hud_buffer) {
        vmaDestroyBuffer(hud_allocator, hud_buffer, hud_allocation);
        hud_buffer = vk::Buffer{};
    }
}

struct SceneSharpenPass::Resources {
    vk::Image input;
    vk::Extent2D size;
    bool srgb = false;
    VideoCore::UniqueImage scratch;
    vk::UniqueImageView input_view;
    vk::UniqueImageView scratch_view;
};

SceneSharpenPass::SceneSharpenPass(const Instance& instance, vk::Image input, vk::Format format,
                                   vk::Extent2D size)
    : resources{std::make_unique<Resources>()} {
    auto& r = *resources;
    r.input = input;
    r.size = size;
    r.srgb = format == vk::Format::eR8G8B8A8Srgb || format == vk::Format::eB8G8R8A8Srgb;
    const auto device = instance.GetDevice();
    r.scratch = VideoCore::UniqueImage(device, instance.GetAllocator());
    r.scratch.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent = {size.width, size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    const auto view = [&](vk::Image image) {
        return Check<"create scene RCAS view">(device.createImageViewUnique(vk::ImageViewCreateInfo{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange = UiHires::ColorRange,
        }));
    };
    r.input_view = view(input);
    r.scratch_view = view(r.scratch);
    pass.Create(device, format);
    pass.CreateHudBuffer(instance.GetAllocator());
}

SceneSharpenPass::~SceneSharpenPass() {
    pass.DestroyHudBuffer();
}

void SceneSharpenPass::Render(vk::CommandBuffer cmd, float gain) {
    auto& r = *resources;
    UiHires::NativeBarrier(cmd, r.input, vk::ImageLayout::eGeneral,
                           vk::ImageLayout::eShaderReadOnlyOptimal);
    // Last frame's scratch is discarded; NativeBarrier also orders the earlier transfer reads.
    UiHires::NativeBarrier(cmd, r.scratch, vk::ImageLayout::eUndefined,
                           vk::ImageLayout::eColorAttachmentOptimal);
    Frame frame{};
    frame.width = r.size.width;
    frame.height = r.size.height;
    frame.image = r.scratch;
    frame.image_view = *r.scratch_view;
    PostProcessingPass::Settings settings{};
    settings.srgb_input = r.srgb;
    pass.Render(cmd, *r.input_view, r.size, frame, settings, {}, {}, {}, gain);
    UiHires::NativeBarrier(cmd, r.scratch, vk::ImageLayout::eGeneral,
                           vk::ImageLayout::eTransferSrcOptimal);
    UiHires::NativeBarrier(cmd, r.input, vk::ImageLayout::eShaderReadOnlyOptimal,
                           vk::ImageLayout::eTransferDstOptimal);
    const vk::ImageCopy copy{
        .srcSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .dstSubresource = {.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .extent = {r.size.width, r.size.height, 1},
    };
    // A raw copy keeps the sRGB or PQ storage; the gamma and the HDR gain run at the present.
    cmd.copyImage(r.scratch, vk::ImageLayout::eTransferSrcOptimal, r.input,
                  vk::ImageLayout::eTransferDstOptimal, copy);
    UiHires::NativeBarrier(cmd, r.input, vk::ImageLayout::eTransferDstOptimal,
                           vk::ImageLayout::eGeneral);
}

} // namespace Vulkan::HostPasses
