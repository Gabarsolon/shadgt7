//  SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
//  SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <memory>
#include <span>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

typedef struct VmaAllocator_T* VmaAllocator;
typedef struct VmaAllocation_T* VmaAllocation;

namespace Vulkan {
class Frame;
class Instance;
}

namespace Vulkan::HostPasses {

/// GPU.dlss_sharpening (0..100 percent, 0 = off) as a gain of 0..1. It is read again every two
/// seconds, so it can be tuned while playing; RYGT_DLSS_SHARPENING overrides it.
float ReadSharpeningGain();

class PostProcessingPass {
public:
    /// GT7-UW: a HUD piece moved off its place: where it is in the HUD picture (x, y, w, h,
    /// 0..1 of the 16:9 HUD) and where it goes on the output (0..1 of the output).
    struct HudWidget {
        float src[4];
        float dst[4];
    };
    static constexpr u32 MaxHudWidgets = 16;

    struct Settings {
        float gamma = 1.0f;
        u32 hdr = 0;
        u32 srgb_input = 0;
        /// GT7-UW: the HUD drawn over black and white is laid over the picture between these
        /// horizontal positions (0..1 of the output).
        u32 hud = 0;
        float hud_x0 = 0.0f;
        float hud_x1 = 1.0f;
    };

    /// FIX-GT7-145: the format of a frame shown on an HDR10 output, the swapchain's own 10 bits.
    /// The surface's 8-bit format cut the PQ picture to 256 levels a channel before the 10-bit
    /// swapchain (measured: 85 to 94% of the window's pixels on the 8-bit grid, 22 to 25% in
    /// the game's picture, which is chance).
    static constexpr vk::Format HdrFrameFormat = vk::Format::eA2B10G10R10UnormPack32;

    void Create(vk::Device device, vk::Format surface_format);
    /// GT7-UW: the buffer the HUD layout goes to the shader in.
    void CreateHudBuffer(VmaAllocator allocator);

    /// With `scene_gain` >= 0 the pass is the scene's sharpening: RCAS with that gain and
    /// nothing else (no gamma, no HDR gain or peak, no HUD), stored in the input's encoding.
    void Render(vk::CommandBuffer cmdbuf, vk::ImageView input, vk::Extent2D input_size,
                Frame& output, Settings settings, vk::ImageView hud_black = {},
                vk::ImageView hud_white = {}, std::span<const HudWidget> hud_widgets = {},
                float scene_gain = -1.0f);
    /// The HUD buffer of a pass that does not live as long as the renderer.
    void DestroyHudBuffer();

private:
    vk::UniquePipeline pipeline{};
    vk::UniquePipeline hdr_pipeline{}; // FIX-GT7-145: for HdrFrameFormat
    vk::UniquePipelineLayout pipeline_layout{};
    vk::UniqueDescriptorSetLayout desc_set_layout{};
    vk::UniqueSampler sampler{};
    // GT7-UW: the HUD layout, one slot per recent frame.
    static constexpr u32 HudSlots = 8;
    static constexpr u32 HudSlotSize = 1024;
    VmaAllocator hud_allocator{};
    vk::Buffer hud_buffer{};
    VmaAllocation hud_allocation{};
    u8* hud_mapped = nullptr;
    u32 hud_slot = 0;
};

/// GPU.dlss_sharpening: sharpens the scene a HD UI proxy holds after DLSS scaled it, before the
/// UI is drawn over it, so the interface is not sharpened. One per proxy; it keeps its own
/// scratch image and retires with the proxy.
class SceneSharpenPass {
public:
    SceneSharpenPass(const Instance& instance, vk::Image input, vk::Format format,
                     vk::Extent2D size);
    ~SceneSharpenPass();
    /// `input` is in the general layout and stays there.
    void Render(vk::CommandBuffer cmd, float gain);

private:
    struct Resources;
    std::unique_ptr<Resources> resources;
    PostProcessingPass pass;
};

} // namespace Vulkan::HostPasses
