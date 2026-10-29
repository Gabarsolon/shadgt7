// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// ryGT: NVIDIA DLSS upscaling, loaded at run time only when the config has "dlss_enabled":
// through Streamline's sl.dlss.dll (NVIDIA's signed files, which the launcher downloads from
// NVIDIA's GitHub; see ryty_dlssg.h), or else through the optional self-built plugin
// dlss\ryGT_dlss.dll (see rygt_dlss_api.h). NVIDIA's DLSS SDK cannot be part of this GPL library.

#pragma once

#include <functional>
#include <string_view>
#include <vector>

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan::RytyDlss {

/// Loads the plugin when DLSS is on; true when it is loaded.
bool Load();

/// Adds the instance extensions DLSS needs that the system offers.
void AddInstanceExtensions(std::vector<const char*>& extensions,
                           const std::vector<vk::ExtensionProperties>& available);

/// Adds the device extensions DLSS needs through add_extension (which checks availability).
void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical_device,
                         const std::function<bool(std::string_view)>& add_extension);

/// Starts DLSS on the created device. False (DLSS off) when the plugin or NGX refuses.
bool Init(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device);

/// Releases DLSS; the device must be idle.
void Shutdown();

/// Scales the game picture into an output image at the window size, like the FSR pass.
class Pass {
public:
    void Create(vk::Device device, VmaAllocator allocator, u32 num_images);

    /// Returns the scaled picture, or input when DLSS is off or cannot scale this size.
    vk::ImageView Render(vk::CommandBuffer cmdbuf, vk::Image input_image, vk::ImageView input,
                         vk::Format input_format, vk::Extent2D input_size,
                         vk::Extent2D output_size);

private:
    struct Img {
        u8 id{};
        bool dirty{true};
        VideoCore::UniqueImage output_image;
        vk::UniqueImageView output_image_view;
    };

    void CreateImages(Img& img) const;

    vk::Device device{};
    VmaAllocator allocator{};
    vk::Extent2D cur_size{};
    u32 cur_image{};
    std::vector<Img> available_imgs;
    bool failed{};
};

} // namespace Vulkan::RytyDlss
