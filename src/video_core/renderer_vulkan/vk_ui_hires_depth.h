// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "video_core/renderer_vulkan/vk_ui_hires.h"

// The depth and the extra color attachments of a UI draw (see vk_ui_hires.h): each gets a proxy
// at the output size, copied up from the native image before the draw and mirrored back to it
// right after, so that the UI draws that test depth or write further outputs keep working.

namespace Vulkan::UiHires {

inline vk::ImageAspectFlags DepthAspects(vk::Format format) {
    switch (format) {
    case vk::Format::eD16Unorm:
    case vk::Format::eX8D24UnormPack32:
    case vk::Format::eD32Sfloat:
        return vk::ImageAspectFlagBits::eDepth;
    case vk::Format::eD16UnormS8Uint:
    case vk::Format::eD24UnormS8Uint:
    case vk::Format::eD32SfloatS8Uint:
        return vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil;
    case vk::Format::eS8Uint:
        return vk::ImageAspectFlagBits::eStencil;
    default:
        return {};
    }
}

inline void DepthBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageAspectFlags aspects,
                         vk::ImageLayout from, vk::ImageLayout to) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = from,
        .newLayout = to,
        .image = image,
        .subresourceRange{.aspectMask = aspects, .levelCount = 1, .layerCount = 1},
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

inline bool CreateDepth(const Instance& instance, Proxy& proxy, vk::Format format) {
    if (proxy.depth && proxy.depth->format == format) {
        return bool(proxy.depth->view);
    }
    if (proxy.depth) {
        Retired retired;
        retired.frame = Get().frames;
        retired.size = proxy.size;
        retired.depth = std::move(proxy.depth);
        Get().retired.push_back(std::move(retired));
    }
    if (!WithinBudget(ImageBytes(proxy.size, format))) {
        return false;
    }
    auto depth = std::make_shared<DepthProxy>();
    depth->format = format;
    depth->aspects = DepthAspects(format);
    depth->image = VideoCore::UniqueImage(instance.GetDevice(), instance.GetAllocator());
    depth->image.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent{.width = proxy.size.width, .height = proxy.size.height, .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    auto view = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
        .image = depth->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange{.aspectMask = depth->aspects, .levelCount = 1, .layerCount = 1},
    });
    if (view.result == vk::Result::eSuccess) {
        depth->view = std::move(view.value);
        LOG_INFO(Render_Vulkan, "HDUI: depth proxy {}x{} {}", proxy.size.width, proxy.size.height,
                 vk::to_string(format));
    }
    proxy.depth = std::move(depth);
    return bool(proxy.depth->view);
}

// Same physical VkFormat in both directions; never reinterpret D32S8 as color or invert Z.
// Separate regions copy depth and stencil. Vulkan depth/stencil blits require nearest filtering.
inline void CopyDepth(vk::CommandBuffer cmd, vk::Image native, Proxy& proxy, bool to_native) {
    auto& depth = *proxy.depth;
    const auto src_layout = vk::ImageLayout::eTransferSrcOptimal;
    const auto dst_layout = vk::ImageLayout::eTransferDstOptimal;
    DepthBarrier(cmd, native, depth.aspects, vk::ImageLayout::eGeneral,
                 to_native ? dst_layout : src_layout);
    DepthBarrier(cmd, depth.image, depth.aspects, depth.layout,
                 to_native ? src_layout : dst_layout);
    std::array<vk::ImageBlit2, 2> regions{};
    u32 count = 0;
    for (const auto aspect : {vk::ImageAspectFlagBits::eDepth, vk::ImageAspectFlagBits::eStencil}) {
        if (!(depth.aspects & aspect)) {
            continue;
        }
        auto region = NativeBlit(to_native ? proxy.size : proxy.native,
                                 to_native ? proxy.native : proxy.size);
        region.srcSubresource.aspectMask = aspect;
        region.dstSubresource.aspectMask = aspect;
        regions[count++] = region;
    }
    cmd.blitImage2(vk::BlitImageInfo2{
        .srcImage = to_native ? vk::Image{depth.image} : native,
        .srcImageLayout = src_layout,
        .dstImage = to_native ? native : vk::Image{depth.image},
        .dstImageLayout = dst_layout,
        .regionCount = count,
        .pRegions = regions.data(),
        .filter = vk::Filter::eNearest,
    });
    DepthBarrier(cmd, native, depth.aspects, to_native ? dst_layout : src_layout,
                 vk::ImageLayout::eGeneral);
    DepthBarrier(cmd, depth.image, depth.aspects, to_native ? src_layout : dst_layout,
                 vk::ImageLayout::eGeneral);
    depth.layout = vk::ImageLayout::eGeneral;
}

inline bool CreateAuxiliary(const Instance& instance, Proxy& proxy, u32 slot, vk::Format format) {
    auto& cached = proxy.auxiliary[slot];
    if (cached && cached->format == format && cached->size == proxy.size) {
        return bool(cached->view);
    }
    if (cached) {
        Retired retired;
        retired.frame = Get().frames;
        retired.size = proxy.size;
        retired.auxiliary[slot] = std::move(cached);
        Get().retired.push_back(std::move(retired));
    }
    if (!WithinBudget(ImageBytes(proxy.size, format))) {
        return false;
    }
    auto color = std::make_shared<AuxiliaryProxy>();
    color->format = format;
    color->size = proxy.size;
    color->image = VideoCore::UniqueImage(instance.GetDevice(), instance.GetAllocator());
    color->image.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent{proxy.size.width, proxy.size.height, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eTransferSrc |
                 vk::ImageUsageFlagBits::eTransferDst,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    auto view = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
        .image = color->image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .subresourceRange = ColorRange,
    });
    if (view.result == vk::Result::eSuccess) {
        color->view = std::move(view.value);
    }
    cached = std::move(color);
    return bool(cached->view);
}

inline void CopyAuxiliary(vk::CommandBuffer cmd, vk::Image native, Proxy& proxy, u32 slot,
                          bool to_native) {
    auto& color = *proxy.auxiliary[slot];
    const auto src = vk::ImageLayout::eTransferSrcOptimal;
    const auto dst = vk::ImageLayout::eTransferDstOptimal;
    NativeBarrier(cmd, native, vk::ImageLayout::eGeneral, to_native ? dst : src);
    NativeBarrier(cmd, color.image, color.layout, to_native ? src : dst);
    const auto region = NativeBlit(to_native ? proxy.size : proxy.native,
                                   to_native ? proxy.native : proxy.size);
    cmd.blitImage2(vk::BlitImageInfo2{
        .srcImage = to_native ? vk::Image{color.image} : native,
        .srcImageLayout = src,
        .dstImage = to_native ? native : vk::Image{color.image},
        .dstImageLayout = dst,
        .regionCount = 1,
        .pRegions = &region,
        .filter = vk::Filter::eNearest,
    });
    NativeBarrier(cmd, native, to_native ? dst : src, vk::ImageLayout::eGeneral);
    NativeBarrier(cmd, color.image, to_native ? src : dst, vk::ImageLayout::eGeneral);
    color.layout = vk::ImageLayout::eGeneral;
}

} // namespace Vulkan::UiHires
