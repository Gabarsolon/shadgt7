// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// GPU.hd_ui, the UI drawn at the output size. After a flip, the first draw into a display buffer
// is the scene composite. Before the next draw into it, DLSS (or a linear blit when DLSS cannot
// scale) scales that picture into a proxy image at the output size, and the remaining draws into
// the display buffer (the UI) render into the proxy with their viewport and scissor scaled. The
// presenter then shows the proxy instead of scaling the display buffer. Unsupported draws resolve
// back and give the proxy up for the frame. Draws that only read the display buffer resolve back
// and keep the proxy. Depth attachments and further color attachments of a UI draw get proxies of
// their own (vk_ui_hires_depth.h). The proxies are only made when the output is bigger than the
// display buffer, and only while their memory stays within MaxProxyBytes.

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <mutex>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/hd_ui_options.h"
#include "common/logging/log.h"
#include "ryty_gpu/ryty_dlss.h"
#include "ryty_gpu/ryty_ui_cache_hires.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {
float RytyDisplayAspect(); // ryty_game_patch.cpp (GTS-009): > 0 with a wider-than-16:9 screen
}

namespace Vulkan::UiHires {

/// The most video memory the proxies of all display buffers, and the ones waiting to be freed,
/// may hold. A draw that would pass it takes the native path, as with the setting off.
constexpr u64 MaxProxyBytes = u64{1} << 30;

/// How many flips a retired proxy waits before it is freed: the GPU may still read it.
constexpr u64 RetireFrames = 8;

enum class Phase {
    Idle,  ///< nothing drawn into the display buffer since its last flip
    Scene, ///< the scene composite was drawn into the display buffer
    Ui,    ///< the proxy holds the picture; display buffer draws go to the proxy
    Off,   ///< the proxy was given up for this frame; the display buffer holds the picture
};

struct DepthProxy {
    VideoCore::UniqueImage image;
    vk::UniqueImageView view;
    vk::Format format{};
    vk::ImageAspectFlags aspects{};
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    vk::Image source{};
    u64 frame = ~u64{0};
    u64 version = ~u64{0};
};

// Extra display MRT outputs use the same raster scale as slot 0. They are mirrored to
// native immediately; consumers outside the display path retain the guest dimensions.
struct AuxiliaryProxy {
    VideoCore::UniqueImage image;
    vk::UniqueImageView view;
    vk::Format format{};
    vk::Extent2D size{};
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    vk::Image source{};
    u64 frame = ~u64{0};
    u64 version = ~u64{0};
};

struct Proxy {
    vk::Extent2D native{}; ///< the display buffer's size as the presenter shows it
    vk::Extent2D size{};   ///< the output size; empty when not used
    vk::Format format{};
    vk::Format present_format{}; ///< the format the presenter samples the proxy as
    VideoCore::UniqueImage image;
    VideoCore::UniqueImage scaled; ///< DLSS output (R16G16B16A16), copied into image
    vk::UniqueImageView scaled_view;
    std::vector<std::pair<std::pair<vk::Format, bool>, vk::UniqueImageView>> views;
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    Phase phase = Phase::Idle;
    bool dlss_failed = false;
    bool dlss_used = false;
    bool native_stale = false; ///< the proxy got draws the display buffer has not
    bool native_dirty = false; ///< native draws must be copied back to the proxy
    std::shared_ptr<DepthProxy> depth;
    std::array<std::shared_ptr<AuxiliaryProxy>, 8> auxiliary;
};

struct Retired {
    VideoCore::UniqueImage image;
    VideoCore::UniqueImage scaled;
    vk::UniqueImageView scaled_view;
    std::vector<std::pair<std::pair<vk::Format, bool>, vk::UniqueImageView>> views;
    u64 frame = 0;
    vk::Extent2D size{};
    vk::Format format{};
    std::shared_ptr<DepthProxy> depth;
    std::array<std::shared_ptr<AuxiliaryProxy>, 8> auxiliary;
};

struct State {
    std::mutex mutex;
    std::unordered_map<VAddr, Proxy> proxies;
    std::deque<Retired> retired; ///< freed some frames later: the GPU may still read them
    u64 frames = 0;
    /// Nonzero while a draw into the proxy is recorded: its viewport and scissor scale.
    float scale = 0.0f;
    /// Lock-free checks for every draw: the display buffer addresses with a proxy, and how
    /// many proxies hold draws their display buffer has not.
    std::array<std::atomic<VAddr>, 8> addresses{};
    std::atomic<u32> stale{};
};

inline State& Get() {
    static State state;
    return state;
}

inline void MarkStale(Proxy& proxy, bool stale) {
    if (proxy.native_stale != stale) {
        proxy.native_stale = stale;
        if (stale) {
            Get().stale.fetch_add(1, std::memory_order_relaxed);
        } else {
            Get().stale.fetch_sub(1, std::memory_order_relaxed);
        }
    }
}

inline bool KnownAddress(VAddr address) {
    for (const auto& known : Get().addresses) {
        if (known.load(std::memory_order_relaxed) == address) {
            return true;
        }
    }
    return false;
}

/// GPU.hd_ui, except with an ultrawide screen: GT7-UW lays the HUD out itself there.
inline bool Enabled() {
    static const bool enabled = [] {
        const bool on = Common::HdUiRequested();
        if (on && RytyDisplayAspect() > 0.0f) {
            LOG_INFO(Render_Vulkan, "HDUI: UI at the window size is off with an ultrawide screen");
            return false;
        }
        return on;
    }();
    return enabled;
}

inline u64 FormatBytes(vk::Format format) {
    switch (format) {
    case vk::Format::eR16G16B16A16Sfloat:
    case vk::Format::eR16G16B16A16Unorm:
    case vk::Format::eD32SfloatS8Uint:
        return 8;
    case vk::Format::eR32G32B32A32Sfloat:
        return 16;
    case vk::Format::eD16Unorm:
        return 2;
    default:
        return 4;
    }
}

inline u64 ImageBytes(vk::Extent2D size, vk::Format format) {
    return u64{size.width} * size.height * FormatBytes(format);
}

inline u64 Footprint(vk::Extent2D size, vk::Format format, bool image, bool scaled,
                     const DepthProxy* depth,
                     const std::array<std::shared_ptr<AuxiliaryProxy>, 8>& auxiliary) {
    u64 bytes = image ? ImageBytes(size, format) : 0;
    if (scaled) {
        bytes += ImageBytes(size, vk::Format::eR16G16B16A16Sfloat);
    }
    if (depth) {
        bytes += ImageBytes(size, depth->format);
    }
    for (const auto& color : auxiliary) {
        if (color) {
            bytes += ImageBytes(color->size, color->format);
        }
    }
    return bytes;
}

/// The video memory of all proxies and retired proxies. With State::mutex held.
inline u64 TotalBytes() {
    const auto& state = Get();
    u64 total = 0;
    for (const auto& [address, proxy] : state.proxies) {
        total += Footprint(proxy.size, proxy.format, bool(proxy.image), bool(proxy.scaled),
                           proxy.depth.get(), proxy.auxiliary);
    }
    for (const auto& old : state.retired) {
        total += Footprint(old.size, old.format, bool(old.image), bool(old.scaled),
                           old.depth.get(), old.auxiliary);
    }
    return total;
}

/// Whether `bytes` more fit in MaxProxyBytes. With State::mutex held.
inline bool WithinBudget(u64 bytes) {
    if (TotalBytes() + bytes <= MaxProxyBytes) {
        return true;
    }
    static std::atomic<bool> logged{};
    if (!logged.exchange(true)) {
        LOG_WARNING(Render_Vulkan,
                    "HDUI: the UI proxies would pass {} MiB of video memory, the UI of this window "
                    "size is drawn at the display buffer's size",
                    MaxProxyBytes >> 20);
    }
    return false;
}

// Returns true once per reason, so a caller can log it once.
inline bool Unsupported(VAddr address, const char* reason, u64 frame) {
    static std::mutex mutex;
    static std::vector<std::string> seen;
    std::scoped_lock lock{mutex};
    for (const auto& known : seen) {
        if (known == reason) {
            return false;
        }
    }
    seen.emplace_back(reason);
    LOG_INFO(Render_Vulkan, "HDUI: display {:#x} falls back to the native UI draw in frame {}: {}",
             address, frame, reason);
    return true;
}

inline void Retire(State& state, Proxy& proxy) {
    if (!proxy.image && !proxy.scaled) {
        return;
    }
    state.retired.push_back(Retired{std::move(proxy.image), std::move(proxy.scaled),
                                    std::move(proxy.scaled_view), std::move(proxy.views),
                                    state.frames, proxy.size, proxy.format, std::move(proxy.depth),
                                    std::move(proxy.auxiliary)});
    proxy.views.clear();
    proxy.layout = vk::ImageLayout::eUndefined;
}

inline vk::ImageView View(const Instance& instance, Proxy& proxy, vk::Format format,
                          bool alpha_one) {
    const auto key = std::make_pair(format, alpha_one);
    for (const auto& [k, view] : proxy.views) {
        if (k == key) {
            return *view;
        }
    }
    auto view = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
        .image = proxy.image,
        .viewType = vk::ImageViewType::e2D,
        .format = format,
        .components{.a = alpha_one ? vk::ComponentSwizzle::eOne : vk::ComponentSwizzle::eIdentity},
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        },
    });
    if (view.result != vk::Result::eSuccess) {
        return {};
    }
    const vk::ImageView handle = *view.value;
    proxy.views.emplace_back(key, std::move(view.value));
    return handle;
}

inline bool CreateImages(const Instance& instance, Proxy& proxy, vk::Format format) {
    if (proxy.image && proxy.format == format) {
        return true;
    }
    Retire(Get(), proxy);
    if (!WithinBudget(ImageBytes(proxy.size, format))) {
        return false;
    }
    proxy.format = format;
    proxy.image = VideoCore::UniqueImage(instance.GetDevice(), instance.GetAllocator());
    proxy.image.Create(vk::ImageCreateInfo{
        .flags = vk::ImageCreateFlagBits::eMutableFormat,
        .imageType = vk::ImageType::e2D,
        .format = format,
        .extent{.width = proxy.size.width, .height = proxy.size.height, .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    proxy.layout = vk::ImageLayout::eUndefined;
    LOG_INFO(Render_Vulkan, "HDUI: proxy {}x{} for a {}x{} display buffer, {}", proxy.size.width,
             proxy.size.height, proxy.native.width, proxy.native.height, vk::to_string(format));
    return true;
}

/// The DLSS output of a proxy, made when DLSS first scales into it (never without DLSS).
inline bool CreateScaled(const Instance& instance, Proxy& proxy) {
    if (proxy.scaled_view) {
        return true;
    }
    if (!WithinBudget(ImageBytes(proxy.size, vk::Format::eR16G16B16A16Sfloat))) {
        return false;
    }
    proxy.scaled = VideoCore::UniqueImage(instance.GetDevice(), instance.GetAllocator());
    proxy.scaled.Create(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent{.width = proxy.size.width, .height = proxy.size.height, .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eTransferSrc,
        .initialLayout = vk::ImageLayout::eUndefined,
    });
    auto view = instance.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
        .image = proxy.scaled,
        .viewType = vk::ImageViewType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .subresourceRange{
            .aspectMask = vk::ImageAspectFlagBits::eColor,
            .levelCount = 1,
            .layerCount = 1,
        },
    });
    if (view.result != vk::Result::eSuccess) {
        proxy.scaled = VideoCore::UniqueImage(instance.GetDevice(), instance.GetAllocator());
        return false;
    }
    proxy.scaled_view = std::move(view.value);
    return true;
}

constexpr vk::ImageSubresourceRange ColorRange{
    .aspectMask = vk::ImageAspectFlagBits::eColor,
    .levelCount = 1,
    .layerCount = 1,
};

inline void ProxyBarrier(vk::CommandBuffer cmdbuf, Proxy& proxy, vk::ImageLayout layout,
                         vk::PipelineStageFlags2 dst_stage, vk::AccessFlags2 dst_access) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = dst_stage,
        .dstAccessMask = dst_access,
        .oldLayout = proxy.layout,
        .newLayout = layout,
        .image = proxy.image,
        .subresourceRange = ColorRange,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    });
    proxy.layout = layout;
}

constexpr vk::PipelineStageFlags2 DrawStages =
    vk::PipelineStageFlagBits2::eColorAttachmentOutput |
    vk::PipelineStageFlagBits2::eFragmentShader | vk::PipelineStageFlagBits2::eVertexShader;
constexpr vk::AccessFlags2 DrawAccess = vk::AccessFlagBits2::eColorAttachmentRead |
                                        vk::AccessFlagBits2::eColorAttachmentWrite |
                                        vk::AccessFlagBits2::eShaderSampledRead;

inline void NativeBarrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout from,
                          vk::ImageLayout to) {
    const vk::ImageMemoryBarrier2 barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = from,
        .newLayout = to,
        .image = image,
        .subresourceRange = ColorRange,
    };
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier});
}

inline vk::ImageBlit2 NativeBlit(vk::Extent2D from, vk::Extent2D to) {
    return vk::ImageBlit2{
        .srcSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .srcOffsets =
            std::array{vk::Offset3D{}, vk::Offset3D{s32(from.width), s32(from.height), 1}},
        .dstSubresource{.aspectMask = vk::ImageAspectFlagBits::eColor, .layerCount = 1},
        .dstOffsets = std::array{vk::Offset3D{}, vk::Offset3D{s32(to.width), s32(to.height), 1}},
    };
}

/// The scene composite in the native display image (general layout) becomes the proxy's
/// picture: DLSS to the output size, or a linear blit. `allow_dlss` is off for a picture that
/// has UI in it already.
inline void UpscaleNative(const Instance& instance, vk::CommandBuffer cmd, vk::Image image,
                          vk::ImageView view, vk::Format format, Proxy& proxy,
                          bool allow_dlss = true) {
    const bool dlss = allow_dlss && !proxy.dlss_failed &&
                      RytyDlss::CanScale(proxy.native, proxy.size) && CreateScaled(instance, proxy);
    const auto input_layout =
        dlss ? vk::ImageLayout::eShaderReadOnlyOptimal : vk::ImageLayout::eTransferSrcOptimal;
    NativeBarrier(cmd, image, vk::ImageLayout::eGeneral, input_layout);
    const bool scaled = dlss && RytyDlss::ScaleInto(cmd, image, view, format, proxy.native,
                                                    proxy.scaled, *proxy.scaled_view, proxy.size);
    if (dlss && !scaled) {
        proxy.dlss_failed = true;
        NativeBarrier(cmd, image, input_layout, vk::ImageLayout::eTransferSrcOptimal);
    }
    proxy.dlss_used = scaled;
    ProxyBarrier(cmd, proxy, vk::ImageLayout::eTransferDstOptimal,
                 vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite);
    const auto region = NativeBlit(scaled ? proxy.size : proxy.native, proxy.size);
    if (scaled) {
        NativeBarrier(cmd, proxy.scaled, vk::ImageLayout::eGeneral,
                      vk::ImageLayout::eTransferSrcOptimal);
    }
    cmd.blitImage2(vk::BlitImageInfo2{
        .srcImage = scaled ? vk::Image{proxy.scaled} : image,
        .srcImageLayout = vk::ImageLayout::eTransferSrcOptimal,
        .dstImage = proxy.image,
        .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
        .regionCount = 1,
        .pRegions = &region,
        .filter = scaled ? vk::Filter::eNearest : vk::Filter::eLinear,
    });
    NativeBarrier(cmd, image, scaled ? input_layout : vk::ImageLayout::eTransferSrcOptimal,
                  vk::ImageLayout::eGeneral);
    ProxyBarrier(cmd, proxy, vk::ImageLayout::eGeneral, DrawStages, DrawAccess);
    proxy.phase = Phase::Ui;
    proxy.native_dirty = false;
}

/// The proxy's picture goes back into the native display image; with `keep`, the proxy stays in
/// use (the display image only gets a copy for what reads it).
inline void ResolveNative(vk::CommandBuffer cmd, vk::Image image, Proxy& proxy, bool keep) {
    if (proxy.native_stale) {
        ProxyBarrier(cmd, proxy, vk::ImageLayout::eTransferSrcOptimal,
                     vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferRead);
        NativeBarrier(cmd, image, vk::ImageLayout::eGeneral, vk::ImageLayout::eTransferDstOptimal);
        const auto region = NativeBlit(proxy.size, proxy.native);
        cmd.blitImage2(vk::BlitImageInfo2{
            .srcImage = proxy.image,
            .srcImageLayout = vk::ImageLayout::eTransferSrcOptimal,
            .dstImage = image,
            .dstImageLayout = vk::ImageLayout::eTransferDstOptimal,
            .regionCount = 1,
            .pRegions = &region,
            .filter = vk::Filter::eLinear,
        });
        NativeBarrier(cmd, image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eGeneral);
        ProxyBarrier(cmd, proxy, vk::ImageLayout::eGeneral, DrawStages, DrawAccess);
        MarkStale(proxy, false);
    }
    if (!keep) {
        proxy.phase = Phase::Off;
    }
}

struct Shown {
    vk::ImageView view;
    vk::Extent2D size;
};

/// At a flip of the display buffer at `address`: the proxy to show instead, if it holds the
/// picture. Also sizes the proxy for the next frame from the display buffer and output sizes.
inline std::optional<Shown> TakeFrame(const Instance& instance, vk::CommandBuffer cmdbuf,
                                      VAddr address, vk::Extent2D native, vk::Extent2D output,
                                      vk::Format view_format, vk::Image native_image) {
    if (!Enabled()) {
        return std::nullopt;
    }
    auto& state = Get();
    std::scoped_lock lock{state.mutex};
    ++state.frames;
    while (!state.retired.empty() && state.frames - state.retired.front().frame > RetireFrames) {
        state.retired.pop_front();
    }
    auto& proxy = state.proxies[address];
    if (!KnownAddress(address)) {
        for (auto& known : state.addresses) {
            VAddr empty = 0;
            if (known.compare_exchange_strong(empty, address)) {
                break;
            }
        }
    }
    std::optional<Shown> shown;
    // A native draw was last: copy it back before showing the proxy. A picture that already
    // contains UI never goes through DLSS.
    if (proxy.native_dirty && proxy.image && proxy.size == output && proxy.native == native &&
        proxy.phase == Phase::Scene) {
        UpscaleNative(instance, cmdbuf, native_image, {}, view_format, proxy, false);
    }
    if (proxy.phase == Phase::Ui && proxy.image && proxy.size == output &&
        proxy.native == native) {
        ProxyBarrier(cmdbuf, proxy, vk::ImageLayout::eShaderReadOnlyOptimal,
                     vk::PipelineStageFlagBits2::eFragmentShader |
                         vk::PipelineStageFlagBits2::eComputeShader,
                     vk::AccessFlagBits2::eShaderRead);
        shown = Shown{View(instance, proxy, view_format, true), proxy.size};
    }
    proxy.phase = Phase::Idle;
    proxy.native_dirty = false;
    MarkStale(proxy, false);
    // The output must be bigger than the display buffer, and have its aspect ratio.
    const bool wanted = output.width > native.width && output.height > native.height &&
                        std::abs(s64(output.width) * native.height -
                                 s64(output.height) * native.width) <= s64(native.height);
    const vk::Extent2D size = wanted ? output : vk::Extent2D{};
    Ryty::UiCache::Output(native.width, native.height, output.width, output.height);
    if (proxy.size != size || proxy.native != native || proxy.present_format != view_format) {
        Retire(state, proxy);
        proxy.present_format = view_format;
        proxy.size = size;
        proxy.native = native;
        proxy.dlss_failed = false;
    }
    return shown;
}

/// Viewport and scissor of a draw into the proxy.
inline void ScaleViewport(vk::Viewport& viewport, vk::Rect2D& scissor) {
    const float scale = Get().scale;
    if (scale == 0.0f) {
        return;
    }
    viewport.x *= scale;
    viewport.y *= scale;
    viewport.width *= scale;
    viewport.height *= scale;
    // Round the scissor outwards, so that no pixel of the guest's rectangle is cut off.
    const s64 x0 = s64(std::floor(scissor.offset.x * scale));
    const s64 y0 = s64(std::floor(scissor.offset.y * scale));
    const s64 x1 = s64(std::ceil((s64(scissor.offset.x) + scissor.extent.width) * scale));
    const s64 y1 = s64(std::ceil((s64(scissor.offset.y) + scissor.extent.height) * scale));
    scissor.offset.x = s32(std::clamp<s64>(x0, 0, 32767));
    scissor.offset.y = s32(std::clamp<s64>(y0, 0, 32767));
    scissor.extent.width = u32(std::clamp<s64>(x1 - scissor.offset.x, 0, 32767));
    scissor.extent.height = u32(std::clamp<s64>(y1 - scissor.offset.y, 0, 32767));
}

} // namespace Vulkan::UiHires
