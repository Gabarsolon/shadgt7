// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <windows.h>

#include <algorithm>
#include <cstring>
#include <string>

#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "ryty_gpu/ryty_dlss.h"
#include "ryty_gpu/ryty_dlssg.h"
#include "ryty_gpu/rygt_dlss_api.h"
#include "video_core/renderer_vulkan/vk_platform.h"

namespace Vulkan::RytyDlss {

namespace {

struct Plugin {
    HMODULE module{};
    PFN_RygtDlss_GetInstanceExtensions get_instance_extensions{};
    PFN_RygtDlss_GetDeviceExtensions get_device_extensions{};
    PFN_RygtDlss_Init init{};
    PFN_RygtDlss_Supports supports{};
    PFN_RygtDlss_Evaluate evaluate{};
    PFN_RygtDlss_Shutdown shutdown{};
    PFN_RygtDlss_LastError last_error{};
    bool tried{};
    bool ready{}; // Init succeeded
};
// Upscaling through Streamline's sl.dlss.dll (ryty_dlssg.cpp) instead of the plugin.
bool g_streamline{};
Plugin g_plugin;

std::string LastError() {
    return g_plugin.last_error ? g_plugin.last_error() : std::string{"(no plugin)"};
}

} // namespace

bool Load() {
    if (g_plugin.tried) {
        return g_plugin.module != nullptr;
    }
    g_plugin.tried = true;
    if (!EmulatorSettings.IsDlssEnabled()) {
        return false;
    }
    if (RytyDlssG::UpscaleRequested()) {
        // NVIDIA's own sl.dlss.dll is in the dlss folder: upscale through Streamline.
        return false;
    }
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir{exe};
    dir = dir.substr(0, dir.find_last_of(L"\\/"));
    const std::wstring path = dir + L"\\dlss\\ryGT_dlss.dll";
    HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        LOG_WARNING(Render_Vulkan,
                    "ryGT DLSS: dlss_enabled is on but neither dlss\\sl.dlss.dll (the "
                    "launcher's Download NVIDIA DLSS button) nor dlss\\ryGT_dlss.dll could be "
                    "loaded (error {}); using FSR",
                    GetLastError());
        return false;
    }
    const auto version = reinterpret_cast<PFN_RygtDlss_ApiVersion>(
        reinterpret_cast<void*>(GetProcAddress(module, "RygtDlss_ApiVersion")));
    if (!version || version() != RYGT_DLSS_API_VERSION) {
        LOG_WARNING(Render_Vulkan, "ryGT DLSS: the plugin has another interface version; using FSR");
        FreeLibrary(module);
        return false;
    }
#define LOAD(field, name)                                                                          \
    g_plugin.field = reinterpret_cast<decltype(g_plugin.field)>(                                   \
        reinterpret_cast<void*>(GetProcAddress(module, name)))
    LOAD(get_instance_extensions, "RygtDlss_GetInstanceExtensions");
    LOAD(get_device_extensions, "RygtDlss_GetDeviceExtensions");
    LOAD(init, "RygtDlss_Init");
    LOAD(supports, "RygtDlss_Supports");
    LOAD(evaluate, "RygtDlss_Evaluate");
    LOAD(shutdown, "RygtDlss_Shutdown");
    LOAD(last_error, "RygtDlss_LastError");
#undef LOAD
    if (!g_plugin.get_instance_extensions || !g_plugin.get_device_extensions || !g_plugin.init ||
        !g_plugin.supports || !g_plugin.evaluate || !g_plugin.shutdown || !g_plugin.last_error) {
        LOG_WARNING(Render_Vulkan, "ryGT DLSS: the plugin is missing functions; using FSR");
        FreeLibrary(module);
        g_plugin = Plugin{.tried = true};
        return false;
    }
    g_plugin.module = module;
    LOG_INFO(Render_Vulkan, "ryGT DLSS: plugin loaded");
    return true;
}

void AddInstanceExtensions(std::vector<const char*>& extensions,
                           const std::vector<vk::ExtensionProperties>& available) {
    if (!Load()) {
        return;
    }
    uint32_t count = 0;
    const char* const* names = nullptr;
    if (g_plugin.get_instance_extensions(&count, &names) != 0) {
        LOG_WARNING(Render_Vulkan, "ryGT DLSS: {}", LastError());
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        const bool offered = std::ranges::any_of(available, [&](const vk::ExtensionProperties& p) {
            return std::strcmp(p.extensionName, names[i]) == 0;
        });
        const bool present = std::ranges::any_of(
            extensions, [&](const char* e) { return std::strcmp(e, names[i]) == 0; });
        if (offered && !present) {
            extensions.push_back(names[i]);
        }
        LOG_INFO(Render_Vulkan, "ryGT DLSS: instance extension {}{}", names[i],
                 offered ? "" : " (not offered)");
    }
}

void AddDeviceExtensions(vk::Instance instance, vk::PhysicalDevice physical_device,
                         const std::function<bool(std::string_view)>& add_extension) {
    if (!Load()) {
        return;
    }
    uint32_t count = 0;
    const char* const* names = nullptr;
    if (g_plugin.get_device_extensions(instance, physical_device,
                                       VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr, &count,
                                       &names) != 0) {
        LOG_WARNING(Render_Vulkan, "ryGT DLSS: {}", LastError());
        return;
    }
    for (uint32_t i = 0; i < count; ++i) {
        add_extension(names[i]);
    }
}

bool Init(vk::Instance instance, vk::PhysicalDevice physical_device, vk::Device device) {
    if (g_plugin.ready || g_streamline) {
        return true;
    }
    if (RytyDlssG::InitUpscale()) {
        g_streamline = true;
        return true;
    }
    if (!g_plugin.module) {
        return false;
    }
    if (g_plugin.init(instance, physical_device, device,
                      VULKAN_HPP_DEFAULT_DISPATCHER.vkGetInstanceProcAddr,
                      VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr, nullptr) != 0) {
        LOG_WARNING(Render_Vulkan, "ryGT DLSS: {}; using FSR", LastError());
        return false;
    }
    g_plugin.ready = true;
    LOG_INFO(Render_Vulkan, "ryGT DLSS: ready");
    return true;
}

void Shutdown() {
    if (g_plugin.ready) {
        g_plugin.shutdown();
        g_plugin.ready = false;
    }
}

void Pass::Create(vk::Device device_, VmaAllocator allocator_, u32 num_images) {
    device = device_;
    allocator = allocator_;
    available_imgs.resize(num_images);
    for (u32 i = 0; i < num_images; ++i) {
        available_imgs[i].id = static_cast<u8>(i);
        available_imgs[i].output_image = VideoCore::UniqueImage(device, allocator_);
    }
}

vk::ImageView Pass::Render(vk::CommandBuffer cmdbuf, vk::Image input_image, vk::ImageView input,
                           vk::Format input_format, vk::Extent2D input_size,
                           vk::Extent2D output_size) {
    if ((!g_plugin.ready && !g_streamline) || failed) {
        return input;
    }
    const bool supported =
        g_streamline ? RytyDlssG::UpscaleSupports(input_size, output_size)
                     : g_plugin.supports(input_size.width, input_size.height, output_size.width,
                                         output_size.height) != 0;
    if (!supported) {
        return input;
    }
    if (output_size != cur_size) {
        cur_size = output_size;
        for (auto& img : available_imgs) {
            img.dirty = true;
        }
    }
    auto& img = available_imgs[cur_image];
    cur_image = (cur_image + 1) % available_imgs.size();
    if (img.dirty) {
        CreateImages(img);
    }

    constexpr vk::ImageSubresourceRange range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    // The game picture is already readable by shaders; make it visible to DLSS's compute work, and
    // the output writable.
    const vk::MemoryBarrier2 input_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
    };
    const vk::ImageMemoryBarrier2 output_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = vk::AccessFlagBits2::eShaderRead,
        .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .dstAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .oldLayout = vk::ImageLayout::eUndefined,
        .newLayout = vk::ImageLayout::eGeneral,
        .image = img.output_image,
        .subresourceRange = range,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &input_barrier,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &output_barrier,
    });

    const RygtDlssImage in{
        .image = static_cast<VkImage>(input_image),
        .view = static_cast<VkImageView>(input),
        .format = static_cast<VkFormat>(input_format),
        .width = input_size.width,
        .height = input_size.height,
    };
    const RygtDlssImage out{
        .image = static_cast<VkImage>(static_cast<vk::Image>(img.output_image)),
        .view = static_cast<VkImageView>(img.output_image_view.get()),
        .format = VK_FORMAT_R16G16B16A16_SFLOAT,
        .width = output_size.width,
        .height = output_size.height,
    };
    const bool ok =
        g_streamline
            ? RytyDlssG::Upscale(device, allocator, cmdbuf, input_image, input, input_format,
                                 input_size, img.output_image, img.output_image_view.get(),
                                 vk::Format::eR16G16B16A16Sfloat, output_size)
            : g_plugin.evaluate(static_cast<VkCommandBuffer>(cmdbuf), &in, &out, 0.0f) == 0;
    if (!ok) {
        // The output was not written; stop trying and let FSR (or plain scaling) take over.
        LOG_ERROR(Render_Vulkan, "ryGT DLSS: {}; falling back to FSR",
                  g_streamline ? std::string{"Streamline's DLSS failed"} : LastError());
        failed = true;
    }

    const vk::ImageMemoryBarrier2 return_barrier{
        .srcStageMask = vk::PipelineStageFlagBits2::eComputeShader,
        .srcAccessMask = vk::AccessFlagBits2::eShaderStorageWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
        .oldLayout = vk::ImageLayout::eGeneral,
        .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
        .image = img.output_image,
        .subresourceRange = range,
    };
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &return_barrier,
    });
    // Logged again whenever the game or window size changes.
    static vk::Extent2D logged_in{}, logged_out{};
    if (ok && (input_size != logged_in || output_size != logged_out)) {
        logged_in = input_size;
        logged_out = output_size;
        LOG_INFO(Render_Vulkan, "ryGT DLSS: scaling {}x{} to {}x{}", input_size.width,
                 input_size.height, output_size.width, output_size.height);
    }
    return ok ? img.output_image_view.get() : input;
}

void Pass::CreateImages(Img& img) const {
    img.dirty = false;
    img.output_image_view.reset();
    img.output_image.Destroy();
    const vk::ImageCreateInfo info{
        .imageType = vk::ImageType::e2D,
        .format = vk::Format::eR16G16B16A16Sfloat,
        .extent{.width = cur_size.width, .height = cur_size.height, .depth = 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = vk::SampleCountFlagBits::e1,
        .usage = vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eSampled |
                 vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eColorAttachment,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    img.output_image.Create(info);
    SetObjectName(device, static_cast<vk::Image>(img.output_image), "DLSS Output Image #{}",
                  img.id);
    img.output_image_view = Check<"create dlss output image view">(
        device.createImageViewUnique(vk::ImageViewCreateInfo{
            .image = img.output_image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            },
        }));
}

} // namespace Vulkan::RytyDlss
