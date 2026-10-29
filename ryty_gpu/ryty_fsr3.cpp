// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <windows.h>

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "ryty_gpu/ryty_upscale.h"
#include "ryty_gpu/ryty_fsr3.h"
#include "video_core/renderer_vulkan/vk_platform.h"

#include "ryty_gpu/ffx_api/ffx_api.h"
#include "ryty_gpu/ffx_api/ffx_upscale.h"
#include "ryty_gpu/ffx_api/vk/ffx_api_vk.h"

namespace Vulkan::RytyFsr3 {

namespace {

struct Api {
    HMODULE module{};
    PfnFfxCreateContext create_context{};
    PfnFfxDestroyContext destroy_context{};
    PfnFfxQuery query{};
    PfnFfxDispatch dispatch{};
    bool tried{};
};
Api g_api;

// The context is created for windows up to this size, so resizing rarely recreates it.
constexpr vk::Extent2D MinUpscaleSize{3840, 2160};

void OnMessage(uint32_t type, const wchar_t* message) {
    std::string text;
    for (const wchar_t* c = message; c && *c; ++c) {
        text.push_back(*c < 0x80 ? static_cast<char>(*c) : '?');
    }
    if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        LOG_ERROR(Render_Vulkan, "FSR 3.1: {}", text);
    } else {
        LOG_WARNING(Render_Vulkan, "FSR 3.1: {}", text);
    }
}

// AMD's backend loads some functions by their extension names (vkGetBufferMemoryRequirements2KHR)
// or from VK_EXT_debug_utils, which the renderer's device may not have enabled, and then calls them
// without checking. Extension names fall back to the core Vulkan function; debug labels do nothing.
void VKAPI_CALL NoBeginLabel(VkCommandBuffer, const VkDebugUtilsLabelEXT*) {}
void VKAPI_CALL NoEndLabel(VkCommandBuffer) {}
VkResult VKAPI_CALL NoObjectName(VkDevice, const VkDebugUtilsObjectNameInfoEXT*) {
    return VK_SUCCESS;
}

PFN_vkVoidFunction VKAPI_CALL GetDeviceProcAddr(VkDevice device, const char* name) {
    const auto real = VULKAN_HPP_DEFAULT_DISPATCHER.vkGetDeviceProcAddr;
    if (const auto function = real(device, name)) {
        return function;
    }
    const std::string_view view{name};
    if (view.ends_with("KHR")) {
        const std::string core{view.substr(0, view.size() - 3)};
        if (const auto function = real(device, core.c_str())) {
            return function;
        }
    }
    if (view == "vkCmdBeginDebugUtilsLabelEXT") {
        return reinterpret_cast<PFN_vkVoidFunction>(&NoBeginLabel);
    }
    if (view == "vkCmdEndDebugUtilsLabelEXT") {
        return reinterpret_cast<PFN_vkVoidFunction>(&NoEndLabel);
    }
    if (view == "vkSetDebugUtilsObjectNameEXT") {
        return reinterpret_cast<PFN_vkVoidFunction>(&NoObjectName);
    }
    LOG_INFO(Render_Vulkan, "FSR 3.1: {} is not available", view);
    return nullptr;
}

FfxApiResource Resource(vk::Image image, vk::Format format, vk::Extent2D size, uint32_t usage,
                        uint32_t state) {
    FfxApiResourceDescription description{};
    description.type = FFX_API_RESOURCE_TYPE_TEXTURE2D;
    description.format = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(format));
    description.width = size.width;
    description.height = size.height;
    description.depth = 1;
    description.mipCount = 1;
    description.flags = FFX_API_RESOURCE_FLAGS_NONE;
    description.usage = usage;
    return ffxApiGetResourceVK(static_cast<VkImage>(image), description, state);
}

} // namespace

PFN_vkGetDeviceProcAddr DeviceProcAddrShim() {
    return &GetDeviceProcAddr;
}

Pass::~Pass() {
    DestroyContext();
}

bool Pass::Load() {
    if (g_api.tried) {
        return g_api.module != nullptr;
    }
    g_api.tried = true;
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir{exe};
    dir = dir.substr(0, dir.find_last_of(L"\\/"));
    const std::wstring path = dir + L"\\amd_fidelityfx_vk.dll";
    HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        LOG_WARNING(Render_Vulkan,
                    "FSR 3.1: fsr3_enabled is on but amd_fidelityfx_vk.dll could not be loaded "
                    "(error {}); using FSR 1",
                    GetLastError());
        return false;
    }
#define LOAD(field, name)                                                                          \
    g_api.field = reinterpret_cast<decltype(g_api.field)>(                                         \
        reinterpret_cast<void*>(GetProcAddress(module, name)))
    LOAD(create_context, "ffxCreateContext");
    LOAD(destroy_context, "ffxDestroyContext");
    LOAD(query, "ffxQuery");
    LOAD(dispatch, "ffxDispatch");
#undef LOAD
    if (!g_api.create_context || !g_api.destroy_context || !g_api.query || !g_api.dispatch) {
        LOG_WARNING(Render_Vulkan, "FSR 3.1: amd_fidelityfx_vk.dll is missing functions; using FSR 1");
        FreeLibrary(module);
        g_api = Api{.tried = true};
        return false;
    }
    g_api.module = module;
    return true;
}

void Pass::Create(vk::Instance, vk::PhysicalDevice physical_device_, vk::Device device_,
                  VmaAllocator allocator_, u32 num_images) {
    physical_device = physical_device_;
    device = device_;
    allocator = allocator_;
    available_imgs.resize(num_images);
    for (u32 i = 0; i < num_images; ++i) {
        available_imgs[i].id = static_cast<u8>(i);
        available_imgs[i].output_image = VideoCore::UniqueImage(device, allocator);
    }
    motion_vectors = VideoCore::UniqueImage(device, allocator);
    depth = VideoCore::UniqueImage(device, allocator);
}

bool Pass::CreateContext(vk::Extent2D input_size, vk::Extent2D output_size) {
    DestroyContext();
    const vk::Extent2D max_output{std::max(output_size.width, MinUpscaleSize.width),
                                  std::max(output_size.height, MinUpscaleSize.height)};
    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice = static_cast<VkDevice>(device);
    backend.vkPhysicalDevice = static_cast<VkPhysicalDevice>(physical_device);
    backend.vkDeviceProcAddr = &GetDeviceProcAddr;

    ffxCreateContextDescUpscale desc{};
    desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE;
    desc.header.pNext = &backend.header;
    desc.flags = 0; // the game picture is display-ready (LDR) color
    desc.maxRenderSize = {input_size.width, input_size.height};
    desc.maxUpscaleSize = {max_output.width, max_output.height};
    desc.fpMessage = OnMessage;

    ffxContext new_context{};
    const auto result = g_api.create_context(&new_context, &desc.header, nullptr);
    if (result != FFX_API_RETURN_OK) {
        LOG_ERROR(Render_Vulkan, "FSR 3.1: creating the upscaler failed (code {}); using FSR 1",
                  result);
        return false;
    }
    context = new_context;
    context_input = input_size;
    context_output = max_output;

    ffxQueryGetProviderVersion version{};
    version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    FfxApiEffectMemoryUsage memory{};
    ffxQueryDescUpscaleGetGPUMemoryUsage memory_query{};
    memory_query.header.type = FFX_API_QUERY_DESC_TYPE_UPSCALE_GPU_MEMORY_USAGE;
    memory_query.gpuMemoryUsageUpscaler = &memory;
    g_api.query(&context, &version.header);
    g_api.query(&context, &memory_query.header);
    LOG_INFO(Render_Vulkan, "FSR 3.1: upscaler {} ready for {}x{} to {}x{} ({} MB of GPU memory)",
             version.versionName ? version.versionName : "(unknown version)", input_size.width,
             input_size.height, max_output.width, max_output.height,
             memory.totalUsageInBytes >> 20);
    return true;
}

void Pass::DestroyContext() {
    if (context && g_api.destroy_context) {
        g_api.destroy_context(&context, nullptr);
    }
    context = nullptr;
}

vk::ImageView Pass::Render(vk::CommandBuffer cmdbuf, vk::Image input_image, vk::ImageView input,
                           vk::Format input_format, vk::Extent2D input_size,
                           vk::Extent2D output_size, float sharpness, bool pq) {
    if (failed || !EmulatorSettings.IsFsr3Enabled()) {
        return input;
    }
    // Like FSR 1: only when the target is larger than the game picture, or the same size for
    // native anti-aliasing (the launcher's Native quality).
    const bool same = input_size == output_size;
    if (output_size.width < input_size.width || output_size.height < input_size.height ||
        (same && !RytyUpscale::NativeAA())) {
        return input;
    }
    // GT Sport's 10-bit picture (A2R10G10B10) has no FidelityFX format. FSR reads it as
    // A2B10G10R10 (the game image allows other views), which swaps red and blue; the output view
    // swaps them back. Its values are display-encoded (sRGB, or PQ for HDR10), not linear.
    const bool swap_red_blue = input_format == vk::Format::eA2R10G10B10UnormPack32;
    const vk::Format read_format = swap_red_blue ? vk::Format::eA2B10G10R10UnormPack32 : input_format;
    const uint32_t color_flags = pq              ? FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_PQ
                                 : swap_red_blue ? FFX_UPSCALE_FLAG_NON_LINEAR_COLOR_SRGB
                                                 : 0u;
    if (ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(read_format)) ==
        FFX_API_SURFACE_FORMAT_UNKNOWN) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Vulkan, "FSR 3.1: the game picture format {} is not supported; "
                        "using FSR 1", vk::to_string(input_format));
        }
        return input;
    }
    if (!Load()) {
        failed = true;
        return input;
    }
    if (!context || input_size != context_input || output_size.width > context_output.width ||
        output_size.height > context_output.height) {
        if (!CreateContext(input_size, output_size)) {
            failed = true;
            return input;
        }
    }
    if (input_size != aux_size) {
        CreateAuxImages(cmdbuf, input_size);
    }
    if (output_size != cur_size || swap_red_blue != cur_swap_red_blue) {
        cur_size = output_size;
        cur_swap_red_blue = swap_red_blue;
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
    // The game picture is already readable by shaders; make it visible to FSR's compute work, and
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

    ffxDispatchDescUpscale dispatch{};
    dispatch.header.type = FFX_API_DISPATCH_DESC_TYPE_UPSCALE;
    dispatch.commandList = static_cast<VkCommandBuffer>(cmdbuf);
    dispatch.color = Resource(input_image, read_format, input_size, FFX_API_RESOURCE_USAGE_READ_ONLY,
                              FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    dispatch.depth = Resource(depth, vk::Format::eR32Sfloat, input_size,
                              FFX_API_RESOURCE_USAGE_READ_ONLY,
                              FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    dispatch.motionVectors = Resource(motion_vectors, vk::Format::eR16G16Sfloat, input_size,
                                      FFX_API_RESOURCE_USAGE_READ_ONLY,
                                      FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    dispatch.output = Resource(img.output_image, vk::Format::eR16G16B16A16Sfloat, output_size,
                               FFX_API_RESOURCE_USAGE_UAV, FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
    dispatch.jitterOffset = {0.0f, 0.0f};
    dispatch.motionVectorScale = {1.0f, 1.0f};
    dispatch.renderSize = {input_size.width, input_size.height};
    dispatch.upscaleSize = {output_size.width, output_size.height};
    dispatch.enableSharpening = sharpness >= 0.0f;
    dispatch.sharpness = std::clamp(sharpness, 0.0f, 1.0f);
    dispatch.frameTimeDelta = 1000.0f / 60.0f;
    dispatch.preExposure = 1.0f;
    // No motion vectors: every frame starts a new history.
    dispatch.reset = true;
    dispatch.cameraNear = 0.1f;
    dispatch.cameraFar = 1000.0f;
    dispatch.cameraFovAngleVertical = 1.0f;
    dispatch.viewSpaceToMetersFactor = 1.0f;
    dispatch.flags = color_flags;
    const auto result = g_api.dispatch(&context, &dispatch.header);
    const bool ok = result == FFX_API_RETURN_OK;
    if (!ok) {
        // The output was not written; stop trying and let FSR 1 take over.
        LOG_ERROR(Render_Vulkan, "FSR 3.1: upscaling failed (code {}); using FSR 1", result);
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
        LOG_INFO(Render_Vulkan, "FSR 3.1: scaling {}x{} to {}x{}", input_size.width,
                 input_size.height, output_size.width, output_size.height);
    }
    return ok ? img.output_image_view.get() : input;
}

void Pass::CreateImages(Img& img) const {
    img.dirty = false;
    img.swap_red_blue = cur_swap_red_blue;
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
                 vk::ImageUsageFlagBits::eStorage,
        .initialLayout = vk::ImageLayout::eUndefined,
    };
    img.output_image.Create(info);
    SetObjectName(device, static_cast<vk::Image>(img.output_image), "FSR 3.1 Output Image #{}",
                  img.id);
    img.output_image_view = Check<"create fsr 3.1 output image view">(
        device.createImageViewUnique(vk::ImageViewCreateInfo{
            .image = img.output_image,
            .viewType = vk::ImageViewType::e2D,
            .format = vk::Format::eR16G16B16A16Sfloat,
            .components = img.swap_red_blue
                              ? vk::ComponentMapping{vk::ComponentSwizzle::eB,
                                                     vk::ComponentSwizzle::eG,
                                                     vk::ComponentSwizzle::eR,
                                                     vk::ComponentSwizzle::eA}
                              : vk::ComponentMapping{},
            .subresourceRange{
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .levelCount = 1,
                .layerCount = 1,
            },
        }));
}

void Pass::CreateAuxImages(vk::CommandBuffer cmdbuf, vk::Extent2D input_size) {
    // Runs when the game picture first appears; GT Sport keeps one size after that.
    aux_size = input_size;
    motion_vectors.Destroy();
    depth.Destroy();
    const auto make = [&](VideoCore::UniqueImage& image, vk::Format format, const char* name) {
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent{.width = input_size.width, .height = input_size.height, .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        SetObjectName(device, static_cast<vk::Image>(image), "FSR 3.1 {}", name);
    };
    make(motion_vectors, vk::Format::eR16G16Sfloat, "Motion Vectors");
    make(depth, vk::Format::eR32Sfloat, "Depth");

    constexpr vk::ImageSubresourceRange range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    std::array<vk::ImageMemoryBarrier2, 2> to_clear{};
    std::array<vk::ImageMemoryBarrier2, 2> to_read{};
    const vk::Image images[] = {motion_vectors, depth};
    for (size_t i = 0; i < 2; ++i) {
        to_clear[i] = vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eNone,
            .dstStageMask = vk::PipelineStageFlagBits2::eClear,
            .dstAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .oldLayout = vk::ImageLayout::eUndefined,
            .newLayout = vk::ImageLayout::eTransferDstOptimal,
            .image = images[i],
            .subresourceRange = range,
        };
        to_read[i] = vk::ImageMemoryBarrier2{
            .srcStageMask = vk::PipelineStageFlagBits2::eClear,
            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
            .dstStageMask = vk::PipelineStageFlagBits2::eComputeShader,
            .dstAccessMask = vk::AccessFlagBits2::eShaderRead,
            .oldLayout = vk::ImageLayout::eTransferDstOptimal,
            .newLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
            .image = images[i],
            .subresourceRange = range,
        };
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(to_clear.size()),
        .pImageMemoryBarriers = to_clear.data(),
    });
    const vk::ClearColorValue zero{std::array<float, 4>{0.0f, 0.0f, 0.0f, 0.0f}};
    for (const vk::Image image : images) {
        cmdbuf.clearColorImage(image, vk::ImageLayout::eTransferDstOptimal, zero, range);
    }
    cmdbuf.pipelineBarrier2(vk::DependencyInfo{
        .imageMemoryBarrierCount = static_cast<u32>(to_read.size()),
        .pImageMemoryBarriers = to_read.data(),
    });
}

} // namespace Vulkan::RytyFsr3
