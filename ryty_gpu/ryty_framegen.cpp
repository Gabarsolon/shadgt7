// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <windows.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdlib>
#include <string>

#include "common/diagnostics.h"
#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "common/guest_clock.h"
#include "ryty_gpu/ryty_framegen.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "ryty_gpu/ryty_fsr3.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_platform.h"

#include "ryty_gpu/ffx_api/ffx_api.h"
#include "ryty_gpu/ffx_api/ffx_framegeneration.h"
#include "ryty_gpu/ffx_api/vk/ffx_api_vk.h"

namespace Vulkan::RytyFrameGen {

namespace {

struct Api {
    HMODULE module{};
    PfnFfxCreateContext create_context{};
    PfnFfxDestroyContext destroy_context{};
    PfnFfxConfigure configure{};
    PfnFfxQuery query{};
    PfnFfxDispatch dispatch{};
    bool tried{};
};
Api g_api;

bool Load() {
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
                    "Frame generation: amd_fidelityfx_vk.dll could not be loaded (error {}); off",
                    GetLastError());
        return false;
    }
#define LOAD(field, name)                                                                          \
    g_api.field = reinterpret_cast<decltype(g_api.field)>(                                         \
        reinterpret_cast<void*>(GetProcAddress(module, name)))
    LOAD(create_context, "ffxCreateContext");
    LOAD(destroy_context, "ffxDestroyContext");
    LOAD(configure, "ffxConfigure");
    LOAD(query, "ffxQuery");
    LOAD(dispatch, "ffxDispatch");
#undef LOAD
    if (!g_api.create_context || !g_api.destroy_context || !g_api.configure || !g_api.query ||
        !g_api.dispatch) {
        LOG_WARNING(Render_Vulkan, "Frame generation: amd_fidelityfx_vk.dll is missing functions");
        FreeLibrary(module);
        g_api = Api{.tried = true};
        return false;
    }
    g_api.module = module;
    return true;
}

void OnMessage(uint32_t type, const wchar_t* message) {
    std::string text;
    for (const wchar_t* c = message; c && *c; ++c) {
        text.push_back(*c < 0x80 ? static_cast<char>(*c) : '?');
    }
    if (type == FFX_API_MESSAGE_TYPE_ERROR) {
        LOG_ERROR(Render_Vulkan, "Frame generation: {}", text);
    } else {
        LOG_WARNING(Render_Vulkan, "Frame generation: {}", text);
    }
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

// Generation itself runs from FidelityFX's swapchain; this sets how the picture is encoded.
ffxReturnCode_t DispatchGeneration(ffxDispatchDescFrameGeneration* params, void* user) {
    auto* context = static_cast<ffxContext*>(user);
    const bool hdr = params->presentColor.description.format ==
                     FFX_API_SURFACE_FORMAT_R10G10B10A2_UNORM;
    params->backbufferTransferFunction =
        hdr ? FFX_API_BACKBUFFER_TRANSFER_FUNCTION_PQ : FFX_API_BACKBUFFER_TRANSFER_FUNCTION_SRGB;
    params->minMaxLuminance[0] = 0.0001f;
    params->minMaxLuminance[1] = 1000.0f;
    return g_api.dispatch(context, &params->header);
}

} // namespace

std::atomic<u64> g_shown_frames{};

bool Requested() {
    // NVIDIA DLSS frame generation (ryty_dlssg.h) replaces the AMD one when both are on.
    return EmulatorSettings.IsFrameGenEnabled() && !EmulatorSettings.IsDlssgEnabled();
}

u64 ShownFrames() {
    return g_shown_frames.load(std::memory_order_relaxed);
}

FrameGen::~FrameGen() {
    Destroy();
}

bool FrameGen::CreateSwapchain(const Instance& instance, const vk::SwapchainCreateInfoKHR& info,
                               vk::SwapchainKHR& swapchain) {
    Destroy();
    if (!Requested()) {
        return false;
    }
    const vk::Queue async_queue = instance.GetFrameGenQueue(0);
    const vk::Queue present_queue = instance.GetFrameGenQueue(1);
    const vk::Queue acquire_queue = instance.GetFrameGenQueue(2);
    if (!async_queue || !present_queue || !acquire_queue) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_WARNING(Render_Vulkan, "Frame generation: the GPU's graphics queue family has "
                                       "fewer than 4 queues; off");
        }
        return false;
    }
    if (!Load()) {
        return false;
    }
    device = instance.GetDevice();
    allocator = instance.GetAllocator();
    const u32 family = instance.GetGraphicsQueueFamilyIndex();

    VkSwapchainKHR handle = VK_NULL_HANDLE;
    ffxCreateContextDescFrameGenerationSwapChainVK swapchain_desc{};
    swapchain_desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    swapchain_desc.physicalDevice = static_cast<VkPhysicalDevice>(instance.GetPhysicalDevice());
    swapchain_desc.device = static_cast<VkDevice>(device);
    swapchain_desc.swapchain = &handle;
    swapchain_desc.allocator = nullptr;
    swapchain_desc.createInfo = static_cast<VkSwapchainCreateInfoKHR>(info);
    swapchain_desc.gameQueue = {static_cast<VkQueue>(instance.GetGraphicsQueue()), family, nullptr};
    swapchain_desc.asyncComputeQueue = {static_cast<VkQueue>(async_queue), family, nullptr};
    swapchain_desc.presentQueue = {static_cast<VkQueue>(present_queue), family, nullptr};
    swapchain_desc.imageAcquireQueue = {static_cast<VkQueue>(acquire_queue), family, nullptr};
    ffxContext new_swapchain_context{};
    auto result = g_api.create_context(&new_swapchain_context, &swapchain_desc.header, nullptr);
    if (result != FFX_API_RETURN_OK || handle == VK_NULL_HANDLE) {
        LOG_ERROR(Render_Vulkan, "Frame generation: creating its swapchain failed (code {}); off",
                  result);
        return false;
    }
    swapchain_context = new_swapchain_context;

    ffxQueryDescSwapchainReplacementFunctionsVK functions{};
    functions.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    g_api.query(reinterpret_cast<ffxContext*>(&swapchain_context), &functions.header);
    get_images = functions.pOutGetSwapchainImagesKHR;
    acquire = functions.pOutAcquireNextImageKHR;
    present = functions.pOutQueuePresentKHR;
    present_count = reinterpret_cast<void*>(functions.pOutGetLastPresentCountFFXAPI);
    if (!get_images || !acquire || !present) {
        LOG_ERROR(Render_Vulkan, "Frame generation: its swapchain functions are missing; off");
        Destroy();
        return false;
    }

    // Pacing that sleeps between short spins and waits on fences, instead of spinning a core.
    static FfxApiSwapchainFramePacingTuning tuning{0.1f, 0.1f, true, 2, true};
    ffxConfigureDescFrameGenerationSwapChainKeyValueVK pacing{};
    pacing.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_KEYVALUE_VK;
    pacing.key = FFX_API_CONFIGURE_FG_SWAPCHAIN_KEY_FRAMEPACINGTUNING;
    pacing.ptr = &tuning;
    g_api.configure(reinterpret_cast<ffxContext*>(&swapchain_context), &pacing.header);

    ffxCreateBackendVKDesc backend{};
    backend.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_BACKEND_VK;
    backend.vkDevice = static_cast<VkDevice>(device);
    backend.vkPhysicalDevice = static_cast<VkPhysicalDevice>(instance.GetPhysicalDevice());
    backend.vkDeviceProcAddr = RytyFsr3::DeviceProcAddrShim();

    ffxCreateContextDescFrameGeneration framegen_desc{};
    framegen_desc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
    framegen_desc.header.pNext = &backend.header;
    framegen_desc.flags = 0;
    framegen_desc.displaySize = {info.imageExtent.width, info.imageExtent.height};
    framegen_desc.maxRenderSize = {info.imageExtent.width, info.imageExtent.height};
    framegen_desc.backBufferFormat = ffxApiGetSurfaceFormatVK(static_cast<VkFormat>(info.imageFormat));
    ffxContext new_framegen_context{};
    result = g_api.create_context(&new_framegen_context, &framegen_desc.header, nullptr);
    if (result != FFX_API_RETURN_OK) {
        LOG_ERROR(Render_Vulkan, "Frame generation: creating it failed (code {}); off", result);
        Destroy();
        return false;
    }
    framegen_context = new_framegen_context;
    display_size = info.imageExtent;
    format = info.imageFormat;
    frame_id = 0;
    swapchain = vk::SwapchainKHR{handle};

    ffxQueryGetProviderVersion version{};
    version.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    g_api.query(reinterpret_cast<ffxContext*>(&framegen_context), &version.header);
    LOG_INFO(Render_Vulkan, "Frame generation: {} on, {}x{} {}",
             version.versionName ? version.versionName : "FSR 3.1", display_size.width,
             display_size.height, vk::to_string(format));
    (void)OnMessage;
    return true;
}

void FrameGen::Destroy() {
    g_shown_frames.store(0, std::memory_order_relaxed);
    if (framegen_context) {
        ffxContext context = framegen_context;
        g_api.destroy_context(&context, nullptr);
        framegen_context = nullptr;
    }
    if (swapchain_context) {
        // Also destroys the swapchain it made.
        ffxContext context = swapchain_context;
        g_api.destroy_context(&context, nullptr);
        swapchain_context = nullptr;
    }
    get_images = nullptr;
    acquire = nullptr;
    present = nullptr;
}

vk::Result FrameGen::GetImages(vk::Device device_, vk::SwapchainKHR swapchain,
                               std::vector<vk::Image>& images) const {
    u32 count = 0;
    auto result = static_cast<vk::Result>(get_images(static_cast<VkDevice>(device_),
                                                     static_cast<VkSwapchainKHR>(swapchain), &count,
                                                     nullptr));
    if (result != vk::Result::eSuccess) {
        return result;
    }
    std::vector<VkImage> raw(count);
    result = static_cast<vk::Result>(get_images(static_cast<VkDevice>(device_),
                                                static_cast<VkSwapchainKHR>(swapchain), &count,
                                                raw.data()));
    images.assign(raw.begin(), raw.begin() + count);
    return result;
}

vk::Result FrameGen::Acquire(vk::Device device_, vk::SwapchainKHR swapchain,
                             vk::Semaphore semaphore, u32* image_index) const {
    return static_cast<vk::Result>(acquire(static_cast<VkDevice>(device_),
                                           static_cast<VkSwapchainKHR>(swapchain), UINT64_MAX,
                                           static_cast<VkSemaphore>(semaphore), VK_NULL_HANDLE,
                                           image_index));
}

vk::Result FrameGen::Present(vk::Queue queue, const vk::PresentInfoKHR& info) const {
    const VkPresentInfoKHR raw = info;
    return static_cast<vk::Result>(present(static_cast<VkQueue>(queue), &raw));
}

void FrameGen::Prepare(vk::CommandBuffer cmdbuf, vk::SwapchainKHR swapchain, vk::Extent2D size,
                       bool hdr) {
    if (!framegen_context) {
        return;
    }
    if (size != aux_size) {
        CreateAuxImages(cmdbuf, size);
    }
    const auto now = std::chrono::steady_clock::now();
    const float delta_ms =
        last_prepare.time_since_epoch().count() == 0
            ? 1000.0f / 30.0f
            : std::chrono::duration<float, std::milli>(now - last_prepare).count();
    last_prepare = now;
    ++frame_id;
    if (present_count) {
        g_shown_frames.store(reinterpret_cast<PFN_getLastPresentCountFFXAPI>(present_count)(
                                 static_cast<VkSwapchainKHR>(swapchain)),
                             std::memory_order_relaxed);
    }
    // Shown frames (game and generated) against game frames, every 10 s.
    if (present_count && now - last_log >= std::chrono::seconds{10}) {
        const u64 presents = reinterpret_cast<PFN_getLastPresentCountFFXAPI>(present_count)(
            static_cast<VkSwapchainKHR>(swapchain));
        if (last_log.time_since_epoch().count() != 0) {
            const double seconds = std::chrono::duration<double>(now - last_log).count();
            LOG_INFO(Render_Vulkan, "Frame generation: {:.1f} frames shown per second, {:.1f} from "
                                    "the game",
                     (presents - logged_presents) / seconds, (frame_id - logged_frames) / seconds);
        }
        last_log = now;
        logged_presents = presents;
        logged_frames = frame_id;
    }

    ffxConfigureDescFrameGeneration config{};
    config.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    config.swapChain = static_cast<VkSwapchainKHR>(swapchain);
    config.presentCallback = nullptr;
    config.presentCallbackUserContext = nullptr;
    config.frameGenerationCallback = &DispatchGeneration;
    config.frameGenerationCallbackUserContext = &framegen_context;
    // SHADGT_FRAMEGEN_AB=<seconds> (test): generation alternates on and off in one run, with
    // the game's frame rate logged for each.
    static const double ab_period = [] {
        const char* env = Common::DiagEnv("SHADGT_FRAMEGEN_AB");
        return env ? std::atof(env) : 0.0;
    }();
    bool enabled = true;
    if (ab_period > 0.0) {
        static const auto ab_start = now;
        static u64 ab_frames[2]{};
        static double ab_seconds[2]{};
        static auto ab_last = now;
        static auto ab_log = now;
        const double t = std::chrono::duration<double>(now - ab_start).count();
        enabled = static_cast<u64>(t / ab_period) % 2 == 0;
        const double dt = std::chrono::duration<double>(now - ab_last).count();
        ab_last = now;
        ++ab_frames[enabled ? 0 : 1];
        ab_seconds[enabled ? 0 : 1] += dt;
        if (now - ab_log >= std::chrono::seconds{20}) {
            ab_log = now;
            LOG_INFO(Render_Vulkan, "Frame generation A/B: game {:.1f} fps with it, {:.1f} without",
                     ab_frames[0] / std::max(0.001, ab_seconds[0]),
                     ab_frames[1] / std::max(0.001, ab_seconds[1]));
            ab_frames[0] = ab_frames[1] = 0;
            ab_seconds[0] = ab_seconds[1] = 0.0;
        }
    }
    // FIX-104: no generated frames while new shaders compile (and for a second after). With
    // generation on during heavy compiles the driver deadlocked: FFX's present thread waited in
    // the driver while pipeline builds and the recorder waited on its locks (October 16, both
    // renderers; the new one compiles its own shaders at first, so it met it at once).
    {
        static std::chrono::steady_clock::time_point last_compile{};
        if (Common::GuestClock::IsPaused() || Vulkan::CompileSlowMotion() ||
            Vulkan::PipelineBuildsRunning() > 0) {
            last_compile = now;
        }
        if (last_compile != std::chrono::steady_clock::time_point{} &&
            now - last_compile < std::chrono::seconds{1}) {
            enabled = false;
        }
    }
    config.frameGenerationEnabled = enabled;
    config.allowAsyncWorkloads = false;
    config.HUDLessColor = FfxApiResource{};
    config.flags = 0;
    config.onlyPresentGenerated = false;
    config.generationRect = {0, 0, static_cast<int32_t>(size.width),
                             static_cast<int32_t>(size.height)};
    config.frameID = frame_id;
    auto result = g_api.configure(reinterpret_cast<ffxContext*>(&framegen_context), &config.header);
    if (result != FFX_API_RETURN_OK) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_ERROR(Render_Vulkan, "Frame generation: configuring failed (code {})", result);
        }
        return;
    }

    ffxDispatchDescFrameGenerationPrepareCameraInfo camera{};
    camera.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_CAMERAINFO;
    camera.cameraUp[1] = 1.0f;
    camera.cameraRight[0] = 1.0f;
    camera.cameraForward[2] = 1.0f;

    ffxDispatchDescFrameGenerationPrepare prepare{};
    prepare.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE;
    prepare.header.pNext = &camera.header;
    prepare.frameID = frame_id;
    prepare.flags = 0;
    prepare.commandList = static_cast<VkCommandBuffer>(cmdbuf);
    prepare.renderSize = {size.width, size.height};
    prepare.jitterOffset = {0.0f, 0.0f};
    prepare.motionVectorScale = {1.0f, 1.0f};
    prepare.frameTimeDelta = delta_ms;
    prepare.unused_reset = false;
    prepare.cameraNear = 0.1f;
    prepare.cameraFar = 1000.0f;
    prepare.cameraFovAngleVertical = 1.0f;
    prepare.viewSpaceToMetersFactor = 1.0f;
    prepare.depth = Resource(depth, vk::Format::eR32Sfloat, size, FFX_API_RESOURCE_USAGE_READ_ONLY,
                             FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    prepare.motionVectors = Resource(motion_vectors, vk::Format::eR16G16Sfloat, size,
                                     FFX_API_RESOURCE_USAGE_READ_ONLY,
                                     FFX_API_RESOURCE_STATE_PIXEL_COMPUTE_READ);
    result = g_api.dispatch(reinterpret_cast<ffxContext*>(&framegen_context), &prepare.header);
    if (result != FFX_API_RETURN_OK) {
        static bool logged = false;
        if (!logged) {
            logged = true;
            LOG_ERROR(Render_Vulkan, "Frame generation: preparing failed (code {})", result);
        }
    }
    (void)hdr;
}

void FrameGen::CreateAuxImages(vk::CommandBuffer cmdbuf, vk::Extent2D size) {
    // No motion vectors or depth from the game: zero motion and a flat depth image.
    aux_size = size;
    motion_vectors.Destroy();
    depth.Destroy();
    motion_vectors = VideoCore::UniqueImage(device, allocator);
    depth = VideoCore::UniqueImage(device, allocator);
    const auto make = [&](VideoCore::UniqueImage& image, vk::Format image_format) {
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = image_format,
            .extent{.width = size.width, .height = size.height, .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
    };
    make(motion_vectors, vk::Format::eR16G16Sfloat);
    make(depth, vk::Format::eR32Sfloat);

    constexpr vk::ImageSubresourceRange range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const std::array<vk::Image, 2> images = {motion_vectors, depth};
    std::array<vk::ImageMemoryBarrier2, 2> to_clear{};
    std::array<vk::ImageMemoryBarrier2, 2> to_read{};
    for (size_t i = 0; i < images.size(); ++i) {
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

} // namespace Vulkan::RytyFrameGen
