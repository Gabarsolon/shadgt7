// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <windows.h>

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <utility>

#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "ryty_gpu/ryty_dlssg.h"
#include "video_core/texture_cache/image.h"

// Streamline's headers (MIT, ryty_gpu/streamline). Its functions are loaded from
// sl.interposer.dll at run time; the header helpers that need the import library are not used.
#include "ryty_gpu/streamline/sl.h"
#include "ryty_gpu/streamline/sl_consts.h"
#include "ryty_gpu/streamline/sl_dlss_g.h"
#include "ryty_gpu/streamline/sl_pcl.h"
#include "ryty_gpu/streamline/sl_reflex.h"

namespace Vulkan::RytyFrameGen {
extern std::atomic<u64> g_shown_frames; // ryty_framegen.cpp, read by the FPS counter
}

namespace Vulkan::RytyDlssG {

namespace {

using PFunInit = sl::Result (*)(const sl::Preferences&, uint64_t);
using PFunShutdown = sl::Result (*)();
using PFunIsFeatureSupported = sl::Result (*)(sl::Feature, const sl::AdapterInfo&);
using PFunSetTagForFrame = sl::Result (*)(const sl::FrameToken&, const sl::ViewportHandle&,
                                          const sl::ResourceTag*, uint32_t, sl::CommandBuffer*);
using PFunSetConstants = sl::Result (*)(const sl::Constants&, const sl::FrameToken&,
                                        const sl::ViewportHandle&);
using PFunGetFeatureFunction = sl::Result (*)(sl::Feature, const char*, void*&);
using PFunGetNewFrameToken = sl::Result (*)(sl::FrameToken*&, const uint32_t*);

struct Streamline {
    HMODULE module{};
    PFunInit init{};
    PFunShutdown shutdown{};
    PFunIsFeatureSupported is_supported{};
    PFunSetTagForFrame set_tag{};
    PFunSetConstants set_constants{};
    PFunGetFeatureFunction get_feature_function{};
    PFunGetNewFrameToken new_frame_token{};
    PFun_slDLSSGSetOptions* dlssg_set_options{};
    PFun_slDLSSGGetState* dlssg_get_state{};
    int last_status = -1;
    PFun_slReflexSetOptions* reflex_set_options{};
    PFun_slPCLSetMarker* pcl_set_marker{};
    PFN_vkGetInstanceProcAddr get_instance_proc_addr{};
    bool tried{};
    bool features_ready{};
    bool options_set{};
    sl::FrameToken* frame{};
    u32 frame_index{};
    std::mutex mutex;
    // No motion vectors or depth from the game: zero motion and a flat depth image.
    vk::Extent2D aux_size{};
    VideoCore::UniqueImage motion_vectors;
    VideoCore::UniqueImage depth;
    vk::UniqueImageView motion_vectors_view;
    vk::UniqueImageView depth_view;
};
Streamline g_sl;

std::wstring DlssDir() {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir{exe};
    return dir.substr(0, dir.find_last_of(L"\\/")) + L"\\dlss";
}

std::wstring LogDir() {
    wchar_t exe[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, exe, MAX_PATH);
    std::wstring dir{exe};
    return dir.substr(0, dir.find_last_of(L"\\/")) + L"\\user\\log";
}

void SlLog(sl::LogType type, const char* message) {
    if (type == sl::LogType::eError) {
        LOG_ERROR(Render_Vulkan, "Streamline: {}", message ? message : "");
    } else if (type == sl::LogType::eWarn) {
        LOG_WARNING(Render_Vulkan, "Streamline: {}", message ? message : "");
    } else {
        LOG_INFO(Render_Vulkan, "Streamline: {}", message ? message : "");
    }
}

template <typename T>
T* Feature(sl::Feature feature, const char* name) {
    void* function{};
    if (!g_sl.get_feature_function ||
        g_sl.get_feature_function(feature, name, function) != sl::Result::eOk) {
        return nullptr;
    }
    return reinterpret_cast<T*>(function);
}

bool LoadFeatureFunctions() {
    if (g_sl.features_ready) {
        return g_sl.dlssg_set_options != nullptr;
    }
    g_sl.features_ready = true;
    g_sl.dlssg_set_options =
        Feature<PFun_slDLSSGSetOptions>(sl::kFeatureDLSS_G, "slDLSSGSetOptions");
    g_sl.dlssg_get_state = Feature<PFun_slDLSSGGetState>(sl::kFeatureDLSS_G, "slDLSSGGetState");
    g_sl.reflex_set_options =
        Feature<PFun_slReflexSetOptions>(sl::kFeatureReflex, "slReflexSetOptions");
    g_sl.pcl_set_marker = Feature<PFun_slPCLSetMarker>(sl::kFeaturePCL, "slPCLSetMarker");
    if (!g_sl.dlssg_set_options) {
        LOG_ERROR(Render_Vulkan, "ryGT DLSS-G: DLSS frame generation is not available here "
                                 "(see the Streamline lines above: it needs an RTX 40 or newer "
                                 "GPU and Windows' hardware-accelerated GPU scheduling)");
        return false;
    }
    if (g_sl.reflex_set_options) {
        sl::ReflexOptions reflex{};
        reflex.mode = sl::ReflexMode::eLowLatency;
        g_sl.reflex_set_options(reflex);
    }
    return true;
}

void CreateAuxImages(vk::Device device, VmaAllocator allocator, vk::CommandBuffer cmdbuf,
                     vk::Extent2D size) {
    g_sl.aux_size = size;
    g_sl.motion_vectors_view.reset();
    g_sl.depth_view.reset();
    g_sl.motion_vectors.Destroy();
    g_sl.depth.Destroy();
    g_sl.motion_vectors = VideoCore::UniqueImage(device, allocator);
    g_sl.depth = VideoCore::UniqueImage(device, allocator);
    const auto make = [&](VideoCore::UniqueImage& image, vk::Format format,
                          vk::UniqueImageView& view) {
        image.Create(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = format,
            .extent{.width = size.width, .height = size.height, .depth = 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = vk::SampleCountFlagBits::e1,
            .usage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst |
                     vk::ImageUsageFlagBits::eTransferSrc,
            .initialLayout = vk::ImageLayout::eUndefined,
        });
        auto [result, created] = device.createImageViewUnique(vk::ImageViewCreateInfo{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = format,
            .subresourceRange{.aspectMask = vk::ImageAspectFlagBits::eColor,
                              .levelCount = 1,
                              .layerCount = 1},
        });
        if (result == vk::Result::eSuccess) {
            view = std::move(created);
        }
    };
    make(g_sl.motion_vectors, vk::Format::eR16G16Sfloat, g_sl.motion_vectors_view);
    make(g_sl.depth, vk::Format::eR32Sfloat, g_sl.depth_view);

    constexpr vk::ImageSubresourceRange range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const std::array<vk::Image, 2> images = {g_sl.motion_vectors, g_sl.depth};
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
            .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
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

sl::float4x4 Identity() {
    sl::float4x4 m;
    m.setRow(0, sl::float4(1, 0, 0, 0));
    m.setRow(1, sl::float4(0, 1, 0, 0));
    m.setRow(2, sl::float4(0, 0, 1, 0));
    m.setRow(3, sl::float4(0, 0, 0, 1));
    return m;
}

void Marker(sl::PCLMarker marker) {
    if (g_sl.pcl_set_marker && g_sl.frame) {
        g_sl.pcl_set_marker(marker, *g_sl.frame);
    }
}

} // namespace

bool Requested() {
    return EmulatorSettings.IsDlssgEnabled();
}

bool Active() {
    return g_sl.get_instance_proc_addr != nullptr;
}

PFN_vkGetInstanceProcAddr LoadInterposer() {
    if (g_sl.tried) {
        return g_sl.get_instance_proc_addr;
    }
    g_sl.tried = true;
    if (!Requested()) {
        return nullptr;
    }
    const std::wstring dir = DlssDir();
    const std::wstring path = dir + L"\\sl.interposer.dll";
    HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        LOG_ERROR(Render_Vulkan,
                  "ryGT DLSS-G: dlssg_enabled is on but dlss\\sl.interposer.dll could not be "
                  "loaded (error {}); no DLSS frame generation",
                  GetLastError());
        return nullptr;
    }
    const auto get = [&](const char* name) { return GetProcAddress(module, name); };
    g_sl.init = reinterpret_cast<PFunInit>(get("slInit"));
    g_sl.shutdown = reinterpret_cast<PFunShutdown>(get("slShutdown"));
    g_sl.is_supported = reinterpret_cast<PFunIsFeatureSupported>(get("slIsFeatureSupported"));
    g_sl.set_tag = reinterpret_cast<PFunSetTagForFrame>(get("slSetTagForFrame"));
    g_sl.set_constants = reinterpret_cast<PFunSetConstants>(get("slSetConstants"));
    g_sl.get_feature_function =
        reinterpret_cast<PFunGetFeatureFunction>(get("slGetFeatureFunction"));
    g_sl.new_frame_token = reinterpret_cast<PFunGetNewFrameToken>(get("slGetNewFrameToken"));
    const auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(get("vkGetInstanceProcAddr"));
    if (!g_sl.init || !g_sl.set_tag || !g_sl.set_constants || !g_sl.get_feature_function ||
        !g_sl.new_frame_token || !gipa) {
        LOG_ERROR(Render_Vulkan, "ryGT DLSS-G: dlss\\sl.interposer.dll lacks Streamline's API");
        FreeLibrary(module);
        return nullptr;
    }

    static const std::wstring plugin_dir = dir;
    static const std::wstring log_dir = LogDir();
    static const wchar_t* plugin_paths[] = {plugin_dir.c_str()};
    static const sl::Feature features[] = {sl::kFeatureDLSS_G, sl::kFeatureReflex,
                                           sl::kFeaturePCL};
    sl::Preferences pref{};
    pref.showConsole = false;
    pref.logLevel = sl::LogLevel::eDefault;
    pref.pathsToPlugins = plugin_paths;
    pref.numPathsToPlugins = 1;
    pref.pathToLogsAndData = log_dir.c_str();
    pref.logMessageCallback = SlLog;
    // Streamline's interposer stands in for vulkan-1.dll (its vkGetInstanceProcAddr gives the
    // driver's functions except for the few it intercepts), and its device proxy adds the
    // extensions, features and queues DLSS-G needs. No over-the-air downloads.
    pref.flags = sl::PreferenceFlags::eDisableCLStateTracking |
                 sl::PreferenceFlags::eUseManualHooking |
                 sl::PreferenceFlags::eUseFrameBasedResourceTagging;
    pref.featuresToLoad = features;
    pref.numFeaturesToLoad = static_cast<u32>(std::size(features));
    pref.engine = sl::EngineType::eCustom;
    pref.engineVersion = "ryGT";
    pref.projectId = "5f1a2b3c-4d5e-4f60-8a7b-9c0d1e2f3a4b";
    pref.renderAPI = sl::RenderAPI::eVulkan;
    const sl::Result result = g_sl.init(pref, sl::kSDKVersion);
    if (result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "ryGT DLSS-G: Streamline did not start (result {}); no DLSS "
                                 "frame generation",
                  static_cast<int>(result));
        FreeLibrary(module);
        return nullptr;
    }
    g_sl.module = module;
    g_sl.get_instance_proc_addr = gipa;
    LOG_WARNING(Render_Vulkan, "ryGT DLSS-G: Streamline started; Vulkan goes through it");
    return gipa;
}

void Prepare(vk::Device device, VmaAllocator allocator, vk::CommandBuffer cmdbuf,
             vk::Extent2D size) {
    if (!Active()) {
        return;
    }
    std::scoped_lock lock{g_sl.mutex};
    if (!LoadFeatureFunctions()) {
        return;
    }
    if (size != g_sl.aux_size || !g_sl.motion_vectors_view || !g_sl.depth_view) {
        CreateAuxImages(device, allocator, cmdbuf, size);
    }
    ++g_sl.frame_index;
    if (g_sl.new_frame_token(g_sl.frame, &g_sl.frame_index) != sl::Result::eOk || !g_sl.frame) {
        g_sl.frame = nullptr;
        return;
    }
    Marker(sl::PCLMarker::eSimulationStart);
    Marker(sl::PCLMarker::eSimulationEnd);
    Marker(sl::PCLMarker::eRenderSubmitStart);

    const sl::ViewportHandle viewport{0u};
    sl::Constants constants{};
    constants.cameraViewToClip = Identity();
    constants.clipToCameraView = Identity();
    constants.clipToLensClip = Identity();
    constants.clipToPrevClip = Identity();
    constants.prevClipToClip = Identity();
    constants.jitterOffset = sl::float2(0.0f, 0.0f);
    constants.mvecScale = sl::float2(1.0f, 1.0f);
    constants.cameraPinholeOffset = sl::float2(0.0f, 0.0f);
    constants.cameraPos = sl::float3(0.0f, 0.0f, 0.0f);
    constants.cameraUp = sl::float3(0.0f, 1.0f, 0.0f);
    constants.cameraRight = sl::float3(1.0f, 0.0f, 0.0f);
    constants.cameraFwd = sl::float3(0.0f, 0.0f, 1.0f);
    constants.cameraNear = 0.1f;
    constants.cameraFar = 10000.0f;
    constants.cameraFOV = 1.0f;
    constants.cameraAspectRatio = float(size.width) / float(std::max(size.height, 1u));
    constants.depthInverted = sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.reset = sl::Boolean::eFalse;
    g_sl.set_constants(constants, *g_sl.frame, viewport);

    const sl::Extent extent{0, 0, size.width, size.height};
    sl::Resource mvec{sl::ResourceType::eTex2d,
                      static_cast<VkImage>(static_cast<vk::Image>(g_sl.motion_vectors)), nullptr,
                      static_cast<VkImageView>(*g_sl.motion_vectors_view),
                      static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)};
    mvec.width = size.width;
    mvec.height = size.height;
    mvec.nativeFormat = static_cast<uint32_t>(VK_FORMAT_R16G16_SFLOAT);
    mvec.mipLevels = 1;
    mvec.arrayLayers = 1;
    mvec.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sl::Resource dep{sl::ResourceType::eTex2d,
                     static_cast<VkImage>(static_cast<vk::Image>(g_sl.depth)), nullptr,
                     static_cast<VkImageView>(*g_sl.depth_view),
                     static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)};
    dep.width = size.width;
    dep.height = size.height;
    dep.nativeFormat = static_cast<uint32_t>(VK_FORMAT_R32_SFLOAT);
    dep.mipLevels = 1;
    dep.arrayLayers = 1;
    dep.usage = mvec.usage;
    // The backbuffer tag only carries the extent (Streamline knows the swapchain image).
    sl::Resource backbuffer{sl::ResourceType::eTex2d, nullptr, nullptr, nullptr};
    const sl::ResourceTag tags[] = {
        sl::ResourceTag{&backbuffer, sl::kBufferTypeBackbuffer,
                        sl::ResourceLifecycle::eValidUntilPresent, &extent},
        sl::ResourceTag{&mvec, sl::kBufferTypeMotionVectors, sl::ResourceLifecycle::eValidUntilPresent,
                        &extent},
        sl::ResourceTag{&dep, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent,
                        &extent},
    };
    auto* command_buffer =
        reinterpret_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(cmdbuf));
    g_sl.set_tag(*g_sl.frame, viewport, tags, static_cast<u32>(std::size(tags)), command_buffer);

    if (!g_sl.options_set) {
        sl::DLSSGOptions options{};
        options.mode = sl::DLSSGMode::eOn;
        // GT-MFG: frames generated between game frames, RYGT_DLSSG_FRAMES=1..3 (2x to 4x;
        // default 1), no more than the GPU supports (multi-frame generation: RTX 50).
        u32 frames = 1;
        if (const char* env = std::getenv("RYGT_DLSSG_FRAMES"); env && *env) {
            frames = std::clamp<u32>(static_cast<u32>(std::strtoul(env, nullptr, 10)), 1u, 3u);
        }
        u32 most = 1;
        if (g_sl.dlssg_get_state) {
            sl::DLSSGState state{};
            if (g_sl.dlssg_get_state(viewport, state, nullptr) == sl::Result::eOk &&
                state.numFramesToGenerateMax > 0) {
                most = state.numFramesToGenerateMax;
            }
        }
        options.numFramesToGenerate = std::min(frames, most);
        const sl::Result set = g_sl.dlssg_set_options(viewport, options);
        g_sl.options_set = set == sl::Result::eOk;
        LOG_WARNING(Render_Vulkan,
                    "ryGT DLSS-G: frame generation {} (result {}), {} generated frames a game "
                    "frame ({} asked, {} supported)",
                    g_sl.options_set ? "on" : "could not be turned on", static_cast<int>(set),
                    options.numFramesToGenerate, frames, most);
    }
    Marker(sl::PCLMarker::eRenderSubmitEnd);

    // Frames shown (game and generated) for the FPS counter, and DLSS-G's status when it changes.
    if (g_sl.dlssg_get_state) {
        sl::DLSSGState state{};
        if (g_sl.dlssg_get_state(viewport, state, nullptr) == sl::Result::eOk) {
            RytyFrameGen::g_shown_frames.fetch_add(state.numFramesActuallyPresented,
                                                   std::memory_order_relaxed);
            if (static_cast<int>(state.status) != g_sl.last_status) {
                g_sl.last_status = static_cast<int>(state.status);
                LOG_WARNING(Render_Vulkan, "ryGT DLSS-G: status {:#x} (0 = generating)",
                            g_sl.last_status);
            }
        }
    }
}

void PresentStart() {
    if (Active()) {
        std::scoped_lock lock{g_sl.mutex};
        Marker(sl::PCLMarker::ePresentStart);
    }
}

void PresentEnd() {
    if (Active()) {
        std::scoped_lock lock{g_sl.mutex};
        Marker(sl::PCLMarker::ePresentEnd);
    }
}

void Shutdown() {
    std::scoped_lock lock{g_sl.mutex};
    g_sl.motion_vectors_view.reset();
    g_sl.depth_view.reset();
    g_sl.motion_vectors.Destroy();
    g_sl.depth.Destroy();
    g_sl.aux_size = {};
}

} // namespace Vulkan::RytyDlssG
