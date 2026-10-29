// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <windows.h>

#include <array>
#include <atomic>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "common/logging/log.h"
#include "core/emulator_settings.h"
#include "ryty_gpu/ryty_dlssg.h"
#include "video_core/texture_cache/image.h"

// Streamline's headers (MIT, ryty_gpu/streamline). Its functions are loaded from
// sl.interposer.dll at run time; the header helpers that need the import library are not used.
#include "ryty_gpu/streamline/sl.h"
#include "ryty_gpu/streamline/sl_consts.h"
#include "ryty_gpu/streamline/sl_dlss.h"
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
using PFunEvaluateFeature = sl::Result (*)(sl::Feature, const sl::FrameToken&,
                                           const sl::BaseStructure**, uint32_t, sl::CommandBuffer*);

// Zero motion vectors and a flat depth image (the game picture comes without either).
struct AuxImages {
    vk::Extent2D size{};
    VideoCore::UniqueImage motion_vectors;
    VideoCore::UniqueImage depth;
    vk::UniqueImageView motion_vectors_view;
    vk::UniqueImageView depth_view;

    bool Ready(vk::Extent2D wanted) const {
        return wanted == size && motion_vectors_view && depth_view;
    }
    void Destroy() {
        motion_vectors_view.reset();
        depth_view.reset();
        motion_vectors.Destroy();
        depth.Destroy();
        size = {};
    }
};

struct Streamline {
    HMODULE module{};
    PFunInit init{};
    PFunShutdown shutdown{};
    PFunIsFeatureSupported is_supported{};
    PFunSetTagForFrame set_tag{};
    PFunSetConstants set_constants{};
    PFunGetFeatureFunction get_feature_function{};
    PFunGetNewFrameToken new_frame_token{};
    PFunEvaluateFeature evaluate{};
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
    AuxImages fg; // frame generation, at the swapchain size
    // DLSS upscaling (sl.dlss.dll) on its own viewport, at the game picture's size.
    bool upscale_requested{};
    bool upscale_ready{};
    PFun_slDLSSGetOptimalSettings* dlss_get_optimal{};
    PFun_slDLSSSetOptions* dlss_set_options{};
    AuxImages sr;
    vk::Extent2D sr_in{}, sr_out{};
    int sr_mode = -1; // DLSSMode for sr_in -> sr_out, -1 when not supported
    bool sr_options_set{};
    // The frame token the upscaler (draw thread) started for the next presented frame; frame
    // generation's Prepare (present thread) carries on with it instead of starting another.
    // Kept apart from frame, which the present thread's markers use for the frame on screen.
    sl::FrameToken* upscale_frame{};
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

void CreateAuxImages(AuxImages& aux, vk::Device device, VmaAllocator allocator,
                     vk::CommandBuffer cmdbuf,
                     vk::Extent2D size) {
    aux.Destroy();
    aux.size = size;
    aux.motion_vectors = VideoCore::UniqueImage(device, allocator);
    aux.depth = VideoCore::UniqueImage(device, allocator);
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
    make(aux.motion_vectors, vk::Format::eR16G16Sfloat, aux.motion_vectors_view);
    make(aux.depth, vk::Format::eR32Sfloat, aux.depth_view);

    constexpr vk::ImageSubresourceRange range = {
        .aspectMask = vk::ImageAspectFlagBits::eColor,
        .levelCount = 1,
        .layerCount = 1,
    };
    const std::array<vk::Image, 2> images = {aux.motion_vectors, aux.depth};
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

// Constants for a picture without camera data: identity matrices, no jitter, zero motion.
sl::Constants FlatConstants(vk::Extent2D size, bool reset) {
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
    constants.reset = reset ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    return constants;
}

sl::Resource Texture(vk::Image image, vk::ImageView view, vk::ImageLayout layout,
                     vk::Extent2D size, vk::Format format, vk::ImageUsageFlags usage) {
    sl::Resource resource{sl::ResourceType::eTex2d, static_cast<VkImage>(image), nullptr,
                          static_cast<VkImageView>(view), static_cast<uint32_t>(layout)};
    resource.width = size.width;
    resource.height = size.height;
    resource.nativeFormat = static_cast<uint32_t>(format);
    resource.mipLevels = 1;
    resource.arrayLayers = 1;
    resource.flags = 0; // no VkImageCreateFlags (Resource leaves this field uninitialized)
    resource.usage = static_cast<uint32_t>(static_cast<VkImageUsageFlags>(usage));
    return resource;
}

std::wstring DlssFile(const wchar_t* name) {
    return DlssDir() + L"\\" + name;
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

bool UpscaleRequested() {
    if (!EmulatorSettings.IsDlssEnabled()) {
        return false;
    }
    // RYGT_DLSS_STREAMLINE=0 keeps upscaling on the self-built ryGT_dlss.dll plugin.
    if (const char* env = std::getenv("RYGT_DLSS_STREAMLINE"); env && env[0] == '0') {
        return false;
    }
    return GetFileAttributesW(DlssFile(L"sl.dlss.dll").c_str()) != INVALID_FILE_ATTRIBUTES;
}

PFN_vkGetInstanceProcAddr LoadInterposer() {
    if (g_sl.tried) {
        return g_sl.get_instance_proc_addr;
    }
    g_sl.tried = true;
    g_sl.upscale_requested = UpscaleRequested();
    if (!Requested() && !g_sl.upscale_requested) {
        return nullptr;
    }
    const std::wstring dir = DlssDir();
    const std::wstring path = dir + L"\\sl.interposer.dll";
    HMODULE module = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
    if (!module) {
        LOG_ERROR(Render_Vulkan,
                  "ryGT DLSS-G: DLSS is on but dlss\\sl.interposer.dll could not be loaded "
                  "(error {}); no DLSS frame generation or Streamline upscaling",
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
    g_sl.evaluate = reinterpret_cast<PFunEvaluateFeature>(get("slEvaluateFeature"));
    const auto gipa = reinterpret_cast<PFN_vkGetInstanceProcAddr>(get("vkGetInstanceProcAddr"));
    if (!g_sl.init || !g_sl.set_tag || !g_sl.set_constants || !g_sl.get_feature_function ||
        !g_sl.new_frame_token || !g_sl.evaluate || !gipa) {
        LOG_ERROR(Render_Vulkan, "ryGT DLSS-G: dlss\\sl.interposer.dll lacks Streamline's API");
        FreeLibrary(module);
        return nullptr;
    }

    static const std::wstring plugin_dir = dir;
    static const std::wstring log_dir = LogDir();
    static const wchar_t* plugin_paths[] = {plugin_dir.c_str()};
    // Frame generation (DLSS-G with Reflex and PCL) and/or upscaling (DLSS), as asked.
    static std::vector<sl::Feature> features;
    if (Requested()) {
        features.insert(features.end(),
                        {sl::kFeatureDLSS_G, sl::kFeatureReflex, sl::kFeaturePCL});
    }
    if (g_sl.upscale_requested) {
        features.push_back(sl::kFeatureDLSS);
    }
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
    pref.featuresToLoad = features.data();
    pref.numFeaturesToLoad = static_cast<u32>(features.size());
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
    LOG_WARNING(Render_Vulkan, "ryGT DLSS-G: Streamline started ({}{}); Vulkan goes through it",
                Requested() ? "frame generation" : "",
                g_sl.upscale_requested ? (Requested() ? ", upscaling" : "upscaling") : "");
    return gipa;
}

void Prepare(vk::Device device, VmaAllocator allocator, vk::CommandBuffer cmdbuf,
             vk::Extent2D size) {
    if (!Active() || !Requested()) {
        return;
    }
    std::scoped_lock lock{g_sl.mutex};
    if (!LoadFeatureFunctions()) {
        return;
    }
    if (!g_sl.fg.Ready(size)) {
        CreateAuxImages(g_sl.fg, device, allocator, cmdbuf, size);
    }
    if (g_sl.upscale_frame) {
        g_sl.frame = std::exchange(g_sl.upscale_frame, nullptr);
    } else {
        ++g_sl.frame_index;
        if (g_sl.new_frame_token(g_sl.frame, &g_sl.frame_index) != sl::Result::eOk ||
            !g_sl.frame) {
            g_sl.frame = nullptr;
            return;
        }
    }
    Marker(sl::PCLMarker::eSimulationStart);
    Marker(sl::PCLMarker::eSimulationEnd);
    Marker(sl::PCLMarker::eRenderSubmitStart);

    const sl::ViewportHandle viewport{0u};
    const sl::Constants constants = FlatConstants(size, false);
    g_sl.set_constants(constants, *g_sl.frame, viewport);

    const sl::Extent extent{0, 0, size.width, size.height};
    sl::Resource mvec{sl::ResourceType::eTex2d,
                      static_cast<VkImage>(static_cast<vk::Image>(g_sl.fg.motion_vectors)), nullptr,
                      static_cast<VkImageView>(*g_sl.fg.motion_vectors_view),
                      static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)};
    mvec.width = size.width;
    mvec.height = size.height;
    mvec.nativeFormat = static_cast<uint32_t>(VK_FORMAT_R16G16_SFLOAT);
    mvec.mipLevels = 1;
    mvec.arrayLayers = 1;
    mvec.flags = 0;
    mvec.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    sl::Resource dep{sl::ResourceType::eTex2d,
                     static_cast<VkImage>(static_cast<vk::Image>(g_sl.fg.depth)), nullptr,
                     static_cast<VkImageView>(*g_sl.fg.depth_view),
                     static_cast<uint32_t>(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)};
    dep.width = size.width;
    dep.height = size.height;
    dep.nativeFormat = static_cast<uint32_t>(VK_FORMAT_R32_SFLOAT);
    dep.mipLevels = 1;
    dep.arrayLayers = 1;
    dep.flags = 0;
    dep.usage = mvec.usage;
    // The backbuffer tag only carries the extent (Streamline knows the swapchain image).
    sl::Resource backbuffer{sl::ResourceType::eTex2d, nullptr, nullptr, nullptr};
    backbuffer.flags = 0;
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
        // GT-MFG: frames generated between game frames, RYGT_DLSSG_FRAMES=1..5 (2x to 6x;
        // default 1; multi-frame generation needs an RTX 50, 6x a recent driver). DLSS-G
        // reports what the GPU supports only once it runs, so the request goes in as asked and
        // falls back to one generated frame when refused.
        u32 frames = 1;
        if (const char* env = std::getenv("RYGT_DLSSG_FRAMES"); env && *env) {
            frames = std::clamp<u32>(static_cast<u32>(std::strtoul(env, nullptr, 10)), 1u, 5u);
        }
        options.numFramesToGenerate = frames;
        sl::Result set = g_sl.dlssg_set_options(viewport, options);
        if (set != sl::Result::eOk && frames > 1) {
            options.numFramesToGenerate = 1;
            set = g_sl.dlssg_set_options(viewport, options);
        }
        g_sl.options_set = set == sl::Result::eOk;
        LOG_WARNING(Render_Vulkan,
                    "ryGT DLSS-G: frame generation {} (result {}), {} generated frames a game "
                    "frame ({} asked)",
                    g_sl.options_set ? "on" : "could not be turned on", static_cast<int>(set),
                    options.numFramesToGenerate, frames);
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
                LOG_WARNING(Render_Vulkan,
                            "ryGT DLSS-G: status {:#x} (0 = generating), up to {} generated "
                            "frames supported",
                            g_sl.last_status, state.numFramesToGenerateMax);
            }
        }
    }
}

bool InitUpscale() {
    if (!Active() || !g_sl.upscale_requested) {
        return false;
    }
    std::scoped_lock lock{g_sl.mutex};
    if (g_sl.upscale_ready) {
        return true;
    }
    g_sl.dlss_get_optimal =
        Feature<PFun_slDLSSGetOptimalSettings>(sl::kFeatureDLSS, "slDLSSGetOptimalSettings");
    g_sl.dlss_set_options = Feature<PFun_slDLSSSetOptions>(sl::kFeatureDLSS, "slDLSSSetOptions");
    g_sl.upscale_ready = g_sl.dlss_get_optimal && g_sl.dlss_set_options;
    if (g_sl.upscale_ready) {
        LOG_WARNING(Render_Vulkan, "ryGT DLSS: upscaling through Streamline (sl.dlss.dll)");
    } else {
        LOG_ERROR(Render_Vulkan, "ryGT DLSS: Streamline's DLSS upscaling is not available here "
                                 "(see the Streamline lines above: it needs an NVIDIA RTX GPU and "
                                 "dlss\\nvngx_dlss.dll); using FSR");
    }
    return g_sl.upscale_ready;
}

namespace {

// The DLSS quality mode whose render size range holds input for output, or -1.
int FindMode(vk::Extent2D input, vk::Extent2D output) {
    if (input.width > output.width || input.height > output.height) {
        return -1;
    }
    static constexpr sl::DLSSMode modes[] = {
        sl::DLSSMode::eDLAA,           sl::DLSSMode::eUltraQuality,
        sl::DLSSMode::eMaxQuality,     sl::DLSSMode::eBalanced,
        sl::DLSSMode::eMaxPerformance, sl::DLSSMode::eUltraPerformance,
    };
    for (const sl::DLSSMode mode : modes) {
        sl::DLSSOptions options{};
        options.mode = mode;
        options.outputWidth = output.width;
        options.outputHeight = output.height;
        sl::DLSSOptimalSettings settings{};
        if (g_sl.dlss_get_optimal(options, settings) != sl::Result::eOk) {
            continue;
        }
        const bool fits = input.width >= settings.renderWidthMin &&
                          input.width <= settings.renderWidthMax &&
                          input.height >= settings.renderHeightMin &&
                          input.height <= settings.renderHeightMax;
        // DLAA only for a picture already at the output size.
        if (fits && (mode != sl::DLSSMode::eDLAA || input == output)) {
            return static_cast<int>(mode);
        }
    }
    return -1;
}

} // namespace

bool UpscaleSupports(vk::Extent2D input, vk::Extent2D output) {
    if (!g_sl.upscale_ready) {
        return false;
    }
    std::scoped_lock lock{g_sl.mutex};
    if (input != g_sl.sr_in || output != g_sl.sr_out) {
        g_sl.sr_in = input;
        g_sl.sr_out = output;
        g_sl.sr_mode = FindMode(input, output);
        g_sl.sr_options_set = false;
        LOG_WARNING(Render_Vulkan, "ryGT DLSS: {}x{} to {}x{}: {}", input.width, input.height,
                    output.width, output.height,
                    g_sl.sr_mode < 0 ? std::string{"no DLSS mode fits; using FSR"}
                                     : fmt::format("DLSS mode {}", g_sl.sr_mode));
    }
    return g_sl.sr_mode >= 0;
}

bool Upscale(vk::Device device, VmaAllocator allocator, vk::CommandBuffer cmdbuf,
             vk::Image input_image, vk::ImageView input, vk::Format input_format,
             vk::Extent2D input_size, vk::Image output_image, vk::ImageView output,
             vk::Format output_format, vk::Extent2D output_size) {
    if (!UpscaleSupports(input_size, output_size)) {
        return false;
    }
    std::scoped_lock lock{g_sl.mutex};
    const sl::ViewportHandle viewport{1u};
    if (!g_sl.sr_options_set) {
        sl::DLSSOptions options{};
        options.mode = static_cast<sl::DLSSMode>(g_sl.sr_mode);
        options.outputWidth = output_size.width;
        options.outputHeight = output_size.height;
        options.colorBuffersHDR = sl::Boolean::eFalse;
        const sl::Result set = g_sl.dlss_set_options(viewport, options);
        if (set != sl::Result::eOk) {
            LOG_ERROR(Render_Vulkan, "ryGT DLSS: Streamline refused the options (result {})",
                      static_cast<int>(set));
            return false;
        }
        g_sl.sr_options_set = true;
    }
    if (!g_sl.sr.Ready(input_size)) {
        CreateAuxImages(g_sl.sr, device, allocator, cmdbuf, input_size);
    }
    sl::FrameToken* frame{};
    ++g_sl.frame_index;
    if (g_sl.new_frame_token(frame, &g_sl.frame_index) != sl::Result::eOk || !frame) {
        return false;
    }
    // Frame generation (when on) carries on with this frame's token at the present.
    if (Requested()) {
        g_sl.upscale_frame = frame;
    }

    // No motion vectors: every frame starts DLSS's history over, like the plugin.
    g_sl.set_constants(FlatConstants(input_size, true), *frame, viewport);

    constexpr auto aux_usage = vk::ImageUsageFlagBits::eSampled |
                               vk::ImageUsageFlagBits::eTransferDst |
                               vk::ImageUsageFlagBits::eTransferSrc;
    // The A2R10G10B10 picture is a view of the game's A2B10G10R10 render target (the view swaps
    // red and blue). Streamline has no name for A2R10G10B10 (it logged an error each frame), so
    // it is told the image's own format; DLSS reads through the view, which keeps the colors.
    const vk::Format tag_format = input_format == vk::Format::eA2R10G10B10UnormPack32
                                      ? vk::Format::eA2B10G10R10UnormPack32
                                      : input_format;
    sl::Resource color_in = Texture(input_image, input, vk::ImageLayout::eShaderReadOnlyOptimal,
                                    input_size, tag_format, vk::ImageUsageFlagBits::eSampled);
    sl::Resource color_out = Texture(output_image, output, vk::ImageLayout::eGeneral, output_size,
                                     output_format,
                                     vk::ImageUsageFlagBits::eStorage |
                                         vk::ImageUsageFlagBits::eSampled);
    sl::Resource mvec = Texture(g_sl.sr.motion_vectors, *g_sl.sr.motion_vectors_view,
                                vk::ImageLayout::eShaderReadOnlyOptimal, input_size,
                                vk::Format::eR16G16Sfloat, aux_usage);
    sl::Resource depth = Texture(g_sl.sr.depth, *g_sl.sr.depth_view,
                                 vk::ImageLayout::eShaderReadOnlyOptimal, input_size,
                                 vk::Format::eR32Sfloat, aux_usage);
    const sl::Extent in_extent{0, 0, input_size.width, input_size.height};
    const sl::Extent out_extent{0, 0, output_size.width, output_size.height};
    const sl::ResourceTag tags[] = {
        sl::ResourceTag{&color_in, sl::kBufferTypeScalingInputColor,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &in_extent},
        sl::ResourceTag{&color_out, sl::kBufferTypeScalingOutputColor,
                        sl::ResourceLifecycle::eValidUntilEvaluate, &out_extent},
        sl::ResourceTag{&mvec, sl::kBufferTypeMotionVectors,
                        sl::ResourceLifecycle::eValidUntilPresent, &in_extent},
        sl::ResourceTag{&depth, sl::kBufferTypeDepth, sl::ResourceLifecycle::eValidUntilPresent,
                        &in_extent},
    };
    auto* command_buffer =
        reinterpret_cast<sl::CommandBuffer*>(static_cast<VkCommandBuffer>(cmdbuf));
    g_sl.set_tag(*frame, viewport, tags, static_cast<u32>(std::size(tags)), command_buffer);
    if (static bool logged = false; !logged) {
        logged = true;
        LOG_WARNING(Render_Vulkan, "ryGT DLSS: Streamline input {} -> output {}",
                    vk::to_string(input_format), vk::to_string(output_format));
    }
    const sl::BaseStructure* inputs[] = {&viewport};
    const sl::Result result = g_sl.evaluate(sl::kFeatureDLSS, *frame, inputs,
                                            static_cast<u32>(std::size(inputs)), command_buffer);
    if (result != sl::Result::eOk) {
        LOG_ERROR(Render_Vulkan, "ryGT DLSS: Streamline's DLSS failed (result {})",
                  static_cast<int>(result));
        return false;
    }
    return true;
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
    g_sl.fg.Destroy();
    g_sl.sr.Destroy();
    g_sl.sr_in = g_sl.sr_out = {};
    g_sl.sr_mode = -1;
    g_sl.sr_options_set = false;
}

} // namespace Vulkan::RytyDlssG
