// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"
#include "ryty_gpu/ryty_gt7_dlss_camera.h"

VK_DEFINE_HANDLE(VmaAllocator)

namespace Vulkan::RytyDlssG {

/// ryGT: NVIDIA DLSS frame generation (GPU.dlssg_enabled) through Streamline: dlss\sl.interposer.dll
/// and its plugins (sl.dlss_g, sl.reflex, sl.pcl, sl.common, nvngx_dlssg.dll) next to GTSport.exe.
/// Streamline's interposer stands in for vulkan-1.dll: the renderer's dispatcher is loaded from it,
/// so the instance, device, surface and swapchain calls go through Streamline, which adds what
/// DLSS-G needs and shows the generated frames from its own present path. The game picture comes
/// without motion vectors or depth, so zero motion vectors and a flat depth image are tagged.

/// True when DLSS frame generation is on in the config (the AMD frame generation then stays off).
bool Requested();

/// Loads and starts Streamline before the Vulkan instance exists. Returns its
/// vkGetInstanceProcAddr, or null when DLSS-G is off or Streamline is missing or fails (the
/// renderer then uses vulkan-1.dll as usual).
PFN_vkGetInstanceProcAddr LoadInterposer();

/// Whether Streamline is in use (LoadInterposer succeeded).
bool Active();

/// Per presented game frame, before the frame's command buffer is submitted: starts a Streamline
/// frame, records the tags (motion vectors, depth) and sets the constants and options.
void Prepare(vk::Device device, VmaAllocator allocator, vk::CommandBuffer cmdbuf,
             vk::Extent2D size);

/// ryGT: DLSS upscaling (GPU.dlss_enabled) through Streamline's sl.dlss.dll, when it is in the dlss
/// folder (RYGT_DLSS_STREAMLINE=0 keeps the self-built ryGT_dlss.dll plugin instead). It runs on
/// its own viewport with zero motion vectors and a flat depth image, starting DLSS's history over
/// every frame like the plugin.
bool UpscaleRequested();

/// After the device exists: whether Streamline's DLSS upscaling is available.
bool InitUpscale();

/// Whether DLSS has a quality mode that scales input to output.
bool UpscaleSupports(vk::Extent2D input, vk::Extent2D output);

/// Records DLSS into cmdbuf: input (shader-read-only layout) into output (general layout,
/// storage usage). False when nothing usable was written.
bool Upscale(vk::Device device, VmaAllocator allocator, vk::CommandBuffer cmdbuf,
             vk::Image input_image, vk::ImageView input, vk::Format input_format,
             vk::Extent2D input_size, vk::Image output_image, vk::ImageView output,
             vk::Format output_format, vk::Extent2D output_size);

struct TemporalImage {
    vk::Image image{};
    vk::ImageView view{};
    vk::Format format{};
    vk::Extent2D size{};
};
struct TemporalFrame {
    u64 frame{}, tick{};
    TemporalImage color, depth, motion, output;
    Gt7Temporal::Camera camera;
    Gt7Temporal::Matrix previous_projection{}, previous_view{};
    float jitter_x{}, jitter_y{}, mvec_scale = 1, frame_ms = 16.667f;
    bool reset = true, hdr = false;
};
// Temporal SR is isolated from legacy viewport 1 and FG viewport 0. All input images
// must be owned, shader-read-only; output is storage-capable in GENERAL.
bool TemporalSupports(vk::Extent2D input, vk::Extent2D output);
bool EvaluateTemporal(vk::CommandBuffer cmd, const TemporalFrame& frame);
/// What the last temporal evaluation sent to the SDK (for the once-a-second GT7-DLSS log line).
struct TemporalReport {
    u64 budget_warnings{}; // successful evaluations carrying eWarnOutOfVRAM
    int mode{-1};            // sl::DLSSMode value
    const char* mode_name{"none"};
    int preset_requested{0}; // GPU.dlss_preset (0 = runtime default)
    int preset_sent{0};      // the preset sent for this mode
    bool hdr{}, auto_exposure{};
    float pre_exposure{1}, exposure_scale{1};
    float optimal_w{}, optimal_h{}; // the SDK's optimal render size for this mode (info only)
};
TemporalReport LastTemporalReport();
// Free viewport 2 only after both the last SDK use and its GPU tick have
// retired.
bool ReleaseIdleTemporal(u64 completed_tick);

/// Around the swapchain present (Reflex/PCL markers DLSS-G needs to pair frames).
void PresentStart();
void PresentEnd();

/// Releases the images (the device must be idle).
void Shutdown();

} // namespace Vulkan::RytyDlssG
