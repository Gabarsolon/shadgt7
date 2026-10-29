// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/types.h"
#include "video_core/renderer_vulkan/vk_common.h"

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

/// Around the swapchain present (Reflex/PCL markers DLSS-G needs to pair frames).
void PresentStart();
void PresentEnd();

/// Releases the images (the device must be idle).
void Shutdown();

} // namespace Vulkan::RytyDlssG
