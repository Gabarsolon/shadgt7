// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/serdes.h"
#include "shader_recompiler/frontend/fetch_shader.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

// Set before cache readers/workers start; candidate blobs are ignored on other profiles.
void SetSingleSampleBarycentricCacheEnabled(bool enabled);
void SetVertexOrderCheckCacheEnabled(bool enabled); // AMD-307
void SetNativeCubeCacheEnabled(bool enabled);       // FIX-GT7-083

void RegisterShaderMeta(const Shader::Info& info,
                        const std::optional<Shader::Gcn::FetchShaderData>& fetch_shader_data,
                        const Shader::StageSpecialization& spec, size_t perm_hash, size_t perm_idx);
void RegisterShaderBinary(std::vector<u32>&& spv, u64 pgm_hash, size_t perm_idx);
/// FIX-GT7-088: `other_cube_sampling` is set when a whole entry was refused only because the other
/// engine's cube sampling made it (FIX-GT7-083): its index is that translation's, not a free one.
bool LoadShaderMeta(Serialization::Archive& ar, Shader::Info& info,
                    Shader::StageSpecialization& spec, size_t& perm_idx,
                    bool* other_cube_sampling = nullptr);

} // namespace Vulkan
