// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>

#include "common/serdes.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/specialization.h"

namespace Vulkan {

/// FIX-043's record of which bound-resource properties a translation read, stored after the
/// shader's Info from meta version 16. Without it a stored permutation, whose key leaves those
/// properties out, would never match the key rebuilt from the loaded shader.
inline void WriteResourceUsage(Serialization::Writer& out, const Shader::Info& info) {
    out.Write(u8{info.resource_usage_known});
    out.Write(u64{info.buffer_stride_used.to_ullong()});
    out.Write(u64{info.image_srgb_used.to_ullong()});
}

inline void ReadResourceUsage(Serialization::Reader& in, Shader::Info& info) {
    u8 known{};
    u64 buffers{};
    u64 images{};
    in.Read(known);
    in.Read(buffers);
    in.Read(images);
    info.resource_usage_known = known != 0;
    info.buffer_stride_used = std::bitset<Shader::NUM_BUFFERS>{buffers};
    info.image_srgb_used = std::bitset<Shader::NUM_IMAGES>{images};
}

/// FIX-GT7-083: native cube sampling (Profile::native_cube_sampling, the native engine's) makes a
/// pixel shader's ordinary cube samples cube-array samples in its code, bound to cube-array views;
/// without it they sample arrays of six faces, bound as 2D arrays (the classic engine's views).
/// Both engines share the store, so a translation made without it that samples a cube says so
/// (ClassicCubeVersion in its binary version) and isn't read with it on, where the native engine
/// would go on sampling faces; one made with it isn't read with it off when one of its images is a
/// native cube sample, which the classic engine would bind a 2D array to.
inline bool TranslatedWithFaceCubes(const Shader::StageSpecialization& spec, bool native_cube) {
    return !native_cube && spec.runtime_info.hw_stage == Shader::HwStage::Fragment &&
           std::ranges::any_of(spec.images, [](const Shader::ImageSpecialization& image) {
               return image.type == AmdGpu::ImageType::Cube && !image.is_storage;
           });
}

inline bool StoredCubesFit(const Shader::StageSpecialization& spec, const Shader::Info& info,
                           bool face_cubes, bool native_cube) {
    if (spec.runtime_info.hw_stage != Shader::HwStage::Fragment) {
        return true;
    }
    if (native_cube) {
        return !face_cubes;
    }
    return std::ranges::none_of(info.images, [](const Shader::ImageResource& image) {
        return image.post_op == Shader::SharpFetchPostOp::NativeCubeSampling;
    });
}

} // namespace Vulkan
