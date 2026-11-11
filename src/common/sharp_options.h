// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <nlohmann/json.hpp>

#include "common/ryty_config.h"

/**
 * Temporal DLSS (GPU.temporal_dlss) options and the texture mip-bias policy that goes with it.
 * The GPU object of the launcher's settings is read once; samplers are cached, so nothing here
 * may change behind a cached VkSampler except through the one latch below. GPU.dlss_sharpening
 * is the exception that is read again while playing (host_passes/pp_pass.cpp).
 */
namespace Common::Sharp {

inline const nlohmann::json& Gpu() {
    static const nlohmann::json gpu = [] {
        const auto document = nlohmann::json::parse(RytyGameConfigText(), nullptr, false);
        if (document.is_object() && document.contains("GPU") && document["GPU"].is_object()) {
            return document["GPU"];
        }
        return nlohmann::json::object();
    }();
    return gpu;
}

inline float Number(const char* key, const char* env, float fallback, float lo, float hi) {
    float value = fallback;
    const auto& gpu = Gpu();
    if (const auto it = gpu.find(key); it != gpu.end()) {
        if (it->is_number()) {
            value = it->get<float>();
        } else if (it->is_string()) {
            const auto text = it->get<std::string>();
            char* end{};
            const float parsed = std::strtof(text.c_str(), &end);
            if (end != text.c_str() && *end == 0) {
                value = parsed;
            }
        }
    }
    if (const char* e = std::getenv(env); e && *e) {
        char* end{};
        const float parsed = std::strtof(e, &end);
        value = end != e && *end == 0 ? parsed : fallback;
    }
    return std::isfinite(value) ? std::clamp(value, lo, hi) : fallback;
}

inline bool Flag(const char* key, const char* env, bool fallback) {
    bool value = fallback;
    if (const auto it = Gpu().find(key); it != Gpu().end() && it->is_boolean()) {
        value = it->get<bool>();
    }
    if (const char* e = std::getenv(env)) {
        if (!std::strcmp(e, "0")) {
            value = false;
        }
        if (!std::strcmp(e, "1")) {
            value = true;
        }
    }
    return value;
}

struct Settings {
    float texture_bias = Number("texture_lod_bias", "RYGT_TEXTURE_LOD_BIAS", 0, -16, 16);
    /// GPU.dlss_auto_mip_bias: the texture bias that follows the DLSS scale (default on).
    bool auto_mip = Flag("dlss_auto_mip_bias", "RYGT_DLSS_AUTO_MIP_BIAS", true);
    /// GPU.dlss_mip_extra: the extra term of the bias. Default: -1 with temporal DLSS.
    float mip_extra = Number("dlss_mip_extra", "RYGT_DLSS_MIP_EXTRA", 0, -2, 0);
    /// GPU.dlss_preset: 0 = the runtime's default preset.
    int preset = static_cast<int>(Number("dlss_preset", "RYGT_DLSS_PRESET", 0, 0, 13));
    /// GPU.tdlss_hdr: -1 = by the scene, 0 / 1 = force the SDK's colour-buffer encoding flag.
    int hdr = static_cast<int>(Number("tdlss_hdr", "RYGT_TDLSS_HDR", -1, -1, 1));
    float pre_exposure = Number("tdlss_pre_exposure", "RYGT_TDLSS_PRE_EXPOSURE", 1, .001f, 1000);
    float exposure_scale = Number("tdlss_exposure_scale", "RYGT_TDLSS_EXPOSURE_SCALE", 1, .001f,
                                  1000);
    /// GPU.tdlss_auto_exposure: DLSS estimates the exposure itself (default on, as before). Off:
    /// the exposure scale above is used as it is; a test switch for the road's brightness.
    bool auto_exposure = Flag("tdlss_auto_exposure", "RYGT_TDLSS_AUTO_EXPOSURE", true);
    /// An unreplayed native writer: apply only what it changed on top of the reconstructed scene
    /// (true), or replace the reconstructed scene with the upscaled native one.
    bool resync_merge = Flag("tdlss_resync_merge", "RYGT_TDLSS_RESYNC_MERGE", true);
    float merge_lo = Number("tdlss_merge_lo", "RYGT_TDLSS_MERGE_LO", 0.25f, 0.f, 8.f);
    float merge_hi = Number("tdlss_merge_hi", "RYGT_TDLSS_MERGE_HI", 0.9f, 0.01f, 16.f);
    /// GPU.dlss_log: a GT7-DLSS status line once a second (off by default).
    bool dlss_log = Flag("dlss_log", "RYGT_DLSS_LOG", false);
};

// Opt-in investigation of unsupported scene shapes and a bounded, timeline-retired dump.
inline bool Diagnostics() {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_DLSS_DIAG");
        return env && std::strcmp(env, "1") == 0;
    }();
    return enabled;
}

inline const Settings& Get() {
    static const Settings settings;
    return settings;
}

/// log2(render / display) while a DLSS scale is active, 0 otherwise.
inline std::atomic<float> resolution_bias{0.0f};
// UI pixel-footprint estimate and actual private temporal allocations. Retired images count.
inline std::atomic<std::uint64_t> ui_image_bytes{0}, temporal_image_bytes{0};


/// Set once temporal reconstruction has succeeded, and kept: the bias must not follow a
/// momentary presentation gap (menus), or the samplers would change every second.
inline std::atomic<bool> temporal_mip_latched{false};

/// GPU.dlss_sharpening as last read (percent), for the status line.
inline std::atomic<float> sharpen_percent{0.0f};

inline void SetScale(unsigned render, unsigned display, bool active) {
    // NVIDIA's DLSS guide: texture bias = native bias + log2(render / display) - 1 + epsilon.
    const float ratio = active && Get().auto_mip && render && display > render
                            ? std::log2(static_cast<float>(render) / static_cast<float>(display))
                            : 0.0f;
    resolution_bias.store(ratio, std::memory_order_relaxed);
}

/// The extra term. Temporal DLSS accumulates jittered samples, which is what the guide's -1 is
/// for, so it defaults to -1 there; an explicit GPU.dlss_mip_extra wins.
inline float MipExtra() {
    static const bool explicit_extra = Gpu().contains("dlss_mip_extra") || [] {
        const char* e = std::getenv("RYGT_DLSS_MIP_EXTRA");
        return e && *e;
    }();
    return explicit_extra ? Get().mip_extra : -1.0f;
}

/// The bias added to the material samplers' LOD bias. Zero until temporal DLSS has succeeded,
/// so a game without GPU.temporal_dlss samples exactly as upstream does.
inline float ResolutionBias() {
    if (!temporal_mip_latched.load(std::memory_order_relaxed)) {
        return 0.0f;
    }
    const float ratio = resolution_bias.load(std::memory_order_relaxed);
    return ratio == 0.0f ? 0.0f : ratio + MipExtra();
}

} // namespace Common::Sharp
