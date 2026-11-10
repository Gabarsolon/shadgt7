// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>

#if defined(RYTY_GPU) && defined(_WIN32)
#include "common/ryty_config.h"
#endif

namespace Common::Hdr {

struct OutputSettings {
    float gain = 1.0f;
    float peak_nits = 0.0f;
    float metadata_nits = 0.0f;
    float metadata_fall_nits = 400.0f;
};

inline double Number(const nlohmann::json& object, const char* key, double fallback) {
    const auto it = object.find(key);
    if (it == object.end()) {
        return fallback;
    }
    double value = fallback;
    if (it->is_number()) {
        value = it->get<double>();
    } else if (it->is_string()) {
        const auto& text = it->get_ref<const std::string&>();
        std::string_view input{text};
        const auto first = input.find_first_not_of(" \t\r\n");
        if (first == input.npos) {
            return fallback;
        }
        input = input.substr(first, input.find_last_not_of(" \t\r\n") - first + 1);
        const auto parsed = std::from_chars(input.data(), input.data() + input.size(), value);
        if (parsed.ec != std::errc{} || parsed.ptr != input.data() + input.size()) {
            return fallback;
        }
    } else {
        return fallback;
    }
    return std::isfinite(value) ? value : fallback;
}

inline OutputSettings ParseOutputSettings(std::string_view text) {
    OutputSettings settings;
    const auto document = nlohmann::json::parse(text, nullptr, false);
    if (!document.is_object() || !document.contains("GPU") || !document["GPU"].is_object()) {
        return settings;
    }
    const auto& gpu = document["GPU"];
    settings.gain = static_cast<float>(std::clamp(Number(gpu, "hdr_gain", 1.0), 0.1, 8.0));
    settings.peak_nits = static_cast<float>(std::clamp(Number(gpu, "hdr_peak_nits", 0.0), 0.0, 10000.0));
    const double metadata = Number(gpu, "hdr_metadata_nits", 0.0);
    settings.metadata_nits = metadata > 0.0 ? static_cast<float>(std::clamp(metadata, 100.0, 10000.0)) : 0.0f;
    settings.metadata_fall_nits = static_cast<float>(std::clamp(
        Number(gpu, "hdr_metadata_fall_nits", 400.0), 1.0,
        settings.metadata_nits > 0.0f ? double(settings.metadata_nits) : 10000.0));
    return settings;
}

// Called only for HDR output. One snapshot keeps the independently configurable
// controls consistent, without adding file I/O to the SDR presentation path.
inline OutputSettings ReadOutputSettings() {
    static std::mutex mutex;
    std::scoped_lock lock{mutex};
    static OutputSettings settings;
    static auto next_read = std::chrono::steady_clock::time_point{};
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_read) {
        next_read = now + std::chrono::seconds(2);
#if defined(RYTY_GPU) && defined(_WIN32)
        settings = ParseOutputSettings(RytyReadGameConfig());
#endif
    }
    return settings;
}

} // namespace Common::Hdr
