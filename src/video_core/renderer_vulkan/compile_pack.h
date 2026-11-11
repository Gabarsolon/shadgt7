// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// PERF-GT7-108: the graphics compile pack. Every shader object set the driver compiles from SPIR-V
// is also kept as a recipe (its stages' SPIR-V, flags and descriptor set layouts), in a folder per
// GPU (not per driver): <cache>/<serial>.compile_pack/<vendor>-<device>/<key>.pack. With
// RYGT_COMPILE_PACK=1 the recipes the SO-009 cache lacks for this driver are compiled in the
// background at start, on every core below normal priority, so a track already in the pack starts
// with its shader objects ready: after a driver update (the driver's caches start empty), or with a
// pack made on another PC with the same GPU. RYGT_PACK_RECORD=0: no recipes are kept.
// FIX-GT7-136: recording and the start-up compile need shader objects (off by default since
// FIX-GT7-147, RYGT_SHADER_OBJECT=1 turns them on): without them no SPIR-V is kept, so there is
// no recipe to keep or compile.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <vector>

#include <fmt/format.h>
#include <miniz.h>
#include <xxhash.h>

namespace Vulkan::CompilePack {

inline std::atomic<std::uint32_t> g_total{0};   ///< recipes the start-up compile looks at
inline std::atomic<std::uint32_t> g_done{0};    ///< of which done (compiled, cached or failed)
inline std::atomic<bool> g_running{false};      ///< the start-up compile is running

inline bool RecordingOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_PACK_RECORD");
        return !(env && env[0] == '0');
    }();
    return on;
}

inline bool CompileOn() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_COMPILE_PACK");
        return env && env[0] == '1';
    }();
    return on;
}

struct Binding {
    std::uint32_t binding;
    std::uint32_t type;
    std::uint32_t count;
    std::uint32_t stages;
};

struct SetLayout {
    std::uint32_t flags = 0;
    std::vector<Binding> bindings;
};

struct Stage {
    std::uint32_t stage = 0; ///< VkShaderStageFlagBits
    std::uint32_t next = 0;  ///< VkShaderStageFlags
    std::vector<std::uint32_t> code;
};

struct Recipe {
    std::uint64_t key = 0;   ///< SO-008's share key: the SO-009 cache file's name with the driver
    std::uint32_t flags = 0; ///< VkShaderCreateFlagsEXT
    std::uint32_t push_stages = 0;
    std::uint32_t push_size = 0;
    std::vector<SetLayout> sets; ///< one per set, empty ones included
    std::vector<Stage> stages;
};

inline constexpr char Magic[8] = {'R', 'Y', 'G', 'T', 'P', 'A', 'C', 'K'};
inline constexpr std::uint32_t FormatVersion = 1;

namespace detail {
inline void Put(std::vector<std::uint8_t>& out, std::uint32_t value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(value));
}
inline void Put64(std::vector<std::uint8_t>& out, std::uint64_t value) {
    const auto* bytes = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), bytes, bytes + sizeof(value));
}
struct Reader {
    const std::uint8_t* data;
    std::size_t size;
    std::size_t at = 0;
    bool Get(std::uint32_t& value) {
        if (size - at < sizeof(value)) return false;
        std::memcpy(&value, data + at, sizeof(value));
        at += sizeof(value);
        return true;
    }
    bool Get64(std::uint64_t& value) {
        if (size - at < sizeof(value)) return false;
        std::memcpy(&value, data + at, sizeof(value));
        at += sizeof(value);
        return true;
    }
};
} // namespace detail

/// Written beside and renamed, so a reader never sees half a file. False on any failure.
inline bool Write(const std::filesystem::path& file, const Recipe& recipe) {
    std::vector<std::uint8_t> raw;
    detail::Put64(raw, recipe.key);
    detail::Put(raw, recipe.flags);
    detail::Put(raw, recipe.push_stages);
    detail::Put(raw, recipe.push_size);
    detail::Put(raw, static_cast<std::uint32_t>(recipe.sets.size()));
    for (const auto& set : recipe.sets) {
        detail::Put(raw, set.flags);
        detail::Put(raw, static_cast<std::uint32_t>(set.bindings.size()));
        for (const auto& binding : set.bindings) {
            detail::Put(raw, binding.binding);
            detail::Put(raw, binding.type);
            detail::Put(raw, binding.count);
            detail::Put(raw, binding.stages);
        }
    }
    detail::Put(raw, static_cast<std::uint32_t>(recipe.stages.size()));
    for (const auto& stage : recipe.stages) {
        detail::Put(raw, stage.stage);
        detail::Put(raw, stage.next);
        detail::Put(raw, static_cast<std::uint32_t>(stage.code.size()));
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(stage.code.data());
        raw.insert(raw.end(), bytes, bytes + stage.code.size() * sizeof(std::uint32_t));
    }
    mz_ulong packed_size = mz_compressBound(static_cast<mz_ulong>(raw.size()));
    std::vector<std::uint8_t> packed(packed_size);
    if (mz_compress2(packed.data(), &packed_size, raw.data(), static_cast<mz_ulong>(raw.size()),
                     6) != MZ_OK) {
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(file.parent_path(), ec);
    auto temporary = file;
    temporary += fmt::format(".{}", std::hash<std::thread::id>{}(std::this_thread::get_id()));
    bool ok = false;
    if (std::FILE* out = std::fopen(temporary.string().c_str(), "wb")) {
        const std::uint32_t version = FormatVersion;
        const auto raw_size = static_cast<std::uint32_t>(raw.size());
        const auto stored = static_cast<std::uint32_t>(packed_size);
        const std::uint64_t checksum = XXH3_64bits(raw.data(), raw.size());
        ok = std::fwrite(Magic, sizeof(Magic), 1, out) == 1 &&
             std::fwrite(&version, 4, 1, out) == 1 && std::fwrite(&raw_size, 4, 1, out) == 1 &&
             std::fwrite(&stored, 4, 1, out) == 1 && std::fwrite(&checksum, 8, 1, out) == 1 &&
             std::fwrite(packed.data(), 1, stored, out) == stored;
        ok = std::fclose(out) == 0 && ok;
    }
    if (ok) {
        std::filesystem::rename(temporary, file, ec);
        ok = !ec;
    }
    if (!ok) {
        std::filesystem::remove(temporary, ec);
    }
    return ok;
}

/// False for a damaged, foreign or other-version file.
inline bool Read(const std::filesystem::path& file, Recipe& recipe) {
    std::vector<std::uint8_t> packed;
    std::uint32_t version = 0, raw_size = 0, stored = 0;
    std::uint64_t checksum = 0;
    {
        std::FILE* in = std::fopen(file.string().c_str(), "rb");
        if (!in) return false;
        char magic[8]{};
        bool ok = std::fread(magic, sizeof(magic), 1, in) == 1 &&
                  std::memcmp(magic, Magic, sizeof(Magic)) == 0 &&
                  std::fread(&version, 4, 1, in) == 1 && version == FormatVersion &&
                  std::fread(&raw_size, 4, 1, in) == 1 && std::fread(&stored, 4, 1, in) == 1 &&
                  std::fread(&checksum, 8, 1, in) == 1 && raw_size < (64u << 20) &&
                  stored < (64u << 20);
        if (ok) {
            packed.resize(stored);
            ok = std::fread(packed.data(), 1, stored, in) == stored;
        }
        std::fclose(in);
        if (!ok) return false;
    }
    std::vector<std::uint8_t> raw(raw_size);
    mz_ulong out_size = raw_size;
    if (mz_uncompress(raw.data(), &out_size, packed.data(), static_cast<mz_ulong>(stored)) !=
            MZ_OK ||
        out_size != raw_size || XXH3_64bits(raw.data(), raw.size()) != checksum) {
        return false;
    }
    detail::Reader r{raw.data(), raw.size()};
    std::uint32_t sets = 0, stages = 0;
    if (!r.Get64(recipe.key) || !r.Get(recipe.flags) || !r.Get(recipe.push_stages) ||
        !r.Get(recipe.push_size) || !r.Get(sets) || sets > 8) {
        return false;
    }
    recipe.sets.resize(sets);
    for (auto& set : recipe.sets) {
        std::uint32_t count = 0;
        if (!r.Get(set.flags) || !r.Get(count) || count > 4096) return false;
        set.bindings.resize(count);
        for (auto& binding : set.bindings) {
            if (!r.Get(binding.binding) || !r.Get(binding.type) || !r.Get(binding.count) ||
                !r.Get(binding.stages)) {
                return false;
            }
        }
    }
    if (!r.Get(stages) || stages == 0 || stages > 4) return false;
    recipe.stages.resize(stages);
    for (auto& stage : recipe.stages) {
        std::uint32_t words = 0;
        if (!r.Get(stage.stage) || !r.Get(stage.next) || !r.Get(words) ||
            words > (raw.size() - r.at) / sizeof(std::uint32_t)) {
            return false;
        }
        stage.code.resize(words);
        std::memcpy(stage.code.data(), raw.data() + r.at, words * sizeof(std::uint32_t));
        r.at += words * sizeof(std::uint32_t);
    }
    return true;
}

/// The pack folder of this GPU, under the cache folder.
inline std::filesystem::path Dir(const std::filesystem::path& cache_dir, std::string_view serial,
                                  std::uint32_t vendor, std::uint32_t device) {
    return cache_dir / fmt::format("{}.compile_pack", serial) /
           fmt::format("{:04x}-{:04x}", vendor, device);
}

inline std::filesystem::path File(const std::filesystem::path& dir, std::uint64_t key) {
    return dir / fmt::format("{:016x}.pack", key);
}

} // namespace Vulkan::CompilePack
