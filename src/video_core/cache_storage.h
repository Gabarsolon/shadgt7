// SPDX-FileCopyrightText: Copyright 2025 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include "common/path_util.h"
#include "common/singleton.h"
#include "common/types.h"

#include <functional>
#include <thread>
#include <vector>

namespace Storage {

/// FIX-018/020's marks: up to which stored permutation each program was translated before a fix.
[[nodiscard]] std::filesystem::path RetranslationMarksPath();

/// SHD-010 leaves another build's translations in the store (RYGT_KEEP_SHADER_CACHE=1, or a build
/// that knows neither its commit nor its description).
[[nodiscard]] bool StoreMayHoldOtherBuilds();

enum class BlobType : u32 {
    ShaderMeta,
    ShaderBinary,
    PipelineKey,
    ShaderProfile,
};

class DataBase {
public:
    static DataBase& Instance() {
        return *Common::Singleton<DataBase>::Instance();
    }

    void Open();
    /// D10: the store the old renderer fills, read-only next to the new renderer's own store.
    static DataBase& Legacy() {
        static DataBase legacy;
        return legacy;
    }
    /// D10: opens Legacy() for reading (not with an archived cache).
    void OpenLegacy();
    void Close();
    [[nodiscard]] bool IsOpened() const {
        return opened;
    }
    void FinishPreload();

    bool Save(BlobType type, const std::string& name, std::vector<u8>&& data);
    bool Save(BlobType type, const std::string& name, std::vector<u32>&& data);

    void Load(BlobType type, const std::string& name, std::vector<u8>& data);
    void Load(BlobType type, const std::string& name, std::vector<u32>& data);
    /// Whether a blob is stored, without reporting a missing one (on-demand loading probes).
    [[nodiscard]] bool Exists(BlobType type, const std::string& name) const;

    void ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func);

private:
    std::jthread io_worker{};
    std::filesystem::path cache_path{};
    bool opened{};
};

} // namespace Storage
