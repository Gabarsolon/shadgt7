// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include "common/elf_info.h"
#include "common/io_file.h"
#include "common/polyfill_thread.h"
#include "common/scm_rev.h"
#include "common/thread.h"
#include "core/emulator_settings.h"

#include "video_core/buffer_cache/direct_memory.h"
#include "video_core/cache_storage.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

#include <miniz.h>

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>

namespace {

std::mutex submit_mutex{};
u32 num_requests{};
std::condition_variable_any request_cv{};
std::queue<std::packaged_task<void()>> req_queue{};
std::mutex m_request{};

mz_zip_archive zip_ar{};
bool ar_is_read_only{true};

} // namespace

namespace Storage {

namespace {

// SHD-010: the translated shaders and pipeline keys a renderer build stored come from its
// recompiler; a newer release's may translate differently, and its stored binaries would be used
// unchecked. Each store is stamped with the build that filled it (its commit: a release is one
// commit), and a store of another build (or of none, from before the stamp) is cleared once
// when a different build opens it. Only this store: the GPU driver's own caches are left alone.
// RYGT_KEEP_SHADER_CACHE=1 keeps it (tests across builds).
constexpr const char* StampName = "renderer_build.txt";

bool KeepShaderCache() {
    const char* env = std::getenv("RYGT_KEEP_SHADER_CACHE");
    return env && env[0] == '1';
}

std::string ReadStamp(const std::filesystem::path& stamp) {
    std::error_code ec;
    if (!std::filesystem::is_regular_file(stamp, ec)) {
        return {};
    }
    Common::FS::IOFile file{stamp, Common::FS::FileAccessMode::Read};
    std::string text(file.GetSize(), '\0');
    file.ReadRaw<char>(text.data(), text.size());
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
        text.pop_back();
    }
    return text;
}

bool WriteStamp(const std::filesystem::path& stamp) {
    Common::FS::IOFile file{stamp, Common::FS::FileAccessMode::Create};
    if (!file.IsOpen()) {
        return false;
    }
    const std::string text = std::string{Common::g_scm_rev} + "\n";
    return file.WriteRaw<char>(text.data(), text.size()) == text.size();
}

/// The store at `store` (a folder, or an archive with its stamp at `stamp`) as this build's:
/// cleared first when another build filled it. The stamp is written only after the clear.
void ClaimStore(const std::filesystem::path& store, const std::filesystem::path& stamp,
                bool is_folder) {
    const std::string build = Common::g_scm_rev;
    if (build.empty()) {
        return; // a build without its commit: nothing to compare
    }
    const std::string stored = ReadStamp(stamp);
    if (stored == build) {
        return;
    }
    std::error_code ec;
    u64 files = 0, bytes = 0;
    if (is_folder && std::filesystem::is_directory(store, ec)) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(store, ec)) {
            if (entry.is_regular_file(ec) && entry.path().filename() != StampName) {
                ++files;
                bytes += entry.file_size(ec);
            }
        }
    } else if (!is_folder && std::filesystem::is_regular_file(store, ec)) {
        files = 1;
        bytes = std::filesystem::file_size(store, ec);
    }
    if (files != 0 && KeepShaderCache()) {
        LOG_WARNING(Render, "SHD-010: shader store {} is from build {} (this is {}); kept "
                            "(RYGT_KEEP_SHADER_CACHE=1)",
                    store.string(), stored.empty() ? "unknown" : stored, build);
        return;
    }
    if (files != 0) {
        bool cleared = true;
        if (is_folder) {
            for (const auto& entry : std::filesystem::directory_iterator(store, ec)) {
                if (entry.path().filename() == StampName) {
                    continue;
                }
                std::error_code remove_ec;
                std::filesystem::remove_all(entry.path(), remove_ec);
                cleared &= !remove_ec;
            }
        } else {
            std::error_code remove_ec;
            std::filesystem::remove(store, remove_ec);
            cleared = !remove_ec;
        }
        if (!cleared) {
            // The stamp stays as it was: the next start tries again.
            LOG_ERROR(Render, "SHD-010: could not clear the shader store {} of build {}",
                      store.string(), stored.empty() ? "unknown" : stored);
            return;
        }
        LOG_WARNING(Render, "SHD-010: new renderer build {} (store from {}): cleared {} files "
                            "({:.1f} MB) from {}; shaders are translated again as they are needed",
                    build, stored.empty() ? "unknown" : stored, files, bytes / 1048576.0,
                    store.string());
    }
    if (is_folder) {
        std::filesystem::create_directories(store, ec);
    }
    if (!WriteStamp(stamp)) {
        LOG_ERROR(Render, "SHD-010: could not write {}", stamp.string());
    }
}

} // namespace

void ProcessIO(const std::stop_token& stoken) {
    Common::SetCurrentThreadName("shadGT:PipelineCacheIO");

    while (!stoken.stop_requested()) {
        {
            std::unique_lock lk{submit_mutex};
            Common::CondvarWait(request_cv, lk, stoken, [&] { return num_requests; });
        }

        if (stoken.stop_requested()) {
            break;
        }

        while (num_requests) {
            std::packaged_task<void()> request{};
            {
                std::scoped_lock lock{m_request};
                if (req_queue.empty()) {
                    continue;
                }
                request = std::move(req_queue.front());
                req_queue.pop();
            }

            if (request.valid()) {
                request();
                request.get_future().wait();
            }

            --num_requests;
        }
    }
}

constexpr std::string GetBlobFileExtension(BlobType type) {
    switch (type) {
    case BlobType::ShaderMeta: {
        return "meta";
    }
    case BlobType::ShaderBinary: {
        return "spv";
    }
    case BlobType::PipelineKey: {
        return "key";
    }
    case BlobType::ShaderProfile: {
        return "bin";
    }
    default:
        UNREACHABLE();
    }
}

void DataBase::Open() {
    if (opened) {
        return;
    }

    const auto& game_info = Common::ElfInfo::Instance();

    using namespace Common::FS;
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_zero_struct(&zip_ar);

        cache_path = GetUserPath(PathType::CacheDir) /
                     std::filesystem::path{game_info.GameSerial()}.replace_extension(".zip");
        {
            auto stamp = cache_path;
            stamp.replace_extension(".build.txt");
            ClaimStore(cache_path, stamp, false); // SHD-010
        }

        if (!mz_zip_reader_init_file(&zip_ar, cache_path.string().c_str(),
                                     MZ_ZIP_FLAG_READ_ALLOW_WRITING) ||
            !mz_zip_validate_archive(&zip_ar, 0)) {
            LOG_INFO(Render, "Cache archive {} is not found or archive is corrupted",
                     cache_path.string().c_str());
            mz_zip_reader_end(&zip_ar);
            mz_zip_writer_init_file(&zip_ar, cache_path.string().c_str(), 0);
        }
    } else {
        cache_path = GetUserPath(PathType::CacheDir) / game_info.GameSerial();
        // D2: direct-memory shaders differ from the stored ones, which load unchecked.
        if (VideoCore::DirectMemory::Enabled()) {
            // SHD-002: stage descriptor sets: translations of their own.
            cache_path += "-direct-sets";
        }
        if (!std::filesystem::exists(cache_path)) {
            std::filesystem::create_directories(cache_path);
        }
        ClaimStore(cache_path, cache_path / StampName, true); // SHD-010
    }

    io_worker = std::jthread{ProcessIO};
    opened = true;
}

void DataBase::OpenLegacy() {
    if (opened || EmulatorSettings.IsPipelineCacheArchived()) {
        return;
    }
    const auto& game_info = Common::ElfInfo::Instance();
    cache_path = Common::FS::GetUserPath(Common::FS::PathType::CacheDir) / game_info.GameSerial();
    std::error_code ec;
    opened = std::filesystem::is_directory(cache_path, ec);
    // SHD-010: another build's store is not read (the old renderer clears it when it opens it).
    if (opened && ReadStamp(cache_path / StampName) != Common::g_scm_rev && !KeepShaderCache()) {
        LOG_INFO(Render, "SHD-010: old renderer's shader store at {} is from another build; not "
                         "read",
                 cache_path.string());
        opened = false;
        return;
    }
    LOG_INFO(Render, "D10: old renderer's shader store {} at {}", opened ? "found" : "not found",
             cache_path.string());
}

void DataBase::Close() {
    if (!IsOpened()) {
        return;
    }

    io_worker.request_stop();
    io_worker.join();

    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_writer_finalize_archive(&zip_ar);
        mz_zip_writer_end(&zip_ar);
    }

    opened = false;
    LOG_INFO(Render, "Cache dumped");
}

template <typename T>
bool WriteVector(const BlobType type, std::filesystem::path&& path_, std::vector<T>&& v) {
    {
        auto request = std::packaged_task<void()>{[=]() {
            auto path{path_};
            path.replace_extension(GetBlobFileExtension(type));
            if (EmulatorSettings.IsPipelineCacheArchived()) {
                ASSERT_MSG(!ar_is_read_only,
                           "The archive is read-only. Did you forget to call `FinishPreload`?");
                if (!mz_zip_writer_add_mem(&zip_ar, path.string().c_str(), v.data(),
                                           v.size() * sizeof(T), MZ_BEST_COMPRESSION)) {
                    LOG_ERROR(Render, "Failed to add {} to the archive", path.string().c_str());
                }
            } else {
                using namespace Common::FS;
                const auto file = IOFile{path, FileAccessMode::Create};
                file.Write(v);
            }
        }};
        std::scoped_lock lock{m_request};
        req_queue.emplace(std::move(request));
    }

    std::scoped_lock lk{submit_mutex};
    ++num_requests;
    request_cv.notify_one();
    return true;
}

template <typename T>
void LoadVector(BlobType type, std::filesystem::path& path, std::vector<T>& v) {
    using namespace Common::FS;
    path.replace_extension(GetBlobFileExtension(type));
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        int index{-1};
        index = mz_zip_reader_locate_file(&zip_ar, path.string().c_str(), nullptr, 0);
        if (index < 0) {
            LOG_WARNING(Render, "File {} is not found in the archive", path.string().c_str());
            return;
        }
        mz_zip_archive_file_stat stat{};
        mz_zip_reader_file_stat(&zip_ar, index, &stat);
        v.resize(stat.m_uncomp_size / sizeof(T));
        mz_zip_reader_extract_to_mem(&zip_ar, index, v.data(), stat.m_uncomp_size, 0);
    } else {
        const auto file = IOFile{path, FileAccessMode::Read};
        v.resize(file.GetSize() / sizeof(T));
        file.Read(v);
    }
}

bool DataBase::Save(BlobType type, const std::string& name, std::vector<u8>&& data) {
    if (!opened) {
        return false;
    }

    auto path = EmulatorSettings.IsPipelineCacheArchived() ? std::filesystem::path{name}
                                                           : cache_path / name;
    return WriteVector(type, std::move(path), std::move(data));
}

bool DataBase::Save(BlobType type, const std::string& name, std::vector<u32>&& data) {
    if (!opened) {
        return false;
    }

    auto path = EmulatorSettings.IsPipelineCacheArchived() ? std::filesystem::path{name}
                                                           : cache_path / name;
    return WriteVector(type, std::move(path), std::move(data));
}

void DataBase::Load(BlobType type, const std::string& name, std::vector<u8>& data) {
    if (!opened) {
        return;
    }

    auto path = EmulatorSettings.IsPipelineCacheArchived() ? std::filesystem::path{name}
                                                           : cache_path / name;
    return LoadVector(type, path, data);
}

void DataBase::Load(BlobType type, const std::string& name, std::vector<u32>& data) {
    if (!opened) {
        return;
    }

    auto path = EmulatorSettings.IsPipelineCacheArchived() ? std::filesystem::path{name}
                                                           : cache_path / name;
    return LoadVector(type, path, data);
}

bool DataBase::Exists(BlobType type, const std::string& name) const {
    if (!opened) {
        return false;
    }
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        std::filesystem::path path{name};
        path.replace_extension(GetBlobFileExtension(type));
        return mz_zip_reader_locate_file(&zip_ar, path.string().c_str(), nullptr, 0) >= 0;
    }
    auto path = cache_path / name;
    path.replace_extension(GetBlobFileExtension(type));
    std::error_code ec;
    return std::filesystem::exists(path, ec);
}

void DataBase::ForEachBlob(BlobType type, const std::function<void(std::vector<u8>&& data)>& func) {
    const auto& ext = GetBlobFileExtension(type);
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        const auto num_files = mz_zip_reader_get_num_files(&zip_ar);
        for (int index = 0; index < num_files; ++index) {
            std::array<char, MZ_ZIP_MAX_ARCHIVE_FILENAME_SIZE> file_name{};
            file_name.fill(0);
            mz_zip_reader_get_filename(&zip_ar, index, file_name.data(), file_name.size());
            if (std::string{file_name.data()}.ends_with(ext)) {
                mz_zip_archive_file_stat stat{};
                mz_zip_reader_file_stat(&zip_ar, index, &stat);
                std::vector<u8> data(stat.m_uncomp_size);
                mz_zip_reader_extract_to_mem(&zip_ar, index, data.data(), data.size(), 0);
                func(std::move(data));
            }
        }
    } else {
        for (const auto& file_name : std::filesystem::directory_iterator{cache_path}) {
            if (file_name.path().extension().string().ends_with(ext)) {
                using namespace Common::FS;
                const auto& file = IOFile{file_name, FileAccessMode::Read};
                if (file.IsOpen()) {
                    std::vector<u8> data(file.GetSize());
                    file.Read(data);
                    func(std::move(data));
                }
            }
        }
    }
}

void DataBase::FinishPreload() {
    if (EmulatorSettings.IsPipelineCacheArchived()) {
        mz_zip_writer_init_from_reader(&zip_ar, cache_path.string().c_str());
        ar_is_read_only = false;
    }
}

} // namespace Storage
