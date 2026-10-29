// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstdio>
#include <optional>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace Common {

#ifdef _WIN32
/// FIX-230: the folder of the running exe, with its trailing backslash, as UTF-16. Unicode and
/// long-path safe (GetModuleFileNameA turned characters outside the ANSI code page into '?' and
/// failed past 260 characters, so the launcher's settings were silently not read).
inline const std::wstring& RytyExeDirW() {
    static const std::wstring dir = [] {
        std::wstring path(512, L'\0');
        for (;;) {
            const DWORD length =
                GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
            if (length == 0) {
                return std::wstring{};
            }
            if (length < path.size()) {
                path.resize(length);
                break;
            }
            path.resize(path.size() * 2); // truncated: try a bigger buffer
        }
        return path.substr(0, path.find_last_of(L"\\/") + 1);
    }();
    return dir;
}

/// FIX-230: a whole file's bytes (empty when it cannot be read), by a UTF-16 path.
inline std::string ReadFileW(const std::wstring& path) {
    std::string result;
    if (std::FILE* file = _wfopen(path.c_str(), L"rb")) {
        char chunk[4096];
        while (const size_t n = std::fread(chunk, 1, sizeof(chunk), file)) {
            result.append(chunk, n);
        }
        std::fclose(file);
    }
    return result;
}

/// FIX-230: user\custom_configs\CUSA03220.json next to the exe.
inline std::wstring RytyGameConfigPathW() {
    const std::wstring& dir = RytyExeDirW();
    return dir.empty() ? std::wstring{} : dir + L"user\\custom_configs\\CUSA03220.json";
}

/// FIX-230: the launcher's settings as they are now (not cached).
inline std::string RytyReadGameConfig() {
    const std::wstring path = RytyGameConfigPathW();
    return path.empty() ? std::string{} : ReadFileW(path);
}
#endif

/// The launcher's game settings (user\custom_configs\CUSA03220.json next to the exe), read once.
inline const std::string& RytyGameConfigText() {
    static const std::string text = [] {
        std::string result;
#ifdef _WIN32
        result = RytyReadGameConfig(); // FIX-230
#endif
        return result;
    }();
    return text;
}

/// The raw value after "key": in the launcher's settings (true, false, a number), if present.
inline std::optional<std::string> RytyGameConfigValue(const char* key) {
    const std::string& text = RytyGameConfigText();
    const std::string quoted = std::string{"\""} + key + "\"";
    const size_t at = text.find(quoted);
    if (at == std::string::npos) {
        return std::nullopt;
    }
    const size_t begin = text.find_first_not_of(" \t\r\n:", at + quoted.size());
    if (begin == std::string::npos) {
        return std::nullopt;
    }
    const size_t end = text.find_first_of(",}\r\n", begin);
    return text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
}

} // namespace Common
