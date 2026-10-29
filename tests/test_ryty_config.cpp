// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// FIX-230: the launcher's settings are read by Unicode paths (a GT7 folder with characters outside
// the ANSI code page, or a path over 260 characters, used to lose them and boot without the
// native renderer).

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include <gtest/gtest.h>

#include "common/ryty_config.h"

#ifdef _WIN32
TEST(RytyConfig, ExeDirEndsWithSeparatorAndExists) {
    const std::wstring& dir = Common::RytyExeDirW();
    ASSERT_FALSE(dir.empty());
    EXPECT_EQ(dir.back(), L'\\');
    EXPECT_TRUE(std::filesystem::is_directory(std::filesystem::path{dir}));
}

TEST(RytyConfig, ReadsFileUnderNonAsciiFolder) {
    const auto base = std::filesystem::temp_directory_path() /
                      std::filesystem::path{L"ryty-t\u00ebst-\u00fc-\u0442\u0435\u0441\u0442"};
    std::filesystem::create_directories(base);
    const auto file = base / L"CUSA03220.json";
    {
        std::ofstream out{file, std::ios::binary};
        out << R"({"GPU":{"native_renderer":true}})";
    }
    const std::string text = Common::ReadFileW(file.wstring());
    EXPECT_NE(text.find("\"native_renderer\":true"), std::string::npos);
    std::filesystem::remove_all(base);
}

TEST(RytyConfig, ReadsFileUnderLongPath) {
    const auto root = std::filesystem::temp_directory_path() / L"ryty-long";
    auto base = root;
    for (int i = 0; i < 12; ++i) {
        base /= std::wstring(24, static_cast<wchar_t>(L'a' + i));
    }
    const std::wstring extended = L"\\\\?\\" + base.wstring();
    std::filesystem::create_directories(std::filesystem::path{extended});
    const std::wstring file = extended + L"\\CUSA03220.json";
    {
        std::ofstream out{std::filesystem::path{file}, std::ios::binary};
        out << "{}";
    }
    EXPECT_EQ(Common::ReadFileW(file), "{}");
    std::filesystem::remove_all(std::filesystem::path{L"\\\\?\\" + root.wstring()});
}

TEST(RytyConfig, MissingFileReadsEmpty) {
    EXPECT_TRUE(Common::ReadFileW(L"Z:\\no\\such\\folder\\CUSA03220.json").empty());
}
#endif
