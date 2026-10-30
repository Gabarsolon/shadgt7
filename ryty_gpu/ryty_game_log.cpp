// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// GTS-009: GT Sport's own log. The game's native code logs through loggers whose output object is
// a null sink in the release build: its write method (GTSport.exe vtable 0x2971f18, slot 0x20) is
// a bare "ret" (RVA 0x1c62fc0). With RYGT_GAME_LOG=1 that slot points here instead, and every
// message goes to gtsport_game.log next to GTSport.exe. The slot is only changed when it still
// holds the original function (another game version or converter build is left alone).

#include <windows.h>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "common/types.h"

namespace {

// The System V x86-64 va_list the game passes (a pointer to this).
struct SysvVaList {
    u32 gp_offset;
    u32 fp_offset;
    u8* overflow_arg_area;
    u8* reg_save_area;
};

u64 NextInt(SysvVaList* list) {
    u64 value;
    if (list->gp_offset < 48) {
        std::memcpy(&value, list->reg_save_area + list->gp_offset, 8);
        list->gp_offset += 8;
    } else {
        std::memcpy(&value, list->overflow_arg_area, 8);
        list->overflow_arg_area += 8;
    }
    return value;
}

double NextDouble(SysvVaList* list) {
    double value;
    if (list->fp_offset < 176) {
        std::memcpy(&value, list->reg_save_area + list->fp_offset, 8);
        list->fp_offset += 16;
    } else {
        std::memcpy(&value, list->overflow_arg_area, 8);
        list->overflow_arg_area += 8;
    }
    return value;
}

bool Readable(const void* pointer) {
    if (pointer == nullptr) {
        return false;
    }
    MEMORY_BASIC_INFORMATION info{};
    return VirtualQuery(pointer, &info, sizeof(info)) == sizeof(info) && info.State == MEM_COMMIT &&
           (info.Protect & (PAGE_NOACCESS | PAGE_GUARD)) == 0;
}

// printf with the arguments taken from a guest va_list: each conversion is formatted on its own by
// the host's snprintf with the same flags, width, precision and type.
std::string Format(const char* format, SysvVaList* list) {
    std::string out;
    if (!Readable(format)) {
        return "<bad format>";
    }
    for (const char* p = format; *p;) {
        if (*p != '%') {
            out.push_back(*p++);
            continue;
        }
        const char* start = p++;
        if (*p == '%') {
            out.push_back('%');
            ++p;
            continue;
        }
        std::string spec = "%";
        while (*p && std::strchr("-+ #0", *p)) {
            spec.push_back(*p++);
        }
        auto width = [&] {
            if (*p == '*') {
                spec += std::to_string(static_cast<int>(NextInt(list)));
                ++p;
            } else {
                while (*p >= '0' && *p <= '9') {
                    spec.push_back(*p++);
                }
            }
        };
        width();
        if (*p == '.') {
            spec.push_back(*p++);
            width();
        }
        int longs = 0;
        bool size_t_like = false, half = false;
        while (*p && std::strchr("hlLqjzt", *p)) {
            if (*p == 'l' || *p == 'q' || *p == 'L') {
                ++longs;
            } else if (*p == 'j' || *p == 'z' || *p == 't') {
                size_t_like = true;
            } else if (*p == 'h') {
                half = true;
            }
            ++p;
        }
        const char conversion = *p ? *p++ : '\0';
        char buffer[512];
        switch (conversion) {
        case 'd':
        case 'i': {
            const u64 raw = NextInt(list);
            const long long value = (longs || size_t_like) ? static_cast<long long>(raw)
                                                            : static_cast<long long>(static_cast<s32>(raw));
            std::snprintf(buffer, sizeof(buffer), (spec + "lld").c_str(), half ? static_cast<long long>(static_cast<short>(value)) : value);
            out += buffer;
            break;
        }
        case 'u':
        case 'x':
        case 'X':
        case 'o': {
            const u64 raw = NextInt(list);
            const unsigned long long value = (longs || size_t_like) ? raw : static_cast<u32>(raw);
            std::snprintf(buffer, sizeof(buffer), (spec + "ll" + conversion).c_str(), value);
            out += buffer;
            break;
        }
        case 'c':
            out.push_back(static_cast<char>(NextInt(list)));
            break;
        case 'p':
            std::snprintf(buffer, sizeof(buffer), "%#llx", static_cast<unsigned long long>(NextInt(list)));
            out += buffer;
            break;
        case 's': {
            const auto* text = reinterpret_cast<const char*>(NextInt(list));
            if (!Readable(text)) {
                out += "(null)";
                break;
            }
            std::snprintf(buffer, sizeof(buffer), (spec + "s").c_str(), text);
            out += buffer;
            break;
        }
        case 'f':
        case 'F':
        case 'e':
        case 'E':
        case 'g':
        case 'G':
        case 'a':
        case 'A':
            std::snprintf(buffer, sizeof(buffer), (spec + conversion).c_str(), NextDouble(list));
            out += buffer;
            break;
        default:
            out.append(start, p);
            break;
        }
    }
    return out;
}

std::mutex g_lock;
std::FILE* g_file = nullptr;

void PS4_SYSV_ABI WriteMessage(void* sink, const void* tags, const char* file, s32 line, s32 level,
                               s32 flags, const char* format, SysvVaList* args) {
    (void)sink;
    (void)tags;
    (void)flags;
    const std::string message = Format(format, args);
    const char* name = Readable(file) ? file : "?";
    if (const char* slash = std::strrchr(name, '\\')) {
        name = slash + 1;
    }
    std::scoped_lock lock{g_lock};
    if (g_file != nullptr) {
        std::fprintf(g_file, "[%2d] %s:%d %s\n", level, name, line, message.c_str());
        std::fflush(g_file);
    }
}

} // namespace

namespace Ryty {

void InstallGameLog() {
    const char* enabled = std::getenv("RYGT_GAME_LOG");
    if (enabled == nullptr || *enabled == '\0' || *enabled == '0') {
        return;
    }
    auto* base = reinterpret_cast<u8*>(GetModuleHandleA(nullptr));
    constexpr u64 VtableRva = 0x2971f18, WriteSlot = 0x20, NullWriteRva = 0x1c62fc0;
    auto* slot = reinterpret_cast<void**>(base + VtableRva + WriteSlot);
    if (!Readable(slot) || *slot != base + NullWriteRva) {
        std::fprintf(stderr, "[patch] GTS-009 game log: the log sink is not the expected one, not applied\n");
        return;
    }
    char path[MAX_PATH] = {};
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    std::string log_path = path;
    log_path = log_path.substr(0, log_path.find_last_of('\\') + 1) + "gtsport_game.log";
    g_file = std::fopen(log_path.c_str(), "w");
    if (g_file == nullptr) {
        return;
    }
    DWORD old_protect = 0;
    if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old_protect)) {
        return;
    }
    *slot = reinterpret_cast<void*>(&WriteMessage);
    VirtualProtect(slot, sizeof(void*), old_protect, &old_protect);
    std::fprintf(stderr, "[patch] GTS-009 game log -> %s\n", log_path.c_str());
}

} // namespace Ryty
