// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

namespace Gt7HdrStatus {

using Query = int(PS4_SYSV_ABI*)(void*);
std::atomic_bool g_available{false};
Query g_streaming = nullptr;
Query g_share_play = nullptr;

template <bool Streaming>
int PS4_SYSV_ABI QueryStatus(void* output) {
    const bool available = g_available.load(std::memory_order_relaxed);
    if (available && output) {
        const u32 inactive = 0;
        if constexpr (Streaming) {
            std::memcpy(static_cast<u8*>(output) + 4, &inactive, 1);
        } else {
            std::memcpy(output, &inactive, sizeof(inactive));
        }
    }
    const int result = (Streaming ? g_streaming : g_share_play)(output);
    if (available) {
        static std::atomic_uint reports{0};
        if (reports.fetch_add(1, std::memory_order_relaxed) < 4) {
            u32 state = 0;
            if (output) {
                if constexpr (Streaming) {
                    std::memcpy(&state, static_cast<u8*>(output) + 4, 1);
                } else {
                    std::memcpy(&state, output, sizeof(state));
                }
            }
            std::fprintf(stderr, "HDRSTATUS: %s result=%d state=%u field-initialized=1\n",
                         Streaming ? "streaming" : "share-play", result, state);
        }
    }
    return result;
}

void Install(u8* base) {
    const auto settings = nlohmann::json::parse(Common::RytyGameConfigText(), nullptr, false);
    if (!settings.is_object() || !settings.contains("GPU") || !settings["GPU"].is_object()) {
        return;
    }
    const auto& gpu = settings["GPU"];
    const auto allowed = gpu.find("hdr_allowed");
    if (allowed == gpu.end() || !allowed->is_boolean() || !allowed->get<bool>()) {
        return;
    }
    PSF sfo;
    const auto path = std::filesystem::path(Common::RytyExeDirW()) / "app0" / "sce_sys" / "param.sfo";
    // FIX-GT7-146: the title only. APP_VER does not name the build (a 1.71 game installed over
    // its 1.00 base says 01.00, the same game dumped merged says 01.71): the bytes checked
    // below do.
    if (!sfo.Open(path) || sfo.GetString("TITLE_ID") != "CUSA24769") {
        std::fprintf(stderr, "HDRSTATUS: title not CUSA24769; not applied\n");
        return;
    }
    constexpr u64 start = 0x1eed374;
    const std::vector<u8> original{
        0x48, 0x8d, 0x7d, 0x80, 0x88, 0x05, 0x72, 0xd2, 0x50, 0x04,
        0xe8, 0x9d, 0xb3, 0xe7, 0x01, 0x85, 0xc0, 0x75, 0x0a, 0x80,
        0x7d, 0x84, 0x00, 0x0f, 0x85, 0x0d, 0x04, 0x00, 0x00, 0x4c,
        0x8d, 0x7d, 0x80, 0x4c, 0x89, 0xff, 0xe8, 0x93, 0xb3, 0xe7,
        0x01, 0x41, 0x8b, 0x0f, 0x83, 0xf1, 0x02, 0x09, 0xc1, 0x41,
        0x0f, 0x94, 0xc5};
    constexpr std::array<u8, 6> streaming_thunk{0xff, 0x25, 0x52, 0x83, 0x81, 0x01};
    constexpr std::array<u8, 6> share_play_thunk{0xff, 0x25, 0x4a, 0x83, 0x81, 0x01};
    if (!ReadableRange(base + start, original.size()) ||
        std::memcmp(base + start, original.data(), original.size()) != 0 ||
        !ReadableRange(base + 0x3d68720, 0x16) ||
        std::memcmp(base + 0x3d68720, streaming_thunk.data(), streaming_thunk.size()) != 0 ||
        std::memcmp(base + 0x3d68730, share_play_thunk.data(), share_play_thunk.size()) != 0) {
        std::fprintf(stderr, "HDRSTATUS: query shape/import thunks differ; not applied\n");
        return;
    }
    u8* stub = AllocNear(base);
    if (!stub) {
        std::fprintf(stderr, "HDRSTATUS: no nearby stub memory; not applied\n");
        return;
    }
    g_streaming = reinterpret_cast<Query>(base + 0x3d68720);
    g_share_play = reinterpret_cast<Query>(base + 0x3d68730);
    const std::array<void*, 2> hooks{reinterpret_cast<void*>(&QueryStatus<true>),
                                   reinterpret_cast<void*>(&QueryStatus<false>)};
    auto replacement = original;
    for (size_t i = 0; i < hooks.size(); ++i) {
        const u8 jump[]{0xff, 0x25, 0, 0, 0, 0};
        std::memcpy(stub + i * 16, jump, sizeof(jump));
        std::memcpy(stub + i * 16 + sizeof(jump), &hooks[i], sizeof(hooks[i]));
        const size_t offset = i == 0 ? 10 : 36;
        const auto distance = reinterpret_cast<intptr_t>(stub + i * 16) -
                              reinterpret_cast<intptr_t>(base + start + offset + 5);
        if (distance < INT32_MIN || distance > INT32_MAX) {
            std::fprintf(stderr, "HDRSTATUS: stub out of a call's reach; not applied\n");
            VirtualFree(stub, 0, MEM_RELEASE);
            return;
        }
        const s32 relative = static_cast<s32>(distance);
        std::memcpy(replacement.data() + offset + 1, &relative, sizeof(relative));
    }
    DWORD previous = 0;
    if (!VirtualProtect(stub, 0x1000, PAGE_EXECUTE_READ, &previous)) {
        std::fprintf(stderr, "HDRSTATUS: stub could not be made executable; not applied\n");
        VirtualFree(stub, 0, MEM_RELEASE);
        return;
    }
    FlushInstructionCache(GetCurrentProcess(), stub, 32);
    if (!Apply(base, Patch{start, original, replacement, "GT7 HDR status initialization"})) {
        VirtualFree(stub, 0, MEM_RELEASE);
        return;
    }
    std::fprintf(stderr, "HDRSTATUS: CUSA24769 hooks installed before guest start; "
                         "inactive until host/title HDR capability is confirmed\n");
}

} // namespace Gt7HdrStatus
