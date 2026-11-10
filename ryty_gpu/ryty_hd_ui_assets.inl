// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// GPU.hd_ui, the game's 4K UI images. Included in ryty_game_patch.cpp's anonymous namespace, so
// that the build system does not change.

// A line of the HD UI patches on stderr, in the style of the other patches.
void HdUiReport(const std::string& text) {
    std::fprintf(stderr, "[patch] HDUI: %s\n", text.c_str());
    std::fflush(stderr);
}

// GT7 (CUSA24769 v1.71): the UI image flag ([0x1462fda71], high-resolution output and not the
// override at [0x1462ef608]) selects the game's 4K images. When false, mImagePS4 rewrites /4k/
// paths to /2k/. GPU.hd_ui forces the flag at both of its writers (its initial calculation and
// its display-change calculation, keeping the latter's comparison and notifications), and sets
// it now, so that no later writer has to run first. Neo mode, the render size, the layout size
// and TAA do not change.
void InstallHdUiAssetsGt7(u8* base) {
    if (!Common::HdUiRequested()) {
        return;
    }
    // Converter RVAs. Both writers are checked before anything is changed, so that another
    // executable cannot receive half of this patch.
    const Patch patches[] = {
        {0x181bc7c, {0x20, 0xd0, 0x88, 0x05, 0xed, 0x1d, 0xae, 0x04},
         {0xb0, 0x01, 0x88, 0x05, 0xed, 0x1d, 0xae, 0x04}, "GT7 HD UI images (1/2)"},
        {0x18f3751, {0x20, 0xca, 0x40, 0x38, 0xd6, 0x0f, 0x85, 0xc2, 0x02, 0, 0},
         {0xb2, 0x01, 0x40, 0x38, 0xd6, 0x0f, 0x85, 0xc2, 0x02, 0, 0},
         "GT7 HD UI images (2/2)"},
    };
    for (const auto& patch : patches) {
        if (!ReadableRange(base + patch.rva, patch.original.size()) ||
            (std::memcmp(base + patch.rva, patch.original.data(), patch.original.size()) &&
             std::memcmp(base + patch.rva, patch.replacement.data(),
                         patch.replacement.size()))) {
            HdUiReport(fmt::format("not applied: the bytes at {:#x} differ", patch.rva));
            return;
        }
    }
    constexpr u32 flag_rva = 0x62fda71;
    if (!ReadableRange(base + flag_rva, 1) || base[flag_rva] > 1) {
        HdUiReport("not applied: the UI image flag is unreadable or unrecognized");
        return;
    }
    const std::array<u64, 3> offsets{patches[0].rva, patches[1].rva, flag_rva};
    const std::array<size_t, 3> sizes{patches[0].original.size(), patches[1].original.size(), 1};
    std::array<DWORD, 3> protections{};
    size_t writable = 0;
    for (; writable < offsets.size(); ++writable) {
        if (!VirtualProtect(base + offsets[writable], sizes[writable], PAGE_EXECUTE_READWRITE,
                            &protections[writable])) {
            break;
        }
    }
    const bool all_writable = writable == offsets.size();
    if (all_writable) {
        for (const auto& patch : patches) {
            std::memcpy(base + patch.rva, patch.replacement.data(), patch.replacement.size());
            FlushInstructionCache(GetCurrentProcess(), base + patch.rva,
                                  patch.replacement.size());
        }
        base[flag_rva] = 1;
    }
    while (writable) {
        const size_t i = --writable;
        DWORD ignored{};
        VirtualProtect(base + offsets[i], sizes[i], protections[i], &ignored);
    }
    HdUiReport(all_writable ? "the game's 4K UI images are selected"
                            : "not applied: a span could not be made writable");
}
