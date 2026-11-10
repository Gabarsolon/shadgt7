// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

// GPU.hd_ui, the pavilion icons. The game rasterizes these SVG images on the CPU at a size from
// its 1080p display mode, then the UI scales them up. Here the game's own rasterizer makes them
// at the window's density instead (no image is enlarged or filtered), and the width and height
// the game reads from them stay the original, so the UI layout does not change. Included in
// ryty_game_patch.cpp's anonymous namespace, so that the build system does not change.

// The path of the image a mImagePS4 holds (CUSA24769 v1.71), or "" when it cannot be read. The
// path is a std::string at self+0x20: its size is at +0x18 and its capacity at +0x20 inside it,
// its inline or heap data at +8. Only plain local asset names are returned, never an URL.
std::string HdUiImageName(u64 self) {
    u64 size{};
    u64 capacity{};
    u64 address = self + 0x28;
    if (!self || self > UINT64_MAX - 0x48 ||
        !Core::RytyMemory::GuardedCopy(&size, reinterpret_cast<void*>(self + 0x38), 8) ||
        !Core::RytyMemory::GuardedCopy(&capacity, reinterpret_cast<void*>(self + 0x40), 8) ||
        !size || size > 384 || capacity < size || capacity > 0x100000) {
        return {};
    }
    if (capacity >= 16 &&
        !Core::RytyMemory::GuardedCopy(&address, reinterpret_cast<void*>(self + 0x28), 8)) {
        return {};
    }
    std::string name(size, '\0');
    if (!address ||
        !Core::RytyMemory::GuardedCopy(name.data(), reinterpret_cast<void*>(address), size)) {
        return {};
    }
    for (const unsigned char c : name) {
        if (c < 32 || c > 126) {
            return {};
        }
    }
    if (name.find("://") != name.npos) {
        return {};
    }
    return name;
}

namespace PavilionCache {
using DecodeFn = u64(PS4_SYSV_ABI*)(u64, u64, u64, u64, u64, u64);
using SizeFn = u32(PS4_SYSV_ABI*)(u64);
DecodeFn original_decode{};
SizeFn original_width{}, original_height{};
bool active{};
struct Record {
    u64 resource{};
    u32 logical[2]{}, physical[2]{};
};
std::array<Record, 128> records{};
std::array<u16, 256> lookup{}; // slot+1; rebuilt at decode, never allocate in an intrinsic getter
size_t Bucket(u64 p) {
    return size_t(((p >> 4) ^ (p >> 20) ^ (p >> 36)) & 255);
}
void Reindex() noexcept {
    lookup.fill(0);
    for (u32 i = 0; i < records.size(); ++i)
        if (records[i].resource) {
            size_t h = Bucket(records[i].resource);
            while (lookup[h])
                h = (h + 1) & 255;
            lookup[h] = u16(i + 1);
        }
}
std::mutex mutex;
u64 spent{}; // lifetime packed-base-byte budget, never refunded on reuse
std::atomic<u32> messages{};
struct Context {
    u64 self{};
    std::string name;
    int slot = -1;
    Record dimensions;
    Context* previous{};
};
thread_local Context* current{};

void Report(const std::string& text) {
    if (messages.fetch_add(1, std::memory_order_relaxed) < 64)
        HdUiReport(text);
}
// Called at the original SVG-only size conversion. The other four inputs to
// mImagePS4::set_data (PNG/JPEG/etc.) never reach either patched SVG site.
void PS4_SYSV_ABI Size(u64 self, float* xy) noexcept {
    try {
        auto* ctx = current;
        if (!active || !ctx || ctx->self != self || ctx->slot >= 0)
            return;
        const float scale = Ryty::UiCache::output_scale.load(std::memory_order_relaxed);
        if (!(scale > 1.f)) {
            Report(fmt::format(
                "svg-cache unchanged name='{}' reason=output-density-not-ready-or-native scale={}",
                ctx->name, scale));
            return;
        }
        if (!std::isfinite(xy[0]) || !std::isfinite(xy[1]) || xy[0] < 8 || xy[1] < 8 ||
            xy[0] > 256 || xy[1] > 256)
            return;
        Record r;
        for (u32 i = 0; i < 2; ++i) {
            r.logical[i] = u32(std::ceil(xy[i]));
            r.physical[i] = u32(std::ceil(xy[i] * scale));
        }
        const u64 bytes = u64(r.physical[0]) * r.physical[1] * 4;
        {
            std::scoped_lock lock{mutex};
            if (spent + bytes > 8 * 1024 * 1024)
                return;
            for (u32 i = 0; i < records.size(); ++i)
                if (!records[i].resource && !records[i].physical[0]) {
                    ctx->slot = int(i);
                    records[i] = r;
                    spent += bytes;
                    break;
                }
        }
        if (ctx->slot < 0)
            return;
        ctx->dimensions = r;
        xy[0] *= scale;
        xy[1] *= scale;
        Report(fmt::format("svg-cache name='{}' logical={}x{} raster={}x{} density={} "
                           "source=guest-SVG-rasterizer layout=original-intrinsic-size "
                           "baseBytes={} budget=8MiB-lifetime",
                           ctx->name, r.logical[0], r.logical[1], r.physical[0], r.physical[1],
                           scale, bytes));
    } catch (...) {
    } // metadata is committed before xy changes; a failed log cannot undo it
}
// The SVG resource has just been allocated; register before the guest publishes
// it at self+d0. Thus another guest thread never sees new pixels with new logical size.
void PS4_SYSV_ABI Resource(u64 allocation, float*) noexcept {
    try {
        if (!active || allocation < 0x10000)
            return;
        const u64 resource = allocation + 0x10;
        std::scoped_lock lock{mutex};
        // This site runs for every SVG, including unselected/unscaled reloads.
        for (auto& r : records)
            if (r.resource == resource)
                r = {};
        if (auto* ctx = current; ctx && ctx->slot >= 0) {
            auto& r = records[ctx->slot];
            r = ctx->dimensions;
            r.resource = resource;
        }
        Reindex();
    } catch (...) {
    }
}
u64 PS4_SYSV_ABI Decode(u64 self, u64 a1, u64 a2, u64 a3, u64 a4, u64 a5) {
    Context ctx;
    if (active)
        try {
            auto name = HdUiImageName(self);
            if (name.ends_with(".svg") && name.find("/icon/pavilion/") != name.npos) {
                ctx.self = self;
                ctx.name = std::move(name);
            }
        } catch (...) {
        }
    ctx.previous = current;
    current = ctx.self ? &ctx : nullptr;
    struct Restore {
        Context* old;
        ~Restore() {
            current = old;
        }
    } restore{ctx.previous};
    const u64 result = original_decode(self, a1, a2, a3, a4, a5);
    if (ctx.slot >= 0)
        try {
            u64 resource{};
            const u32 w = original_width(self), h = original_height(self);
            Core::RytyMemory::GuardedCopy(&resource, reinterpret_cast<void*>(self + 0xd0), 8);
            Report(fmt::format(
                "svg-ready name='{}' resource={:#x} physical={}x{} expected={}x{} "
                "logical={}x{} nativeResult={} (complete mip/driver allocation unmeasured)",
                ctx.name, resource, w, h, ctx.dimensions.physical[0], ctx.dimensions.physical[1],
                ctx.dimensions.logical[0], ctx.dimensions.logical[1], u32(u8(result))));
            std::scoped_lock lock{mutex};
            auto& r = records[ctx.slot];
            if (!r.resource || r.resource != resource || r.physical[0] != w || r.physical[1] != h) {
                // Keep a registered resource if it was shared/published, but don't retain a
                // never-registered reserved slot. Getter checks also require the physical size.
                if (!r.resource)
                    r = {};
            }
        } catch (...) {
        }
    return result;
}
u32 Logical(u64 self, u32 physical, u32 dimension) noexcept {
    try {
        u64 resource{}, table{};
        if (!active ||
            !Core::RytyMemory::GuardedCopy(&resource, reinterpret_cast<void*>(self + 0xd0), 8) ||
            !resource)
            return physical;
        std::scoped_lock lock{mutex};
        for (size_t h = Bucket(resource); lookup[h]; h = (h + 1) & 255) {
            const auto& r = records[lookup[h] - 1];
            if (r.resource == resource) {
                if (r.physical[dimension] == physical &&
                    Core::RytyMemory::GuardedCopy(&table, reinterpret_cast<void*>(resource), 8) &&
                    table == reinterpret_cast<u64>(GetModuleHandleW(nullptr)) + 0x53592e8)
                    return r.logical[dimension];
                return physical;
            }
        }
    } catch (...) {
    }
    return physical;
}
u32 PS4_SYSV_ABI Width(u64 self) {
    return Logical(self, original_width(self), 0);
}
u32 PS4_SYSV_ABI Height(u64 self) {
    return Logical(self, original_height(self), 1);
}

// Both sites have RSP 16-aligned. Save flags, all nine volatile GPRs and all
// sixteen full YMM registers, plus MXCSR, before the SysV callback. Only Size
// changes saved XMM0.xy. No guest arguments, callee-saved registers or labels change.
std::vector<u8> Stub(u64 callback, bool allocation, u64 resume) {
    std::vector<u8> c;
    const auto emit = [&](std::initializer_list<u8> b) { c.insert(c.end(), b); };
    const auto dword = [&](u32 x) {
        for (int i = 0; i < 4; ++i)
            c.push_back(u8(x >> (8 * i)));
    };
    const auto qword = [&](u64 x) {
        for (int i = 0; i < 8; ++i)
            c.push_back(u8(x >> (8 * i)));
    };
    emit({0x9c, 0x50, 0x51, 0x52, 0x56, 0x57, 0x41, 0x50, 0x41, 0x51, 0x41, 0x52, 0x41, 0x53});
    emit({0x48, 0x81, 0xec});
    dword(0x220);
    for (u32 i = 0; i < 16; ++i) {
        emit({0xc5, u8(i < 8 ? 0xfe : 0x7e), 0x7f, u8(0x84 | ((i & 7) << 3)), 0x24});
        dword(i * 32);
    }
    emit({0x0f, 0xae, 0x9c, 0x24});
    dword(0x200); // stmxcsr
    if (allocation) {
        emit({0x48, 0x8b, 0xbc, 0x24});
        dword(0x260);
    } // original RAX
    else
        emit({0x48, 0x8b, 0xbd, 0x10, 0xfe, 0xff, 0xff}); // self saved at rbp-1f0
    emit({0x48, 0x89, 0xe6, 0xc5, 0xf8, 0x77, 0x49, 0xbb});
    qword(callback);
    emit({0x41, 0xff, 0xd3});
    emit({0x0f, 0xae, 0x94, 0x24});
    dword(0x200); // ldmxcsr
    for (u32 i = 0; i < 16; ++i) {
        emit({0xc5, u8(i < 8 ? 0xfe : 0x7e), 0x6f, u8(0x84 | ((i & 7) << 3)), 0x24});
        dword(i * 32);
    }
    emit({0x48, 0x81, 0xc4});
    dword(0x220);
    emit({0x41, 0x5b, 0x41, 0x5a, 0x41, 0x59, 0x41, 0x58, 0x5f, 0x5e, 0x5a, 0x59, 0x58, 0x9d});
    emit({0xff, 0x25, 0, 0, 0, 0});
    qword(resume);
    return c;
}

void Install(u8* base) {
    if (!Common::HdUiRequested())
        return;
    if (Vulkan::RytyDisplayAspect() > 0) {
        Report(
            "svg-cache not installed: the UI at the window size is off with an ultrawide screen");
        return;
    }
    const std::array<u8, 8> prologue{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56};
    const std::array<u8, 20> frame{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56, 0x41, 0x55,
                                   0x41, 0x54, 0x53, 0x48, 0x81, 0xec, 0x28, 0x02, 0,    0};
    const std::array<u8, 7> getter{0x48, 0x8b, 0xbf, 0xd0, 0, 0, 0};
    const std::array<u8, 6> round{0xc4, 0xe3, 0x79, 0x08, 0xc0, 0x0a};
    const std::array<u8, 10> round_convert{0xc4, 0xe3, 0x79, 0x08, 0xc0,
                                           0x0a, 0xc5, 0xfa, 0x5b, 0xc0};
    const std::array<u8, 7> saved_self{0x4c, 0x89, 0xb5, 0x10, 0xfe, 0xff, 0xff};
    const std::array<u8, 13> allocate{0x49, 0x89, 0xc7, 0x48, 0xb9, 0x01, 0, 0, 0, 0x01, 0, 0, 0};
    for (auto [rva, bytes] :
         std::array<std::pair<u64, std::span<const u8>>, 6>{{{0x18eaae0, frame},
                                                             {0x17f03f0, getter},
                                                             {0x17f0410, getter},
                                                             {0x18eb24f, round_convert},
                                                             {0x18eb2f0, allocate},
                                                             {0x18eab3e, saved_self}}}) {
        if (!ReadableRange(base + rva, bytes.size()) ||
            std::memcmp(base + rva, bytes.data(), bytes.size())) {
            Report(fmt::format("svg-cache not installed: instruction mismatch RVA={:#x}", rva));
            return;
        }
    }
    static Native::NearCode near_code{base};
    auto* size_code = near_code.Alloc(512);
    auto* allocation_code = near_code.Alloc(512);
    if (!size_code || !allocation_code) {
        Report("svg-cache no near memory");
        return;
    }
    // Safe pass-throughs exist before either site is patched, including if a later
    // protection failure prevents rollback. Never leave a jump to an empty stub.
    const auto passthrough = [&](u8* code, u64 rva, std::span<const u8> bytes) {
        std::memcpy(code, bytes.data(), bytes.size());
        const std::array<u8, 6> jump{0xff, 0x25, 0, 0, 0, 0};
        std::memcpy(code + bytes.size(), jump.data(), jump.size());
        const u64 next = reinterpret_cast<u64>(base + rva + bytes.size());
        std::memcpy(code + bytes.size() + jump.size(), &next, 8);
        FlushInstructionCache(GetCurrentProcess(), code, bytes.size() + 14);
    };
    passthrough(size_code, 0x18eb24f, round);
    passthrough(allocation_code, 0x18eb2f0, allocate);
    // A moved-site trampoline is obtained first. Guest threads have not started.
    void* size_resume =
        Native::HookEntry(near_code, base, 0x18db24f, round, size_code, "HDUI SVG size");
    void* alloc_resume = Native::HookEntry(near_code, base, 0x18db2f0, allocate, allocation_code,
                                           "HDUI SVG resource");
    if (!size_resume || !alloc_resume) {
        // Preflight succeeded but a protection/allocation failure must never leave an
        // entry pointing at an empty stub. Restore either site that was installed.
        const auto restore = [&](u64 rva, std::span<const u8> bytes) {
            DWORD old{}, ignored{};
            if (VirtualProtect(base + rva, bytes.size(), PAGE_EXECUTE_READWRITE, &old)) {
                std::memcpy(base + rva, bytes.data(), bytes.size());
                VirtualProtect(base + rva, bytes.size(), old, &ignored);
                FlushInstructionCache(GetCurrentProcess(), base + rva, bytes.size());
            }
        };
        if (size_resume)
            restore(0x18eb24f, round);
        if (alloc_resume)
            restore(0x18eb2f0, allocate);
        Report("svg-cache not installed: hook allocation/protection failure");
        return;
    }
    const auto a = Stub(reinterpret_cast<u64>(&Size), false, reinterpret_cast<u64>(size_resume));
    const auto b =
        Stub(reinterpret_cast<u64>(&Resource), true, reinterpret_cast<u64>(alloc_resume));
    std::memcpy(size_code, a.data(), a.size());
    std::memcpy(allocation_code, b.data(), b.size());
    FlushInstructionCache(GetCurrentProcess(), size_code, a.size());
    FlushInstructionCache(GetCurrentProcess(), allocation_code, b.size());
    original_width = reinterpret_cast<SizeFn>(Native::HookEntry(
        near_code, base, 0x17e03f0, getter, reinterpret_cast<void*>(&Width), "HDUI logical width"));
    original_height = reinterpret_cast<SizeFn>(Native::HookEntry(near_code, base, 0x17e0410, getter,
                                                                 reinterpret_cast<void*>(&Height),
                                                                 "HDUI logical height"));
    original_decode = reinterpret_cast<DecodeFn>(
        Native::HookEntry(near_code, base, 0x18daae0, prologue, reinterpret_cast<void*>(&Decode),
                          "HDUI pavilion decode"));
    active = original_width && original_height && original_decode;
    Ryty::UiCache::enabled = active;
    Report(fmt::format(
        "svg-cache enabled={} key=GPU.hd_ui paths=/icon/pavilion/*.svg "
        "density=output/native-capped-2 scale-before-ceil logical-getters=compensated "
        "budget=8MiB-lifetime-base-128-records scene/Neo/TAA=unchanged; runtime-unverified",
        active));
}
} // namespace PavilionCache
