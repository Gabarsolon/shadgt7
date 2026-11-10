// SPDX-License-Identifier: GPL-2.0-or-later
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <nlohmann/json.hpp>
#include <fmt/format.h>
#include "common/ryty_config.h"
#include "common/types.h"
#include "core/memory.h"
#include "ryty_gpu/native/entry_hook.h"
#include "ryty_gpu/ryty_gt7_dlss_camera.h"
#include "common/hd_ui_options.h"

namespace Vulkan::Gt7Temporal {
const Settings& Config() {
    static const Settings settings = [] {
        Settings s;
        auto gpu = nlohmann::json::object();
        try {
            const auto j = nlohmann::json::parse(Common::RytyGameConfigText());
            if (j.contains("GPU") && j["GPU"].is_object())
                gpu = j["GPU"];
            s.enabled = gpu.value("temporal_dlss", false);
        } catch (...) {
        }
        const auto flag = [](const char* name, bool value) {
            const char* e = std::getenv(name);
            return e && std::strcmp(e, "1") == 0   ? true
                   : e && std::strcmp(e, "0") == 0 ? false
                                                   : value;
        };
        s.enabled = flag("RYGT_TEMPORAL_DLSS", s.enabled);
        // The scene is shown through the HD UI proxy, which draws the UI at the output size;
        // without GPU.hd_ui there is nowhere to put the UI over the reconstructed scene.
        s.enabled = s.enabled && Common::HdUiRequested();
        const auto number = [&](const char* key, const char* env, float fallback, float limit) {
            float v = fallback;
            try {
                v = gpu.value(key, fallback);
            } catch (...) {
            }
            if (const char* e = std::getenv(env)) {
                char* end{};
                const float parsed = std::strtof(e, &end);
                v = end != e && *end == 0 ? parsed : fallback;
            }
            return std::isfinite(v) && std::abs(v) <= limit ? v : fallback;
        };
        s.jitter_sign_x = number("tdlss_jitter_sign_x", "RYGT_TDLSS_JITTER_SIGN_X", -1, 1);
        s.jitter_sign_y = number("tdlss_jitter_sign_y", "RYGT_TDLSS_JITTER_SIGN_Y", -1, 1);
        s.mvec_scale = number("tdlss_mvec_scale", "RYGT_TDLSS_MVEC_SCALE", 1, 4);
        s.motion_jitter = number("tdlss_motion_jitter", "RYGT_TDLSS_MOTION_JITTER", 1, 2);
        return s;
    }();
    return settings;
}
namespace {
using Clock = std::chrono::steady_clock;
struct Record {
    Camera camera;
    Clock::time_point time;
};
std::mutex camera_mutex;
std::deque<Record> records;
uint64_t sequence{};
using RenderFn = void(PS4_SYSV_ABI*)(void*, void*);
using MatrixFn = void*(PS4_SYSV_ABI*)(float*, void*);
RenderFn original_render{};
template <class T>
bool Read(uint64_t at, T& value) {
    return at && Core::RytyMemory::GuardedCopy(&value, reinterpret_cast<void*>(at), sizeof(value));
}
bool Method(uint64_t object, unsigned slot, MatrixFn& fn) {
    uint64_t table{}, address{};
    const uint64_t base = reinterpret_cast<uint64_t>(GetModuleHandleW(nullptr));
    if (!Read(object, table) || !Read(table + slot, address))
        return false;
    // Only game text; never follow an arbitrary writable pointer into a host DLL.
    if (address < base + 0x10000 || address >= base + 0x4000000)
        return false;
    fn = reinterpret_cast<MatrixFn>(address);
    return true;
}
void PS4_SYSV_ABI RenderHook(void* self, void* context) {
    Camera c;
    uint64_t holder{}, graphic{}, camera{};
    MatrixFn projection{}, view{};
    bool accepted = false;
    if (Read(reinterpret_cast<uint64_t>(context) + 0x60, holder) && Read(holder + 8, graphic) &&
        Read(graphic + 0x2c8, camera) && Read(graphic + 0x2e0, c.normalized_jitter) &&
        Method(camera, 0x10, projection) && Method(camera, 0x18, view)) {
        // Original camera, synchronously on the game's setup thread. No late guest virtual calls.
        alignas(32) Matrix p{}, v{};
        projection(p.data(), reinterpret_cast<void*>(camera));
        view(v.data(), reinterpret_cast<void*>(camera));
        Matrix inverse{};
        if (Perspective(p, c.lens) && Invert(v, inverse) && std::isfinite(c.normalized_jitter[0]) &&
            std::isfinite(c.normalized_jitter[1])) {
            accepted = true;
            c.projection = p;
            c.view = v;
            c.identity = reinterpret_cast<uint64_t>(self) ^ camera;
            std::scoped_lock lock{camera_mutex};
            c.sequence = ++sequence;
            records.push_back({c, Clock::now()});
            if (c.sequence <= 8)
                std::fprintf(
                    stderr,
                    "GT7-DLSS camera sequence=%llu view=%llx normalizedJitter=(%g,%g) near=%g "
                    "far=%g fov=%g aspect=%g inverted=%d UNVERIFIED-runtime-contract\n",
                    static_cast<unsigned long long>(c.sequence),
                    static_cast<unsigned long long>(c.identity), c.normalized_jitter[0],
                    c.normalized_jitter[1], c.lens.near_plane, c.lens.far_plane, c.lens.fov,
                    c.lens.aspect, c.lens.inverted);
            while (records.size() > 128)
                records.pop_front();
        }
    }
    static std::atomic<unsigned> rejected{};
    if (!accepted && rejected.fetch_add(1, std::memory_order_relaxed) < 8) {
        std::fprintf(stderr,
                     "GT7-TDLSS camera snapshot rejected context=%p graphic=%llx camera=%llx; "
                     "missing interface or unsupported perspective/view; fallback\n",
                     context, static_cast<unsigned long long>(graphic),
                     static_cast<unsigned long long>(camera));
    }
    original_render(self, context);
}
} // namespace
void InstallCameraHook() {
    if (!Config().enabled)
        return;
    auto* base = reinterpret_cast<u8*>(GetModuleHandleW(nullptr));
    uint64_t slot{};
    // Converted VAs from REPORT: vtable 145462aa0, render 1427e9950.
    if (!Read(reinterpret_cast<uint64_t>(base) + 0x5462aa0, slot) ||
        slot != reinterpret_cast<uint64_t>(base) + 0x27e9950) {
        std::fprintf(stderr, "GT7-TDLSS camera hook unavailable: vtable guard; fallback\n");
        return;
    }
    static Native::NearCode code{base};
    constexpr std::array<u8, 8> prologue{0x55, 0x48, 0x89, 0xe5, 0x41, 0x57, 0x41, 0x56};
    original_render = reinterpret_cast<RenderFn>(
        Native::HookEntry(code, base, 0x27d9950, prologue, reinterpret_cast<void*>(&RenderHook),
                          "GT7-TDLSS TAA camera snapshot"));
}
std::optional<Camera> MatchCamera(const std::array<float, 29>& c, uint64_t after) {
    std::scoped_lock lock{camera_mutex};
    const auto now = Clock::now();
    while (!records.empty() && now - records.front().time > std::chrono::seconds(2))
        records.pop_front();
    std::optional<Camera> found;
    for (const auto& r : records) {
        const auto& camera = r.camera;
        if (camera.sequence <= after || now - r.time > std::chrono::milliseconds(250))
            continue;
        // CPU uploads twice the normalized correction; use both axes and aspect.
        if (std::abs(camera.normalized_jitter[0] * 2 - c[20]) > 1e-6f ||
            std::abs(camera.normalized_jitter[1] * 2 - c[21]) > 1e-6f ||
            std::abs(camera.lens.aspect - c[16] / c[17]) > .02f)
            continue;
        if (found)
            return std::nullopt; // multiple views/frames with this phase are ambiguous
        found = camera;
    }
    static unsigned missing = 0;
    if (!found && missing++ < 12)
        std::fprintf(stderr,
                     "GT7-TDLSS camera-match missing C20=%g C21=%g after=%llu records=%zu "
                     "expectedCPUJitter=(%g,%g)\n",
                     c[20], c[21], static_cast<unsigned long long>(after), records.size(),
                     c[20] * .5f, c[21] * .5f);
    return found;
}
} // namespace Vulkan::Gt7Temporal
