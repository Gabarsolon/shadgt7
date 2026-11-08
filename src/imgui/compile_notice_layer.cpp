// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <imgui.h>

#include "common/perf_monitor.h"
#include "common/types.h"
#include "imgui/compile_notice_layer.h"
#include "imgui/imgui_layer.h"
#include "video_core/renderer_vulkan/build_queue.h"
#include "video_core/renderer_vulkan/compile_pack.h"

namespace ImGui::CompileNotice {

namespace {

using Clock = std::chrono::steady_clock;

// FIX-GT7-100: a first GT7 race load (every shader new) showed one frame for 185 s while the
// draws into one-off targets waited for their pipelines: players took it for a crash. The present
// thread shows the last frame again while a layer asks for it (ShouldKeepDrawing), so the notice
// moves even when the game's own loading screen can't.
constexpr auto ShowAfter = std::chrono::milliseconds{500};

bool Enabled() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_COMPILE_NOTICE");
        return !(env && env[0] == '0');
    }();
    return on;
}

class CompileNoticeUI final : public ImGui::Layer {
public:
    void Draw() override;
    bool ShouldKeepDrawing() override {
        return Stalled(Clock::now());
    }

private:
    /// True while the game has shown no new frame for ShowAfter (after its first one).
    bool Stalled(Clock::time_point now) {
        if (!Enabled()) {
            return false;
        }
        const u64 frames = Common::GetGameFrameTotals().frames;
        const u64 seen = last_frames.load(std::memory_order_relaxed);
        if (frames != seen) {
            last_frames.store(frames, std::memory_order_relaxed);
            changed_ns.store(now.time_since_epoch().count(), std::memory_order_relaxed);
            return false;
        }
        const s64 since = changed_ns.load(std::memory_order_relaxed);
        return frames != 0 && since != 0 &&
               now.time_since_epoch().count() - since >
                   std::chrono::nanoseconds{ShowAfter}.count();
    }

    std::atomic<u64> last_frames{~u64{0}};
    std::atomic<s64> changed_ns{0};
};

CompileNoticeUI g_layer;

void CompileNoticeUI::Draw() {
    const auto now = Clock::now();
    // PERF-GT7-108: the compile pack's start-up compile shows its progress while it runs.
    const bool pack = Enabled() && Vulkan::CompilePack::g_running.load(std::memory_order_acquire);
    const bool stalled = Stalled(now);
    if (!stalled && !pack) {
        return;
    }
    const double stalled_s =
        std::chrono::duration<double>(now.time_since_epoch() -
                                      Clock::duration{changed_ns.load(std::memory_order_relaxed)})
            .count();
    const u32 jobs = Vulkan::g_build_jobs.load(std::memory_order_relaxed);
    char text[96];
    if (!stalled) {
        std::snprintf(text, sizeof(text), "Compiling shader pack  %u / %u",
                      Vulkan::CompilePack::g_done.load(std::memory_order_relaxed),
                      Vulkan::CompilePack::g_total.load(std::memory_order_relaxed));
    } else if (jobs != 0) {
        std::snprintf(text, sizeof(text), "Compiling shaders (%u)  %.0f s", jobs, stalled_s);
    } else {
        std::snprintf(text, sizeof(text), "Loading  %.0f s", stalled_s);
    }

    const auto& io = ImGui::GetIO();
    const float scale = std::max(1.0f, io.DisplaySize.y / 1080.0f);
    const float margin = 24.0f * scale;
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - margin, io.DisplaySize.y - margin),
                            ImGuiCond_Always, ImVec2(1.0f, 1.0f));
    ImGui::SetNextWindowBgAlpha(0.6f);
    const ImGuiWindowFlags flags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoInputs |
                                   ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoFocusOnAppearing |
                                   ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoMove |
                                   ImGuiWindowFlags_AlwaysAutoResize;
    if (ImGui::Begin("##compile_notice", nullptr, flags)) {
        ImGui::SetWindowFontScale(scale);
        // A spinner: a quarter circle turning once a second.
        const float radius = ImGui::GetTextLineHeight() * 0.45f;
        const ImVec2 pos = ImGui::GetCursorScreenPos();
        const ImVec2 center(pos.x + radius, pos.y + ImGui::GetTextLineHeight() * 0.5f);
        const float turn = static_cast<float>(std::fmod(ImGui::GetTime(), 1.0)) * 6.2831853f;
        auto* draw_list = ImGui::GetWindowDrawList();
        draw_list->PathArcTo(center, radius, turn, turn + 4.7f, 16);
        draw_list->PathStroke(IM_COL32(255, 255, 255, 230), 0, radius * 0.35f);
        ImGui::Dummy(ImVec2(radius * 2.0f, ImGui::GetTextLineHeight()));
        ImGui::SameLine(0.0f, radius);
        ImGui::TextUnformatted(text);
    }
    ImGui::End();
}

} // namespace

void Register() {
    ImGui::Layer::AddLayer(&g_layer);
}

void Unregister() {
    ImGui::Layer::RemoveLayer(&g_layer);
}

} // namespace ImGui::CompileNotice
