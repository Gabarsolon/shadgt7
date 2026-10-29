// SPDX-FileCopyrightText: Copyright 2025-2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "common/assert.h"
#include "common/debug.h"
#include "common/guest_clock.h"
#include "common/thread.h"
#include "core/debug_state.h"
#include "core/emulator_settings.h"
#include "core/libraries/kernel/time.h"
#include "common/nvtx.h"
#include "core/libraries/videoout/driver.h"
#include "core/libraries/videoout/fps_lock.h"
#include "core/libraries/videoout/videoout_error.h"
#include "imgui/renderer/imgui_core.h"
#include "video_core/amdgpu/liverpool.h"
#include "video_core/renderer_vulkan/vk_presenter.h"
#include "ryty_gpu/ryty_diag.h"

extern std::unique_ptr<Vulkan::Presenter> presenter;
extern std::unique_ptr<AmdGpu::Liverpool> liverpool;

namespace Libraries::VideoOut {

constexpr static bool Is32BppPixelFormat(PixelFormat format) {
    switch (format) {
    case PixelFormat::A8R8G8B8Srgb:
    case PixelFormat::A8B8G8R8Srgb:
    case PixelFormat::A2R10G10B10:
    case PixelFormat::A2R10G10B10Srgb:
    case PixelFormat::A2R10G10B10Bt2020Pq:
        return true;
    default:
        return false;
    }
}

constexpr u32 PixelFormatBpp(PixelFormat pixel_format) {
    switch (pixel_format) {
    case PixelFormat::A16R16G16B16Float:
        return 8;
    default:
        return 4;
    }
}

VideoOutDriver::VideoOutDriver(u32 width, u32 height) {
    main_port.resolution.full_width = width;
    main_port.resolution.full_height = height;
    main_port.resolution.pane_width = width;
    main_port.resolution.pane_height = height;
    present_thread = std::jthread([&](std::stop_token token) { PresentThread(token); });
}

VideoOutDriver::~VideoOutDriver() = default;

int VideoOutDriver::Open(const ServiceThreadParams* params) {
    if (main_port.is_open) {
        return ORBIS_VIDEO_OUT_ERROR_RESOURCE_BUSY;
    }
    main_port.is_open = true;
    liverpool->SetVoPort(&main_port);
    return 1;
}

void VideoOutDriver::Close(s32 handle) {
    std::scoped_lock lock{mutex};

    // Mark as closed
    main_port.is_open = false;
    main_port.flip_rate = 0;
    main_port.prev_index = -1;

    // Clear port information
    std::memset(main_port.buffer_labels.data(), 0, sizeof(main_port.buffer_labels));
    std::memset(main_port.groups.data(), 0, sizeof(main_port.groups));
    std::memset(&main_port.vblank_status, 0, sizeof(main_port.vblank_status));
    main_port.flip_status = FlipStatus{};

    // Re-initialize buffers
    std::memset(main_port.buffer_slots.data(), 0, sizeof(main_port.buffer_slots));
    for (auto& buffer : main_port.buffer_slots) {
        buffer.group_index = -1;
    }

    // Clear events
    for (auto event : main_port.flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.flip_events.clear();
    for (auto event : main_port.vblank_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->RemoveEvent(static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                                Kernel::OrbisKernelEvent::Filter::VideoOut);
        }
    }
    main_port.vblank_events.clear();
}

VideoOutPort* VideoOutDriver::GetPort(int handle) {
    if (handle != 1) [[unlikely]] {
        return nullptr;
    }
    return &main_port;
}

int VideoOutDriver::RegisterBuffers(VideoOutPort* port, s32 startIndex, void* const* addresses,
                                    s32 bufferNum, const BufferAttribute* attribute) {
    const s32 group_index = port->FindFreeGroup();
    if (group_index >= MaxDisplayBufferGroups) {
        return ORBIS_VIDEO_OUT_ERROR_NO_EMPTY_SLOT;
    }

    if (startIndex + bufferNum > MaxDisplayBuffers || startIndex > MaxDisplayBuffers ||
        bufferNum > MaxDisplayBuffers) {
        LOG_ERROR(Lib_VideoOut,
                  "Attempted to register too many buffers startIndex = {}, bufferNum = {}",
                  startIndex, bufferNum);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    const s32 end_index = startIndex + bufferNum;
    if (bufferNum > 0 &&
        std::any_of(port->buffer_slots.begin() + startIndex, port->buffer_slots.begin() + end_index,
                    [](auto& buffer) { return buffer.group_index != -1; })) {
        return ORBIS_VIDEO_OUT_ERROR_SLOT_OCCUPIED;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "startIndex = {}, bufferNum = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             startIndex, bufferNum, GetPixelFormatString(attribute->pixel_format),
             attribute->aspect_ratio, static_cast<u32>(attribute->tiling_mode), attribute->width,
             attribute->height, attribute->pitch_in_pixel, attribute->option);

    auto& group = port->groups[group_index];
    std::memcpy(&group.attrib, attribute, sizeof(BufferAttribute));
    group.is_occupied = true;

    for (u32 i = 0; i < bufferNum; i++) {
        const uintptr_t address = reinterpret_cast<uintptr_t>(addresses[i]);
        port->buffer_slots[startIndex + i] = VideoOutBuffer{
            .group_index = group_index,
            .address_left = address,
            .address_right = 0,
        };

        // Reset flip label also when registering buffer
        port->buffer_labels[startIndex + i] = 0;
        port->SignalVoLabel();

        presenter->RegisterVideoOutSurface(group, address);
        LOG_INFO(Lib_VideoOut, "buffers[{}] = {:#x}", i + startIndex, address);
    }

    return group_index;
}

int VideoOutDriver::UnregisterBuffers(VideoOutPort* port, s32 attributeIndex) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    auto& group = port->groups[attributeIndex];
    group.is_occupied = false;

    for (auto& buffer : port->buffer_slots) {
        if (buffer.group_index != attributeIndex) {
            continue;
        }
        buffer.group_index = -1;
    }

    return ORBIS_OK;
}

int VideoOutDriver::ChangeBufferAttribute(VideoOutPort* port, s32 attributeIndex,
                                          const BufferAttribute* attribute) {
    if (attributeIndex >= MaxDisplayBufferGroups || !port->groups[attributeIndex].is_occupied) {
        LOG_ERROR(Lib_VideoOut, "Invalid attribute index {}", attributeIndex);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }

    if (attribute->reserved0 != 0 || attribute->reserved1 != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid reserved members");
        return ORBIS_VIDEO_OUT_ERROR_INVALID_VALUE;
    }
    if (attribute->aspect_ratio != 0) {
        LOG_ERROR(Lib_VideoOut, "Invalid aspect ratio = {}", attribute->aspect_ratio);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_ASPECT_RATIO;
    }
    if (attribute->width > attribute->pitch_in_pixel) {
        LOG_ERROR(Lib_VideoOut, "Buffer width {} is larger than pitch {}", attribute->width,
                  attribute->pitch_in_pixel);
        return ORBIS_VIDEO_OUT_ERROR_INVALID_PITCH;
    }
    if (attribute->tiling_mode < TilingMode::Tile || attribute->tiling_mode > TilingMode::Linear) {
        LOG_ERROR(Lib_VideoOut, "Invalid tilingMode = {}",
                  static_cast<u32>(attribute->tiling_mode));
        return ORBIS_VIDEO_OUT_ERROR_INVALID_TILING_MODE;
    }

    LOG_INFO(Lib_VideoOut,
             "attributeIndex = {}, pixelFormat = {}, aspectRatio = {}, "
             "tilingMode = {}, width = {}, height = {}, pitchInPixel = {}, option = {:#x}",
             attributeIndex, GetPixelFormatString(attribute->pixel_format), attribute->aspect_ratio,
             static_cast<u32>(attribute->tiling_mode), attribute->width, attribute->height,
             attribute->pitch_in_pixel, attribute->option);

    std::unique_lock lock{port->port_mutex};
    std::memcpy(&port->groups[attributeIndex].attrib, attribute, sizeof(BufferAttribute));
    return 0;
}

// PERF-DIAG-002: flip pacing. Submit-to-flip latency, Present() time, and queued flips at each
// vblank, reported every 2 s.
namespace {
struct FlipStats {
    std::mutex mutex;
    std::chrono::steady_clock::time_point window{};
    u32 flips{}, vblanks{}, idle_vblanks{}, max_queue{};
    double latency_ms{}, max_latency_ms{}, present_ms{}, max_present_ms{};
    u32 paced{}, pace_timeouts{}; // PERF-111
    double paced_ms{};
};
FlipStats g_flip_stats;

// PERF-075: SHADGT_FPS_LOCK=<fps> (30 in run_gt7.cmd and Run-GTSportPerformance.ps1) paces
// flips evenly at that rate; GT Sport's races vary between 25 and 40 FPS, which shows as judder.
u32 FpsLock() {
    static const u32 fps_lock = [] {
        const char* env = std::getenv("SHADGT_FPS_LOCK");
        const u32 fps = env ? static_cast<u32>(std::strtoul(env, nullptr, 10)) : 0;
        if (fps != 0) {
            LOG_WARNING(Lib_VideoOut, "PERF-075: flips locked to at most {} per second", fps);
        }
        return fps;
    }();
    return fps_lock;
}

// SHADGT_FPS_LOCK_AFTER=<seconds>: the lock starts that long after the first vblank or flip (a
// test can go through the menus at full rate first).
bool FpsLockActive() {
    static const auto lock_from = [] {
        const char* env = std::getenv("SHADGT_FPS_LOCK_AFTER");
        return std::chrono::steady_clock::now() +
               std::chrono::seconds(env ? std::strtoul(env, nullptr, 10) : 0);
    }();
    return FpsLock() != 0 && std::chrono::steady_clock::now() >= lock_from;
}

void ReportFlipStats() {
    auto& s = g_flip_stats;
    const auto now = std::chrono::steady_clock::now();
    if (s.window == std::chrono::steady_clock::time_point{}) {
        s.window = now;
        return;
    }
    if (now - s.window < std::chrono::seconds{2}) {
        return;
    }
    LOG_WARNING(Lib_VideoOut,
                "Flips in {:.1f} s: {} flips, {} vblanks ({} with no flip ready), max queued {}, "
                "submit-to-flip avg {:.2f} ms max {:.2f} ms, Present avg {:.2f} ms max {:.2f} ms; "
                "PERF-111: {} submits paced, avg wait {:.2f} ms, {} timeouts",
                std::chrono::duration<double>(now - s.window).count(), s.flips, s.vblanks,
                s.idle_vblanks, s.max_queue, s.flips ? s.latency_ms / s.flips : 0.0,
                s.max_latency_ms, s.flips ? s.present_ms / s.flips : 0.0, s.max_present_ms,
                s.paced, s.paced ? s.paced_ms / s.paced : 0.0, s.pace_timeouts);
    s.flips = s.vblanks = s.idle_vblanks = s.max_queue = 0;
    s.latency_ms = s.max_latency_ms = s.present_ms = s.max_present_ms = 0;
    s.paced = s.pace_timeouts = 0;
    s.paced_ms = 0;
    s.window = now;
}
} // namespace

void VideoOutDriver::Flip(const Request& req) {
    // Update HDR status before presenting.
    presenter->SetHDR(req.port->is_hdr);
    // FIX-193: the game learns of the flip once the GPU has drawn the frame.
    const auto wait_start = std::chrono::steady_clock::now(); // DIAG-GT7-007
    if (req.index != -1) {
        presenter->WaitFrameDrawn(req.frame);
    }

    // Present the frame.
    const auto present_start = std::chrono::steady_clock::now();
    Common::Nvtx::Mark("flip"); // DIAG-056
    {
        Common::Nvtx::Scope nvtx{"present"};
        presenter->Present(req.frame);
    }
    {
        const auto now = std::chrono::steady_clock::now();
        RyDiag::Timeline('P', RyDiag::Seconds(req.submitted), RyDiag::Seconds(wait_start),
                         RyDiag::Seconds(present_start), RyDiag::Seconds(now),
                         u64(u32(req.index))); // DIAG-GT7-007
        std::scoped_lock lk{g_flip_stats.mutex};
        auto& st = g_flip_stats;
        const double present_ms =
            std::chrono::duration<double, std::milli>(now - present_start).count();
        const double latency_ms =
            std::chrono::duration<double, std::milli>(now - req.submitted).count();
        ++st.flips;
        st.present_ms += present_ms;
        st.max_present_ms = std::max(st.max_present_ms, present_ms);
        st.latency_ms += latency_ms;
        st.max_latency_ms = std::max(st.max_latency_ms, latency_ms);
    }

    // Update flip status.
    auto* port = req.port;
    {
        std::unique_lock lock{port->port_mutex};
        auto& flip_status = port->flip_status;
        flip_status.count++;
        flip_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
        flip_status.tsc = Libraries::Kernel::sceKernelReadTsc();
        flip_status.flip_arg = req.flip_arg;
        flip_status.current_buffer = req.index;
        if (req.eop) {
            --flip_status.gc_queue_num;
        }
        --flip_status.flip_pending_num;
    }

    // Trigger flip events for the port.
    for (auto event : port->flip_events) {
        auto equeue = Kernel::GetEqueue(event);
        if (equeue != nullptr) {
            equeue->TriggerEvent(
                static_cast<u64>(OrbisVideoOutInternalEventId::Flip),
                Kernel::OrbisKernelEvent::Filter::VideoOut,
                reinterpret_cast<void*>(static_cast<u64>(OrbisVideoOutInternalEventId::Flip) |
                                        (req.flip_arg << 16)));
        }
    }

    // Reset prev flip label
    if (port->prev_index != -1) {
        port->buffer_labels[port->prev_index] = 0;
        port->SignalVoLabel();
    }
    // save to prev buf index
    port->prev_index = req.index;

    if (req.eop) {
        // PERF-111: a submit-and-flip waiting for this frame to be shown goes on.
        if (eop_flips_in_flight.fetch_sub(1, std::memory_order_acq_rel) <= 0) {
            eop_flips_in_flight.store(0, std::memory_order_release);
        }
        std::scoped_lock lk{pace_mutex};
        pace_cv.notify_all();
    }
}

void VideoOutDriver::NoteEopFlipRequested() {
    eop_flips_in_flight.fetch_add(1, std::memory_order_acq_rel);
}

void VideoOutDriver::PaceFlipSubmit() {
    // Opt-in while untested: SHADGT_FPS_PACE=1.
    static const bool pace = [] {
        const char* env = std::getenv("SHADGT_FPS_PACE");
        return env && env[0] == '1';
    }();
    if (!pace || !FpsLockActive()) {
        return;
    }
    // The frame just submitted may stay in flight; the ones before it must be on screen. A flip
    // that does not come (frames held for a compile, a closed port) holds the game a few frames
    // at most.
    const auto start = std::chrono::steady_clock::now();
    bool shown = true;
    {
        std::unique_lock lk{pace_mutex};
        shown = pace_cv.wait_for(lk, std::chrono::milliseconds(250), [this] {
            return eop_flips_in_flight.load(std::memory_order_acquire) <= 1;
        });
    }
    const double waited_ms = std::chrono::duration<double, std::milli>(
                                 std::chrono::steady_clock::now() - start)
                                 .count();
    std::scoped_lock lk{g_flip_stats.mutex};
    ++g_flip_stats.paced;
    g_flip_stats.paced_ms += waited_ms;
    if (!shown) {
        ++g_flip_stats.pace_timeouts;
    }
}

void VideoOutDriver::DrawBlankFrame() {
    const auto empty_frame = presenter->PrepareBlankFrame(true);
    presenter->Present(empty_frame, false, false);
}

bool VideoOutDriver::DrawLastFrame() {
    const auto frame = presenter->PrepareLastFrame();
    if (frame != nullptr) {
        presenter->Present(frame, true);
        return true;
    }
    return false;
}

bool VideoOutDriver::SubmitFlip(VideoOutPort* port, s32 index, s64 flip_arg,
                                bool is_eop /*= false*/) {
    RyDiag::Mark(RyDiag::Zone::GameFlip, u64(index));
    RyDiag::Timeline('R', RyDiag::Seconds(std::chrono::steady_clock::now()), 0.0, 0.0, 0.0,
                     u64(u32(index)) | (u64(is_eop) << 32)); // DIAG-GT7-007
    {
        std::unique_lock lock{port->port_mutex};
        if (index != -1 && port->flip_status.flip_pending_num > 16) {
            LOG_ERROR(Lib_VideoOut, "Flip queue is full");
            return false;
        }

        if (is_eop) {
            ++port->flip_status.gc_queue_num;
        }
        ++port->flip_status.flip_pending_num; // integral GPU and CPU pending flips counter
        port->flip_status.submit_tsc = Libraries::Kernel::sceKernelReadTsc();
    }

    if (!is_eop) {
        // Non EOP flips can arrive from any thread so ask GPU thread to perform them
        liverpool->SendCommand([=, this]() { SubmitFlipInternal(port, index, flip_arg, is_eop); });
    } else {
        SubmitFlipInternal(port, index, flip_arg, is_eop);
    }

    return true;
}

void VideoOutDriver::SubmitFlipInternal(VideoOutPort* port, s32 index, s64 flip_arg, bool is_eop) {
    Vulkan::Frame* frame;
    if (index == -1) {
        frame = presenter->PrepareBlankFrame(false);
    } else {
        const auto& buffer = port->buffer_slots[index];
        ASSERT_MSG(buffer.group_index >= 0, "Trying to flip an unregistered buffer!");
        const auto& group = port->groups[buffer.group_index];
        frame = presenter->PrepareFrame(group, buffer.address_left);
    }

    const auto queued = std::chrono::steady_clock::now();
    {
        std::scoped_lock lock{mutex};
        requests.push({
            .frame = frame,
            .port = port,
            .flip_arg = flip_arg,
            .index = index,
            .eop = is_eop,
            .submitted = queued,
        });
    }
    RyDiag::Timeline('Q', RyDiag::Seconds(queued), 0.0, 0.0, 0.0,
                     u64(u32(index)) | (u64(is_eop) << 32)); // DIAG-GT7-007
    requests_cv.notify_one(); // GT-FF
}

// GT-FF: RYGT_FAST_FLIPS=1 shows each frame as soon as it is submitted (no faster than the
// flip rate) instead of at the next vblank, which held a frame up to a vblank period and made
// frame times alternate between one and two periods whenever the game ran below the refresh
// rate. Vblanks still tick at their own rate for the game.
static bool FastFlips() {
    static const bool on = [] {
        const char* env = std::getenv("RYGT_FAST_FLIPS");
        return env && env[0] == '1';
    }();
    return on;
}

void VideoOutDriver::PresentThread(std::stop_token token) {
    const std::chrono::nanoseconds vblank_period(1000000000 /
                                                 EmulatorSettings.GetVblankFrequency());

    Common::SetCurrentThreadName("shadGT:PresentThread");
    Common::SetCurrentThreadRealtime(vblank_period);
    // GT7-DISPLAY: the screen stays on while the game presents (the request ends with this thread).
    Common::KeepDisplayAwake();

    Common::AccurateTimer timer{vblank_period};

    const auto receive_request = [this] -> Request {
        std::scoped_lock lk{mutex};
        if (!requests.empty()) {
            const auto request = requests.front();
            requests.pop();
            return request;
        }
        return {};
    };

    while (!token.stop_requested()) {
        timer.Start();
        const auto tick_start = std::chrono::steady_clock::now(); // GT-FF

        if (DebugState.IsGuestThreadsPaused()) {
            DrawLastFrame();
            timer.End();
            continue;
        }
        if (Common::GuestClock::IsPaused()) {
            // FIX-044: no vblank passes while the guest clocks are held for a shader or
            // pipeline compile, as no time passes for the game. PERF-100: the last frame is
            // shown again, with the compile notice drawn over it (or the splash screen, before
            // the game's first frame).
            if (timer.GetTotalWait().count() < 0) {
                // Before the game's first frame a blank frame, which shows the splash screen;
                // after it the last frame (skipped while it is taken).
                // DIAG-126 (SHADGT_NO_LAST_FRAME=1): compile pauses after the first frame keep
                // what is on screen instead of showing the last frame again.
                static const bool no_last_frame = [] {
                    const char* env = std::getenv("SHADGT_NO_LAST_FRAME");
                    return env && env[0] == '1';
                }();
                if (!main_port.is_open || !presenter->HasLastFrame()) {
                    DrawBlankFrame();
                } else if (!no_last_frame) {
                    DrawLastFrame();
                }
            }
            timer.End();
            continue;
        }

        // Check if it's time to take a request.
        auto& vblank_status = main_port.vblank_status;
        const bool locked = FpsLockActive(); // PERF-075
        const int flip_rate = LockedFlipRate(main_port.flip_rate, locked ? FpsLock() : 0,
                                             EmulatorSettings.GetVblankFrequency());
        if (vblank_status.count % (flip_rate + 1) == 0) {
            size_t queued_flips = 0; // DIAG-GT7-007
            {
                std::scoped_lock lk{mutex, g_flip_stats.mutex};
                ++g_flip_stats.vblanks;
                queued_flips = requests.size();
                g_flip_stats.max_queue =
                    std::max<u32>(g_flip_stats.max_queue, static_cast<u32>(requests.size()));
                if (requests.empty()) {
                    ++g_flip_stats.idle_vblanks;
                }
                ReportFlipStats();
            }
            // GT-FF: with fast flips a frame shown since the last vblank counts against the rate.
            const auto flip_interval = vblank_period * (flip_rate + 1);
            const bool may_flip = !FastFlips() || std::chrono::steady_clock::now() -
                                                          last_flip_time >=
                                                      flip_interval - std::chrono::milliseconds(1);
            const auto request = may_flip ? receive_request() : Request{};
            RyDiag::Timeline('V', RyDiag::Seconds(tick_start), double(queued_flips), 0.0, 0.0,
                             request ? 1 : 0); // DIAG-GT7-007
            if (!request) {
                if (timer.GetTotalWait().count() < 0) { // Dont draw too fast
                    if (!main_port.is_open) {
                        DrawBlankFrame();
                    } else if (ImGui::Core::MustKeepDrawing()) {
                        DrawLastFrame();
                    }
                }
            } else {
                Flip(request);
                FRAME_END;
                last_flip_time = std::chrono::steady_clock::now();
            }
        }

        {
            // Needs lock here as can be concurrently read by `sceVideoOutGetVblankStatus`
            std::scoped_lock lock{main_port.vo_mutex};

            // Trigger flip events for the port
            for (auto event : main_port.vblank_events) {
                auto equeue = Kernel::GetEqueue(event);
                if (equeue != nullptr) {
                    equeue->TriggerEvent(
                        static_cast<u64>(OrbisVideoOutInternalEventId::Vblank),
                        Kernel::OrbisKernelEvent::Filter::VideoOut,
                        reinterpret_cast<void*>(
                            static_cast<u64>(OrbisVideoOutInternalEventId::Vblank) |
                            (vblank_status.count << 16)));
                }
            }

            // Update vblank status
            vblank_status.count++;
            vblank_status.process_time = Libraries::Kernel::sceKernelGetProcessTime();
            vblank_status.tsc = Libraries::Kernel::sceKernelReadTsc();
            main_port.vblank_cv.notify_all();
        }

        if (FastFlips()) {
            // GT-FF: until the next vblank, frames are shown as they arrive, no closer together
            // than the flip rate allows (the timer below keeps the vblank period exact).
            const auto flip_interval = vblank_period * (flip_rate + 1) - std::chrono::milliseconds(1);
            const auto deadline = tick_start + vblank_period - std::chrono::microseconds(500);
            while (!token.stop_requested() && std::chrono::steady_clock::now() < deadline) {
                {
                    std::unique_lock lk{mutex};
                    if (!requests_cv.wait_until(lk, deadline, [this] { return !requests.empty(); })) {
                        break;
                    }
                }
                const auto allowed = last_flip_time + flip_interval;
                if (std::chrono::steady_clock::now() < allowed) {
                    if (allowed >= deadline) {
                        break; // the next vblank takes it
                    }
                    std::this_thread::sleep_until(allowed);
                }
                const auto request = receive_request();
                if (!request) {
                    continue;
                }
                Flip(request);
                FRAME_END;
                last_flip_time = std::chrono::steady_clock::now();
            }
        }

        timer.End();
    }
}

} // namespace Libraries::VideoOut
