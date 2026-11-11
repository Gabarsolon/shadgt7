// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

namespace Vulkan::Gt7Diagnostic {

constexpr std::uint64_t MaxBundleBytes = 256ull << 20;

// Poll can run at the TAA draw or the final scene handoff. Start on the next frame
// so the trace includes the scene's geometry and motion producers.
struct CaptureRequest {
    std::uint64_t earliest_frame{};
    bool armed{};

    void Arm(std::uint64_t observed_frame) noexcept {
        earliest_frame = observed_frame + 1;
        armed = true;
    }

    bool Ready(std::uint64_t frame) const noexcept { return armed && frame >= earliest_frame; }

    bool Take(std::uint64_t frame) noexcept {
        if (!Ready(frame)) {
            return false;
        }
        armed = false;
        return true;
    }
};

inline std::optional<std::uint64_t> ReadbackBytes(unsigned width, unsigned height, unsigned texel,
                                                  std::uint64_t remaining) noexcept {
    if (!width || !height || width > 8192 || height > 8192 || !texel || texel > 16) {
        return std::nullopt;
    }
    const auto bytes = std::uint64_t(width) * height * texel;
    if (bytes > remaining || bytes > std::uint64_t(std::numeric_limits<std::streamsize>::max())) {
        return std::nullopt;
    }
    return bytes;
}

inline bool MappedSpan(const void* data, std::uint64_t bytes, std::uint64_t allocated) noexcept {
    return data && bytes && bytes <= allocated && bytes <= MaxBundleBytes;
}

// Only retired readbacks reach this queue. Their owner keeps the mapping alive through I/O.
struct Write {
    unsigned bundle{};
    std::string name;
    std::string text;
    std::shared_ptr<void> owner;
    const void* data{};
    std::uint64_t bytes{};
};

static_assert(std::is_nothrow_move_constructible_v<Write>);

class Writer {
public:
    using Report = void (*)(bool, unsigned, std::string_view, std::uint64_t) noexcept;

    Writer(std::filesystem::path directory, std::filesystem::path trigger, Report report)
        : directory{std::move(directory)}, trigger{std::move(trigger)}, report{report},
          thread{[this](std::stop_token stop) { Run(stop); }} {}

    ~Writer() {
        thread.request_stop();
        changed.notify_all();
    }

    bool TakeRequest() noexcept { return requested.exchange(false, std::memory_order_relaxed); }

    bool Failed(unsigned bundle) const noexcept {
        return bundle > 3 || (failures.load(std::memory_order_relaxed) & (1u | (1u << bundle)));
    }

    void Fail(unsigned bundle, std::string_view reason, std::uint64_t error = 0) noexcept {
        const auto bit = 1u << (bundle <= 3 ? bundle : 0);
        if (!(failures.fetch_or(bit, std::memory_order_relaxed) & bit) && report) {
            report(false, bundle, reason, error);
        }
    }

    // One recorder produces tasks and one writer consumes them. Publication is lock-free.
    bool Enqueue(Write write) noexcept {
        const auto at = produced.load(std::memory_order_relaxed);
        const auto next = (at + 1) % QueueSize;
        if (next == consumed.load(std::memory_order_acquire) || Failed(write.bundle))
            return false;
        pending[at].emplace(std::move(write));
        produced.store(next, std::memory_order_release);
        changed.notify_one();
        return true;
    }

private:
    void PollTrigger() {
        if (requested.load(std::memory_order_relaxed) || requests >= 3 || Failed(0))
            return;
        std::error_code error;
        const bool exists = std::filesystem::is_regular_file(trigger, error);
        if (error && error != std::errc::no_such_file_or_directory) {
            Fail(0, "trigger lookup", error.value());
            return;
        }
        if (!exists)
            return;
        if (std::filesystem::remove(trigger, error)) {
            ++requests;
            requested.store(true, std::memory_order_relaxed);
        } else if (error) {
            Fail(0, "trigger removal", error.value());
        }
    }

    void Store(const Write& write) noexcept {
        try {
            if (Failed(write.bundle))
                return;
            const std::filesystem::path name{write.name};
            if (write.name.empty() || write.name.size() > 192 || name.filename() != name) {
                Fail(write.bundle, "invalid output name");
                return;
            }
            if (write.bytes &&
                (!write.owner || !MappedSpan(write.data, write.bytes, write.bytes))) {
                Fail(write.bundle, "invalid retired readback");
                return;
            }
            if (write.text.size() > (4u << 20)) {
                Fail(write.bundle, "metadata limit");
                return;
            }
            std::error_code error;
            std::filesystem::create_directories(directory, error);
            if (error) {
                Fail(write.bundle, "output directory", error.value());
                return;
            }
            std::ofstream file(directory / name, std::ios::binary | std::ios::trunc);
            if (!file.is_open()) {
                Fail(write.bundle, "output open");
                return;
            }
            if (write.bytes) {
                file.write(static_cast<const char*>(write.data), std::streamsize(write.bytes));
            } else if (!write.text.empty()) {
                file.write(write.text.data(), std::streamsize(write.text.size()));
            }
            file.close();
            if (file.fail()) {
                Fail(write.bundle, "output write or close");
                return;
            }
            if (write.bytes && report)
                report(true, write.bundle, write.name, write.bytes);
        } catch (...) {
            Fail(write.bundle, "output exception");
        }
    }

    void Run(std::stop_token stop) noexcept {
        try {
            auto next_trigger = std::chrono::steady_clock::time_point{};
            while (!stop.stop_requested()) {
                std::optional<Write> write;
                const auto at = consumed.load(std::memory_order_relaxed);
                if (at != produced.load(std::memory_order_acquire)) {
                    write.emplace(std::move(*pending[at]));
                    pending[at].reset();
                    consumed.store((at + 1) % QueueSize, std::memory_order_release);
                } else {
                    std::unique_lock lock{wait_mutex};
                    changed.wait_for(lock, std::chrono::milliseconds(100), [&] {
                        return produced.load(std::memory_order_acquire) != at ||
                               stop.stop_requested();
                    });
                }
                if (write)
                    Store(*write);
                const auto now = std::chrono::steady_clock::now();
                if (now >= next_trigger) {
                    next_trigger = now + std::chrono::milliseconds(500);
                    PollTrigger();
                }
            }
        } catch (...) {
            Fail(0, "writer exception");
        }
    }

    std::filesystem::path directory;
    std::filesystem::path trigger;
    Report report{};
    std::atomic<unsigned> failures{};
    std::atomic<bool> requested{};
    unsigned requests{};
    static constexpr unsigned QueueSize = 64;
    std::atomic<unsigned> produced{}, consumed{};
    std::mutex wait_mutex;
    std::condition_variable changed;
    std::array<std::optional<Write>, QueueSize> pending;
    std::jthread thread;
};

} // namespace Vulkan::Gt7Diagnostic
