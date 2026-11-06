#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

namespace Vulkan {

template <typename Chunk, typename Completed, typename Wait, typename Release, typename Allocate>
Chunk RecoverUploadMemory(std::uint64_t size, std::uint64_t current_tick, bool allow_reuse,
                          std::vector<Chunk>& free, std::deque<Chunk>& retiring,
                          Completed&& completed, Wait&& wait, Release&& release,
                          Allocate&& allocate) {
    Chunk reusable;
    const auto reclaim = [&](Chunk&& chunk) {
        if (allow_reuse && chunk.buffer->Mapped() && chunk.buffer->Size() >= size &&
            (!reusable.buffer || chunk.buffer->Size() < reusable.buffer->Size())) {
            if (reusable.buffer) {
                release(std::move(reusable.buffer));
            }
            reusable = std::move(chunk);
        } else {
            release(std::move(chunk.buffer));
        }
    };
    for (auto& chunk : free) {
        reclaim(std::move(chunk));
    }
    free.clear();
    const auto reclaim_completed = [&] {
        const auto done = completed();
        for (auto it = retiring.begin(); it != retiring.end();) {
            if (it->retire_tick <= done && it->retire_tick < current_tick) {
                reclaim(std::move(*it));
                it = retiring.erase(it);
            } else {
                ++it;
            }
        }
    };
    reclaim_completed();
    if (reusable.buffer) {
        return reusable;
    }
    std::uint64_t last_submitted = 0;
    for (const auto& chunk : retiring) {
        if (chunk.retire_tick < current_tick) {
            last_submitted = std::max(last_submitted, chunk.retire_tick);
        }
    }
    if (last_submitted != 0) {
        wait(last_submitted);
        reclaim_completed();
    }
    return reusable.buffer ? std::move(reusable) : allocate();
}

}
