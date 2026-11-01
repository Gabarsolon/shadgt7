// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <atomic>
#include <cstdint>
#include <exception>
#include <future>
#include <mutex>
#include <unordered_map>
#include <utility>

namespace Vulkan {

/// Built values by key, kept until exit.
template <typename Value>
class SharedBuildMap {
public:
    Value Find(std::uint64_t key) {
        std::scoped_lock lk{mutex};
        const auto it = values.find(key);
        return it != values.end() ? it->second : Value{};
    }

    void Insert(std::uint64_t key, Value value) {
        std::scoped_lock lk{mutex};
        values.emplace(key, value);
    }

    /// PERF-GT7-041 (RYGT_GPL=1): the value of the key, built by the first caller only; a caller
    /// that needs one another build is working on waits for it instead of building it again (a
    /// race start queues many cars that share their pixel shaders). Second: it was reused.
    /// FIX-GT7-030: with may_wait false (a draw is waiting for this build) the caller builds its
    /// own copy instead of waiting for a slower build. If the build throws, its waiters get the
    /// exception and the next caller builds again.
    template <typename Create>
    std::pair<Value, bool> GetOrCreate(std::uint64_t key, bool may_wait, Create&& create) {
        std::shared_future<Value> wait_for;
        std::promise<Value> promise;
        bool owner = false;
        {
            std::scoped_lock lk{mutex};
            if (const auto it = values.find(key); it != values.end()) {
                return {it->second, true};
            }
            if (const auto it = pending.find(key); it == pending.end()) {
                pending.emplace(key, promise.get_future().share());
                owner = true;
            } else if (may_wait) {
                wait_for = it->second;
            }
        }
        if (wait_for.valid()) {
            ++shared_waits;
            return {wait_for.get(), true};
        }
        if (!owner) {
            const Value value = create();
            Insert(key, value);
            return {value, false};
        }
        const Value value = [&] {
            try {
                return create();
            } catch (...) {
                {
                    std::scoped_lock lk{mutex};
                    pending.erase(key);
                }
                promise.set_exception(std::current_exception());
                throw;
            }
        }();
        {
            std::scoped_lock lk{mutex};
            values.emplace(key, value);
            pending.erase(key);
        }
        promise.set_value(value);
        return {value, false};
    }

    std::atomic<std::uint64_t> shared_waits{};

private:
    std::mutex mutex;
    std::unordered_map<std::uint64_t, Value> values;
    std::unordered_map<std::uint64_t, std::shared_future<Value>> pending;
};

} // namespace Vulkan
