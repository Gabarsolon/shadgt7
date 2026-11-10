// SPDX-FileCopyrightText: Copyright 2021 yuzu Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "common/recycling_pool.h"

#if defined(_M_X64) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

namespace Common {

/// General purpose function wrapper similar to std::function.
/// Unlike std::function, the captured values don't have to be copyable.
/// This class can be moved but not copied.
template <typename ResultType, typename... Args>
class UniqueFunction {
    class CallableBase {
    public:
        virtual ~CallableBase() = default;
        /// PERF-GT7-126: memory the callable will read that lies behind its captures' pointers.
        const void* hint_at[2]{};
        std::uint32_t hint_bytes[2]{};
        virtual ResultType operator()(Args&&...) = 0;

        // PERF-056: callables are often made on one thread and destroyed on another.
        static void* operator new(std::size_t size) {
            return RecyclingPool::Allocate(size);
        }
        static void operator delete(void* block, std::size_t size) {
            RecyclingPool::Free(block, size);
        }
        static void* operator new(std::size_t size, std::align_val_t alignment) {
            return ::operator new(size, alignment);
        }
        static void operator delete(void* block, std::size_t, std::align_val_t alignment) {
            ::operator delete(block, alignment);
        }
    };

    template <typename Functor>
    class Callable final : public CallableBase {
    public:
        template <typename F>
        explicit Callable(F&& functor_) : functor{std::forward<F>(functor_)} {}
        ~Callable() override = default;

        ResultType operator()(Args&&... args) override {
            if constexpr (std::is_void_v<ResultType>) {
                functor(std::forward<Args>(args)...);
            } else {
                return functor(std::forward<Args>(args)...);
            }
        }

    private:
        Functor functor;
    };

public:
    UniqueFunction() = default;

    template <typename Functor>
        requires(!std::is_same_v<std::remove_cvref_t<Functor>, UniqueFunction> &&
                 std::is_invocable_r_v<ResultType, std::decay_t<Functor>&, Args...>)
    UniqueFunction(Functor&& functor)
        // FIX-GT7-098: a named const lambda deduces Functor as const Lambda&.
        // Storing that reference left queued commands pointing into a dead local
        // or coroutine frame (occlusion writes then corrupted host heap objects).
        // Own the decayed callable; borrowing requires an explicit std::ref.
        : callable{std::make_unique<Callable<std::decay_t<Functor>>>(
              std::forward<Functor>(functor))} {}

    UniqueFunction& operator=(UniqueFunction&& rhs) noexcept = default;
    UniqueFunction(UniqueFunction&& rhs) noexcept = default;

    UniqueFunction& operator=(const UniqueFunction&) = delete;
    UniqueFunction(const UniqueFunction&) = delete;

    ResultType operator()(Args&&... args) const {
        return (*callable)(std::forward<Args>(args)...);
    }

    explicit operator bool() const noexcept {
        return static_cast<bool>(callable);
    }

    /// PERF-GT7-124: asks for the first `bytes` of the callable (its captures) to be brought into
    /// the cache, without waiting for them. For a consumer about to run a callable another
    /// thread made. Nothing on processors other than x86-64.
    void Prefetch(std::size_t bytes) const noexcept {
#if defined(_M_X64) || defined(__x86_64__)
        const char* const at = reinterpret_cast<const char*>(callable.get());
        for (std::size_t offset = 0; at && offset < bytes; offset += 64) {
            _mm_prefetch(at + offset, _MM_HINT_T0);
        }
#else
        static_cast<void>(bytes);
#endif
    }

    /// PERF-GT7-126: names up to two blocks the callable reads through its captures' pointers.
    /// The blocks must stay valid until the callable has run. Called by who made the callable.
    void SetPrefetchHint(std::size_t slot, const void* at, std::size_t bytes) noexcept {
        if (callable && slot < 2) {
            callable->hint_at[slot] = at;
            callable->hint_bytes[slot] = static_cast<std::uint32_t>(bytes);
        }
    }

    /// PERF-GT7-126: asks for the hinted blocks. It reads the callable itself, so it is best
    /// called once an earlier Prefetch of the callable has had time to arrive.
    void PrefetchHints() const noexcept {
#if defined(_M_X64) || defined(__x86_64__)
        if (!callable) {
            return;
        }
        for (std::size_t slot = 0; slot < 2; ++slot) {
            const char* const at = static_cast<const char*>(callable->hint_at[slot]);
            for (std::size_t offset = 0; at && offset < callable->hint_bytes[slot]; offset += 64) {
                _mm_prefetch(at + offset, _MM_HINT_T0);
            }
        }
#endif
    }

private:
    std::unique_ptr<CallableBase> callable;
};

} // namespace Common
