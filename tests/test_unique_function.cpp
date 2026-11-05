// SPDX-FileCopyrightText: Copyright 2026 shadGT
// SPDX-License-Identifier: GPL-2.0-or-later
#include <array>
#include <functional>
#include <memory>
#include <thread>
#include <type_traits>
#include <gtest/gtest.h>
#include "common/unique_function.h"

static_assert(!std::is_copy_constructible_v<Common::UniqueFunction<void>>);
static_assert(std::is_move_constructible_v<Common::UniqueFunction<void>>);

TEST(UniqueFunction, VoidWrapperDiscardsResult) {
    unsigned calls = 0;
    Common::UniqueFunction<void> queued{[&calls] { return ++calls; }};
    queued();
    EXPECT_EQ(calls, 1u);
}

TEST(UniqueFunction, ConstLvalueIsOwnedSnapshot) {
    struct Command {
        mutable unsigned address;
        mutable unsigned pairs;
        mutable unsigned value;
        std::array<unsigned, 3> operator()() const { return {address, pairs, value}; }
    };
    const Command original{123, 8, 456};
    Common::UniqueFunction<std::array<unsigned, 3>> queued{original};
    // A deterministic lifetime regression without dereferencing freed storage:
    // changing the source object must not change a previously queued copy.
    original.address = 999;
    original.pairs = 0xffffffff;
    original.value = 0;
    const auto result = queued();
    EXPECT_EQ(result[0], 123u);
    EXPECT_EQ(result[1], 8u);
    EXPECT_EQ(result[2], 456u);
}

TEST(UniqueFunction, NamedConstLambdaOutlivesProducerThread) {
    Common::UniqueFunction<unsigned> queued;
    std::jthread producer{[&] {
        const auto local = [address = 123u, pairs = 8u, counter = 456u] {
            return address + pairs + counter;
        };
        queued = Common::UniqueFunction<unsigned>{local};
    }};
    producer.join();
    EXPECT_EQ(queued(), 587u);
}

TEST(UniqueFunction, MoveOnlyCaptureSurvivesWrapperMove) {
    Common::UniqueFunction<unsigned> first{[p = std::make_unique<unsigned>(91)] { return *p; }};
    auto second = std::move(first);
    EXPECT_FALSE(static_cast<bool>(first));
    EXPECT_EQ(second(), 91u);
}

TEST(UniqueFunction, ReferenceWrapperExplicitlyBorrows) {
    struct Counter {
        unsigned value = 0;
        unsigned operator()() { return ++value; }
    } original;
    Common::UniqueFunction<unsigned> queued{std::ref(original)};
    EXPECT_EQ(queued(), 1u);
    EXPECT_EQ(original.value, 1u);
}

TEST(UniqueFunction, MutableLvalueCallableIsCopied) {
    auto local = [value = 0u]() mutable { return ++value; };
    Common::UniqueFunction<unsigned> queued{local};
    EXPECT_EQ(queued(), 1u);
    EXPECT_EQ(queued(), 2u);
    EXPECT_EQ(local(), 1u);
}

namespace {
unsigned Add(unsigned a, unsigned b) { return a + b; }
}
TEST(UniqueFunction, FunctionReferenceDecaysToPointer) {
    Common::UniqueFunction<unsigned, unsigned, unsigned> queued{Add};
    EXPECT_EQ(queued(7u, 8u), 15u);
}
