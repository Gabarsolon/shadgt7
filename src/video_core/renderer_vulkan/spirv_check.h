// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <span>
#include <unordered_map>
#include <utility>

#include "common/types.h"

namespace Vulkan {

/// PERF-032: whether a stored SPIR-V module is whole before it goes to the driver: the SPIR-V
/// magic, instructions whose word counts end exactly at the end of the data, and OpFunctionEnd
/// last (modules end with their last function). A file cut short when a session was killed while
/// storing it fails this.
inline bool IsCompleteSpirv(std::span<const u32> words) {
    constexpr u32 Magic = 0x07230203;
    constexpr u32 HeaderWords = 5;
    constexpr u32 OpFunctionEnd = 56;
    if (words.size() <= HeaderWords || words[0] != Magic) {
        return false;
    }
    size_t at = HeaderWords;
    u32 last_opcode = 0;
    while (at < words.size()) {
        const u32 count = words[at] >> 16;
        if (count == 0 || count > words.size() - at) {
            return false;
        }
        last_opcode = words[at] & 0xFFFF;
        at += count;
    }
    return last_opcode == OpFunctionEnd;
}

/// FIX-GT7-011: how a module makes a float atomic min/max from integer atomics (without
/// VK_EXT_shader_atomic_float2): both atomics, then an OpSelect by the value's sign (before
/// FIX-226, or RYGT_FLOAT_ATOMIC_EXACT=0), or only the one the sign picks, its result from an
/// OpPhi (FIX-226). The pairs are UMin + SMax (max) and UMax + SMin (min), each result maybe
/// through one OpBitcast. Words of a whole module (IsCompleteSpirv).
enum class FloatAtomicEmulation { None, BothAtomics, OneAtomic };

inline FloatAtomicEmulation FindFloatAtomicEmulation(std::span<const u32> words) {
    constexpr u32 HeaderWords = 5;
    constexpr u32 OpBitcast = 124, OpSelect = 169, OpPhi = 245;
    constexpr u32 OpAtomicSMin = 236, OpAtomicUMin = 237, OpAtomicSMax = 238, OpAtomicUMax = 239;
    const auto each = [&](auto&& visit) {
        for (size_t at = HeaderWords; at < words.size();) {
            const u32 count = words[at] >> 16;
            if (count == 0 || count > words.size() - at) {
                return;
            }
            visit(words[at] & 0xFFFF, words.subspan(at + 1, count - 1));
            at += count;
        }
    };
    // Result id: the atomic's opcode, or a bitcast's source.
    std::unordered_map<u32, std::pair<u32, u32>> defs;
    bool atomics = false;
    each([&](u32 op, std::span<const u32> operands) {
        if (op >= OpAtomicSMin && op <= OpAtomicUMax && operands.size() >= 2) {
            defs.emplace(operands[1], std::pair{op, 0u});
            atomics = true;
        } else if (op == OpBitcast && operands.size() >= 3) {
            defs.emplace(operands[1], std::pair{op, operands[2]});
        }
    });
    if (!atomics) {
        return FloatAtomicEmulation::None;
    }
    const auto atomic_of = [&](u32 id) -> u32 {
        auto it = defs.find(id);
        if (it != defs.end() && it->second.first == OpBitcast) {
            it = defs.find(it->second.second);
        }
        return it != defs.end() && it->second.first != OpBitcast ? it->second.first : 0u;
    };
    const auto pair = [&](u32 a, u32 b) {
        const u32 x = atomic_of(a);
        const u32 y = atomic_of(b);
        const auto is = [&](u32 p, u32 q) { return (x == p && y == q) || (x == q && y == p); };
        return is(OpAtomicUMin, OpAtomicSMax) || is(OpAtomicUMax, OpAtomicSMin);
    };
    bool both = false;
    bool one = false;
    each([&](u32 op, std::span<const u32> operands) {
        if (op == OpSelect && operands.size() == 5 && pair(operands[3], operands[4])) {
            both = true;
        } else if (op == OpPhi && operands.size() == 6 && pair(operands[2], operands[4])) {
            one = true;
        }
    });
    return both  ? FloatAtomicEmulation::BothAtomics
           : one ? FloatAtomicEmulation::OneAtomic
                 : FloatAtomicEmulation::None;
}

} // namespace Vulkan
