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

/// FIX-GT7-011: how a module makes a float atomic min/max of a buffer from integer atomics
/// (without VK_EXT_shader_atomic_float2): both atomics, each result bitcast to a float, then an
/// OpSelect by the value's sign (before FIX-226, or RYGT_FLOAT_ATOMIC_EXACT=0), or only the one
/// the sign picks, the two results merged by an OpPhi that is bitcast to a float (FIX-226). The
/// pairs are UMin + SMax (max) and UMax + SMin (min). FIX-GT7-013: image atomics (on an
/// OpImageTexelPointer) are not counted: FIX-226 left their emulation as it was, so a module of
/// them would be translated again into the same form every session. Nor are atomics whose
/// results are selected or merged as integers. Words of a whole module (IsCompleteSpirv).
enum class FloatAtomicEmulation { None, BothAtomics, OneAtomic };

inline FloatAtomicEmulation FindFloatAtomicEmulation(std::span<const u32> words) {
    constexpr u32 HeaderWords = 5;
    constexpr u32 OpImageTexelPointer = 60, OpBitcast = 124, OpSelect = 169, OpPhi = 245;
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
    // Image texel pointers first (an atomic's pointer is defined before it).
    std::unordered_map<u32, u32> texel_pointers;
    std::unordered_map<u32, u32> atomics; // result id: opcode, buffer atomics only
    std::unordered_map<u32, u32> bitcasts; // result id: source id
    each([&](u32 op, std::span<const u32> operands) {
        if (op == OpImageTexelPointer && operands.size() >= 2) {
            texel_pointers.emplace(operands[1], 0u);
        } else if (op >= OpAtomicSMin && op <= OpAtomicUMax && operands.size() >= 3 &&
                   !texel_pointers.contains(operands[2])) {
            atomics.emplace(operands[1], op);
        } else if (op == OpBitcast && operands.size() >= 3) {
            bitcasts.emplace(operands[1], operands[2]);
        }
    });
    if (atomics.empty()) {
        return FloatAtomicEmulation::None;
    }
    const auto atomic_of = [&](u32 id) -> u32 {
        const auto it = atomics.find(id);
        return it != atomics.end() ? it->second : 0u;
    };
    const auto bitcast_atomic_of = [&](u32 id) -> u32 {
        const auto it = bitcasts.find(id);
        return it != bitcasts.end() ? atomic_of(it->second) : 0u;
    };
    const auto pair = [&](u32 x, u32 y) {
        const auto is = [&](u32 p, u32 q) { return (x == p && y == q) || (x == q && y == p); };
        return is(OpAtomicUMin, OpAtomicSMax) || is(OpAtomicUMax, OpAtomicSMin);
    };
    bool both = false;
    std::unordered_map<u32, u32> phis; // OpPhi results merging such a pair
    each([&](u32 op, std::span<const u32> operands) {
        if (op == OpSelect && operands.size() == 5 &&
            pair(bitcast_atomic_of(operands[3]), bitcast_atomic_of(operands[4]))) {
            both = true;
        } else if (op == OpPhi && operands.size() == 6 &&
                   pair(atomic_of(operands[2]), atomic_of(operands[4]))) {
            phis.emplace(operands[1], 0u);
        }
    });
    if (both) {
        return FloatAtomicEmulation::BothAtomics;
    }
    for (const auto& [id, source] : bitcasts) {
        if (phis.contains(source)) {
            return FloatAtomicEmulation::OneAtomic;
        }
    }
    return FloatAtomicEmulation::None;
}

} // namespace Vulkan
