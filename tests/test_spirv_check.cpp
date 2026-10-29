// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <initializer_list>
#include <vector>

#include <gtest/gtest.h>

#include "video_core/renderer_vulkan/spirv_check.h"

using Vulkan::FindFloatAtomicEmulation;
using Vulkan::FloatAtomicEmulation;

namespace {
// A module's words: the header, then the given instructions (opcode, operands).
std::vector<u32> Module(std::initializer_list<std::vector<u32>> instructions) {
    std::vector<u32> words{0x07230203, 0x00010600, 0, 100, 0};
    for (const auto& inst : instructions) {
        words.push_back(u32(inst.size()) << 16 | inst[0]);
        words.insert(words.end(), inst.begin() + 1, inst.end());
    }
    words.push_back(1u << 16 | 56); // OpFunctionEnd
    return words;
}
constexpr u32 OpBitcast = 124, OpSelect = 169, OpPhi = 245;
constexpr u32 OpAtomicSMin = 236, OpAtomicUMin = 237, OpAtomicSMax = 238, OpAtomicUMax = 239;
} // namespace

TEST(SpirvCheck, FindsBothAtomicsBeforeFix226) {
    // Atomic max: UMin and SMax (pointer 9, scope 10, semantics 11, value 12), both bitcast to
    // float, then an OpSelect by the sign (13).
    const auto module = Module({
        {OpAtomicUMin, 1, 20, 9, 10, 11, 12},
        {OpBitcast, 2, 21, 20},
        {OpAtomicSMax, 1, 22, 9, 10, 11, 12},
        {OpBitcast, 2, 23, 22},
        {OpSelect, 2, 24, 13, 21, 23},
    });
    EXPECT_EQ(FindFloatAtomicEmulation(module), FloatAtomicEmulation::BothAtomics);
}

TEST(SpirvCheck, FindsOneAtomicOfFix226) {
    // Atomic min: UMax in one branch (label 30), SMin in the other (31), an OpPhi of both.
    const auto module = Module({
        {OpAtomicUMax, 1, 20, 9, 10, 11, 12},
        {OpAtomicSMin, 1, 22, 9, 10, 11, 12},
        {OpPhi, 1, 24, 20, 30, 22, 31},
        {OpBitcast, 2, 25, 24},
    });
    EXPECT_EQ(FindFloatAtomicEmulation(module), FloatAtomicEmulation::OneAtomic);
}

TEST(SpirvCheck, IgnoresOtherAtomicsAndSelects) {
    const auto module = Module({
        {OpAtomicUMin, 1, 20, 9, 10, 11, 12},
        {OpAtomicUMax, 1, 22, 9, 10, 11, 12},
        {OpSelect, 1, 24, 13, 20, 22}, // an integer min and max: not a float emulation pair
        {OpSelect, 1, 25, 13, 14, 15},
    });
    EXPECT_EQ(FindFloatAtomicEmulation(module), FloatAtomicEmulation::None);
    EXPECT_EQ(FindFloatAtomicEmulation(Module({{OpBitcast, 2, 21, 20}})),
              FloatAtomicEmulation::None);
}
