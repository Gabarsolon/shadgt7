// SPDX-License-Identifier: GPL-2.0-or-later
// Regression checks for wave64 reads beneath wave-aligned invocation guards.
#include <cstdio>
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"
#include "shader_recompiler/runtime_info.h"

using namespace Shader;

namespace Common {
bool PerfFeatureEnabled(u32) { return true; }
} // namespace Common

int main() {
    int tests = 0;
    int failures = 0;
    for (u32 threads : {64u, 128u}) {
        for (u32 subgroup : {32u, 64u}) {
            for (bool global : {false, true}) {
                for (bool constant_left : {false, true}) {
                    for (int comparison = 0; comparison < 4; ++comparison) {
                        for (u32 boundary : {0u, 31u, 32u, 63u, 64u, 65u, 127u, 128u}) {
                            Common::ObjectPool<IR::Inst> pool{64};
                            IR::Block header{pool}, body{pool}, merge{pool};
                            IR::IREmitter ir{header};
                            IR::U32 index = ir.GetAttributeU32(IR::Attribute::LocalInvocationId, 0);
                            if (global) {
                                index = IR::U32{ir.IAdd(ir.ShiftLeftLogical(
                                    ir.GetAttributeU32(IR::Attribute::WorkgroupId, 0), ir.Imm32(6)),
                                    index)};
                            }
                            const auto lhs = constant_left ? ir.Imm32(boundary) : index;
                            const auto rhs = constant_left ? index : ir.Imm32(boundary);
                            const IR::U1 condition = comparison == 0 ? ir.ILessThan(lhs, rhs, false)
                                : comparison == 1 ? ir.ILessThanEqual(lhs, rhs, false)
                                : comparison == 2 ? ir.IGreaterThan(lhs, rhs, false)
                                                  : ir.IGreaterThanEqual(lhs, rhs, false);
                            IR::IREmitter body_ir{body};
                            [[maybe_unused]] const auto read = body_ir.ReadLane(index, ir.Imm32(63));
                            Info info{};
                            info.hw_stage = HwStage::Compute;
                            IR::Program program{info};
                            program.blocks = {&header, &body, &merge};
                            using Node = IR::AbstractSyntaxNode;
                            program.syntax_list = {
                                {.data = {.block = &header}, .type = Node::Type::Block},
                                {.data = {.if_node = {condition, &body, &merge}}, .type = Node::Type::If},
                                {.data = {.block = &body}, .type = Node::Type::Block},
                                {.data = {.end_if = {&merge}}, .type = Node::Type::EndIf},
                                {.data = {.block = &merge}, .type = Node::Type::Block},
                            };
                            RuntimeInfo runtime{};
                            runtime.Initialize(HwStage::Compute, SwStage::Compute);
                            runtime.hw.cs.workgroup_size = {threads, 1, 1};
                            Profile profile{};
                            profile.subgroup_size = subgroup;
                            Optimization::LowerWave64BallotPass(program, runtime, profile);
                            // Derive uniformity by evaluating the guard over the entire wave,
                            // independently of the compiler's aligned-boundary recognition.
                            const auto predicate = [&](u32 lane) {
                                const u32 a = constant_left ? boundary : lane;
                                const u32 b = constant_left ? lane : boundary;
                                return comparison == 0 ? a < b : comparison == 1 ? a <= b
                                     : comparison == 2 ? a > b : a >= b;
                            };
                            bool uniform = true;
                            for (u32 lane = 1; lane < 64; ++lane)
                                uniform &= predicate(lane) == predicate(0);
                            // This conservative optimization only recognizes aligned cuts;
                            // guards always true/false for this workgroup may be left alone.
                            const bool aligned = (boundary % 64 == 0 &&
                                ((!constant_left && (comparison == 0 || comparison == 3)) ||
                                 (constant_left && (comparison == 1 || comparison == 2)))) ||
                                (boundary % 64 == 63 &&
                                ((!constant_left && (comparison == 1 || comparison == 2)) ||
                                 (constant_left && (comparison == 0 || comparison == 3))));
                            const bool expected = uniform && aligned && threads == 64 && subgroup == 32;
                            const bool lowered = info.shared_memory_scratch_size != 0;
                            ++tests;
                            if (lowered != expected) {
                                ++failures;
                                std::printf("FAIL threads=%u subgroup=%u global=%d left=%d cmp=%d boundary=%u\n",
                                            threads, subgroup, global, constant_left, comparison, boundary);
                            }
                        }
                    }
                }
            }
        }
    }
    std::printf("%d wave guard checks, %d failures\n", tests, failures);
    return failures != 0;
}
