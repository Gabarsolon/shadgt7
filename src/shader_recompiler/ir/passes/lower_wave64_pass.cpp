// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <string>
#include <string_view>
#include <unordered_set>
#include <fmt/format.h>
#include <magic_enum/magic_enum.hpp>
#include "common/logging/classes.h"
#include "common/perf_monitor.h"
#include "shader_recompiler/info.h"
#include "shader_recompiler/ir/basic_block.h"
#include "shader_recompiler/ir/breadth_first_search.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/program.h"
#include "shader_recompiler/profile.h"

namespace Shader::Optimization {

// A global X index in a workgroup whose X dimension is a multiple of 64 has
// the form 64 * wave + lane. Comparing it with a wave-aligned boundary cannot
// split a guest wave, even though the index itself varies between invocations.
static bool IsWaveIndex(const IR::Value& value, const RuntimeInfo& runtime) {
    if (value.IsImmediate()) {
        return false;
    }
    const auto* inst = value.Inst();
    if (inst->GetOpcode() == IR::Opcode::GetAttributeU32) {
        const auto attr = inst->Arg(0).Attribute();
        return attr == IR::Attribute::LocalInvocationIndex ||
               (attr == IR::Attribute::LocalInvocationId && inst->Arg(1).IsImmediate() &&
                inst->Arg(1).U32() == 0 && runtime.hw.cs.workgroup_size[0] % 64 == 0);
    }
    if (inst->GetOpcode() != IR::Opcode::IAdd32) {
        return false;
    }
    const auto aligned_group_offset = [](const IR::Value& v) {
        if (v.IsImmediate()) {
            return v.U32() % 64 == 0;
        }
        const auto* shift = v.Inst();
        if (shift->GetOpcode() != IR::Opcode::ShiftLeftLogical32 ||
            !shift->Arg(1).IsImmediate() || shift->Arg(1).U32() < 6 ||
            shift->Arg(1).U32() >= 32 || shift->Arg(0).IsImmediate()) {
            return false;
        }
        const auto* group = shift->Arg(0).Inst();
        return group->GetOpcode() == IR::Opcode::GetAttributeU32 &&
               group->Arg(0).Attribute() == IR::Attribute::WorkgroupId;
    };
    return (aligned_group_offset(inst->Arg(0)) && IsWaveIndex(inst->Arg(1), runtime)) ||
           (aligned_group_offset(inst->Arg(1)) && IsWaveIndex(inst->Arg(0), runtime));
}

static bool IsWaveUniformComparison(const IR::Inst& inst, const RuntimeInfo& runtime) {
    const auto& wg = runtime.hw.cs.workgroup_size;
    // The lowering exchanges the two halves through workgroup barriers. A guard
    // that selects whole waves can still split a workgroup containing several waves.
    if (wg[0] * wg[1] * wg[2] != 64) {
        return false;
    }
    using IR::Opcode;
    const auto op = inst.GetOpcode();
    if (op != Opcode::ULessThan32 && op != Opcode::ULessThanEqual32 &&
        op != Opcode::UGreaterThan32 && op != Opcode::UGreaterThanEqual32) {
        return false;
    }
    const bool constant_left = inst.Arg(0).IsImmediate();
    const IR::Value boundary = inst.Arg(constant_left ? 0 : 1);
    const IR::Value index = inst.Arg(constant_left ? 1 : 0);
    if (!boundary.IsImmediate() || !IsWaveIndex(index, runtime)) {
        return false;
    }
    const bool starts_wave = constant_left
                                ? op == Opcode::UGreaterThan32 || op == Opcode::ULessThanEqual32
                                : op == Opcode::ULessThan32 || op == Opcode::UGreaterThanEqual32;
    return (boundary.U32() & 63) == (starts_wave ? 0u : 63u);
}

static bool IsDivergentCondition(const IR::U1& condition) {
    if (condition.IsImmediate()) {
        return false;
    }
    const IR::Inst* const condition_inst = condition.Inst();
    return IR::BreadthFirstSearch(condition_inst,
                                  [](const IR::Inst* inst) -> std::optional<bool> {
                                      switch (inst->GetOpcode()) {
                                      case IR::Opcode::LaneId:
                                          return true;
                                      case IR::Opcode::GetAttributeU32:
                                          if (inst->Arg(0).Attribute() ==
                                              IR::Attribute::LocalInvocationId) {
                                              return true;
                                          }
                                          break;
                                      default:
                                          break;
                                      }
                                      return std::nullopt;
                                  })
        .value_or(false);
}

// FIX-020: whether a loop condition can differ between the invocations of a GCN wave. GCN
// loops are often driven by scalar branches: SCC, or VCC/EXEC masks built from scalar values and
// EXEC itself. Those take the same path in every active invocation. Values that differ per lane
// (lane and invocation ids, shuffles, atomics, the bits of a ballot of a per-lane condition) make
// the loop divergent. Wave-wide reductions (lane reads, ballots) end the search: their result is
// the same everywhere.
// Offline inspection (tests/tools): when set, records why conditions are divergent.
std::string* g_wave64_trace = nullptr;

static bool TraceDivergence(const IR::Inst* inst) {
    if (g_wave64_trace) {
        *g_wave64_trace +=
            fmt::format(" [divergent at {}]", magic_enum::enum_name(inst->GetOpcode()));
    }
    return true;
}

static bool IsDivergentLoopCondition(const IR::U1& condition, const RuntimeInfo& runtime) {
    if (condition.IsImmediate()) {
        return false;
    }
    std::vector<const IR::Inst*> stack{condition.Inst()};
    std::unordered_set<const IR::Inst*> visited{condition.Inst()};
    bool in_inverse_ballot = false;
    const auto push_args = [&](const IR::Inst* inst) {
        for (size_t arg = 0; arg < inst->NumArgs(); ++arg) {
            const IR::Value value{inst->Arg(arg)};
            if (!value.IsImmediate() && visited.insert(value.Inst()).second) {
                stack.push_back(value.Inst());
            }
        }
    };
    while (!stack.empty()) {
        const IR::Inst* inst = stack.back();
        stack.pop_back();
        if (IsWaveUniformComparison(*inst, runtime)) {
            continue;
        }
        switch (inst->GetOpcode()) {
        case IR::Opcode::LaneId:
        case IR::Opcode::MaskedBitCount32:
        case IR::Opcode::WriteLane:
        case IR::Opcode::Shuffle:
        case IR::Opcode::ShuffleXor:
        case IR::Opcode::QuadBroadcast:
        case IR::Opcode::DataAppend:
        case IR::Opcode::DataConsume:
            return TraceDivergence(inst);
        case IR::Opcode::GetAttributeU32: {
            const auto attribute = inst->Arg(0).Attribute();
            if (attribute == IR::Attribute::LocalInvocationId ||
                attribute == IR::Attribute::LocalInvocationIndex) {
                return TraceDivergence(inst);
            }
            break;
        }
        case IR::Opcode::GetExec:
            // EXEC by itself is a per-invocation condition (an EXEC scope).
            return TraceDivergence(inst);
        case IR::Opcode::ReadLane:
        case IR::Opcode::ReadFirstLane:
        case IR::Opcode::BallotFindLsb:
            continue;
        case IR::Opcode::ConditionRef:
            // Holds a branch condition so passes keep it; the value is its argument.
            push_args(inst);
            continue;
        case IR::Opcode::InverseBallot:
            // The invocation's bit of a mask: the same in every active invocation only when the
            // mask is made of EXEC and wave-wide values.
            in_inverse_ballot = true;
            break;
        case IR::Opcode::Ballot: {
            const IR::Value arg{inst->Arg(0)};
            const bool exec_ballot =
                !arg.IsImmediate() && arg.Inst()->GetOpcode() == IR::Opcode::GetExec;
            if (in_inverse_ballot && !exec_ballot) {
                return TraceDivergence(inst);
            }
            continue;
        }
        default:
            if (inst->MayHaveSideEffects()) {
                // Atomics and other memory operations return per-invocation values.
                return TraceDivergence(inst);
            }
            break;
        }
        push_args(inst);
    }
    return false;
}

static std::vector<IR::Block*> FindUniformBlocks(const IR::Program& program,
                                               const RuntimeInfo& runtime) {
    using Type = IR::AbstractSyntaxNode::Type;
    static const bool uniform_loops_enabled = Common::PerfFeatureEnabled(31);

    // FIX-020: loops whose exit conditions take the same path in every invocation, by merge block.
    // Ballots and lane reads in them used to be left as 32-wide operations on NVIDIA, so a wave64
    // scan in a loop (GT Sport's grass prefix sums, cs 0x766d0f18) read lane 63 from the wrong
    // subgroup and produced wrong offsets.
    std::unordered_set<const IR::Block*> divergent_loops;
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        if (node.type != Type::Repeat && node.type != Type::Break) {
            continue;
        }
        const bool repeat = node.type == Type::Repeat;
        if (g_wave64_trace) {
            *g_wave64_trace += repeat ? "Repeat:" : "Break:";
        }
        const IR::U1 cond = repeat ? node.data.repeat.cond : node.data.break_node.cond;
        const bool divergent = !uniform_loops_enabled || IsDivergentLoopCondition(cond, runtime);
        if (g_wave64_trace) {
            *g_wave64_trace += divergent ? " divergent;" : " uniform;";
        }
        if (divergent) {
            divergent_loops.insert(repeat ? node.data.repeat.merge : node.data.break_node.merge);
        }
    }

    struct ConditionalScope {
        const IR::Block* merge;
        bool divergent;
    };

    std::vector<IR::Block*> blocks;
    std::vector<ConditionalScope> conditionals;
    std::vector<const IR::Block*> loops;
    u32 divergence_depth{};
    u32 divergent_loop_depth{};
    for (const IR::AbstractSyntaxNode& node : program.syntax_list) {
        switch (node.type) {
        case Type::If: {
            // FIX-020: an if on wave-wide values (lane reads, SCC) is uniform too; the older
            // search followed a lane read's per-lane input and called it divergent.
            if (g_wave64_trace) {
                *g_wave64_trace += "If:";
            }
            const bool divergent = uniform_loops_enabled
                                       ? IsDivergentLoopCondition(node.data.if_node.cond, runtime)
                                       : IsDivergentCondition(node.data.if_node.cond);
            if (g_wave64_trace) {
                *g_wave64_trace += divergent ? " divergent;" : " uniform;";
            }
            conditionals.push_back({node.data.if_node.merge, divergent});
            divergence_depth += static_cast<u32>(divergent);
            break;
        }
        case Type::EndIf:
            ASSERT(!conditionals.empty() && conditionals.back().merge == node.data.end_if.merge);
            divergence_depth -= static_cast<u32>(conditionals.back().divergent);
            conditionals.pop_back();
            break;
        case Type::Loop:
            loops.push_back(node.data.loop.merge);
            divergent_loop_depth += divergent_loops.contains(node.data.loop.merge);
            break;
        case Type::Repeat:
            if (loops.empty() || loops.back() != node.data.repeat.merge) {
                return {};
            }
            divergent_loop_depth -= divergent_loops.contains(loops.back());
            loops.pop_back();
            break;
        case Type::Block:
            if (divergence_depth == 0 && divergent_loop_depth == 0) {
                blocks.push_back(node.data.block);
            }
            break;
        default:
            break;
        }
    }
    if (!conditionals.empty() || !loops.empty()) {
        return {};
    }
    return blocks;
}

static IR::Inst* FindBallotForMaskedBitCount(const IR::Inst& mbcnt) {
    IR::Value value = mbcnt.Arg(0);
    if (value.IsImmediate()) {
        return nullptr;
    }
    IR::Inst* inst = value.Inst();
    if (inst->GetOpcode() != IR::Opcode::CompositeExtractU32x2 || inst->Arg(0).IsImmediate()) {
        return nullptr;
    }
    inst = inst->Arg(0).Inst();
    if (inst->GetOpcode() != IR::Opcode::UnpackUint2x32 || inst->Arg(0).IsImmediate()) {
        return nullptr;
    }
    inst = inst->Arg(0).Inst();
    return inst->GetOpcode() == IR::Opcode::Ballot ? inst : nullptr;
}

// FIX-GT7-028: a read of a lane past the host's subgroup (subgroupBroadcast of lane 32 on NVIDIA) has
// an undefined result, and its lowering needs a barrier, which control flow that differs between
// invocations cannot hold. GT7's occlusion pyramid reads lane 32 inside ifs on the invocation's
// position, when it makes a level from an 8x8 tile: the second row of the 4x2 level was lost, so the
// two coarsest levels said the whole screen was covered and buildings in view were eliminated (black,
// with the depth of their occluder). Those ifs are GCN EXEC scopes around arithmetic: without the
// branch every invocation runs the arithmetic, which nothing else sees, the ifs directly inside test
// the removed condition as well, and the lane read is in uniform control flow. A branch whose own
// blocks ask which invocations are active is kept: without it, more of them are.
// RYGT_LANE_READ_UNBRANCH=0: as before.
static bool ReadsActiveLanes(const IR::Inst& inst) {
    switch (inst.GetOpcode()) {
    case IR::Opcode::ReadFirstLane:
    case IR::Opcode::Ballot:
    case IR::Opcode::GroupAny:
    case IR::Opcode::GetExec:
        return true;
    default:
        return false;
    }
}

static void UnbranchUpperLaneReads(IR::Program& program, const RuntimeInfo& runtime,
                                   const Profile& profile) {
    static const bool enabled = [] {
        const char* env = std::getenv("RYGT_LANE_READ_UNBRANCH");
        return !(env && env[0] == '0');
    }();
    if (!enabled) {
        return;
    }
    using Type = IR::AbstractSyntaxNode::Type;
    auto& list = program.syntax_list;
    // The ifs that differ between invocations around a read of an upper lane.
    std::vector<size_t> open;
    std::unordered_set<size_t> scopes;
    u32 loops = 0;
    for (size_t i = 0; i < list.size(); ++i) {
        switch (list[i].type) {
        case Type::If:
            open.push_back(i);
            break;
        case Type::EndIf:
            open.pop_back();
            break;
        case Type::Loop:
            ++loops;
            break;
        case Type::Repeat:
            --loops;
            break;
        case Type::Block:
            if (loops == 0 && !open.empty() &&
                std::ranges::any_of(list[i].data.block->Instructions(), [&](const IR::Inst& inst) {
                    return inst.GetOpcode() == IR::Opcode::ReadLane && inst.Arg(1).IsImmediate() &&
                           inst.Arg(1).U32() >= profile.subgroup_size;
                })) {
                for (const size_t scope : open) {
                    if (IsDivergentLoopCondition(list[scope].data.if_node.cond, runtime)) {
                        scopes.insert(scope);
                    }
                }
            }
            break;
        default:
            break;
        }
    }
    if (scopes.empty()) {
        return;
    }
    // Without their branch they must change nothing: no side effects in their own blocks and no phi
    // after them. The ifs directly inside get the conditions that were around them.
    open.clear();
    const auto refuse = [&](std::string_view why) {
        LOG_WARNING(Render_Recompiler,
                    "FIX-GT7-028: shader {:#x}: upper lane read left in its branch: {}",
                    program.info.pgm_hash, why);
    };
    struct InnerIf {
        size_t node;
        std::vector<size_t> around;
    };
    std::vector<InnerIf> inner_ifs;
    for (size_t i = 0; i < list.size(); ++i) {
        const bool inside =
            std::ranges::any_of(open, [&](size_t scope) { return scopes.contains(scope); });
        switch (list[i].type) {
        case Type::If:
            if (scopes.contains(i)) {
                if (std::ranges::any_of(list[i].data.if_node.merge->Instructions(),
                                        [](const IR::Inst& inst) { return IR::IsPhi(inst); })) {
                    return refuse("a value leaves the branch");
                }
            } else if (!open.empty() && scopes.contains(open.back())) {
                if (i == 0 || list[i - 1].type != Type::Block) {
                    return refuse("an inner if has no block before it");
                }
                InnerIf inner{i, {}};
                std::ranges::copy_if(open, std::back_inserter(inner.around),
                                     [&](size_t scope) { return scopes.contains(scope); });
                inner_ifs.push_back(std::move(inner));
            }
            open.push_back(i);
            break;
        case Type::EndIf:
            open.pop_back();
            break;
        case Type::Block:
            if (!open.empty() && scopes.contains(open.back()) &&
                std::ranges::any_of(list[i].data.block->Instructions(), [](const IR::Inst& inst) {
                    return inst.MayHaveSideEffects() &&
                           inst.GetOpcode() != IR::Opcode::ConditionRef;
                })) {
                return refuse("the branch has side effects");
            }
            if (!open.empty() && scopes.contains(open.back()) &&
                std::ranges::any_of(list[i].data.block->Instructions(), ReadsActiveLanes)) {
                return refuse("the branch reads which lanes are active");
            }
            break;
        case Type::Loop:
        case Type::Repeat:
        case Type::Break:
        case Type::Return:
        case Type::Unreachable:
            if (inside) {
                return refuse("the branch has other control flow");
            }
            break;
        default:
            break;
        }
    }
    for (const InnerIf& inner : inner_ifs) {
        IR::Block* const before = list[inner.node - 1].data.block;
        IR::IREmitter ir{*before, before->end()};
        IR::U1 condition = list[inner.node].data.if_node.cond;
        for (const size_t scope : inner.around) {
            condition = ir.LogicalAnd(list[scope].data.if_node.cond, condition);
        }
        list[inner.node].data.if_node.cond = ir.ConditionRef(condition);
    }
    for (const size_t scope : scopes) {
        list[scope].data.if_node.cond = IR::U1{IR::Value{true}};
    }
    LOG_WARNING(Render_Recompiler,
                "FIX-GT7-028: shader {:#x}: {} branches removed for an upper lane read, {} inner ifs "
                "test their conditions",
                program.info.pgm_hash, scopes.size(), inner_ifs.size());
}

void LowerWave64BallotPass(IR::Program& program, const RuntimeInfo& runtime_info,
                           const Profile& profile) {
    if (program.info.hw_stage != HwStage::Compute || profile.subgroup_size == 64) {
        return;
    }

    const auto [size_x, size_y, size_z] = runtime_info.hw.cs.workgroup_size;
    const u32 num_threads = size_x * size_y * size_z;
    if (num_threads <= 32) {
        return;
    }

    UnbranchUpperLaneReads(program, runtime_info, profile);

    std::vector<IR::Inst*> worklist;
    const auto uniform_blocks = FindUniformBlocks(program, runtime_info);
    for (IR::Block* block : program.blocks) {
        const bool is_uniform = std::ranges::contains(uniform_blocks, block);
        const auto push_worklist = [&](IR::Inst& inst) {
            if (is_uniform) {
                worklist.push_back(&inst);
            } else {
                LOG_WARNING(Render_Recompiler, "{} instruction in non uniform control flow",
                            inst.GetOpcode());
                if (g_wave64_trace) {
                    *g_wave64_trace +=
                        fmt::format(" NOT LOWERED: {};", magic_enum::enum_name(inst.GetOpcode()));
                }
            }
        };
        for (IR::Inst& inst : block->Instructions()) {
            if (inst.GetOpcode() == IR::Opcode::ReadLane && inst.Arg(1).IsImmediate()) {
                push_worklist(inst);
            } else if (inst.GetOpcode() == IR::Opcode::Ballot) {
                const auto is_unpack = [](const IR::Use& use) {
                    return use.user->GetOpcode() == IR::Opcode::UnpackUint2x32;
                };
                if (std::ranges::any_of(inst.Uses(), is_unpack)) {
                    push_worklist(inst);
                }
            } else if (inst.GetOpcode() == IR::Opcode::MaskedBitCount32) {
                IR::Inst* const ballot = FindBallotForMaskedBitCount(inst);
                if (ballot == nullptr ||
                    std::ranges::contains(uniform_blocks, ballot->GetParent())) {
                    worklist.push_back(&inst);
                }
            }
        }
    }
    if (worklist.empty()) {
        return;
    }

    const u32 scratch_base = Common::AlignUp(runtime_info.hw.cs.shared_memory_size, sizeof(u64));
    const u32 scratch_size =
        (Common::AlignUp(num_threads, 64) / profile.subgroup_size) * sizeof(u32);
    program.info.shared_memory_scratch_size =
        scratch_base + scratch_size - runtime_info.hw.cs.shared_memory_size;

    for (IR::Inst* inst : worklist) {
        LOG_INFO(Render_Recompiler, "Lowering {} instruction for wave64", inst->GetOpcode());
        IR::IREmitter ir{*inst->GetParent(), IR::Block::InstructionList::s_iterator_to(*inst)};
        const IR::U32 invocation_index = ir.GetAttributeU32(IR::Attribute::LocalInvocationIndex);
        const IR::U32 subgroup_id = ir.ShiftRightLogical(invocation_index, ir.Imm32(5));
        if (inst->GetOpcode() == IR::Opcode::Ballot) {
            const IR::U32 mask_low =
                IR::U32{ir.CompositeExtract(ir.UnpackUint2x32(ir.Ballot(IR::U1{inst->Arg(0)})), 0)};
            const IR::U32 offset =
                ir.IAdd(ir.Imm32(scratch_base), ir.ShiftLeftLogical(subgroup_id, ir.Imm32(2u)));
            ir.WriteShared(32, mask_low, offset);
            ir.Barrier();
            // FIX-188: 8-byte aligned (#65); a 64-bit access dropped those bits already.
            const IR::U64 mask =
                IR::U64{ir.LoadShared(64, false, ir.BitwiseAnd(offset, ir.Imm32(~7u)))};
            ir.Barrier();
            inst->ReplaceUsesWithAndRemove(mask);
        } else if (inst->GetOpcode() == IR::Opcode::ReadLane) {
            const IR::U32 lane32 = ir.BitwiseAnd(IR::U32{inst->Arg(1)}, ir.Imm32(31));
            const IR::U32 half = ir.ShiftRightLogical(IR::U32{inst->Arg(1)}, ir.Imm32(5u));
            const IR::U32 offset =
                ir.IAdd(ir.Imm32(scratch_base), ir.ShiftLeftLogical(subgroup_id, ir.Imm32(2u)));
            ir.WriteShared(32, ir.ReadLane(IR::U32{inst->Arg(0)}, lane32), offset);
            ir.Barrier();
            const IR::U32 value =
                IR::U32{ir.LoadShared(32, false,
                                      ir.IAdd(ir.BitwiseAnd(offset, ir.Imm32(~7u)),
                                              ir.ShiftLeftLogical(half, ir.Imm32(2u))))};
            ir.Barrier();
            inst->ReplaceUsesWithAndRemove(value);
        } else if (inst->GetOpcode() == IR::Opcode::MaskedBitCount32) {
            const IR::U32 subgroup_invocation_id = ir.BitwiseAnd(invocation_index, ir.Imm32(63));
            const IR::U64 mask = ir.ISub(
                ir.ShiftLeftLogical(ir.Imm64(u64{1}), subgroup_invocation_id), ir.Imm64(u64{1}));
            const IR::U32 thread_mask{
                ir.CompositeExtract(ir.UnpackUint2x32(mask), inst->Arg(2).U1() ? 1u : 0u)};
            const IR::U32 masked_value{
                ir.BitCount(ir.BitwiseAnd(IR::U32{inst->Arg(0)}, thread_mask))};
            inst->ReplaceUsesWithAndRemove(ir.IAdd(masked_value, IR::U32{inst->Arg(1)}));
        }
    }
}

} // namespace Shader::Optimization
