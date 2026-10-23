// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <algorithm>
#include "common/assert.h"
#include "common/logging/log.h"
#include "shader_recompiler/frontend/control_flow_graph.h"
#include "shader_recompiler/ir/condition.h"

namespace Shader::Gcn {

struct Compare {
    bool operator()(const Block& lhs, u32 rhs) const noexcept {
        return lhs.begin < rhs;
    }

    bool operator()(u32 lhs, const Block& rhs) const noexcept {
        return lhs < rhs.begin;
    }

    bool operator()(const Block& lhs, const Block& rhs) const noexcept {
        return lhs.begin < rhs.begin;
    }
};

static IR::Condition MakeCondition(const GcnInst& inst) {
    if (inst.IsCmpx()) {
        return IR::Condition::Execnz;
    }

    switch (inst.opcode) {
    case Opcode::S_CBRANCH_SCC0:
        return IR::Condition::Scc0;
    case Opcode::S_CBRANCH_SCC1:
        return IR::Condition::Scc1;
    case Opcode::S_CBRANCH_VCCZ:
        return IR::Condition::Vccz;
    case Opcode::S_CBRANCH_VCCNZ:
        return IR::Condition::Vccnz;
    case Opcode::S_CBRANCH_EXECZ:
        return IR::Condition::Execz;
    case Opcode::S_CBRANCH_EXECNZ:
        return IR::Condition::Execnz;
    default:
        return IR::Condition::True;
    }
}

static bool IgnoresExecMask(const GcnInst& inst) {
    // EXEC mask does not affect scalar instructions or branches.
    switch (inst.category) {
    case InstCategory::ScalarALU:
    case InstCategory::ScalarMemory:
    case InstCategory::FlowControl:
        return true;
    default:
        break;
    }
    // Read/Write Lane instructions are not affected either.
    switch (inst.opcode) {
    case Opcode::V_READLANE_B32:
    case Opcode::V_WRITELANE_B32:
    case Opcode::V_READFIRSTLANE_B32:
        return true;
    default:
        break;
    }
    return false;
}

static bool ChangesExec(const GcnInst& inst) {
    switch (inst.opcode) {
    case Opcode::S_AND_SAVEEXEC_B64:
    case Opcode::S_OR_SAVEEXEC_B64:
    case Opcode::S_XOR_SAVEEXEC_B64:
    case Opcode::S_ANDN2_SAVEEXEC_B64:
    case Opcode::S_ORN2_SAVEEXEC_B64:
    case Opcode::S_NAND_SAVEEXEC_B64:
    case Opcode::S_NOR_SAVEEXEC_B64:
    case Opcode::S_XNOR_SAVEEXEC_B64:
        return true;
    default:
        break;
    }
    if (inst.IsCmpx()) {
        return true;
    }
    for (u32 i = 0; i < inst.dst_count; ++i) {
        if (inst.dst[i].field == OperandField::ExecLo || inst.dst[i].field == OperandField::ExecHi) {
            return true;
        }
    }
    return false;
}

static bool IsWholeWaveEnable(const GcnInst& inst) {
    if (inst.opcode == Opcode::S_ORN2_SAVEEXEC_B64) {
        // exec = src | ~exec: every lane when src is exec.
        return inst.src[0].field == OperandField::ExecLo;
    }
    if (inst.opcode == Opcode::S_OR_SAVEEXEC_B64) {
        return inst.src[0].field == OperandField::SignedConstIntNeg &&
               inst.src[0].code == static_cast<u32>(OperandField::SignedConstIntNeg); // -1
    }
    return false;
}

// FIX-190: invocations stand for lanes, and s_cbranch_execz took an invocation whose lane is off
// past the branch body. GCN branches only when every lane is off, so a body that turns every lane
// on to sum a value over the wave runs on all of them: GT7's Hi-Z test (cs 0xc959ee53) sums its
// per-lane votes this way, and without the inactive invocations the sum missed lanes, visible
// terrain was marked occluded and mountains floated. Such a branch is dropped: every invocation
// goes through the body, the part before the enable and after the restore in exec scopes. Only
// bodies that stay one block until a later s_cbranch_execz to the same target qualify, so the
// invocations whose lane is off leave there and every other path keeps its branch.
std::vector<WholeWaveRegion> FindWholeWaveRegions(std::span<const GcnInst> list) {
    std::vector<u32> pcs(list.size() + 1);
    for (size_t i = 0; i < list.size(); ++i) {
        pcs[i + 1] = pcs[i] + list[i].length;
    }
    std::vector<u32> targets;
    for (size_t i = 0; i < list.size(); ++i) {
        const auto& inst = list[i];
        if (inst.opcode == Opcode::S_SETPC_B64 || inst.opcode == Opcode::S_SWAPPC_B64) {
            return {}; // targets not known here
        }
        if (inst.IsConditionalBranch() || inst.IsUnconditionalBranch()) {
            targets.push_back(inst.BranchTarget(pcs[i]));
        }
    }
    const auto is_branch = [](const GcnInst& inst) {
        return inst.IsConditionalBranch() || inst.IsUnconditionalBranch() ||
               inst.opcode == Opcode::S_ENDPGM;
    };
    const auto opens_scope = [](const GcnInst& inst) {
        return inst.opcode == Opcode::S_AND_SAVEEXEC_B64 || inst.IsCmpx();
    };

    std::vector<WholeWaveRegion> regions;
    for (size_t branch = 1; branch < list.size(); ++branch) {
        const auto& inst = list[branch];
        if (inst.opcode != Opcode::S_CBRANCH_EXECZ || !opens_scope(list[branch - 1])) {
            continue;
        }
        const u32 target = inst.BranchTarget(pcs[branch]);
        if (target <= pcs[branch]) {
            continue;
        }
        // The body up to the enable and on to the restore: no branch and no other exec change.
        size_t index = branch + 1;
        while (index < list.size() && !is_branch(list[index]) && !ChangesExec(list[index])) {
            ++index;
        }
        if (index == list.size() || !IsWholeWaveEnable(list[index])) {
            continue;
        }
        const size_t enable = index;
        const auto& saved = list[enable].dst[0];
        ++index;
        while (index < list.size() && !is_branch(list[index]) && !ChangesExec(list[index])) {
            ++index;
        }
        if (index == list.size() || list[index].opcode != Opcode::S_MOV_B64 ||
            list[index].dst[0].field != OperandField::ExecLo ||
            list[index].src[0].field != saved.field || list[index].src[0].code != saved.code) {
            continue;
        }
        const size_t restore = index;
        // Exec-masked instructions after the restore are scoped by it, unless another scope opens
        // after them (that opening would end the restore's scope unwrapped).
        bool masked = false;
        bool reopened = false;
        ++index;
        while (index < list.size() && !is_branch(list[index])) {
            if (opens_scope(list[index])) {
                reopened = true;
            } else if (ChangesExec(list[index])) {
                break;
            } else if (!IgnoresExecMask(list[index]) && !reopened) {
                masked = true;
            }
            ++index;
        }
        if (index == list.size() || list[index].opcode != Opcode::S_CBRANCH_EXECZ ||
            list[index].BranchTarget(pcs[index]) != target || (masked && reopened)) {
            continue;
        }
        // One block from the scope's opening to that branch: no branch lands inside.
        const u32 first = pcs[branch];
        const u32 last = pcs[index];
        if (std::ranges::any_of(targets, [&](u32 t) { return t > first && t <= last; })) {
            continue;
        }
        regions.push_back({branch, enable, restore});
    }
    return regions;
}

static std::optional<u32> ResolveSetPcTarget(std::span<const GcnInst> list, u32 setpc_index,
                                             std::span<const u32> pc_map) {
    if (setpc_index < 3) {
        return std::nullopt;
    }

    const auto& getpc = list[setpc_index - 3];
    const auto& arith = list[setpc_index - 2];
    const auto& setpc = list[setpc_index];

    if (getpc.opcode != Opcode::S_GETPC_B64 ||
        !(arith.opcode == Opcode::S_ADD_U32 || arith.opcode == Opcode::S_SUB_U32) ||
        setpc.opcode != Opcode::S_SETPC_B64)
        return std::nullopt;

    if (getpc.dst[0].code != setpc.src[0].code || arith.dst[0].code != setpc.src[0].code)
        return std::nullopt;

    if (arith.src_count < 2 || arith.src[1].field != OperandField::LiteralConst)
        return std::nullopt;

    const u32 imm = arith.src[1].code;

    const s32 signed_offset =
        (arith.opcode == Opcode::S_ADD_U32) ? static_cast<s32>(imm) : -static_cast<s32>(imm);

    const u32 base_pc = pc_map[setpc_index - 3] + getpc.length;

    const u32 result_pc = static_cast<u32>(static_cast<s32>(base_pc) + signed_offset);
    LOG_DEBUG(Render_Recompiler, "SetPC target: {} + {} = {}", base_pc, signed_offset, result_pc);
    return result_pc & ~0x3u;
}

static constexpr size_t LabelReserveSize = 32;

CFG::CFG(Common::ObjectPool<Block>& block_pool_, std::span<const GcnInst> inst_list_)
    : block_pool{block_pool_}, inst_list{inst_list_} {
    index_to_pc.resize(inst_list.size() + 1);
    labels.reserve(LabelReserveSize);
    EmitLabels();
    EmitBlocks();
    LinkBlocks();
    SplitDivergenceScopes();
    RemoveUnreachableBlocks();
}

void CFG::EmitLabels() {
    // Always set a label at entry point.
    u32 pc = 0;
    AddLabel(pc);

    // FIX-190: branches around whole-wave regions are dropped, so they end no block.
    const auto regions = FindWholeWaveRegions(inst_list);
    const auto dropped = [&](u32 index) {
        return std::ranges::any_of(regions, [index](const auto& r) { return r.branch == index; });
    };

    // Iterate instruction list and add labels to branch targets.
    for (u32 i = 0; i < inst_list.size(); i++) {
        index_to_pc[i] = pc;
        const GcnInst inst = inst_list[i];
        if (inst.IsConditionalBranch() && dropped(i)) {
            LOG_DEBUG(Render_Recompiler, "FIX-190: branch at {:#x} kept for every invocation", pc);
        } else if (inst.IsUnconditionalBranch()) {
            u32 target = inst.BranchTarget(pc);
            if (inst.opcode == Opcode::S_SETPC_B64) {
                if (auto t = ResolveSetPcTarget(inst_list, i, index_to_pc)) {
                    target = *t;
                } else {
                    ASSERT_MSG(
                        false,
                        "S_SETPC_B64 without a resolvable offset at PC {:#x} (Index {}): Involved "
                        "instructions not recognized or invalid pattern",
                        pc, i);
                }
            }
            AddLabel(target);
            // Emit this label so that the block ends with the branching instruction
            AddLabel(pc + inst.length);
        } else if (inst.IsConditionalBranch()) {
            const u32 true_label = inst.BranchTarget(pc);
            const u32 false_label = pc + inst.length;
            if (true_label != false_label) {
                AddLabel(true_label);
                AddLabel(false_label);
            }
        } else if (inst.opcode == Opcode::S_ENDPGM) {
            const u32 next_label = pc + inst.length;
            AddLabel(next_label);
        }

        pc += inst.length;
    }
    index_to_pc[inst_list.size()] = pc;

    // Sort labels to make sure block insertion is correct.
    std::ranges::sort(labels);
}

void CFG::SplitDivergenceScopes() {
    // FIX-190: in a whole-wave region the dropped branch closes nothing, the enable closes the
    // body's scope (every invocation runs the region) and the restore opens the next one.
    const auto regions = FindWholeWaveRegions(inst_list);
    const auto in_regions = [&](size_t WholeWaveRegion::* member, size_t index) {
        return std::ranges::any_of(regions, [&](const auto& r) { return r.*member == index; });
    };
    const auto is_open_scope = [&](const GcnInst& inst, size_t index) {
        // An open scope instruction is an instruction that modifies EXEC
        // but also saves the previous value to restore later. This indicates
        // we are entering a scope.
        return inst.opcode == Opcode::S_AND_SAVEEXEC_B64 ||
               // While this instruction does not save EXEC it is often used paired
               // with SAVEEXEC to mask the threads that didn't pass the condition
               // of initial branch.
               (inst.opcode == Opcode::S_ANDN2_B64 && inst.dst[0].field == OperandField::ExecLo) ||
               inst.IsCmpx() || in_regions(&WholeWaveRegion::restore, index);
    };
    const auto is_close_scope = [&](const GcnInst& inst, size_t index) {
        if (in_regions(&WholeWaveRegion::branch, index)) {
            return false;
        }
        // Closing an EXEC scope can be either a branch instruction
        // (typical case when S_AND_SAVEEXEC_B64 is right before a branch)
        // or by a move instruction to EXEC that restores the backup.
        return (inst.opcode == Opcode::S_MOV_B64 && inst.dst[0].field == OperandField::ExecLo) ||
               // Sometimes compiler might insert instructions between the SAVEEXEC and the branch.
               // Those instructions need to be wrapped in the condition as well so allow branch
               // as end scope instruction.
               inst.opcode == Opcode::S_CBRANCH_EXECZ || inst.opcode == Opcode::S_ENDPGM ||
               (inst.opcode == Opcode::S_ANDN2_B64 && inst.dst[0].field == OperandField::ExecLo) ||
               in_regions(&WholeWaveRegion::enable, index);
    };

    for (auto blk = blocks.begin(); blk != blocks.end(); blk++) {
        auto next_blk = std::next(blk);
        s32 curr_begin = -1;
        for (size_t index = blk->begin_index; index <= blk->end_index; index++) {
            const auto& inst = inst_list[index];
            const bool is_close = is_close_scope(inst, index);
            if ((is_close || index == blk->end_index) && curr_begin != -1) {
                // If there are no instructions inside scope don't do anything. The closing
                // instruction can open the next scope (an else after an empty if); keep it.
                // FIX-018, from upstream shadPS4 c34ca21f.
                if (index - curr_begin == 1 && is_close) {
                    curr_begin = is_open_scope(inst, index) ? static_cast<s32>(index) : -1;
                    continue;
                }
                // If all instructions in the scope ignore exec masking, we shouldn't insert a
                // scope.
                const auto start = inst_list.begin() + curr_begin + 1;
                if (!std::ranges::all_of(start, inst_list.begin() + index + !is_close,
                                         IgnoresExecMask)) {
                    // Determine the first instruction affected by the exec mask.
                    do {
                        ++curr_begin;
                    } while (IgnoresExecMask(inst_list[curr_begin]));

                    // Determine the last instruction affected by the exec mask.
                    s32 curr_end = index;
                    while (IgnoresExecMask(inst_list[curr_end])) {
                        --curr_end;
                    }

                    // Create a new block for the divergence scope.
                    Block* block = block_pool.Create();
                    block->begin = index_to_pc[curr_begin];
                    block->end = index_to_pc[curr_end];
                    block->begin_index = curr_begin;
                    block->end_index = curr_end;
                    block->end_inst = inst_list[curr_end];
                    block->num_predecessors = 1;
                    blocks.insert_before(next_blk, *block);

                    // If we are inside the parent block, make an epilogue block and jump to it.
                    if (curr_end != blk->end_index) {
                        Block* epi_block = block_pool.Create();
                        epi_block->begin = index_to_pc[curr_end + 1];
                        epi_block->end = blk->end;
                        epi_block->begin_index = curr_end + 1;
                        epi_block->end_index = blk->end_index;
                        epi_block->end_inst = blk->end_inst;
                        epi_block->cond = blk->cond;
                        epi_block->end_class = blk->end_class;
                        epi_block->branch_true = blk->branch_true;
                        epi_block->branch_false = blk->branch_false;
                        epi_block->num_predecessors = 2;
                        blocks.insert_before(next_blk, *epi_block);

                        // Have divergence block always jump to epilogue block.
                        block->cond = IR::Condition::True;
                        block->branch_true = epi_block;
                        block->branch_false = nullptr;

                        // If the parent block fails to enter divergence block make it jump to
                        // epilogue too
                        blk->branch_false = epi_block;
                    } else {
                        // No epilogue block is needed since the divergence block
                        // also ends the parent block. Inherit the end condition.
                        auto& parent_blk = *blk;
                        ASSERT(blk->cond == IR::Condition::True && blk->branch_true);
                        block->cond = IR::Condition::True;
                        block->branch_true = blk->branch_true;
                        block->branch_false = nullptr;

                        // If the parent block didn't enter the divergence scope
                        // have it jump directly to the next one
                        blk->branch_false = blk->branch_true;
                        blk->branch_true->num_predecessors++;
                    }

                    // Shrink parent block to end right before curr_begin
                    // and make it jump to divergence block
                    --curr_begin;
                    blk->end = index_to_pc[curr_begin];
                    blk->end_index = curr_begin;
                    blk->end_inst = inst_list[curr_begin];
                    blk->cond = IR::Condition::Execnz;
                    blk->end_class = EndClass::Branch;
                    blk->branch_true = block;
                }
                // Reset scope begin.
                curr_begin = -1;
            }
            // Mark a potential start of an exec scope.
            if (is_open_scope(inst, index)) {
                curr_begin = index;
            }
        }
    }
}

void CFG::EmitBlocks() {
    for (auto it = labels.cbegin(); it != labels.cend(); ++it) {
        const Label start = *it;
        const auto next_it = std::next(it);
        const bool is_last = (next_it == labels.cend());
        if (is_last) {
            // Last label is special.
            return;
        }
        // The end label is the start instruction of next block.
        // The end instruction of this block is the previous one.
        const Label end = *next_it;
        const size_t end_index = GetIndex(end) - 1;
        const auto& end_inst = inst_list[end_index];

        // Insert block between the labels using the last instruction
        // as an indicator for branching type.
        Block* block = block_pool.Create();
        block->begin = start;
        block->end = end;
        block->begin_index = GetIndex(start);
        block->end_index = end_index;
        block->end_inst = end_inst;
        block->cond = MakeCondition(end_inst);
        blocks.insert(*block);
    }
}

void CFG::LinkBlocks() {
    const auto get_block = [this](u32 address) {
        auto it = blocks.find(address, Compare{});
        ASSERT_MSG(it != blocks.cend() && it->begin == address);
        return &*it;
    };

    for (auto it = blocks.begin(); it != blocks.end(); it++) {
        auto& block = *it;
        const auto end_inst{block.end_inst};

        // If the block doesn't end with a branch we simply
        // need to link with the next block.
        if (!end_inst.IsTerminateInstruction()) {
            auto* next_block = get_block(block.end);
            block.branch_true = next_block;
            block.end_class = EndClass::Branch;
            next_block->num_predecessors++;
            continue;
        }

        // Find the branch targets from the instruction and link the blocks.
        // Note: Block end address is one instruction after end_inst.
        const u32 branch_pc = block.end - end_inst.length;
        u32 target_pc = 0;
        if (end_inst.opcode == Opcode::S_SETPC_B64) {
            auto tgt = ResolveSetPcTarget(inst_list, block.end_index, index_to_pc);
            ASSERT_MSG(tgt,
                       "S_SETPC_B64 without a resolvable offset at PC {:#x} (Index {}): Involved "
                       "instructions not recognized or invalid pattern",
                       branch_pc, block.end_index);
            target_pc = *tgt;
        } else {
            target_pc = end_inst.BranchTarget(branch_pc);
        }

        if (end_inst.IsUnconditionalBranch()) {
            auto* target_block = get_block(target_pc);
            block.branch_true = target_block;
            block.end_class = EndClass::Branch;
            target_block->num_predecessors++;
        } else if (end_inst.IsConditionalBranch()) {
            auto* end_block = get_block(block.end);
            block.end_class = EndClass::Branch;
            if (target_pc != block.end) {
                auto* target_block = get_block(target_pc);
                block.branch_true = target_block;
                block.branch_false = end_block;
                target_block->num_predecessors++;
            } else {
                block.branch_true = end_block;
                block.cond = IR::Condition::True;
            }
            end_block->num_predecessors++;
        } else if (end_inst.opcode == Opcode::S_ENDPGM) {
            block.end_class = EndClass::Exit;
        } else {
            UNREACHABLE();
        }
    }
}

void CFG::RemoveUnreachableBlocks() {
    for (auto it = std::next(blocks.begin()); it != blocks.end();) {
        if (it->num_predecessors == 0) {
            LOG_WARNING(Render_Recompiler, "Removing unreachable block begin={:#x}", it->begin);
            it = blocks.erase(it);
        } else {
            it++;
        }
    }
}

std::string CFG::Dot() const {
    int node_uid{0};

    const auto name_of = [](const Block& block) { return fmt::format("\"{:#x}\"", block.begin); };

    std::string dot{"digraph shader {\n"};
    dot += fmt::format("\tsubgraph cluster_{} {{\n", 0);
    dot += fmt::format("\t\tnode [style=filled];\n");
    for (const Block& block : blocks) {
        const std::string name{name_of(block)};
        const auto add_branch = [&](Block* branch, bool add_label) {
            dot += fmt::format("\t\t{}->{}", name, name_of(*branch));
            if (add_label && block.cond != IR::Condition::True &&
                block.cond != IR::Condition::False) {
                dot += fmt::format(" [label=\"{}\"]", block.cond);
            }
            dot += '\n';
        };
        dot += fmt::format("\t\t{};\n", name);
        switch (block.end_class) {
        case EndClass::Branch:
            if (block.cond != IR::Condition::False) {
                add_branch(block.branch_true, true);
            }
            if (block.cond != IR::Condition::True) {
                add_branch(block.branch_false, false);
            }
            break;
        case EndClass::Exit:
            dot += fmt::format("\t\t{}->N{};\n", name, node_uid);
            dot +=
                fmt::format("\t\tN{} [label=\"Exit\"][shape=square][style=stripped];\n", node_uid);
            ++node_uid;
            break;
        }
    }
    dot += "\t\tlabel = \"main\";\n\t}\n";
    if (blocks.empty()) {
        dot += "Start;\n";
    } else {
        dot += fmt::format("\tStart -> {};\n", name_of(*blocks.begin()));
    }
    dot += fmt::format("\tStart [shape=diamond];\n");
    dot += "}\n";
    return dot;
}

} // namespace Shader::Gcn
