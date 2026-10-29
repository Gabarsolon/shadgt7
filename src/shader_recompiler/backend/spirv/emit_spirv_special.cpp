// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "shader_recompiler/backend/spirv/emit_spirv_instructions.h"
#include "shader_recompiler/backend/spirv/spirv_emit_context.h"
#include "shader_recompiler/ir/debug_print.h"
#include "shader_recompiler/ir/microinstruction.h"

namespace Shader::Backend::SPIRV {

// FIX-227: GT7's car hull shaders (one input control point per patch, 16 patches a VGT group)
// read the LS outputs of the other patches of their group from LDS. A Vulkan hull shader only
// sees its own patch's inputs; reads past them are undefined, and on NVIDIA some patches got wrong
// positions for their occlusion test and were culled in view (black patches on the car in the
// race start's close-ups). The renderer runs such a draw twice: first with the writer bit in the
// push data, each patch storing its inputs into a scratch buffer and drawing nothing; after a
// barrier, the real draw reads its inputs from the scratch buffer as GCN reads LDS.
static void EmitLdsEmulationPrologue(EmitContext& ctx) {
    const auto& tcs = ctx.runtime_info.sw.tcs;
    const u32 num_cp = std::max(tcs.num_input_control_points, 1u);
    const u32 num_attrs = (tcs.ls_stride + 15) / 16;
    ASSERT_MSG(num_attrs <= 4, "FIX-227: LS stride {} past the scratch's 64 bytes a vertex",
               tcs.ls_stride);
    const Id push_ptr = ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1]);
    const auto push_word = [&](u32 index) {
        return ctx.OpLoad(ctx.U32[1],
                          ctx.OpAccessChain(push_ptr, ctx.push_data_block,
                                            ctx.ConstU32(PushData::LdsScratchIndex),
                                            ctx.ConstU32(index)));
    };
    const Id lo = push_word(0);
    const Id hi = push_word(1);
    const Id writer = ctx.OpINotEqual(
        ctx.U1[1], ctx.OpBitwiseAnd(ctx.U32[1], lo, ctx.u32_one_value), ctx.u32_zero_value);
    const Id lo_aligned = ctx.OpBitwiseAnd(ctx.U32[1], lo, ctx.ConstU32(~3u));
    ctx.lds_scratch_base =
        ctx.OpBitcast(ctx.U64, ctx.OpCompositeConstruct(ctx.U32[2], lo_aligned, hi));
    const Id prim = ctx.OpLoad(ctx.U32[1], ctx.primitive_id);
    const Id own = ctx.OpIMul(ctx.U32[1], prim, ctx.ConstU32(num_cp));
    if (tcs.lds_emulation == 1) {
        const Id first = ctx.OpISub(
            ctx.U32[1], prim,
            ctx.OpUMod(ctx.U32[1], prim, ctx.ConstU32(std::max<u32>(tcs.num_patches, 1u))));
        ctx.lds_vertex_base = ctx.OpIMul(ctx.U32[1], first, ctx.ConstU32(num_cp));
    } else {
        ctx.lds_vertex_base = own;
    }
    const auto scratch_pointer = [&](Id dword) {
        const Id address = ctx.OpIAdd(
            ctx.U64, ctx.lds_scratch_base,
            ctx.OpUConvert(ctx.U64, ctx.OpShiftLeftLogical(ctx.U32[1], dword, ctx.ConstU32(2u))));
        return ctx.OpConvertUToPtr(ctx.physical_pointer_type_f32, address);
    };
    // The writer pass: invocation 0 stores the patch's inputs, the tessellation levels are 0
    // (nothing is drawn) and the shader ends.
    const Id write_label = ctx.OpLabel();
    const Id merge_label = ctx.OpLabel();
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(writer, write_label, merge_label);
    ctx.AddLabel(write_label);
    const Id first_invocation = ctx.OpIEqual(
        ctx.U1[1], ctx.OpLoad(ctx.U32[1], ctx.invocation_id), ctx.u32_zero_value);
    const Id store_label = ctx.OpLabel();
    const Id stored_label = ctx.OpLabel();
    ctx.OpSelectionMerge(stored_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(first_invocation, store_label, stored_label);
    ctx.AddLabel(store_label);
    const Id in_ptr = ctx.TypePointer(spv::StorageClass::Input, ctx.F32[1]);
    for (u32 cp = 0; cp < num_cp; ++cp) {
        for (u32 attr = 0; attr < num_attrs; ++attr) {
            for (u32 comp = 0; comp < 4; ++comp) {
                const Id value = ctx.OpLoad(
                    ctx.F32[1], ctx.OpAccessChain(in_ptr, ctx.input_attr_array, ctx.ConstU32(cp),
                                                  ctx.ConstU32(attr), ctx.ConstU32(comp)));
                const Id dword = ctx.OpIAdd(
                    ctx.U32[1],
                    ctx.OpIMul(ctx.U32[1], ctx.OpIAdd(ctx.U32[1], own, ctx.ConstU32(cp)),
                               ctx.ConstU32(num_attrs * 4)),
                    ctx.ConstU32(attr * 4 + comp));
                ctx.OpStore(scratch_pointer(dword), value, spv::MemoryAccessMask::Aligned, 4u);
            }
        }
    }
    ctx.OpBranch(stored_label);
    ctx.AddLabel(stored_label);
    for (u32 i = 0; Sirit::ValidId(ctx.output_tess_level_outer) && i < 4; ++i) {
        ctx.OpStore(ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_outer, ctx.ConstU32(i)),
                    ctx.ConstF32(0.0f));
    }
    for (u32 i = 0; Sirit::ValidId(ctx.output_tess_level_inner) && i < 2; ++i) {
        ctx.OpStore(ctx.OpAccessChain(ctx.output_f32, ctx.output_tess_level_inner, ctx.ConstU32(i)),
                    ctx.ConstF32(0.0f));
    }
    ctx.OpReturn();
    ctx.AddLabel(merge_label);
}

void EmitPrologue(EmitContext& ctx) {
    if (ctx.hw_stage == HwStage::Fragment) {
        ctx.DefineAmdPerVertexAttribs();
        if (ctx.runtime_info.hw.fs.quad_sample_mask) {
            const Id coordinates =
                ctx.OpConvertFToU(ctx.U32[4], ctx.OpLoad(ctx.F32[4], ctx.frag_coord));
            const Id x = ctx.OpBitwiseAnd(
                ctx.U32[1], ctx.OpCompositeExtract(ctx.U32[1], coordinates, 0u), ctx.u32_one_value);
            const Id y = ctx.OpBitwiseAnd(
                ctx.U32[1], ctx.OpCompositeExtract(ctx.U32[1], coordinates, 1u), ctx.u32_one_value);
            const Id row_ptr = ctx.OpAccessChain(
                ctx.TypePointer(spv::StorageClass::PushConstant, ctx.U32[1]), ctx.push_data_block,
                ctx.ConstU32(PushData::QuadSampleMaskIndex), y);
            const Id row = ctx.OpLoad(ctx.U32[1], row_ptr);
            const Id shift = ctx.OpShiftLeftLogical(ctx.U32[1], x, ctx.ConstU32(4u));
            ctx.quad_sample_mask = ctx.OpBitwiseAnd(
                ctx.U32[1], ctx.OpShiftRightLogical(ctx.U32[1], row, shift), ctx.ConstU32(0xffffu));
            ctx.OpStore(ctx.OpAccessChain(ctx.output_u32, ctx.sample_mask, ctx.u32_zero_value),
                        ctx.quad_sample_mask);
        }
    }
    if (ctx.sw_stage == SwStage::TessellationControl && ctx.runtime_info.sw.tcs.lds_emulation) {
        EmitLdsEmulationPrologue(ctx); // FIX-227
    }
    if (ctx.info.loads.Get(IR::Attribute::WorkgroupIndex)) {
        ctx.DefineWorkgroupIndex();
    }
    ctx.DefineBufferProperties();
}

void ConvertDepthMode(EmitContext& ctx) {
    const Id type{ctx.F32[1]};
    const Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id screen_depth{ctx.OpFMul(type, ctx.OpFAdd(type, z, w), ctx.Constant(type, 0.5f))};
    const Id vector{ctx.OpCompositeInsert(ctx.F32[4], screen_depth, position, 2u)};
    ctx.OpStore(ctx.output_position, vector);
}

void ConvertPositionToClipSpace(EmitContext& ctx) {
    ASSERT_MSG(!ctx.info.stores.GetAny(IR::Attribute::ViewportIndex),
               "Multi-viewport with shader clip space conversion not yet implemented.");

    const Id type{ctx.F32[1]};
    Id position{ctx.OpLoad(ctx.F32[4], ctx.output_position)};
    const Id x{ctx.OpCompositeExtract(type, position, 0u)};
    const Id y{ctx.OpCompositeExtract(type, position, 1u)};
    const Id z{ctx.OpCompositeExtract(type, position, 2u)};
    const Id w{ctx.OpCompositeExtract(type, position, 3u)};
    const Id xoffset_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                           ctx.push_data_block,
                                           ctx.ConstU32(PushData::XOffsetIndex))};
    const Id xoffset{ctx.OpLoad(type, xoffset_ptr)};
    const Id yoffset_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                           ctx.push_data_block,
                                           ctx.ConstU32(PushData::YOffsetIndex))};
    const Id yoffset{ctx.OpLoad(type, yoffset_ptr)};
    const Id xscale_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                          ctx.push_data_block,
                                          ctx.ConstU32(PushData::XScaleIndex))};
    const Id xscale{ctx.OpLoad(type, xscale_ptr)};
    const Id yscale_ptr{ctx.OpAccessChain(ctx.TypePointer(spv::StorageClass::PushConstant, type),
                                          ctx.push_data_block,
                                          ctx.ConstU32(PushData::YScaleIndex))};
    const Id yscale{ctx.OpLoad(type, yscale_ptr)};
    const Id vport_w =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_width / 2, 8_KB)));
    const Id wnd_x = ctx.OpFAdd(type, ctx.OpFMul(type, x, xscale), xoffset);
    const Id ndc_x = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_x, vport_w), ctx.Constant(type, 1.f));
    const Id vport_h =
        ctx.Constant(type, float(std::min<u32>(ctx.profile.max_viewport_height / 2, 8_KB)));
    const Id wnd_y = ctx.OpFAdd(type, ctx.OpFMul(type, y, yscale), yoffset);
    const Id ndc_y = ctx.OpFSub(type, ctx.OpFDiv(type, wnd_y, vport_h), ctx.Constant(type, 1.f));
    const Id vector{ctx.OpCompositeConstruct(ctx.F32[4], std::array<Id, 4>({ndc_x, ndc_y, z, w}))};
    ctx.OpStore(ctx.output_position, vector);
}

void EmitEpilogue(EmitContext& ctx) {
    if (ctx.hw_stage == HwStage::Vertex &&
        ctx.runtime_info.hw.vs.emulate_depth_negative_one_to_one) {
        ConvertDepthMode(ctx);
    }
    if (ctx.hw_stage == HwStage::Vertex && ctx.runtime_info.hw.vs.clip_disable) {
        ConvertPositionToClipSpace(ctx);
    }
}

void EmitDiscard(EmitContext& ctx) {
    ctx.OpDemoteToHelperInvocationEXT();
}

void EmitDiscardCond(EmitContext& ctx, Id condition) {
    const Id kill_label{ctx.OpLabel()};
    const Id merge_label{ctx.OpLabel()};
    ctx.OpSelectionMerge(merge_label, spv::SelectionControlMask::MaskNone);
    ctx.OpBranchConditional(condition, kill_label, merge_label);
    ctx.AddLabel(kill_label);
    ctx.OpDemoteToHelperInvocationEXT();
    ctx.OpBranch(merge_label);
    ctx.AddLabel(merge_label);
}

void EmitEmitVertex(EmitContext& ctx) {
    ctx.OpEmitVertex();
}

void EmitEmitPrimitive(EmitContext& ctx) {
    ctx.OpEndPrimitive();
}

void EmitEmitVertex(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitEndPrimitive(EmitContext& ctx, const IR::Value& stream) {
    UNREACHABLE_MSG("Geometry streams");
}

void EmitDebugPrint(EmitContext& ctx, IR::Inst* inst, Id fmt, Id arg0, Id arg1, Id arg2, Id arg3) {
    IR::DebugPrintFlags flags = inst->Flags<IR::DebugPrintFlags>();
    std::array<Id, IR::DEBUGPRINT_NUM_FORMAT_ARGS> fmt_args = {arg0, arg1, arg2, arg3};
    auto fmt_args_span = std::span<Id>(fmt_args.begin(), fmt_args.begin() + flags.num_args);
    ctx.OpDebugPrintf(fmt, fmt_args_span);
}

Id EmitMemtime(EmitContext& ctx) {
    if (ctx.profile.supports_shader_subgroup_clock) {
        return ctx.OpReadClockKHR(ctx.U64, ctx.ConstU32(std::to_underlying(spv::Scope::Subgroup)));
    } else {
        return ctx.Constant(ctx.U64, 1U);
    }
}

} // namespace Shader::Backend::SPIRV
