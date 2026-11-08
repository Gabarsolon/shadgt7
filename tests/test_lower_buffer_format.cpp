// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <fmt/format.h>
#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/ir/program.h"
#include "video_core/amdgpu/resource.h"

namespace {

using Shader::IR::Opcode;
using Shader::IR::Type;
using Shader::IR::Value;

// FIX-GT7-126: a V# whose data format the lowering has no case for (15) is read as a null buffer.
class LowerBufferFormat : public testing::Test {
protected:
    Common::ObjectPool<Shader::IR::Inst> inst_pool{64};
    Shader::IR::Block block{inst_pool};
    Shader::IR::IREmitter ir{block};
    Shader::Info info{};
    Shader::IR::Program program{info};

    void BindBuffer(u32 data_format, AmdGpu::NumberFormat num_format) {
        AmdGpu::Buffer buffer{};
        buffer.base_address = 0x1000;
        buffer.num_records = 64;
        buffer.dst_sel_x = u32(AmdGpu::CompSwizzle::Red);
        buffer.dst_sel_y = u32(AmdGpu::CompSwizzle::Green);
        buffer.dst_sel_z = u32(AmdGpu::CompSwizzle::Blue);
        buffer.dst_sel_w = u32(AmdGpu::CompSwizzle::Alpha);
        buffer.num_format = u32(num_format);
        buffer.data_format = data_format;
        const auto dwords = std::bit_cast<std::array<u32, 4>>(buffer);
        info.flattened_ud_buf.assign(dwords.begin(), dwords.end());

        auto& desc = info.buffers.emplace_back();
        desc.sharp_fetch.summary = Shader::SharpFetch<AmdGpu::Buffer>::Summary::SingleLoad;
        desc.sharp_fetch.offsets[0] = 0;
        desc.is_formatted = true;
        ASSERT_EQ(u32(desc.GetSharp(info).data_format), data_format);
    }

    void Lower() {
        program.blocks.push_back(&block);
        Shader::Optimization::LowerBufferFormatToRaw(program);
    }

    u32 Count(Opcode opcode) {
        u32 count = 0;
        for (const auto& inst : block.Instructions()) {
            count += inst.GetOpcode() == opcode ? 1 : 0;
        }
        return count;
    }
};

TEST_F(LowerBufferFormat, UnknownDataFormatLoadReadsZeros) {
    BindBuffer(15, AmdGpu::NumberFormat::Float);
    const auto load = ir.LoadBufferFormat(ir.Imm32(0U), ir.Imm32(0U), {});
    const std::array extracts{ir.CompositeExtract(load, 0), ir.CompositeExtract(load, 3)};

    Lower();

    EXPECT_EQ(load.Inst()->GetOpcode(), Opcode::Void);
    EXPECT_EQ(Count(Opcode::LoadBufferU32) + Count(Opcode::LoadBufferF32), 0U);
    const auto replacement = extracts[0].Inst()->Arg(0);
    ASSERT_EQ(replacement.Type(), Type::F32x4);
    ASSERT_EQ(replacement.Inst()->GetOpcode(), Opcode::CompositeConstructF32x4);
    for (u32 i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(replacement.Inst()->Arg(i).F32(), 0.0f);
    }
    EXPECT_EQ(extracts[1].Inst()->Arg(0), replacement);
}

TEST_F(LowerBufferFormat, UnknownDataFormatStoreIsDropped) {
    BindBuffer(15, AmdGpu::NumberFormat::Float);
    const auto data =
        ir.CompositeConstruct(ir.Imm32(1.0f), ir.Imm32(2.0f), ir.Imm32(3.0f), ir.Imm32(4.0f));
    ir.StoreBufferFormat(ir.Imm32(0U), ir.Imm32(0U), data, {});

    Lower();

    EXPECT_EQ(Count(Opcode::StoreBufferFormatF32), 0U);
    EXPECT_EQ(Count(Opcode::StoreBufferU32) + Count(Opcode::StoreBufferF32), 0U);
}

TEST_F(LowerBufferFormat, KnownDataFormatIsStillLowered) {
    BindBuffer(u32(AmdGpu::DataFormat::Format32), AmdGpu::NumberFormat::Float);
    const auto load = ir.LoadBufferFormat(ir.Imm32(0U), ir.Imm32(0U), {});
    const auto x = ir.CompositeExtract(load, 0);

    Lower();

    EXPECT_EQ(load.Inst()->GetOpcode(), Opcode::Void);
    EXPECT_EQ(Count(Opcode::LoadBufferF32), 1U);
    EXPECT_NE(x.Inst()->Arg(0).Inst()->GetOpcode(), Opcode::LoadBufferFormatF32);
}

TEST(PixelFormatName, UnknownValuesPrintTheirNumber) {
    EXPECT_EQ(fmt::format("{}", AmdGpu::DataFormat::Format32), "Format32");
    EXPECT_EQ(fmt::format("{}", AmdGpu::DataFormat(15)), "DataFormat(15)");
    EXPECT_EQ(fmt::format("{}", AmdGpu::DataFormat(63)), "DataFormat(63)");
    EXPECT_EQ(fmt::format("{}", AmdGpu::NumberFormat::Float), "Float");
    EXPECT_EQ(fmt::format("{}", AmdGpu::NumberFormat(8)), "NumberFormat(8)");
}

} // namespace
