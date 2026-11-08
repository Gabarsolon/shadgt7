// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <gtest/gtest.h>

#include "common/object_pool.h"
#include "shader_recompiler/ir/ir_emitter.h"
#include "shader_recompiler/ir/passes/ir_passes.h"
#include "shader_recompiler/profile.h"

namespace {

using Shader::IR::Opcode;
using Shader::IR::Type;
using Shader::IR::Value;

class FMaskResourcePatching : public testing::Test {
protected:
    Common::ObjectPool<Shader::IR::Inst> inst_pool{64};
    Shader::IR::Block block{inst_pool};
    Shader::IR::IREmitter ir{block};
    Shader::Info info{};
    Shader::Optimization::ResourceDiscoveryList resources;

    void Patch(const Value& value) {
        AmdGpu::Image image = AmdGpu::Image::Null(false);
        image.base_address = 1;
        image.data_format = u64(AmdGpu::DataFormat::FormatFmask64_S16_F8);
        const auto dwords = std::bit_cast<std::array<u32, 8>>(image);

        auto& usage = resources.emplace_back();
        usage.user = value.Inst();
        usage.sharps[0].num_dwords = dwords.size();
        for (u32 i = 0; i < dwords.size(); ++i) {
            usage.sharps[0].dwords[i] = ir.Imm32(dwords[i]);
        }
        Shader::Optimization::ResourcePatchingPass(info, resources, Shader::Profile{});
    }

    void ExpectFourComponentFallback(const Value& value) {
        std::array<Value, 4> extracts;
        for (u32 i = 0; i < extracts.size(); ++i) {
            extracts[i] = ir.CompositeExtract(value, i);
        }

        Patch(value);

        EXPECT_EQ(value.Inst()->GetOpcode(), Opcode::Void);
        ASSERT_TRUE(info.images.empty());
        const auto replacement = extracts[0].Inst()->Arg(0);
        ASSERT_EQ(replacement.Type(), Type::F32x4);
        ASSERT_EQ(replacement.Inst()->GetOpcode(), Opcode::CompositeConstructF32x4);
        const std::array<u32, 2> identity_words{0x76543210U, 0xfedcba98U};
        for (u32 i = 0; i < identity_words.size(); ++i) {
            const auto word = replacement.Inst()->Arg(i);
            ASSERT_EQ(word.Inst()->GetOpcode(), Opcode::BitCastF32U32);
            EXPECT_EQ(word.Inst()->Arg(0).U32(), identity_words[i]);
        }
        for (u32 i = 2; i < 4; ++i) {
            const auto component = replacement.Inst()->Arg(i);
            ASSERT_EQ(component.Type(), Type::F32);
            EXPECT_FLOAT_EQ(component.F32(), 0.0f);
        }
        for (u32 i = 0; i < extracts.size(); ++i) {
            EXPECT_EQ(extracts[i].Inst()->Arg(0), replacement);
            EXPECT_EQ(extracts[i].Inst()->Arg(1).U32(), i);
        }
    }
};

TEST_F(FMaskResourcePatching, ImageReadPreservesAllFourComponentConsumers) {
    const auto coords =
        ir.CompositeConstruct(ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U), ir.Imm32(0U));
    ExpectFourComponentFallback(ir.ImageRead(ir.Imm32(0U), coords, {}, {}, {}));
}

TEST_F(FMaskResourcePatching, RawSamplePreservesAllFourComponentConsumers) {
    const auto coords =
        ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f));
    ExpectFourComponentFallback(
        ir.ImageSampleRaw(ir.Imm32(0U), ir.Imm32(0U), coords, coords, coords, ir.Imm32(0.0f), {}));
}

TEST_F(FMaskResourcePatching, LodQueryReturnsTwoFloatingPointComponents) {
    const auto coords =
        ir.CompositeConstruct(ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f), ir.Imm32(0.0f));
    const auto query = ir.ImageQueryLod(ir.Imm32(0U), coords, {});
    const std::array extracts{ir.CompositeExtract(query, 0), ir.CompositeExtract(query, 1)};

    Patch(query);

    EXPECT_EQ(query.Inst()->GetOpcode(), Opcode::Void);
    EXPECT_TRUE(info.images.empty());
    const auto replacement = extracts[0].Inst()->Arg(0);
    ASSERT_EQ(replacement.Type(), Type::F32x2);
    ASSERT_EQ(replacement.Inst()->GetOpcode(), Opcode::CompositeConstructF32x2);
    for (u32 i = 0; i < extracts.size(); ++i) {
        EXPECT_EQ(extracts[i].Inst()->Arg(0), replacement);
        EXPECT_EQ(extracts[i].Inst()->Arg(1).U32(), i);
        const auto component = replacement.Inst()->Arg(i);
        ASSERT_EQ(component.Type(), Type::F32);
        EXPECT_FLOAT_EQ(component.F32(), 0.0f);
    }
}

// FIX-GT7-125: a sample with the array flag on an MSAA array T# (GT7, 11-08: its translations
// stopped at the sample coordinates, and its draws were never drawn). It reads the multisampled
// array at (x, y, layer), the layer rounded, not scaled by a dimension.
TEST(MsaaArrayResourcePatching, ArraySampleReadsTheLayerOfAMultisampledArray) {
    Common::ObjectPool<Shader::IR::Inst> inst_pool{64};
    Shader::IR::Block block{inst_pool};
    Shader::IR::IREmitter ir{block};
    Shader::Info info{};
    info.hw_stage = Shader::HwStage::Fragment;
    info.sw_stage = Shader::SwStage::Fragment;
    Shader::Optimization::ResourceDiscoveryList resources;

    Shader::IR::TextureInstInfo flags{};
    flags.is_array.Assign(1);
    const auto address =
        ir.CompositeConstruct(ir.Imm32(0.25f), ir.Imm32(0.5f), ir.Imm32(2.6f), ir.Imm32(0.0f));
    const auto sample = ir.ImageSampleRaw(ir.Imm32(0U), ir.Imm32(0U), address, address, address,
                                          ir.Imm32(0.0f), flags);
    const auto use = ir.CompositeExtract(sample, 0);

    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.base_address = 0x1000;
    image.type = u64(AmdGpu::ImageType::Color2DMsaaArray);
    image.last_level = 2; // 4 samples
    image.last_array = 3;
    const auto image_dwords = std::bit_cast<std::array<u32, 8>>(image);
    auto& usage = resources.emplace_back();
    usage.user = sample.Inst();
    usage.sharps[0].num_dwords = image_dwords.size();
    for (u32 i = 0; i < image_dwords.size(); ++i) {
        usage.sharps[0].dwords[i] = ir.Imm32(image_dwords[i]);
    }
    usage.sharps[1].num_dwords = 4;
    for (u32 i = 0; i < 4; ++i) {
        usage.sharps[1].dwords[i] = ir.Imm32(0U);
    }

    bool failed = false;
    try {
        Shader::Optimization::ResourcePatchingPass(info, resources, Shader::Profile{});
    } catch (...) {
        failed = true;
    }
    ASSERT_TRUE(!failed);
    ASSERT_EQ(info.images.size(), 1u);

    // The sample's value now comes from a read of the image.
    const auto read = use.Inst()->Arg(0);
    ASSERT_TRUE(!read.IsImmediate());
    ASSERT_EQ(read.Inst()->GetOpcode(), Opcode::ImageRead);
    const auto coords = read.Inst()->Arg(1);
    ASSERT_TRUE(!coords.IsImmediate());
    ASSERT_EQ(coords.Inst()->GetOpcode(), Opcode::CompositeConstructU32x3);
    // The layer: 2.6 rounded to 3, with no dimension in between.
    const auto layer = coords.Inst()->Arg(2);
    ASSERT_TRUE(!layer.IsImmediate());
    ASSERT_EQ(layer.Inst()->GetOpcode(), Opcode::ConvertU32F32);
    const auto rounded = layer.Inst()->Arg(0);
    ASSERT_TRUE(!rounded.IsImmediate());
    EXPECT_EQ(rounded.Inst()->GetOpcode(), Opcode::FPRoundEven32);
}

} // namespace
