// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <cstdlib>
#include <gtest/gtest.h>
#include "video_core/renderer_vulkan/shader_usage_serde.h"
#include "video_core/renderer_vulkan/spirv_check.h"
#include "video_core/renderer_vulkan/vk_pipeline_cache.h"

namespace {

Shader::StageSpecialization MakeSpec(const Shader::Info& info) {
    Shader::StageSpecialization spec;
    spec.info = &info;
    spec.runtime_info.Initialize(info.hw_stage, info.sw_stage);
    spec.buffers.resize(info.buffers.size());
    spec.images.resize(info.images.size());
    spec.samplers.resize(info.samplers.size());
    return spec;
}

std::unique_ptr<Shader::Info> MakeFragmentInfo(bool flatbuf) {
    auto info = std::make_unique<Shader::Info>();
    info->hw_stage = Shader::HwStage::Fragment;
    info->sw_stage = Shader::SwStage::Fragment;
    if (flatbuf) {
        info->buffers.push_back({.buffer_type = Shader::BufferType::Flatbuf});
    }
    info->images.emplace_back();
    info->samplers.emplace_back();
    return info;
}

} // namespace

TEST(ShaderProgram, KeepsDistinctResourceLayoutsAndStableMetadataForPermutations) {
    Vulkan::Program program;
    auto ordinary = MakeFragmentInfo(false);
    const auto* ordinary_ptr = ordinary.get();
    auto ordinary_spec = MakeSpec(*ordinary);
    program.AddPermut({}, std::move(ordinary_spec), std::move(ordinary));
    auto flatbuf = MakeFragmentInfo(true);
    const auto* flatbuf_ptr = flatbuf.get();
    auto flatbuf_spec = MakeSpec(*flatbuf);
    program.AddPermut({}, std::move(flatbuf_spec), std::move(flatbuf));

    // Force the permutation list out of its inline storage, as GT Sport does.
    for (size_t i = 0; i < Vulkan::Program::MaxPermutations; ++i) {
        auto info = MakeFragmentInfo(false);
        auto spec = MakeSpec(*info);
        program.AddPermut({}, std::move(spec), std::move(info));
    }
    EXPECT_EQ(program.modules[0].info.get(), ordinary_ptr);
    EXPECT_EQ(program.modules[1].info.get(), flatbuf_ptr);
    EXPECT_EQ(program.modules[0].spec.info, ordinary_ptr);
    EXPECT_EQ(program.modules[1].spec.info, flatbuf_ptr);

    Shader::Backend::Bindings ordinary_end;
    Shader::Backend::Bindings flatbuf_end;
    ordinary_ptr->AddBindings(ordinary_end);
    flatbuf_ptr->AddBindings(flatbuf_end);
    EXPECT_EQ(ordinary_end.unified, 2);
    EXPECT_EQ(flatbuf_end.unified, 3);
    EXPECT_EQ(flatbuf_end.buffer, 1);
}

TEST(ShaderProgram, DoesNotReuseOpaqueShaderPermutationForQuadMaskedDraw) {
    Vulkan::Program program;
    auto opaque = MakeFragmentInfo(false);
    auto opaque_spec = MakeSpec(*opaque);
    program.AddPermut({}, std::move(opaque_spec), std::move(opaque));

    auto masked = MakeFragmentInfo(false);
    auto masked_spec = MakeSpec(*masked);
    masked_spec.runtime_info.hw.fs.quad_sample_mask = true;
    EXPECT_FALSE(program.modules[0].spec == masked_spec);
    program.AddPermut({}, std::move(masked_spec), std::move(masked));

    auto next_draw = MakeSpec(*program.modules[1].info);
    next_draw.runtime_info.hw.fs.quad_sample_mask = true;
    EXPECT_FALSE(program.modules[0].spec == next_draw);
    EXPECT_TRUE(program.modules[1].spec == next_draw);
}

TEST(ShaderProgram, RetainsEachPreloadedPermutationsMetadataAtItsOriginalIndex) {
    Vulkan::Program program;
    auto flatbuf = MakeFragmentInfo(true);
    const auto* flatbuf_ptr = flatbuf.get();
    auto flatbuf_spec = MakeSpec(*flatbuf);
    program.InsertPermut({}, std::move(flatbuf_spec), std::move(flatbuf), 5);
    auto ordinary = MakeFragmentInfo(false);
    const auto* ordinary_ptr = ordinary.get();
    auto ordinary_spec = MakeSpec(*ordinary);
    program.InsertPermut({}, std::move(ordinary_spec), std::move(ordinary), 0);

    EXPECT_EQ(program.modules[5].info.get(), flatbuf_ptr);
    EXPECT_EQ(program.modules[5].spec.info, flatbuf_ptr);
    EXPECT_EQ(program.modules[0].info.get(), ordinary_ptr);
    EXPECT_EQ(program.modules[0].spec.info, ordinary_ptr);
    EXPECT_FALSE(program.modules[1].spec.Valid());
    EXPECT_TRUE(program.modules[5].spec.Valid());
}

TEST(ShaderSpecialization, RejectsDifferentResourceCountsInEitherComparisonDirection) {
    auto ordinary = MakeFragmentInfo(false);
    auto flatbuf = MakeFragmentInfo(true);
    auto ordinary_spec = MakeSpec(*ordinary);
    auto flatbuf_spec = MakeSpec(*flatbuf);
    EXPECT_FALSE(ordinary_spec == flatbuf_spec);
    EXPECT_FALSE(flatbuf_spec == ordinary_spec);

    auto extra_image = ordinary_spec;
    extra_image.images.emplace_back();
    EXPECT_FALSE(ordinary_spec == extra_image);
    EXPECT_FALSE(extra_image == ordinary_spec);
    auto extra_sampler = ordinary_spec;
    extra_sampler.samplers.emplace_back();
    EXPECT_FALSE(ordinary_spec == extra_sampler);
    EXPECT_FALSE(extra_sampler == ordinary_spec);
    Shader::StageSpecialization unloaded;
    EXPECT_FALSE(ordinary_spec == unloaded);
    EXPECT_FALSE(unloaded == ordinary_spec);
}

TEST(ShaderSpecialization, NewlyActiveResourcesMustMatchTheCompiledPermutation) {
    auto info = MakeFragmentInfo(false);
    auto compiled = MakeSpec(*info);
    compiled.images[0].type = AmdGpu::ImageType::Color2D;
    auto current = compiled;
    current.bitset[0] = true;
    current.images[0].type = AmdGpu::ImageType::Color2DArray;
    EXPECT_FALSE(compiled == current);

    // An inactive current resource can retain the existing module's declaration.
    EXPECT_TRUE(current == compiled);
}

TEST(ShaderBindings, CountsMipFallbackDescriptorsWhenReusingAModule) {
    Shader::Info info;
    AmdGpu::Image image = AmdGpu::Image::Null(false);
    image.base_level = 2;
    image.last_level = 5;
    Shader::ImageResource resource;
    resource.mip_fallback_mode = Shader::MipStorageFallbackMode::DynamicIndex;
    std::memcpy(resource.sharp_fetch.immediates.data(), &image, sizeof(image));
    info.images.push_back(resource);
    info.images.emplace_back();
    info.buffers.emplace_back();
    info.samplers.emplace_back();
    Shader::Backend::Bindings binding{.unified = 7, .buffer = 3};
    info.AddBindings(binding);
    // A compiled module reserves four mip descriptors, one ordinary image, a buffer and sampler.
    EXPECT_EQ(binding.unified, 14);
    EXPECT_EQ(binding.buffer, 4);
}

namespace {

// FIX-043: specializations built the way the pipeline cache builds them, from a translated
// shader's Info and the V#/T#s bound for a draw.
AmdGpu::Buffer BoundBuffer(u32 stride) {
    AmdGpu::Buffer buffer{};
    buffer.base_address = 0x10000;
    buffer.num_records = 256;
    buffer.stride = stride;
    return buffer;
}

void Bind(Shader::BufferResource& resource, const AmdGpu::Buffer& buffer) {
    std::memcpy(resource.sharp_fetch.immediates.data(), &buffer, sizeof(buffer));
}

void Bind(Shader::ImageResource& resource, const AmdGpu::Image& image) {
    std::memcpy(resource.sharp_fetch.immediates.data(), &image, sizeof(image));
}

Shader::StageSpecialization KeyFor(const Shader::Info& info) {
    Shader::RuntimeInfo runtime_info{};
    runtime_info.Initialize(info.hw_stage, info.sw_stage);
    return Shader::StageSpecialization(info, runtime_info, Shader::Profile{}, {});
}

Shader::Info TranslatedFragmentWithBuffer(bool usage_known) {
    Shader::Info info;
    info.hw_stage = Shader::HwStage::Fragment;
    info.sw_stage = Shader::SwStage::Fragment;
    info.buffers.emplace_back();
    info.resource_usage_known = usage_known;
    return info;
}

} // namespace

TEST(ShaderSpecialization, StrideOfABufferWithoutIndexedAddressingDoesNotSplitPermutations) {
    auto info = TranslatedFragmentWithBuffer(true);
    Bind(info.buffers[0], BoundBuffer(16));
    const auto compiled = KeyFor(info);
    Bind(info.buffers[0], BoundBuffer(48));
    EXPECT_TRUE(compiled == KeyFor(info));

    // Indexed addressing multiplies by the stride, so it stays in the key.
    info.buffer_stride_used.set(0);
    Bind(info.buffers[0], BoundBuffer(16));
    const auto indexed = KeyFor(info);
    Bind(info.buffers[0], BoundBuffer(48));
    EXPECT_FALSE(indexed == KeyFor(info));
}

TEST(ShaderSpecialization, KeepsEveryBufferPropertyWhenTranslationUsageIsUnknown) {
    // A shader loaded from storage carries no usage flags.
    auto info = TranslatedFragmentWithBuffer(false);
    Bind(info.buffers[0], BoundBuffer(16));
    const auto compiled = KeyFor(info);
    Bind(info.buffers[0], BoundBuffer(48));
    EXPECT_FALSE(compiled == KeyFor(info));
}

TEST(ShaderSpecialization, ThreadIdAddressingAndCoherenceSplitPermutations) {
    auto info = TranslatedFragmentWithBuffer(true);
    Bind(info.buffers[0], BoundBuffer(16));
    const auto compiled = KeyFor(info);

    auto add_tid = BoundBuffer(16);
    add_tid.add_tid_enable = 1;
    Bind(info.buffers[0], add_tid);
    EXPECT_FALSE(compiled == KeyFor(info));

    auto coherent = BoundBuffer(16);
    coherent.mtype = 3;
    Bind(info.buffers[0], coherent);
    EXPECT_FALSE(compiled == KeyFor(info));
}

TEST(ShaderSpecialization, UnboundBufferIsComparedWithTheDescriptorTranslationSaw) {
    auto info = TranslatedFragmentWithBuffer(true);
    // Translated while the slot held an empty, unswizzled V#.
    auto empty = BoundBuffer(16);
    empty.num_records = 0;
    Bind(info.buffers[0], empty);
    const auto compiled = KeyFor(info);
    EXPECT_FALSE(compiled.bitset[0]);
    Bind(info.buffers[0], BoundBuffer(64));
    EXPECT_TRUE(compiled == KeyFor(info));

    // Swizzled addressing in the empty V# went into the code, so a plain V# needs new code.
    auto swizzled = empty;
    swizzled.swizzle_enable = 1;
    Bind(info.buffers[0], swizzled);
    const auto compiled_swizzled = KeyFor(info);
    Bind(info.buffers[0], BoundBuffer(64));
    EXPECT_FALSE(compiled_swizzled == KeyFor(info));

    // A typed buffer read through a null V# returns zeros; a bound one needs new code.
    info.buffers[0].is_formatted = true;
    Bind(info.buffers[0], AmdGpu::Buffer::Null());
    const auto compiled_typed = KeyFor(info);
    auto typed = BoundBuffer(16);
    typed.data_format = static_cast<u32>(AmdGpu::DataFormat::Format32);
    Bind(info.buffers[0], typed);
    EXPECT_FALSE(compiled_typed == KeyFor(info));
}

TEST(ShaderSpecialization, SrgbFormatSplitsPermutationsOnlyWhenItDecidesADegamma) {
    Shader::Info info;
    info.hw_stage = Shader::HwStage::Fragment;
    info.sw_stage = Shader::SwStage::Fragment;
    info.images.emplace_back();
    info.resource_usage_known = true;
    auto image = AmdGpu::Image::Null(false);
    image.base_address = 0x100;
    Bind(info.images[0], image);
    const auto compiled = KeyFor(info);
    ASSERT_TRUE(compiled.bitset[0]);
    auto srgb = image;
    srgb.num_format = static_cast<u64>(AmdGpu::NumberFormat::Srgb);
    Bind(info.images[0], srgb);
    EXPECT_TRUE(compiled == KeyFor(info));

    info.image_srgb_used.set(0);
    Bind(info.images[0], image);
    const auto degamma = KeyFor(info);
    Bind(info.images[0], srgb);
    EXPECT_FALSE(degamma == KeyFor(info));
}

TEST(ShaderStore, ReloadedTranslationMatchesItsStoredKeyOnlyWithItsResourceUsage) {
    // PERF-032: a translation stored with FIX-043's narrower key, loaded in a later session.
    auto translated = TranslatedFragmentWithBuffer(true);
    Bind(translated.buffers[0], BoundBuffer(16));
    const auto stored_key = KeyFor(translated);

    Serialization::Archive ar;
    Serialization::Writer out{ar};
    Vulkan::WriteResourceUsage(out, translated);
    auto bytes = ar.TakeOff();

    auto loaded = TranslatedFragmentWithBuffer(false);
    Serialization::Archive in_ar{std::vector<u8>{bytes}};
    Serialization::Reader in{in_ar};
    Vulkan::ReadResourceUsage(in, loaded);
    EXPECT_TRUE(loaded.resource_usage_known);
    // A draw binding the same buffer with another stride reuses the stored translation.
    Bind(loaded.buffers[0], BoundBuffer(48));
    EXPECT_TRUE(stored_key == KeyFor(loaded));

    // Without the usage (meta versions before 16) the rebuilt key keeps the stride and the
    // stored narrower key could never match.
    auto old_format = TranslatedFragmentWithBuffer(false);
    Bind(old_format.buffers[0], BoundBuffer(48));
    EXPECT_FALSE(stored_key == KeyFor(old_format));
}

TEST(ShaderStore, ResourceUsageRoundTripsEveryBufferAndImageFlag) {
    Shader::Info info;
    info.resource_usage_known = true;
    info.buffer_stride_used.set(0).set(17).set(Shader::NUM_BUFFERS - 1);
    info.image_srgb_used.set(3).set(Shader::NUM_IMAGES - 1);
    Serialization::Archive ar;
    Serialization::Writer out{ar};
    Vulkan::WriteResourceUsage(out, info);
    Serialization::Archive in_ar{ar.TakeOff()};
    Serialization::Reader in{in_ar};
    Shader::Info loaded;
    Vulkan::ReadResourceUsage(in, loaded);
    EXPECT_EQ(loaded.resource_usage_known, true);
    EXPECT_EQ(loaded.buffer_stride_used, info.buffer_stride_used);
    EXPECT_EQ(loaded.image_srgb_used, info.image_srgb_used);
}

TEST(ShaderStore, AcceptsOnlyWholeSpirvModules) {
    // Header, OpCapability Shader, OpFunction / OpLabel / OpReturn / OpFunctionEnd.
    const std::vector<u32> module{0x07230203,
                                  0x00010600,
                                  0,
                                  16,
                                  0,
                                  (2u << 16) | 17,
                                  1,
                                  (5u << 16) | 54,
                                  1,
                                  2,
                                  0,
                                  3,
                                  (2u << 16) | 248,
                                  4,
                                  (1u << 16) | 253,
                                  (1u << 16) | 56};
    EXPECT_TRUE(Vulkan::IsCompleteSpirv(module));
    // Cut short inside an instruction, or at an instruction boundary before the function ends.
    EXPECT_FALSE(Vulkan::IsCompleteSpirv(std::span{module}.first(module.size() - 3)));
    EXPECT_FALSE(Vulkan::IsCompleteSpirv(std::span{module}.first(module.size() - 1)));
    EXPECT_FALSE(Vulkan::IsCompleteSpirv(std::span{module}.first(10)));
    auto not_spirv = module;
    not_spirv[0] = 0;
    EXPECT_FALSE(Vulkan::IsCompleteSpirv(not_spirv));
    auto zero_count = module;
    zero_count[5] = 17;
    EXPECT_FALSE(Vulkan::IsCompleteSpirv(zero_count));
    EXPECT_FALSE(Vulkan::IsCompleteSpirv({}));
}

// FIX-GT7-083: both engines share the store; an ordinary pixel-shader cube sample is a cube-array
// sample with native cube sampling (the native engine) and samples six faces of a 2D array without
// it (the classic engine's views). A translation is read only by the side it fits.
TEST(ShaderStore, StoredCubeSamplesAreReadOnlyByTheCubeSamplingTheyWereMadeWith) {
    auto info = MakeFragmentInfo(false);
    auto spec = MakeSpec(*info);
    spec.images[0].type = AmdGpu::ImageType::Cube;
    // Made without it: tagged; the classic engine reads it, the native one translates again.
    const bool faces = Vulkan::TranslatedWithFaceCubes(spec, false);
    EXPECT_TRUE(faces);
    EXPECT_TRUE(Vulkan::StoredCubesFit(spec, *info, faces, false));
    EXPECT_FALSE(Vulkan::StoredCubesFit(spec, *info, faces, true));
    // Made with it, the cube a native cube sample: not tagged; the classic engine translates again.
    info->images[0].post_op = Shader::SharpFetchPostOp::NativeCubeSampling;
    const bool native = Vulkan::TranslatedWithFaceCubes(spec, true);
    EXPECT_FALSE(native);
    EXPECT_TRUE(Vulkan::StoredCubesFit(spec, *info, native, true));
    EXPECT_FALSE(Vulkan::StoredCubesFit(spec, *info, native, false));
    // A stored cube image, or no cube: both read it.
    auto other_info = MakeFragmentInfo(false);
    auto other = MakeSpec(*other_info);
    other.images[0].type = AmdGpu::ImageType::Cube;
    other.images[0].is_storage = true;
    EXPECT_FALSE(Vulkan::TranslatedWithFaceCubes(other, false));
    other.images[0] = {};
    EXPECT_FALSE(Vulkan::TranslatedWithFaceCubes(other, false));
    EXPECT_TRUE(Vulkan::StoredCubesFit(other, *other_info, false, true));
    EXPECT_TRUE(Vulkan::StoredCubesFit(other, *other_info, false, false));
    // Other stages have no native cube samples.
    Shader::Info compute;
    compute.hw_stage = Shader::HwStage::Compute;
    compute.sw_stage = Shader::SwStage::Compute;
    compute.images.emplace_back();
    auto compute_spec = MakeSpec(compute);
    compute_spec.images[0].type = AmdGpu::ImageType::Cube;
    EXPECT_FALSE(Vulkan::TranslatedWithFaceCubes(compute_spec, false));
    EXPECT_TRUE(Vulkan::StoredCubesFit(compute_spec, compute, false, true));
}

// These focused header tests do not initialize the emulator's logging or fatal-error backend.
void Common::Log::VLog(Class, Level, const char* file, int line, const char*, fmt::string_view,
                       fmt::format_args) {
    ADD_FAILURE() << "Unexpected shader metadata assertion at " << file << ':' << line;
}

void assert_fail_impl() {
    std::abort();
}

void unreachable_impl() {
    std::abort();
}

// StageSpecialization parses a fetch shader only for vertex stages, which these tests do not use.
bool Shader::Gcn::ParseFetchShader(const Shader::Info&, Shader::Gcn::FetchShaderData&) {
    ADD_FAILURE() << "Unexpected fetch shader parse";
    return false;
}

namespace {

// PERF-GT7-058: what StageSpecialization::PackInputs packs for a stage.
std::vector<u64> PackFor(const Shader::Info& info) {
    std::vector<u64> words;
    Shader::Gcn::FetchShaderData fetch;
    Shader::StageSpecialization::PackInputs(info, words, fetch);
    return words;
}

} // namespace

TEST(ShaderSpecialization, PackedInputsIgnoreAddressesAndKeepWhatSplitsPermutations) {
    auto info = TranslatedFragmentWithBuffer(true);
    info.buffer_stride_used.set(0);
    Bind(info.buffers[0], BoundBuffer(16));
    const auto compiled = KeyFor(info);
    const auto packed = PackFor(info);

    // Another address and size: the same pack and the same specialization.
    auto moved = BoundBuffer(16);
    moved.base_address = 0x7650000;
    moved.num_records = 4096;
    Bind(info.buffers[0], moved);
    EXPECT_EQ(packed, PackFor(info));
    EXPECT_TRUE(compiled == KeyFor(info));

    // What splits permutations changes the pack.
    auto stride = BoundBuffer(48);
    auto add_tid = BoundBuffer(16);
    add_tid.add_tid_enable = 1;
    auto coherent = BoundBuffer(16);
    coherent.mtype = 3;
    auto swizzled = BoundBuffer(16);
    swizzled.swizzle_enable = 1;
    for (const auto& changed : {stride, add_tid, coherent, swizzled}) {
        Bind(info.buffers[0], changed);
        EXPECT_FALSE(compiled == KeyFor(info));
        EXPECT_NE(packed, PackFor(info));
    }

    // Unbound, or bound empty by its address (FIX-GT7-024): another pack.
    auto unbound = BoundBuffer(16);
    unbound.num_records = 0;
    Bind(info.buffers[0], unbound);
    EXPECT_NE(packed, PackFor(info));
    auto empty = BoundBuffer(16);
    empty.base_address = 0;
    Bind(info.buffers[0], empty);
    EXPECT_NE(packed, PackFor(info));
}

TEST(ShaderSpecialization, PackedImageInputsIgnoreAddressesAndKeepTheFormat) {
    Shader::Info info;
    info.hw_stage = Shader::HwStage::Fragment;
    info.sw_stage = Shader::SwStage::Fragment;
    info.images.emplace_back();
    info.resource_usage_known = true;
    auto image = AmdGpu::Image::Null(false);
    image.base_address = 0x1000;
    Bind(info.images[0], image);
    const auto compiled = KeyFor(info);
    const auto packed = PackFor(info);

    auto moved = image;
    moved.base_address = 0x9876;
    Bind(info.images[0], moved);
    EXPECT_EQ(packed, PackFor(info));
    EXPECT_TRUE(compiled == KeyFor(info));

    auto integer = image;
    integer.num_format = static_cast<u64>(AmdGpu::NumberFormat::Uint);
    Bind(info.images[0], integer);
    EXPECT_FALSE(compiled == KeyFor(info));
    EXPECT_NE(packed, PackFor(info));

    auto unbound = image;
    unbound.base_address = 0;
    Bind(info.images[0], unbound);
    EXPECT_NE(packed, PackFor(info));
}

// FIX-GT7-082: a vertex stage reads its attributes' V#s from guest memory each time, not from the
// flattened buffer: a pack recorded after a full match could hold a V# that changed in between
// (float to integer), and later draws would take the float module with no full comparison. Such
// stages always get a full comparison.
TEST(ShaderSpecialization, PackedInputsNeverStandForVertexAttributesReadFromMemory) {
    Shader::Info info;
    info.hw_stage = Shader::HwStage::Vertex;
    info.sw_stage = Shader::SwStage::Vertex;
    info.has_fetch_shader = true;
    EXPECT_FALSE(Shader::StageSpecialization::Packable(info));
    info.has_fetch_shader = false;
    EXPECT_TRUE(Shader::StageSpecialization::Packable(info));
    info.hw_stage = Shader::HwStage::Fragment;
    info.sw_stage = Shader::SwStage::Fragment;
    EXPECT_TRUE(Shader::StageSpecialization::Packable(info));
    info.sw_stage = Shader::SwStage::TessellationControl;
    EXPECT_FALSE(Shader::StageSpecialization::Packable(info));
}

// FIX-GT7-124: tessellation constants pinned as their draw had them.
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#endif

namespace {

Shader::SwTessControlRuntimeInfo TcsRegisters(u32 input_cp, u8 lds_emulation, u8 patches) {
    Shader::SwTessControlRuntimeInfo tcs{};
    tcs.num_input_control_points = input_cp;
    tcs.lds_emulation = lds_emulation;
    tcs.num_patches = patches;
    return tcs;
}

Shader::TessellationDataConstantBuffer TessConstants(u32 ls_stride, u32 cp_stride, u32 patches,
                                                     u32 output_base) {
    Shader::TessellationDataConstantBuffer c{};
    c.ls_stride = ls_stride;
    c.hs_cp_stride = cp_stride;
    c.num_patches = patches;
    c.hs_output_base = output_base;
    return c;
}

// Guest memory is at the guest's addresses, and a V#'s base address has 40 bits: a page below
// 2^40 stands for it.
u8* LowGuestPage() {
    static u8* const page = []() -> u8* {
        for (u64 hint = 0x2000000000ULL; hint < (1ULL << 40); hint += 0x1000000000ULL) {
#ifdef _WIN32
            void* p = VirtualAlloc(reinterpret_cast<void*>(hint), 0x10000, MEM_RESERVE | MEM_COMMIT,
                                   PAGE_READWRITE);
#else
            void* p = mmap(reinterpret_cast<void*>(hint), 0x10000, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (p == MAP_FAILED) {
                p = nullptr;
            }
#endif
            if (p && reinterpret_cast<u64>(p) + 0x10000 <= (1ULL << 40)) {
                return static_cast<u8*>(p);
            }
        }
        return nullptr;
    }();
    return page;
}

// A hull shader's constants V# at dword 16 of the table user data registers 2-3 point at, and
// the constants it points at; returns them.
Shader::TessellationDataConstantBuffer* PlaceTessConstants(
    u8* page, std::array<u32, Shader::ShaderParams::NumShaderUserData>& user_data) {
    auto* constants = reinterpret_cast<Shader::TessellationDataConstantBuffer*>(page + 0x100);
    *constants = TessConstants(0x28, 0x30, 16, 0x280);
    AmdGpu::Buffer sharp{};
    sharp.base_address = reinterpret_cast<u64>(constants);
    std::memcpy(page + 16 * 4, &sharp, sizeof(sharp));
    std::memcpy(page + 20 * 4, &sharp, sizeof(sharp));
    const u64 table = reinterpret_cast<u64>(page);
    std::memcpy(&user_data[2], &table, sizeof(table));
    return constants;
}

} // namespace

// The constants of GT7's draws (DIAG-059 lines of the 11-05 to 11-08 logs, with their FIX-227
// registers) are taken as their draws'.
TEST(TessConstantsPin, ConstantsOfGt7DrawsFitTheirRegisters) {
    EXPECT_TRUE(TcsRegisters(1, 2, 16).FitsConstants(TessConstants(0x28, 0x30, 16, 0x280)));
    EXPECT_TRUE(TcsRegisters(22, 2, 11).FitsConstants(TessConstants(0x18, 0xb0, 11, 0x16b0)));
    // Passing its inputs through: no output base.
    EXPECT_TRUE(TcsRegisters(4, 2, 64).FitsConstants(TessConstants(0x24, 0x24, 64, 0)));
    // Without the LDS emulation the registers' patches are not kept.
    EXPECT_TRUE(TcsRegisters(1, 0, 0).FitsConstants(TessConstants(0x8, 0x30, 16, 0x80)));
    // An 80-byte LS stride (FIX-GT7-119) is a draw's too.
    EXPECT_TRUE(TcsRegisters(20, 2, 12).FitsConstants(TessConstants(80, 0xd0, 12, 20 * 80 * 12)));
}

// What workers read after GT7 wrote that memory again (11-08 logs) is not taken as a draw's.
TEST(TessConstantsPin, ConstantsOfAnotherDrawDoNotFit) {
    const auto regs = TcsRegisters(1, 2, 16);
    EXPECT_FALSE(regs.FitsConstants(TessConstants(0x3F800000, 0x30, 16, 0x280)));  // 1.0f
    EXPECT_FALSE(regs.FitsConstants(TessConstants(0x10028000, 0x1000e329, 0, 0))); // addresses
    EXPECT_FALSE(regs.FitsConstants(TessConstants(0, 0, 0, 0))); // unmapped: zeros
    EXPECT_FALSE(regs.FitsConstants(TessConstants(0x2a, 0x30, 16, 0x2a0)));
    // The same program's constants of a draw with other patches.
    EXPECT_FALSE(regs.FitsConstants(TessConstants(0x28, 0x30, 11, 0x1b8)));
    // Another program's: sane, but not these registers'.
    EXPECT_FALSE(TcsRegisters(22, 2, 11).FitsConstants(TessConstants(0x28, 0x30, 16, 0x280)));
    EXPECT_FALSE(TcsRegisters(1, 0, 0).FitsConstants(TessConstants(0x18, 0xb0, 11, 0x16b0)));
}

TEST(TessConstantsPin, PinnedConstantsStandForMemoryWrittenAgain) {
    u8* page = LowGuestPage();
    ASSERT_TRUE(page != nullptr);
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    auto* constants = PlaceTessConstants(page, user_data);
    const auto pin = Shader::Info::ReadTessConstants(user_data, 2, 16);
    ASSERT_TRUE(pin.has_value());
    EXPECT_EQ(pin->sharp_address, reinterpret_cast<u64>(page) + 16 * 4);
    EXPECT_EQ(pin->constants.ls_stride, 0x28u);
    EXPECT_EQ(pin->constants.hs_output_base, 0x280u);

    // The game writes that memory again for a later draw.
    *constants = TessConstants(0x3F800000, 0x10028000, 0, 0);
    Shader::Info info;
    info.sw_stage = Shader::SwStage::TessellationControl;
    info.user_data = user_data;
    info.tess_consts_ptr_base = static_cast<Shader::IR::ScalarReg>(2);
    info.tess_consts_dword_offset = 16;
    Shader::TessellationDataConstantBuffer read{};
    info.ReadTessConstantBuffer(read);
    EXPECT_EQ(read.ls_stride, 0x3F800000u);
    info.pinned_tess_constants = pin;
    info.ReadTessConstantBuffer(read);
    EXPECT_EQ(read.ls_stride, 0x28u);
    EXPECT_EQ(read.hs_cp_stride, 0x30u);
    EXPECT_EQ(read.hs_output_base, 0x280u);
    // A pin made for another V# is not these constants.
    info.tess_consts_dword_offset = 20;
    info.ReadTessConstantBuffer(read);
    EXPECT_EQ(read.ls_stride, 0x3F800000u);
}

TEST(TessConstantsPin, ReadsAVSharpInTheUserDataRegisters) {
    u8* page = LowGuestPage();
    ASSERT_TRUE(page != nullptr);
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    auto* constants = reinterpret_cast<Shader::TessellationDataConstantBuffer*>(page + 0x200);
    *constants = TessConstants(0x18, 0xb0, 11, 0x16b0);
    AmdGpu::Buffer sharp{};
    sharp.base_address = reinterpret_cast<u64>(constants);
    std::memcpy(&user_data[4], &sharp, sizeof(sharp));
    const u32 in_registers = static_cast<u32>(Shader::IR::NumScalarRegs);
    const auto pin = Shader::Info::ReadTessConstants(user_data, in_registers, 4);
    ASSERT_TRUE(pin.has_value());
    EXPECT_EQ(pin->sharp_address, 0u);
    EXPECT_EQ(pin->constants.hs_output_base, 0x16b0u);
    // Past the registers, or no V#: nothing to pin.
    EXPECT_FALSE(Shader::Info::ReadTessConstants(user_data, in_registers,
                                                 static_cast<s32>(user_data.size()) - 3)
                     .has_value());
    EXPECT_FALSE(Shader::Info::ReadTessConstants(user_data, in_registers, 8).has_value());
    EXPECT_FALSE(Shader::Info::ReadTessConstants(user_data, 2, -1).has_value());
}

// PERF-GT7-111: a recorded translation with pinned constants logs the reads they stand for, in
// the order the unpinned reads come, so the manifest's replay gives the pinned constants.
TEST(TessConstantsPin, PinnedReadsAreLoggedAsTheReadsTheyStandFor) {
    u8* page = LowGuestPage();
    ASSERT_TRUE(page != nullptr);
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    auto* constants = PlaceTessConstants(page, user_data);
    const auto pin = Shader::Info::ReadTessConstants(user_data, 2, 16);
    ASSERT_TRUE(pin.has_value());
    *constants = TessConstants(0x3F800000, 0x10028000, 0, 0);
    Shader::Info info;
    info.user_data = user_data;
    info.tess_consts_ptr_base = static_cast<Shader::IR::ScalarReg>(2);
    info.tess_consts_dword_offset = 16;
    Shader::TessellationDataConstantBuffer read{};
    Common::GuestReadLog unpinned, pinned;
    Common::t_guest_read_log = &unpinned;
    info.ReadTessConstantBuffer(read);
    info.pinned_tess_constants = pin;
    Common::t_guest_read_log = &pinned;
    info.ReadTessConstantBuffer(read);
    Common::t_guest_read_log = nullptr;
    ASSERT_EQ(unpinned.spans.size(), 2u);
    ASSERT_EQ(pinned.spans.size(), 2u);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(pinned.spans[i].address, unpinned.spans[i].address);
        EXPECT_EQ(pinned.spans[i].size, unpinned.spans[i].size);
    }
    Common::GuestReadLog replay = pinned;
    replay.replay = true;
    info.pinned_tess_constants.reset();
    Common::t_guest_read_log = &replay;
    info.ReadTessConstantBuffer(read);
    Common::t_guest_read_log = nullptr;
    EXPECT_EQ(read.ls_stride, 0x28u);
    EXPECT_EQ(replay.misses, 0u);
}

// A domain shader reads only the control point stride: GT7's (0x20 to 0x150 bytes) are taken, what
// a late read finds (1.0f as the stride: 66 million vec4 inputs) is not.
TEST(TessConstantsPin, DomainShaderConstantsFit) {
    using Tes = Shader::SwTessEvalRuntimeInfo;
    EXPECT_TRUE(Tes::FitsConstants(TessConstants(0x28, 0x30, 16, 0x280)));
    EXPECT_TRUE(Tes::FitsConstants(TessConstants(0x20, 0x150, 8, 0x1400)));
    EXPECT_FALSE(Tes::FitsConstants(TessConstants(0x28, 0x3F800000, 16, 0x280)));
    EXPECT_FALSE(Tes::FitsConstants(TessConstants(0, 0, 0, 0)));
    EXPECT_FALSE(Tes::FitsConstants(TessConstants(0x28, 0x32, 16, 0x280)));
    EXPECT_FALSE(Tes::FitsConstants(TessConstants(0x28, 0x30, 0, 0x280)));
}

// The read a translation makes without a pin gives the V# with where it was read: pinned again
// (the worker does, for its key and the manifest's record), it stands for the translation's own
// read after the game wrote that memory again, at the same addresses.
TEST(TessConstantsPin, ATranslationsOwnReadStandsForItsKey) {
    u8* page = LowGuestPage();
    ASSERT_TRUE(page != nullptr);
    std::array<u32, Shader::ShaderParams::NumShaderUserData> user_data{};
    auto* constants = PlaceTessConstants(page, user_data);
    Shader::Info info;
    info.user_data = user_data;
    info.tess_consts_ptr_base = static_cast<Shader::IR::ScalarReg>(2);
    info.tess_consts_dword_offset = 16;
    Common::GuestReadLog translation;
    Common::t_guest_read_log = &translation;
    const auto used = info.ReadTessConstantsAt();
    Common::t_guest_read_log = nullptr;
    const auto expected = Shader::Info::ReadTessConstants(user_data, 2, 16);
    ASSERT_TRUE(expected.has_value());
    EXPECT_EQ(used.sharp_address, expected->sharp_address);
    EXPECT_EQ(u64(used.sharp.base_address), u64(expected->sharp.base_address));
    EXPECT_EQ(used.constants.ls_stride, 0x28u);
    ASSERT_EQ(translation.spans.size(), 2u);

    *constants = TessConstants(0x3F800000, 0x10028000, 0, 0);
    info.pinned_tess_constants = used;
    Common::GuestReadLog key;
    Common::t_guest_read_log = &key;
    Shader::TessellationDataConstantBuffer read{};
    info.ReadTessConstantBuffer(read);
    Common::t_guest_read_log = nullptr;
    EXPECT_EQ(read.ls_stride, 0x28u);
    ASSERT_EQ(key.spans.size(), 2u);
    for (size_t i = 0; i < 2; ++i) {
        EXPECT_EQ(key.spans[i].address, translation.spans[i].address);
        EXPECT_EQ(key.spans[i].size, translation.spans[i].size);
    }
    EXPECT_EQ(0, std::memcmp(key.bytes.data(), translation.bytes.data(), key.bytes.size()));
}
