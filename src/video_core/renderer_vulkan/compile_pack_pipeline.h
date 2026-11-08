// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

// PERF-GT7-108: compile pack recipes of pipeline libraries (the pre-rasterization and fragment
// shader parts PERF-017 compiles for pipelines that are not shader objects: blended ones, SO-017).
// The create info as the driver saw it, field by field, with the stages' SPIR-V and the part's
// layout; replayed at start into the driver's VkPipelineCache, which keeps it on disk. A chained
// structure this file doesn't know means no recipe: a replay never differs from the original.

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string_view>
#include <vector>

#include "video_core/renderer_vulkan/compile_pack.h"
#include "video_core/renderer_vulkan/vk_common.h"

namespace Vulkan::CompilePack {

/// The layout of a library part: its sets (absent ones null), push constants, flags.
struct LayoutDesc {
    std::uint32_t flags = 0;
    std::uint32_t push_stages = 0;
    std::uint32_t push_size = 0;
    std::vector<std::optional<SetLayout>> sets;
};

namespace lib {
struct Out {
    std::vector<std::uint8_t> bytes;
    template <typename T>
    void Put(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
        bytes.insert(bytes.end(), p, p + sizeof(T));
    }
};
struct In {
    const std::uint8_t* data;
    std::size_t size;
    std::size_t at = 0;
    bool ok = true;
    template <typename T>
    T Get() {
        T value{};
        if (!ok || size - at < sizeof(T)) {
            ok = false;
            return value;
        }
        std::memcpy(&value, data + at, sizeof(T));
        at += sizeof(T);
        return value;
    }
};
inline constexpr std::uint32_t Version = 3; // 2: the attachment sample counts; 3: subgroup sizes
} // namespace lib

/// Serializes a library part's create info (pNext: the library info, then the rendering info).
/// `code_of` gives a stage module's SPIR-V (empty: not kept). False when anything isn't known.
template <typename CodeOf>
bool SerializeLibrary(const vk::GraphicsPipelineCreateInfo& info, const LayoutDesc& layout,
                      CodeOf&& code_of, std::vector<std::uint8_t>& result,
                      bool ignore_specialization = false) {
    lib::Out o;
    o.Put(lib::Version);
    // pNext: VkGraphicsPipelineLibraryCreateInfoEXT -> VkPipelineRenderingCreateInfo.
    const auto* library = static_cast<const VkGraphicsPipelineLibraryCreateInfoEXT*>(info.pNext);
    if (!library || library->sType != VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT) {
        return false;
    }
    const auto* rendering = static_cast<const VkPipelineRenderingCreateInfo*>(library->pNext);
    if (!rendering || rendering->sType != VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO ||
        rendering->colorAttachmentCount > 8) {
        return false;
    }
    // VK_AMD_mixed_attachment_samples / VK_NV_framebuffer_mixed_samples' sample counts, if any.
    const auto* samples = static_cast<const VkAttachmentSampleCountInfoAMD*>(rendering->pNext);
    if (samples && (samples->sType != VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_AMD ||
                    samples->pNext || samples->colorAttachmentCount > 8)) {
        return false;
    }
    o.Put(static_cast<std::uint32_t>(library->flags));
    o.Put(rendering->viewMask);
    o.Put(rendering->colorAttachmentCount);
    for (std::uint32_t i = 0; i < rendering->colorAttachmentCount; ++i) {
        o.Put(static_cast<std::uint32_t>(rendering->pColorAttachmentFormats[i]));
    }
    o.Put(static_cast<std::uint32_t>(rendering->depthAttachmentFormat));
    o.Put(static_cast<std::uint32_t>(rendering->stencilAttachmentFormat));
    o.Put(static_cast<std::uint8_t>(samples != nullptr));
    if (samples) {
        o.Put(samples->colorAttachmentCount);
        for (std::uint32_t i = 0; i < samples->colorAttachmentCount; ++i) {
            o.Put(static_cast<std::uint32_t>(samples->pColorAttachmentSamples[i]));
        }
        o.Put(static_cast<std::uint32_t>(samples->depthStencilAttachmentSamples));
    }
    o.Put(static_cast<std::uint32_t>(static_cast<VkPipelineCreateFlags>(info.flags)));
    // Stages: entry "main", no specialization; pNext only AMD-301's required subgroup size.
    o.Put(info.stageCount);
    for (std::uint32_t i = 0; i < info.stageCount; ++i) {
        const auto& stage = info.pStages[i];
        if ((stage.pSpecializationInfo && !ignore_specialization) || !stage.pName ||
            std::string_view{stage.pName} != "main" || stage.flags) {
            return false;
        }
        std::uint32_t subgroup = 0;
        if (const auto* next = static_cast<const VkBaseInStructure*>(stage.pNext)) {
            if (next->sType !=
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO ||
                next->pNext) {
                return false;
            }
            subgroup = reinterpret_cast<const VkPipelineShaderStageRequiredSubgroupSizeCreateInfo*>(next)
                           ->requiredSubgroupSize;
        }
        const std::vector<std::uint32_t> code = code_of(stage.module);
        if (code.empty()) {
            return false;
        }
        o.Put(static_cast<std::uint32_t>(stage.stage));
        o.Put(subgroup);
        o.Put(static_cast<std::uint32_t>(code.size()));
        const auto* p = reinterpret_cast<const std::uint8_t*>(code.data());
        o.bytes.insert(o.bytes.end(), p, p + code.size() * sizeof(std::uint32_t));
    }
    // Vertex input: absent with VK_EXT_vertex_input_dynamic_state (no divisors otherwise).
    o.Put(static_cast<std::uint8_t>(info.pVertexInputState != nullptr));
    if (const auto* v = info.pVertexInputState) {
        if (v->pNext || v->flags) return false;
        o.Put(v->vertexBindingDescriptionCount);
        for (std::uint32_t i = 0; i < v->vertexBindingDescriptionCount; ++i) {
            const auto& b = v->pVertexBindingDescriptions[i];
            o.Put(b.binding);
            o.Put(b.stride);
            o.Put(static_cast<std::uint32_t>(b.inputRate));
        }
        o.Put(v->vertexAttributeDescriptionCount);
        for (std::uint32_t i = 0; i < v->vertexAttributeDescriptionCount; ++i) {
            const auto& a = v->pVertexAttributeDescriptions[i];
            o.Put(a.location);
            o.Put(a.binding);
            o.Put(static_cast<std::uint32_t>(a.format));
            o.Put(a.offset);
        }
    }
    o.Put(static_cast<std::uint8_t>(info.pInputAssemblyState != nullptr));
    if (const auto* a = info.pInputAssemblyState) {
        if (a->pNext || a->flags) return false;
        o.Put(static_cast<std::uint32_t>(a->topology));
        o.Put(static_cast<std::uint32_t>(a->primitiveRestartEnable));
    }
    o.Put(static_cast<std::uint8_t>(info.pTessellationState != nullptr));
    if (const auto* t = info.pTessellationState) {
        if (t->pNext || t->flags) return false;
        o.Put(t->patchControlPoints);
    }
    o.Put(static_cast<std::uint8_t>(info.pViewportState != nullptr));
    if (const auto* v = info.pViewportState) {
        if (v->flags || v->pViewports || v->pScissors) return false;
        o.Put(v->viewportCount);
        o.Put(v->scissorCount);
        const auto* clip = static_cast<const VkPipelineViewportDepthClipControlCreateInfoEXT*>(v->pNext);
        if (clip && (clip->sType != VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT ||
                     clip->pNext)) {
            return false;
        }
        o.Put(static_cast<std::uint8_t>(clip != nullptr));
        if (clip) o.Put(static_cast<std::uint32_t>(clip->negativeOneToOne));
    }
    o.Put(static_cast<std::uint8_t>(info.pRasterizationState != nullptr));
    if (const auto* r = info.pRasterizationState) {
        if (r->flags) return false;
        o.Put(static_cast<std::uint32_t>(r->depthClampEnable));
        o.Put(static_cast<std::uint32_t>(r->rasterizerDiscardEnable));
        o.Put(static_cast<std::uint32_t>(r->polygonMode));
        o.Put(static_cast<std::uint32_t>(r->cullMode));
        o.Put(static_cast<std::uint32_t>(r->frontFace));
        o.Put(static_cast<std::uint32_t>(r->depthBiasEnable));
        o.Put(r->depthBiasConstantFactor);
        o.Put(r->depthBiasClamp);
        o.Put(r->depthBiasSlopeFactor);
        o.Put(r->lineWidth);
        // Chain: provoking vertex and depth clip, in order, each at most once.
        std::int32_t provoking = -1;
        std::int32_t clip = -1;
        std::vector<std::uint32_t> order;
        for (const auto* next = static_cast<const VkBaseInStructure*>(r->pNext); next;
             next = next->pNext) {
            if (next->sType ==
                    VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT &&
                provoking < 0) {
                provoking = static_cast<std::int32_t>(
                    reinterpret_cast<const VkPipelineRasterizationProvokingVertexStateCreateInfoEXT*>(next)
                        ->provokingVertexMode);
                order.push_back(1);
            } else if (next->sType ==
                           VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT &&
                       clip < 0) {
                const auto* c =
                    reinterpret_cast<const VkPipelineRasterizationDepthClipStateCreateInfoEXT*>(next);
                if (c->flags) return false;
                clip = static_cast<std::int32_t>(c->depthClipEnable);
                order.push_back(2);
            } else {
                return false;
            }
        }
        o.Put(static_cast<std::uint32_t>(order.size()));
        for (const auto kind : order) {
            o.Put(kind);
            o.Put(kind == 1 ? provoking : clip);
        }
    }
    o.Put(static_cast<std::uint8_t>(info.pMultisampleState != nullptr));
    if (const auto* m = info.pMultisampleState) {
        if (m->pNext || m->flags) return false;
        o.Put(static_cast<std::uint32_t>(m->rasterizationSamples));
        o.Put(static_cast<std::uint32_t>(m->sampleShadingEnable));
        o.Put(m->minSampleShading);
        o.Put(static_cast<std::uint32_t>(m->alphaToCoverageEnable));
        o.Put(static_cast<std::uint32_t>(m->alphaToOneEnable));
        const std::uint32_t words = m->pSampleMask ? (std::uint32_t(m->rasterizationSamples) + 31) / 32 : 0;
        o.Put(words);
        for (std::uint32_t i = 0; i < words; ++i) o.Put(m->pSampleMask[i]);
    }
    o.Put(static_cast<std::uint8_t>(info.pDepthStencilState != nullptr));
    if (const auto* d = info.pDepthStencilState) {
        if (d->pNext || d->flags) return false;
        // Field by field into a zeroed copy: the structure's padding (after sType) and its null
        // pNext would otherwise enter the bytes, and the content hash that names the file.
        VkPipelineDepthStencilStateCreateInfo copy{};
        copy.sType = static_cast<VkStructureType>(d->sType);
        copy.depthTestEnable = d->depthTestEnable;
        copy.depthWriteEnable = d->depthWriteEnable;
        copy.depthCompareOp = static_cast<VkCompareOp>(d->depthCompareOp);
        copy.depthBoundsTestEnable = d->depthBoundsTestEnable;
        copy.stencilTestEnable = d->stencilTestEnable;
        copy.front = static_cast<const VkStencilOpState&>(d->front);
        copy.back = static_cast<const VkStencilOpState&>(d->back);
        copy.minDepthBounds = d->minDepthBounds;
        copy.maxDepthBounds = d->maxDepthBounds;
        o.Put(copy);
    }
    o.Put(static_cast<std::uint8_t>(info.pColorBlendState != nullptr));
    if (const auto* c = info.pColorBlendState) {
        if (c->pNext || c->flags || c->attachmentCount > 8) return false;
        o.Put(static_cast<std::uint32_t>(c->logicOpEnable));
        o.Put(static_cast<std::uint32_t>(c->logicOp));
        o.Put(c->attachmentCount);
        for (std::uint32_t i = 0; i < c->attachmentCount; ++i) o.Put(c->pAttachments[i]);
        for (const float constant : c->blendConstants) o.Put(constant);
    }
    o.Put(static_cast<std::uint8_t>(info.pDynamicState != nullptr));
    if (const auto* d = info.pDynamicState) {
        if (d->pNext || d->flags) return false;
        o.Put(d->dynamicStateCount);
        for (std::uint32_t i = 0; i < d->dynamicStateCount; ++i) {
            o.Put(static_cast<std::uint32_t>(d->pDynamicStates[i]));
        }
    }
    if (info.renderPass || info.basePipelineHandle || info.subpass != 0) {
        return false;
    }
    // The layout.
    o.Put(layout.flags);
    o.Put(layout.push_stages);
    o.Put(layout.push_size);
    o.Put(static_cast<std::uint32_t>(layout.sets.size()));
    for (const auto& set : layout.sets) {
        o.Put(static_cast<std::uint8_t>(set.has_value()));
        if (!set) continue;
        o.Put(set->flags);
        o.Put(static_cast<std::uint32_t>(set->bindings.size()));
        for (const auto& b : set->bindings) o.Put(b);
    }
    result = std::move(o.bytes);
    return true;
}

/// A replayable library part: owns what its create info points to.
struct LibraryReplay {
    VkGraphicsPipelineLibraryCreateInfoEXT library{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT};
    VkPipelineRenderingCreateInfo rendering{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    std::vector<VkFormat> color_formats;
    std::optional<VkAttachmentSampleCountInfoAMD> sample_counts;
    std::vector<VkSampleCountFlagBits> color_samples;
    std::vector<std::pair<VkShaderStageFlagBits, std::vector<std::uint32_t>>> stages;
    std::vector<std::uint32_t> subgroups; ///< per stage, 0: none
    std::optional<VkPipelineVertexInputStateCreateInfo> vertex_input;
    std::vector<VkVertexInputBindingDescription> vertex_bindings;
    std::vector<VkVertexInputAttributeDescription> vertex_attributes;
    std::optional<VkPipelineInputAssemblyStateCreateInfo> input_assembly;
    std::optional<VkPipelineTessellationStateCreateInfo> tessellation;
    std::optional<VkPipelineViewportStateCreateInfo> viewport;
    std::optional<VkPipelineViewportDepthClipControlCreateInfoEXT> clip_control;
    std::optional<VkPipelineRasterizationStateCreateInfo> raster;
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT provoking{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT};
    VkPipelineRasterizationDepthClipStateCreateInfoEXT depth_clip{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_DEPTH_CLIP_STATE_CREATE_INFO_EXT};
    std::vector<std::uint32_t> raster_order;
    std::optional<VkPipelineMultisampleStateCreateInfo> multisample;
    std::vector<VkSampleMask> sample_mask;
    std::optional<VkPipelineDepthStencilStateCreateInfo> depth_stencil;
    std::optional<VkPipelineColorBlendStateCreateInfo> color_blend;
    std::vector<VkPipelineColorBlendAttachmentState> blend_attachments;
    std::optional<VkPipelineDynamicStateCreateInfo> dynamic;
    std::vector<VkDynamicState> dynamic_states;
    VkPipelineCreateFlags flags = 0;
    LayoutDesc layout;
};

inline bool DeserializeLibrary(const std::vector<std::uint8_t>& bytes, LibraryReplay& r) {
    lib::In in{bytes.data(), bytes.size()};
    if (in.Get<std::uint32_t>() != lib::Version) return false;
    r.library.flags = in.Get<std::uint32_t>();
    r.rendering.viewMask = in.Get<std::uint32_t>();
    const auto colors = in.Get<std::uint32_t>();
    if (!in.ok || colors > 8) return false;
    for (std::uint32_t i = 0; i < colors; ++i) r.color_formats.push_back(VkFormat(in.Get<std::uint32_t>()));
    r.rendering.depthAttachmentFormat = VkFormat(in.Get<std::uint32_t>());
    r.rendering.stencilAttachmentFormat = VkFormat(in.Get<std::uint32_t>());
    if (in.Get<std::uint8_t>()) {
        VkAttachmentSampleCountInfoAMD counts{VK_STRUCTURE_TYPE_ATTACHMENT_SAMPLE_COUNT_INFO_AMD};
        const auto n = in.Get<std::uint32_t>();
        if (!in.ok || n > 8) return false;
        for (std::uint32_t i = 0; i < n; ++i) {
            r.color_samples.push_back(VkSampleCountFlagBits(in.Get<std::uint32_t>()));
        }
        counts.depthStencilAttachmentSamples = VkSampleCountFlagBits(in.Get<std::uint32_t>());
        r.sample_counts = counts;
    }
    r.flags = in.Get<std::uint32_t>();
    const auto stage_count = in.Get<std::uint32_t>();
    if (!in.ok || stage_count == 0 || stage_count > 5) return false;
    for (std::uint32_t i = 0; i < stage_count; ++i) {
        const auto stage = VkShaderStageFlagBits(in.Get<std::uint32_t>());
        r.subgroups.push_back(in.Get<std::uint32_t>());
        const auto words = in.Get<std::uint32_t>();
        if (!in.ok || words > (in.size - in.at) / 4) return false;
        std::vector<std::uint32_t> code(words);
        std::memcpy(code.data(), in.data + in.at, words * 4);
        in.at += words * 4;
        r.stages.emplace_back(stage, std::move(code));
    }
    if (in.Get<std::uint8_t>()) {
        VkPipelineVertexInputStateCreateInfo v{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
        const auto nb = in.Get<std::uint32_t>();
        if (!in.ok || nb > 64) return false;
        for (std::uint32_t i = 0; i < nb; ++i) {
            VkVertexInputBindingDescription b{};
            b.binding = in.Get<std::uint32_t>();
            b.stride = in.Get<std::uint32_t>();
            b.inputRate = VkVertexInputRate(in.Get<std::uint32_t>());
            r.vertex_bindings.push_back(b);
        }
        const auto na = in.Get<std::uint32_t>();
        if (!in.ok || na > 64) return false;
        for (std::uint32_t i = 0; i < na; ++i) {
            VkVertexInputAttributeDescription a{};
            a.location = in.Get<std::uint32_t>();
            a.binding = in.Get<std::uint32_t>();
            a.format = VkFormat(in.Get<std::uint32_t>());
            a.offset = in.Get<std::uint32_t>();
            r.vertex_attributes.push_back(a);
        }
        r.vertex_input = v;
    }
    if (in.Get<std::uint8_t>()) {
        VkPipelineInputAssemblyStateCreateInfo a{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
        a.topology = VkPrimitiveTopology(in.Get<std::uint32_t>());
        a.primitiveRestartEnable = in.Get<std::uint32_t>();
        r.input_assembly = a;
    }
    if (in.Get<std::uint8_t>()) {
        VkPipelineTessellationStateCreateInfo t{VK_STRUCTURE_TYPE_PIPELINE_TESSELLATION_STATE_CREATE_INFO};
        t.patchControlPoints = in.Get<std::uint32_t>();
        r.tessellation = t;
    }
    if (in.Get<std::uint8_t>()) {
        VkPipelineViewportStateCreateInfo v{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
        v.viewportCount = in.Get<std::uint32_t>();
        v.scissorCount = in.Get<std::uint32_t>();
        if (in.Get<std::uint8_t>()) {
            VkPipelineViewportDepthClipControlCreateInfoEXT c{
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT};
            c.negativeOneToOne = in.Get<std::uint32_t>();
            r.clip_control = c;
        }
        r.viewport = v;
    }
    if (in.Get<std::uint8_t>()) {
        VkPipelineRasterizationStateCreateInfo s{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
        s.depthClampEnable = in.Get<std::uint32_t>();
        s.rasterizerDiscardEnable = in.Get<std::uint32_t>();
        s.polygonMode = VkPolygonMode(in.Get<std::uint32_t>());
        s.cullMode = in.Get<std::uint32_t>();
        s.frontFace = VkFrontFace(in.Get<std::uint32_t>());
        s.depthBiasEnable = in.Get<std::uint32_t>();
        s.depthBiasConstantFactor = in.Get<float>();
        s.depthBiasClamp = in.Get<float>();
        s.depthBiasSlopeFactor = in.Get<float>();
        s.lineWidth = in.Get<float>();
        const auto links = in.Get<std::uint32_t>();
        if (!in.ok || links > 2) return false;
        for (std::uint32_t i = 0; i < links; ++i) {
            const auto kind = in.Get<std::uint32_t>();
            const auto value = in.Get<std::int32_t>();
            if (kind == 1) r.provoking.provokingVertexMode = VkProvokingVertexModeEXT(value);
            else if (kind == 2) r.depth_clip.depthClipEnable = VkBool32(value);
            else return false;
            r.raster_order.push_back(kind);
        }
        r.raster = s;
    }
    if (in.Get<std::uint8_t>()) {
        VkPipelineMultisampleStateCreateInfo m{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
        m.rasterizationSamples = VkSampleCountFlagBits(in.Get<std::uint32_t>());
        m.sampleShadingEnable = in.Get<std::uint32_t>();
        m.minSampleShading = in.Get<float>();
        m.alphaToCoverageEnable = in.Get<std::uint32_t>();
        m.alphaToOneEnable = in.Get<std::uint32_t>();
        const auto words = in.Get<std::uint32_t>();
        if (!in.ok || words > 2) return false;
        for (std::uint32_t i = 0; i < words; ++i) r.sample_mask.push_back(in.Get<VkSampleMask>());
        r.multisample = m;
    }
    if (in.Get<std::uint8_t>()) {
        auto d = in.Get<VkPipelineDepthStencilStateCreateInfo>();
        d.pNext = nullptr;
        r.depth_stencil = d;
    }
    if (in.Get<std::uint8_t>()) {
        VkPipelineColorBlendStateCreateInfo c{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
        c.logicOpEnable = in.Get<std::uint32_t>();
        c.logicOp = VkLogicOp(in.Get<std::uint32_t>());
        const auto count = in.Get<std::uint32_t>();
        if (!in.ok || count > 8) return false;
        for (std::uint32_t i = 0; i < count; ++i) {
            r.blend_attachments.push_back(in.Get<VkPipelineColorBlendAttachmentState>());
        }
        for (float& constant : c.blendConstants) constant = in.Get<float>();
        r.color_blend = c;
    }
    if (in.Get<std::uint8_t>()) {
        VkPipelineDynamicStateCreateInfo d{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
        const auto count = in.Get<std::uint32_t>();
        if (!in.ok || count > 128) return false;
        for (std::uint32_t i = 0; i < count; ++i) r.dynamic_states.push_back(VkDynamicState(in.Get<std::uint32_t>()));
        r.dynamic = d;
    }
    r.layout.flags = in.Get<std::uint32_t>();
    r.layout.push_stages = in.Get<std::uint32_t>();
    r.layout.push_size = in.Get<std::uint32_t>();
    const auto sets = in.Get<std::uint32_t>();
    if (!in.ok || sets > 8) return false;
    for (std::uint32_t s = 0; s < sets; ++s) {
        if (!in.Get<std::uint8_t>()) {
            r.layout.sets.emplace_back();
            continue;
        }
        SetLayout set;
        set.flags = in.Get<std::uint32_t>();
        const auto count = in.Get<std::uint32_t>();
        if (!in.ok || count > 4096) return false;
        for (std::uint32_t i = 0; i < count; ++i) set.bindings.push_back(in.Get<Binding>());
        r.layout.sets.emplace_back(std::move(set));
    }
    return in.ok && in.at == in.size;
}

/// Stores serialized bytes in a pack file (the recipe format's header, kind 'L').
inline bool WriteLibraryFile(const std::filesystem::path& file, const std::vector<std::uint8_t>& raw) {
    Recipe wrapper; // reuses Write's compression and header through one stage of raw words
    wrapper.key = XXH3_64bits(raw.data(), raw.size());
    wrapper.flags = 0x4C494200u; // "LIB": a library recipe
    auto& stage = wrapper.stages.emplace_back();
    stage.code.resize((raw.size() + 3) / 4 + 1);
    stage.code[0] = static_cast<std::uint32_t>(raw.size());
    std::memcpy(stage.code.data() + 1, raw.data(), raw.size());
    return Write(file, wrapper);
}

inline bool ReadLibraryFile(const std::filesystem::path& file, std::vector<std::uint8_t>& raw) {
    Recipe wrapper;
    if (!Read(file, wrapper) || wrapper.flags != 0x4C494200u || wrapper.stages.size() != 1 ||
        wrapper.stages[0].code.empty()) {
        return false;
    }
    const auto size = wrapper.stages[0].code[0];
    if (size > (wrapper.stages[0].code.size() - 1) * 4) return false;
    raw.resize(size);
    std::memcpy(raw.data(), wrapper.stages[0].code.data() + 1, size);
    return true;
}

} // namespace Vulkan::CompilePack
