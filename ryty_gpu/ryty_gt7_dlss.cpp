// SPDX-License-Identifier: GPL-2.0-or-later
#include "ryty_gpu/ryty_gt7_dlss.h"
#include "common/dlss_memory.h"
#include "ryty_gpu/ryty_dlss_budget.h"
#include "ryty_gpu/ryty_gt7_dlss_diagnostic.h"

#include <array>
#include <bit>
#include <chrono>
#include <cstring>
#include <limits>
#include <map>
#include <filesystem>
#include <fstream>
#include "common/path_util.h"
#include <string>
#include <half.hpp>
#include <vk_mem_alloc.h>
#include "common/logging/log.h"
#include "common/sharp_options.h"
#include "core/emulator_settings.h"
#include "ryty_gpu/ryty_dlss.h"
#include "ryty_gpu/ryty_gt7_dlss_merge_spv.h"
#include "ryty_gpu/ryty_gt7_dlss_motion_spv.h"
#include "ryty_gpu/ryty_gt7_dlss_swap_spv.h"
#include "video_core/renderer_vulkan/vk_instance.h"
#include "video_core/renderer_vulkan/vk_scheduler.h"
#include "video_core/texture_cache/image.h"

namespace Vulkan {
namespace {
using namespace Gt7Temporal;
void Barrier(vk::CommandBuffer cmd, vk::Image image, vk::ImageAspectFlags aspects,
             vk::ImageLayout from, vk::ImageLayout to) {
    const vk::ImageMemoryBarrier2 b{
        .srcStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .srcAccessMask = from == vk::ImageLayout::eUndefined
                             ? vk::AccessFlags2{}
                             : vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .dstStageMask = vk::PipelineStageFlagBits2::eAllCommands,
        .dstAccessMask = vk::AccessFlagBits2::eMemoryRead | vk::AccessFlagBits2::eMemoryWrite,
        .oldLayout = from,
        .newLayout = to,
        .image = image,
        .subresourceRange = {.aspectMask = aspects, .levelCount = 1, .layerCount = 1}};
    cmd.pipelineBarrier2(
        vk::DependencyInfo{.imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b});
}
struct Texture {
    VideoCore::UniqueImage image;
    vk::UniqueImageView view;
    vk::Format format{};
    vk::Extent2D size{};
    vk::ImageAspectFlags aspects{vk::ImageAspectFlagBits::eColor};
    vk::ImageLayout layout = vk::ImageLayout::eUndefined;
    bool usage_storage{};
    // These are private frame-ring images. On a retired slot, release them to the driver,
    // rather than moving their memory into the unrelated global image recycler.
    void Release() {
        view.reset();
        if (image.image) {
            vmaDestroyImage(image.allocator, image.image, image.allocation);
            image.image = vk::Image{};
            image.allocation = VK_NULL_HANDLE;
            image.size_bytes = 0;
        }
    }
    ~Texture() { Release(); }
    RytyDlssG::TemporalImage Descriptor() const {
        return {image, *view, format, size};
    }
    void Transit(vk::CommandBuffer cmd, vk::ImageLayout next) {
        Barrier(cmd, image, aspects, layout, next);
        layout = next;
    }
    bool Create(const Instance& gpu, vk::Format fmt, vk::Extent2D extent,
                vk::ImageAspectFlags aspect, vk::ImageUsageFlags extra = {}) {
        if (image && format == fmt && size == extent && aspects == aspect)
            return true;
        Release();
        image = VideoCore::UniqueImage{gpu.GetDevice(), gpu.GetAllocator()};
        image.device = gpu.GetDevice();
        const vk::ImageCreateInfo info{.imageType = vk::ImageType::e2D,
                                       .format = fmt,
                                       .extent = {extent.width, extent.height, 1},
                                       .mipLevels = 1,
                                       .arrayLayers = 1,
                                       .samples = vk::SampleCountFlagBits::e1,
                                       .usage = vk::ImageUsageFlagBits::eSampled |
                                                vk::ImageUsageFlagBits::eTransferSrc |
                                                vk::ImageUsageFlagBits::eTransferDst | extra};
        const auto raw = static_cast<VkImageCreateInfo>(info);
        const VmaAllocationCreateInfo alloc{.flags = VMA_ALLOCATION_CREATE_WITHIN_BUDGET_BIT,
                                            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkImage handle{};
        VmaAllocation memory{};
        VmaAllocationInfo allocated{};
        if (vmaCreateImage(gpu.GetAllocator(), &raw, &alloc, &handle, &memory, &allocated) !=
            VK_SUCCESS)
            return false;
        image.image = handle;
        image.allocation = memory;
        image.image_ci = info;
        image.size_bytes = allocated.size;
        auto v = gpu.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
            .image = image,
            .viewType = vk::ImageViewType::e2D,
            .format = fmt,
            .subresourceRange = {.aspectMask = aspect & ~vk::ImageAspectFlagBits::eStencil,
                                 .levelCount = 1,
                                 .layerCount = 1}});
        if (v.result != vk::Result::eSuccess) {
            Release();
            return false;
        }
        view = std::move(v.value);
        format = fmt;
        size = extent;
        aspects = aspect;
        layout = vk::ImageLayout::eUndefined;
        usage_storage = bool(extra & vk::ImageUsageFlagBits::eStorage);
        return true;
    }
    void Copy(vk::CommandBuffer cmd, const Gt7DlssPass::Input& source) {
        Barrier(cmd, source.image, source.aspects, vk::ImageLayout::eGeneral,
                vk::ImageLayout::eTransferSrcOptimal);
        Transit(cmd, vk::ImageLayout::eTransferDstOptimal);
        const auto aspect = aspects & ~vk::ImageAspectFlagBits::eStencil;
        cmd.copyImage(source.image, vk::ImageLayout::eTransferSrcOptimal, image,
                      vk::ImageLayout::eTransferDstOptimal,
                      vk::ImageCopy{.srcSubresource = {aspect, 0, 0, 1},
                                    .dstSubresource = {aspect, 0, 0, 1},
                                    .extent = {size.width, size.height, 1}});
        Barrier(cmd, source.image, source.aspects, vk::ImageLayout::eTransferSrcOptimal,
                vk::ImageLayout::eGeneral);
        Transit(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    }
};
struct Probe {
    VmaAllocator allocator{};
    VkBuffer buffer{};
    VmaAllocation allocation{};
    void* mapped{};
    u64 tick{}, frame{}, epoch{};
    std::array<float, 2> correction{}, jitter{};
    bool inverted{};
    vk::Extent2D size{};
    ~Probe() {
        if (buffer)
            vmaDestroyBuffer(allocator, buffer, allocation);
    }
};
constexpr unsigned ProbeTiles = 9;
constexpr unsigned ProbeBytes = ProbeTiles * 512;
unsigned TexelBytes(vk::Format format) {
    switch (format) {
    case vk::Format::eR16G16B16A16Sfloat: return 8;
    case vk::Format::eB10G11R11UfloatPack32:
    case vk::Format::eA2B10G10R10UnormPack32:
    case vk::Format::eA2R10G10B10UnormPack32:
    case vk::Format::eR8G8B8A8Unorm:
    case vk::Format::eR8G8B8A8Srgb:
    case vk::Format::eB8G8R8A8Unorm:
    case vk::Format::eB8G8R8A8Srgb:
    case vk::Format::eR16G16Sfloat:
    case vk::Format::eR32Sfloat:
    case vk::Format::eD32Sfloat:
    case vk::Format::eD32SfloatS8Uint: return 4; // the depth aspect copies as packed float32
    default: return 0;
    }
}
std::filesystem::path DiagnosticDirectory() {
    return Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "dlss-diagnostic";
}

void Blit(vk::CommandBuffer cmd, vk::Image source, vk::Extent2D in,
          vk::Image target, vk::Extent2D out) {
    const vk::ImageBlit region{
        .srcSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .srcOffsets = std::array{vk::Offset3D{}, vk::Offset3D{int(in.width), int(in.height), 1}},
        .dstSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
        .dstOffsets = std::array{vk::Offset3D{}, vk::Offset3D{int(out.width), int(out.height), 1}}};
    cmd.blitImage(source, vk::ImageLayout::eTransferSrcOptimal, target,
                  vk::ImageLayout::eTransferDstOptimal, region, vk::Filter::eLinear);
}
} // namespace

struct SharpReadback {
    VmaAllocator allocator{};
    VkBuffer buffer{};
    VmaAllocation allocation{};
    void* mapped{};
    u64 bytes{}, allocated_bytes{}, tick{}, frame{};
    unsigned bundle{};
    std::string name, metadata;
    ~SharpReadback() {
        if (buffer) vmaDestroyBuffer(allocator, buffer, allocation);
    }
};

static void ReportDiagnosticIo(bool success, unsigned bundle, std::string_view detail,
                               std::uint64_t value) noexcept {
    try {
        if (success) {
            LOG_WARNING(Render_Vulkan, "GT7-DLSS dump-retired: bundle={} file={} bytes={}", bundle,
                        detail, value);
        } else {
            LOG_WARNING(Render_Vulkan, "GT7-DLSS dump-aborted: bundle={} reason={} error={}",
                        bundle, detail, value);
        }
    } catch (...) {
        // Diagnostics cannot propagate a logging/allocation failure into the renderer.
    }
}

struct Gt7DlssPass::Impl {
    const Instance& gpu;
    Scheduler& scheduler;
    struct Slot {
        Texture color, depth, motion, branch, output, scene, ref;
        vk::UniqueImageView color_read, scene_read;
        std::vector<vk::UniqueImageView> merge_views; // native views of this frame's resyncs
        u64 tick{}, frame = ~u64{}, epoch{};
        std::chrono::steady_clock::time_point used = std::chrono::steady_clock::now();
        u64 Bytes() const {
            u64 bytes = 0;
            for (const auto* texture : {&color, &depth, &motion, &branch, &output, &scene, &ref}) {
                if (texture->image) bytes += texture->image.size_bytes;
            }
            return bytes;
        }
    };
    std::vector<std::unique_ptr<Slot>> slots;
    Slot* current{};
    std::int64_t retry_after{};
    u64 budget_recoveries{}, budget_trims{};
    bool budget_was_refused{};
    const char* budget_reason = "ready";
    std::chrono::steady_clock::time_point housekeeping_last{}, inactive_since{};
    u64 sdk_releases{};
    u64 memory_refusals{}, slots_trimmed{}, logged_budget_warnings{};
    u64 OwnedBytes() const {
        u64 bytes = 0;
        for (const auto& slot : slots) bytes += slot->Bytes();
        return bytes;
    }
    void TrimSlots(bool pressure = false, const Slot* preserve = nullptr) {
        const auto now = std::chrono::steady_clock::now();
        const bool inactive = !scene_identity && !branch_ready;
        if (inactive && current && now - current->used > std::chrono::seconds(2) &&
            scheduler.IsFree(current->tick))
            current = nullptr;
        size_t remaining = slots.size();
        std::erase_if(slots, [&](const auto& slot) {
            if (slot.get() == current || slot.get() == preserve || !scheduler.IsFree(slot->tick))
                return false;
            const bool old_scene = slot->epoch != epoch;
            const bool idle = now - slot->used > std::chrono::seconds(2);
            if (pressure || old_scene || (remaining > 2 && idle) || (inactive && idle)) {
                --remaining;
                ++slots_trimmed;
                return true;
            }
            return false;
        });
        Common::Sharp::temporal_image_bytes.store(OwnedBytes(), std::memory_order_relaxed);
    }
    std::vector<std::unique_ptr<Probe>> probes;
    std::vector<std::unique_ptr<SharpReadback>> sharp_readbacks;
    std::unique_ptr<Gt7Diagnostic::Writer> dump_io;
    bool dump_io_tried{};
    unsigned dump_skipped{};
    bool DumpActive(u64 frame) const noexcept {
        return dump_io && dump_frame == frame && !dump_io->Failed(dump_id);
    }
    void AbortDump(std::string_view reason) noexcept {
        if (dump_io)
            dump_io->Fail(dump_id, reason);
        dump_frame = ~u64{};
        dump_request.armed = false;
    }
    bool QueueText(std::string name, std::string text) noexcept {
        if (dump_io && dump_io->Enqueue({dump_id, std::move(name), std::move(text)}))
            return true;
        AbortDump("metadata queue");
        return false;
    }

    Gt7Diagnostic::CaptureRequest dump_request;
    unsigned dump_id{};
    u64 dump_frame = ~u64{}, dump_bytes{}, dump_resync = ~u64{};
    u64 dump_trace_frame = ~u64{}, dump_draws{}, dump_geometry{}, dump_missing_motion{};
    unsigned dump_before_evaluation{}, dump_geometry_before{}, dump_no_motion_export{};
    unsigned dump_trace_dropped{};
    bool dump_trace_starts_before{};
    u64 dump_previous_color_frame = ~u64{};
    Gt7Temporal::Matrix dump_previous_projection{}, dump_previous_view{};
    std::string dump_draw_state;


    unsigned fallback_lines{};
    u64 scene_key{}, epoch{}, last_taa = ~u64{}, taa_draws{},
                              qualified_draws{}, captures{}, evaluations{}, successes{},
                              numeric_passes{}, numeric_failures{};
    bool epoch_pending = true;
    std::string qualification_reason = "no-full-scene-TAA", epoch_reason = "first-scene";
    std::map<std::string, u64> fallbacks, resets;
    std::map<std::string, u64> fallback_frames, fallback_last_frame, logged_fallback_frames;
    // Why no draw was taken as the final scene composite (NoteCompositeReject).
    std::map<std::string, u64> reject_frames, reject_last_frame, logged_reject_frames;
    // Why the unchanged-scene reuse in Evaluate did not hold (first failing term, once per frame),
    // and the frames that kept the last reconstructed scene without a new one (CarryScene).
    std::map<std::string, u64> reuse_reject_frames, reuse_reject_last_frame, logged_reuse_rejects;
    u64 carried{}, logged_carried{}, carried_frame = ~u64{};
    std::array<u64, 3> present_frames{}, logged_present_frames{};
    u64 present_frame = ~u64{}, present_flips{}, logged_present_flips{};
    u64 reset_requests{}, logged_reset_requests{}, unchanged_reuse{}, logged_unchanged_reuse{};
    int present_mode = -1;
    vk::Extent2D spatial_source{};
    const char* spatial_filter = "none";
    u64 spatial_source_frame = ~u64{};
    std::array<u64, 2> spatial_filters{}, logged_spatial_filters{};
    u64 spatial_grants{}, spatial_denials{}, logged_spatial_grants{}, logged_spatial_denials{};
    vk::ComponentMapping consumer_mapping{};
    struct BufferWrite {
        VAddr address;
        u64 size, version;
        bool exact;
    };
    struct UploadProof {
        VAddr address;
        u64 size, version, image;
        u32 count;
    };
    std::vector<BufferWrite> buffer_writes;
    std::map<const void *, UploadProof> uploaded;
    u64 buffer_frame = ~u64{};
    bool buffer_overflow{};
    std::chrono::steady_clock::time_point last_probe{}, last_evaluate_log{},
        last_summary = std::chrono::steady_clock::now();
    vk::UniqueDescriptorSetLayout set_layout;
    vk::UniquePipelineLayout pipeline_layout;
    vk::UniquePipeline motion_pipeline, swap_pipeline;
    vk::Extent2D input{}, output{};
    bool supported{}, branch_ready{}, attempted{}, camera_valid{}, numeric_valid{};
    bool reconstructed{}, scene_dirty{}, display_hdr{};
    u64 injected{}, scene_resyncs{}, scene_replays{}, composites{};
    Input native_scene{};
    u64 captured = ~u64{}, evaluated = ~u64{}, camera_sequence{}, scene_version{},
        display_version{};
    VAddr display_address{};
    const void* scene_identity{};
    std::array<const void*, 2> history_pair{};
    Camera camera{}, previous{};
    std::array<float, 29> constants{};
    History history;
    std::chrono::steady_clock::time_point last_success{};
    // Detail-preserving resync (ryty_gt7_dlss_merge.comp).
    vk::UniqueDescriptorSetLayout merge_set_layout;
    vk::UniquePipelineLayout merge_layout;
    std::array<vk::UniquePipeline, 2> merge_pipeline;
    vk::UniqueSampler merge_sampler;
    u64 resync_merged{}, resync_replaced{}, resync_bypassed{}, last_dirty_ps{};
    std::map<u64, u64> dirty_writers; // pixel shader hash -> resyncs it caused
    // GPU time of the SDK evaluation (timestamp queries around slEvaluateFeature).
    static constexpr unsigned TimeSlots = 8;
    struct TimeSlot {
        u64 tick{};
        bool pending{};
    };
    vk::UniqueQueryPool time_pool;
    std::array<TimeSlot, TimeSlots> time_slots{};
    bool time_failed{};
    double eval_ms_last{}, eval_ms_sum{}, eval_ms_max{};
    u64 eval_ms_count{};
    // What the last evaluation used, for the GT7-DLSS line.
    struct LastEval {
        float jitter_x{}, jitter_y{}, c20{}, c21{};
        bool reset{}, ok{};
        const char* reset_reason = "none";
        vk::Format depth_format{}, color_format{};
        bool depth_aspect{}, inverted{};
        float near_plane{}, far_plane{}, fov{}, mvec_scale{};
        u64 frame{};
    } last_eval;
    struct Window {
        u64 evaluations{}, successes{}, injected{}, merged{}, replaced{}, bypassed{}, replays{},
            fallbacks{}, captures{};
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    } window;
    Impl(const Instance& i, Scheduler& s) : gpu{i}, scheduler{s} {}
    int BeginTime(vk::CommandBuffer cmd) {
        if (time_failed)
            return -1;
        if (!time_pool) {
            auto pool = gpu.GetDevice().createQueryPoolUnique(vk::QueryPoolCreateInfo{
                .queryType = vk::QueryType::eTimestamp, .queryCount = TimeSlots * 2});
            if (pool.result != vk::Result::eSuccess) {
                time_failed = true;
                return -1;
            }
            time_pool = std::move(pool.value);
        }
        for (unsigned i = 0; i < TimeSlots; ++i) {
            if (time_slots[i].pending)
                continue;
            cmd.resetQueryPool(*time_pool, i * 2, 2);
            cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *time_pool, i * 2);
            return int(i);
        }
        return -1; // all slots still waiting for the GPU: skip this sample
    }
    void EndTime(vk::CommandBuffer cmd, int slot) {
        if (slot < 0)
            return;
        cmd.writeTimestamp2(vk::PipelineStageFlagBits2::eAllCommands, *time_pool, unsigned(slot) * 2 + 1);
        time_slots[slot] = {scheduler.CurrentTick(), true};
    }
    void CollectTime() {
        if (!time_pool)
            return;
        const double period_ms = double(gpu.GetImageLimits().timestampPeriod) / 1e6;
        for (unsigned i = 0; i < TimeSlots; ++i) {
            if (!time_slots[i].pending || !scheduler.IsFree(time_slots[i].tick))
                continue;
            std::array<u64, 2> stamps{};
            const vk::Result result = gpu.GetDevice().getQueryPoolResults(
                *time_pool, i * 2, 2, sizeof(stamps), stamps.data(), sizeof(u64),
                vk::QueryResultFlagBits::e64);
            if (result == vk::Result::eNotReady)
                continue;
            time_slots[i].pending = false;
            if (result != vk::Result::eSuccess || stamps[1] < stamps[0])
                continue;
            eval_ms_last = double(stamps[1] - stamps[0]) * period_ms;
            eval_ms_sum += eval_ms_last;
            eval_ms_max = std::max(eval_ms_max, eval_ms_last);
            ++eval_ms_count;
        }
    }
    void QueueDump(vk::CommandBuffer cmd, u64 frame, const char* role, vk::Image image,
                   vk::ImageAspectFlags aspects, vk::ImageLayout layout, vk::Format format,
                   vk::Extent2D size, const char* note) noexcept {
        if (!DumpActive(frame))
            return;
        try {
            const std::string label = role ? role : "unnamed";
            const unsigned texel = TexelBytes(format);
            const auto remaining = Gt7Diagnostic::MaxBundleBytes - dump_bytes;
            const auto bytes =
                Gt7Diagnostic::ReadbackBytes(size.width, size.height, texel, remaining);
            const auto copy_aspect = aspects & ~vk::ImageAspectFlagBits::eStencil;
            if (!cmd || !image || !bytes || layout == vk::ImageLayout::eUndefined ||
                (copy_aspect != vk::ImageAspectFlagBits::eColor &&
                 copy_aspect != vk::ImageAspectFlagBits::eDepth)) {
                if (dump_skipped++ < 8)
                    LOG_WARNING(Render_Vulkan,
                                "GT7-DLSS dump-skipped: bundle={} role={} "
                                "reason=unavailable-plane-or-limit",
                                dump_id, label);
                return;
            }
            // Complete all potentially allocating CPU work before recording GPU references.
            sharp_readbacks.reserve(sharp_readbacks.size() + 1);
            auto rb = std::make_unique<SharpReadback>();
            rb->allocator = gpu.GetAllocator();
            rb->bytes = *bytes;
            rb->bundle = dump_id;
            const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0,
                                          rb->bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            const VmaAllocationCreateInfo alloc{.flags =
                                                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                                    VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                                .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
            VmaAllocationInfo allocated{};
            if (vmaCreateBuffer(rb->allocator, &info, &alloc, &rb->buffer, &rb->allocation,
                                &allocated) != VK_SUCCESS ||
                !rb->buffer || !rb->allocation) {
                AbortDump("readback allocation");
                return;
            }
            rb->mapped = allocated.pMappedData;
            rb->allocated_bytes = allocated.size;
            if (!Gt7Diagnostic::MappedSpan(rb->mapped, rb->bytes, rb->allocated_bytes)) {
                AbortDump("readback mapping");
                return;
            }
            rb->frame = frame;
            rb->name = fmt::format("dlss-{:02}-f{}-{}-{}x{}", dump_id, frame, label, size.width,
                                   size.height);
            const nlohmann::json metadata{
                {"bundle", dump_id},      {"frame", frame},
                {"role", label},          {"width", size.width},
                {"height", size.height},  {"format", vk::to_string(format)},
                {"bytesPerPixel", texel}, {"rowPitch", size.width * texel},
                {"littleEndian", true},   {"note", note ? std::string(note) : std::string{}}};
            rb->metadata = metadata.dump(2);
            rb->tick = scheduler.CurrentTick();
            auto* pending = rb.get();
            sharp_readbacks.push_back(std::move(rb));
            dump_bytes += *bytes;
            // Keep the buffer until this tick completes even if a later bundle step fails.
            Barrier(cmd, image, aspects, layout, vk::ImageLayout::eTransferSrcOptimal);
            cmd.copyImageToBuffer(image, vk::ImageLayout::eTransferSrcOptimal, pending->buffer,
                                  vk::BufferImageCopy{.imageSubresource = {copy_aspect, 0, 0, 1},
                                                      .imageExtent = {size.width, size.height, 1}});
            Barrier(cmd, image, aspects, vk::ImageLayout::eTransferSrcOptimal, layout);
            const vk::BufferMemoryBarrier2 host{
                .srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                .dstAccessMask = vk::AccessFlagBits2::eHostRead,
                .buffer = pending->buffer,
                .offset = 0,
                .size = pending->bytes};
            cmd.pipelineBarrier2({.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &host});
        } catch (...) {
            AbortDump("readback preparation");
        }
    }
    // The detail-preserving resync. False: the caller falls back to the old replace.
    bool MergeScene(vk::CommandBuffer cmd, const Input& native) {
        const auto scene_format = current->scene.format;
        const int variant = scene_format == vk::Format::eR16G16B16A16Sfloat      ? 0
                            : scene_format == vk::Format::eB10G11R11UfloatPack32 ? 1
                                                                                 : -1;
        if (variant < 0 || native.format != scene_format || !(current->scene.usage_storage))
            return false;
        const auto device = gpu.GetDevice();
        if (!merge_layout) {
            const std::array<vk::DescriptorSetLayoutBinding, 3> bindings{{
                {0, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute},
                {1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute},
                {2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eCompute}}};
            auto set = device.createDescriptorSetLayoutUnique(
                {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
                 .bindingCount = u32(bindings.size()), .pBindings = bindings.data()});
            if (set.result != vk::Result::eSuccess)
                return false;
            merge_set_layout = std::move(set.value);
            const vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute, 0, 8};
            auto layout = device.createPipelineLayoutUnique({.setLayoutCount = 1,
                .pSetLayouts = &merge_set_layout.get(), .pushConstantRangeCount = 1,
                .pPushConstantRanges = &push});
            if (layout.result != vk::Result::eSuccess)
                return false;
            merge_layout = std::move(layout.value);
            auto sampler = device.createSamplerUnique(vk::SamplerCreateInfo{
                .magFilter = vk::Filter::eLinear, .minFilter = vk::Filter::eLinear,
                .mipmapMode = vk::SamplerMipmapMode::eNearest,
                .addressModeU = vk::SamplerAddressMode::eClampToEdge,
                .addressModeV = vk::SamplerAddressMode::eClampToEdge,
                .addressModeW = vk::SamplerAddressMode::eClampToEdge, .maxLod = 0.f});
            if (sampler.result != vk::Result::eSuccess)
                return false;
            merge_sampler = std::move(sampler.value);
        }
        if (!merge_pipeline[variant]) {
            const auto* code = variant ? GT7_TDLSS_MERGE_R11G11B10_SPV : GT7_TDLSS_MERGE_RGBA16F_SPV;
            const size_t bytes = variant ? sizeof(GT7_TDLSS_MERGE_R11G11B10_SPV)
                                         : sizeof(GT7_TDLSS_MERGE_RGBA16F_SPV);
            auto module = device.createShaderModuleUnique({.codeSize = bytes, .pCode = code});
            if (module.result != vk::Result::eSuccess)
                return false;
            auto pipeline = device.createComputePipelineUnique(
                {}, vk::ComputePipelineCreateInfo{
                        .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                  .module = *module.value, .pName = "main"},
                        .layout = *merge_layout});
            if (pipeline.result != vk::Result::eSuccess)
                return false;
            merge_pipeline[variant] = std::move(pipeline.value);
        }
        if (!current->ref.Create(gpu, scene_format, input, vk::ImageAspectFlagBits::eColor))
            return false;
        auto native_view = device.createImageViewUnique(vk::ImageViewCreateInfo{
            .image = native.image, .viewType = vk::ImageViewType::e2D, .format = native.format,
            .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
        if (native_view.result != vk::Result::eSuccess)
            return false;
        // The reference: the owned scene scaled down the way the native mirror was, so
        // (native - reference) is exactly what the unreplayed writer changed.
        current->scene.Transit(cmd, vk::ImageLayout::eTransferSrcOptimal);
        current->ref.Transit(cmd, vk::ImageLayout::eTransferDstOptimal);
        Blit(cmd, current->scene.image, output, current->ref.image, input);
        current->ref.Transit(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        current->scene.Transit(cmd, vk::ImageLayout::eGeneral);
        Barrier(cmd, native.image, native.aspects, vk::ImageLayout::eGeneral,
                vk::ImageLayout::eGeneral);
        const vk::DescriptorImageInfo owned_info{.imageView = *current->scene.view,
                                                 .imageLayout = vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo native_info{.sampler = *merge_sampler,
            .imageView = *native_view.value, .imageLayout = vk::ImageLayout::eGeneral};
        const vk::DescriptorImageInfo ref_info{.sampler = *merge_sampler,
            .imageView = *current->ref.view, .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal};
        const std::array<vk::WriteDescriptorSet, 3> writes{{
            {.dstBinding = 0, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &owned_info},
            {.dstBinding = 1, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &native_info},
            {.dstBinding = 2, .descriptorCount = 1,
             .descriptorType = vk::DescriptorType::eCombinedImageSampler, .pImageInfo = &ref_info}}};
        const auto& options = Common::Sharp::Get();
        const std::array<float, 2> limits{options.merge_lo, std::max(options.merge_hi, options.merge_lo + .01f)};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *merge_pipeline[variant]);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *merge_layout, 0, writes);
        cmd.pushConstants(*merge_layout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(limits),
                          limits.data());
        cmd.dispatch((output.width + 7) / 8, (output.height + 7) / 8, 1);
        Barrier(cmd, current->scene.image, current->scene.aspects, vk::ImageLayout::eGeneral,
                vk::ImageLayout::eGeneral);
        current->merge_views.push_back(std::move(native_view.value));
        current->tick = scheduler.CurrentTick();
        scheduler.GetDynamicState().Invalidate();
        return true;
    }
    bool Normalize(vk::CommandBuffer cmd) {
        if (!motion_pipeline) {
            const auto device = gpu.GetDevice();
            const vk::DescriptorSetLayoutBinding binding{0, vk::DescriptorType::eStorageImage, 1,
                                                         vk::ShaderStageFlagBits::eCompute};
            auto set = device.createDescriptorSetLayoutUnique(
                {.flags = vk::DescriptorSetLayoutCreateFlagBits::ePushDescriptor,
                 .bindingCount = 1,
                 .pBindings = &binding});
            if (set.result != vk::Result::eSuccess)
                return false;
            set_layout = std::move(set.value);
            const vk::PushConstantRange push{vk::ShaderStageFlagBits::eCompute, 0, 8};
            auto layout = device.createPipelineLayoutUnique({.setLayoutCount = 1,
                                                             .pSetLayouts = &set_layout.get(),
                                                             .pushConstantRangeCount = 1,
                                                             .pPushConstantRanges = &push});
            if (layout.result != vk::Result::eSuccess)
                return false;
            pipeline_layout = std::move(layout.value);
            auto module = device.createShaderModuleUnique(
                {.codeSize = sizeof(GT7_TDLSS_MOTION_SPV), .pCode = GT7_TDLSS_MOTION_SPV});
            if (module.result != vk::Result::eSuccess)
                return false;
            auto pipeline = device.createComputePipelineUnique(
                {},
                vk::ComputePipelineCreateInfo{.stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                                                        .module = *module.value,
                                                        .pName = "main"},
                                              .layout = *pipeline_layout});
            if (pipeline.result != vk::Result::eSuccess)
                return false;
            motion_pipeline = std::move(pipeline.value);
        }
        current->motion.Transit(cmd, vk::ImageLayout::eGeneral);
        const vk::DescriptorImageInfo image{.imageView = *current->motion.view,
                                            .imageLayout = vk::ImageLayout::eGeneral};
        const vk::WriteDescriptorSet write{.dstBinding = 0,
                                           .descriptorCount = 1,
                                           .descriptorType = vk::DescriptorType::eStorageImage,
                                           .pImageInfo = &image};
        const std::array<float, 2> correction{constants[20] * Config().motion_jitter,
                                              constants[21] * Config().motion_jitter};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *motion_pipeline);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *pipeline_layout, 0, write);
        cmd.pushConstants(*pipeline_layout, vk::ShaderStageFlagBits::eCompute, 0,
                          sizeof(correction), correction.data());
        cmd.dispatch((input.width + 7) / 8, (input.height + 7) / 8, 1);
        current->motion.Transit(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        return true;
    }
    void ProbeInputs(vk::CommandBuffer cmd, u64 frame) {
        const auto now = std::chrono::steady_clock::now();
        if (!camera_valid || probes.size() >= 2 ||
            (last_probe.time_since_epoch().count() && now - last_probe < std::chrono::seconds(1)))
            return;
        last_probe = now;
        auto p = std::make_unique<Probe>();
        p->allocator = gpu.GetAllocator();
        const VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, nullptr, 0, ProbeBytes,
                                      VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        const VmaAllocationCreateInfo alloc{.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                                     VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                            .usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST};
        VmaAllocationInfo allocated{};
        if (vmaCreateBuffer(p->allocator, &info, &alloc, &p->buffer, &p->allocation, &allocated) !=
            VK_SUCCESS)
            return;
        p->mapped = allocated.pMappedData;
        p->frame = frame;
        p->epoch = epoch;
        p->tick = scheduler.CurrentTick();
        p->size = input;
        p->correction = {constants[20] * Config().motion_jitter,
                         constants[21] * Config().motion_jitter};
        p->jitter = {-constants[20] * constants[14], -constants[21] * constants[15]};
        p->inverted = camera.lens.inverted;
        for (auto* texture : {&current->depth, &current->motion})
            texture->Transit(cmd, vk::ImageLayout::eTransferSrcOptimal);
        for (unsigned tile = 0; tile < ProbeTiles; ++tile) {
            const vk::Offset3D at{int(input.width * (.2f + .3f * (tile % 3))) - 4,
                                  int(input.height * (.25f + .275f * (tile / 3))) - 4, 0};
            for (unsigned role = 0; role < 2; ++role) {
                auto& t = role == 0 ? current->depth : current->motion;
                const vk::BufferImageCopy region{
                    .bufferOffset = (tile * 2 + role) * 256u,
                    .imageSubresource = {t.aspects & ~vk::ImageAspectFlagBits::eStencil, 0, 0, 1},
                    .imageOffset = at,
                    .imageExtent = {8, 8, 1}};
                cmd.copyImageToBuffer(t.image, vk::ImageLayout::eTransferSrcOptimal, p->buffer,
                                      region);
            }
        }
        for (auto* texture : {&current->depth, &current->motion})
            texture->Transit(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
        const vk::BufferMemoryBarrier2 host{.srcStageMask = vk::PipelineStageFlagBits2::eTransfer,
                                            .srcAccessMask = vk::AccessFlagBits2::eTransferWrite,
                                            .dstStageMask = vk::PipelineStageFlagBits2::eHost,
                                            .dstAccessMask = vk::AccessFlagBits2::eHostRead,
                                            .buffer = p->buffer,
                                            .offset = 0,
                                            .size = ProbeBytes};
        cmd.pipelineBarrier2(
            vk::DependencyInfo{.bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &host});
        probes.push_back(std::move(p));
    }
};

Gt7DlssPass::Gt7DlssPass(const Instance& i, Scheduler& s) : impl{std::make_unique<Impl>(i, s)} {
    const auto& c = Config();
    LOG_INFO(Render_Vulkan,
             "GT7-DLSS: temporal reconstruction armed (jitterSign=({}, {}) mvecMultiplier={} "
             "motionJitter={})",
             c.jitter_sign_x, c.jitter_sign_y, c.mvec_scale, c.motion_jitter);
}
Gt7DlssPass::~Gt7DlssPass() {
    // Resize/reuse and destruction are all timeline based. No device-idle wait.
    if (impl)
        impl->scheduler.DeferOperation([retired = std::move(impl)] {});
}
void Gt7DlssPass::SetDisplay(vk::Extent2D input, vk::Extent2D output, bool hdr) {
    auto& p = *impl;
    Common::Sharp::SetScale(input.width, output.width,
                            EmulatorSettings.IsDlssEnabled() && RytyDlss::CanScale(input, output));
    // Display PQ is produced by the game composite AFTER linear scene reconstruction. The scene
    // is shown through the HD UI proxy, which only exists for an output bigger than the display.
    const bool supported = output.width > input.width && output.height > input.height &&
                           input.width && input.height && output.width <= 8192 &&
                           output.height <= 8192;
    if (p.input != input || p.output != output || p.supported != supported || p.display_hdr != hdr) {
        Invalidate(p.display_hdr != hdr ? "display-encoding-change" : "scene-output-resize");
        p.captured = ~u64{};
        SceneEpoch("scene-output-resize");
        LOG_INFO(Render_Vulkan,
                 "GT7-DLSS: display {}x{} output {}x{} displayPQ={} temporalAllowed={}", input.width,
                 input.height, output.width, output.height, hdr, supported);
    }
    p.input = input;
    p.output = output;
    p.supported = supported;
    p.display_hdr = hdr;
}
void Gt7DlssPass::Invalidate(const char* reason) {
    ++impl->resets[reason];
    impl->history.Invalidate(reason);
    if (std::string_view(reason) == "TAA-initialize" ||
        std::string_view(reason) == "game-history-replaced-or-cleared")
        SceneEpoch(reason);
    impl->branch_ready = false;
    impl->scene_identity = nullptr;
}
void Gt7DlssPass::NoteCompositeReject(u64 frame, const char* reason, u64 ps) {
    auto& p = *impl;
    std::string key = ps ? fmt::format("{:#x}:{}", ps, reason) : std::string{reason};
    const auto last = p.reject_last_frame.find(key);
    if (last == p.reject_last_frame.end() || last->second != frame) {
        p.reject_last_frame[key] = frame;
        ++p.reject_frames[key];
    }
}
bool Gt7DlssPass::CarryScene(u64 frame, const void* identity, u64 version) {
    auto& p = *impl;
    static const bool enabled = [] {
        const char* value = std::getenv("RYGT_DLSS_CARRY_SCENE");
        return !(value && value[0] == '0');
    }();
    if (!enabled || !Config().enabled || !p.current || !p.reconstructed || p.captured == ~u64{} ||
        p.captured >= frame || !p.scene_identity || p.scene_identity != identity || p.scene_dirty ||
        p.scene_version != version)
        return false;
    p.captured = frame;
    p.carried_frame = frame;
    p.branch_ready = false;
    ++p.carried;
    return true;
}
void Gt7DlssPass::Fallback(u64 frame, const char* reason, bool invalidate) {
    if (invalidate)
        Invalidate(reason);
    ++impl->fallbacks[reason];
    const auto previous = impl->fallback_last_frame.find(reason);
    if (previous == impl->fallback_last_frame.end() || previous->second != frame) {
        impl->fallback_last_frame[reason] = frame;
        ++impl->fallback_frames[reason];
    }
    impl->qualification_reason = reason;
    if (Common::Sharp::Get().dlss_log &&
        (impl->fallback_lines++ < 64 || impl->fallback_lines % 240 == 0))
        LOG_INFO(Render_Vulkan,
                 "GT7-DLSS fallback frame={} reason={} nativeResultKept=1 sceneInjected={}", frame,
                 reason, Reconstructed(frame));
}
std::string_view Gt7DlssPass::QualificationReason() const {
    return impl->qualification_reason;
}
void Gt7DlssPass::SceneEpoch(const char *reason) {
    auto &p = *impl;
    p.epoch_pending = true;
    p.numeric_valid = false;
    p.qualification_reason = reason;
    p.epoch_reason = reason;
    p.branch_ready = false;
    p.scene_identity = nullptr;
}
void Gt7DlssPass::NoteDraw(u64 frame, bool taa, bool full_scene, u64 scene_key) {
    auto &p = *impl;
    if (taa) {
        ++p.taa_draws;
        if (full_scene) {
            if (p.scene_key != scene_key) {
                SceneEpoch("scene-resources-change");
                p.scene_key = scene_key;
            }
            if (p.last_taa == ~u64{} || frame > p.last_taa + 2)
                SceneEpoch("full-scene-TAA-gap");
            if (p.epoch_pending) {
                ++p.epoch;
                p.epoch_pending = false;
                p.last_probe = {};
                p.history.Invalidate("new-scene-epoch");
                ++p.resets["new-scene-epoch"];
                if (Common::Sharp::Get().dlss_log)
                    LOG_INFO(Render_Vulkan, "GT7-DLSS epoch={} frame={} reason={}", p.epoch,
                             frame, p.epoch_reason);
            }
            p.last_taa = frame;
        }
    }
}
void Gt7DlssPass::ConsumerMapping(vk::ComponentMapping mapping) {
    impl->consumer_mapping = mapping;
}
void Gt7DlssPass::NoteBufferWrite(u64 frame, VAddr address, u64 size, u64 version, bool exact) {
    if (!Config().enabled || !size || address > std::numeric_limits<VAddr>::max() - size)
        return;
    auto &p = *impl;
    if (p.buffer_frame != frame) {
        p.buffer_frame = frame;
        p.buffer_writes.clear();
        p.uploaded.clear();
        p.buffer_overflow = false;
    }
    // Any later overlapping write revokes an earlier upload's proof.
    std::erase_if(p.uploaded, [&](const auto &pair) {
        const auto &u = pair.second;
        return address < u.address + u.size && u.address < address + size;
    });
    if (p.buffer_writes.size() >= 4096)
        p.buffer_overflow = true;
    if (p.buffer_overflow) {
        p.uploaded.clear();
        return;
    }
    p.buffer_writes.push_back({address, size, version, exact});
}
void Gt7DlssPass::NoteUpload(const void *identity, VAddr address, u64 size, u64 frame, u32 kind,
                             u32 count, u64 image) {
    auto &p = *impl;
    RevokeUpload(identity);
    // Kinds 1..3 read GPU memory in command order. Kind 4 is a CPU snapshot.
    if (!Config().enabled || p.buffer_frame != frame || p.buffer_overflow || kind < 1 || kind > 3 ||
        p.uploaded.size() >= 4096 || !size || address > std::numeric_limits<VAddr>::max() - size)
        return;
    for (auto it = p.buffer_writes.rbegin(); it != p.buffer_writes.rend(); ++it) {
        if (it->address >= address + size || address >= it->address + it->size)
            continue;
        // Match native GpuWrote's whole-image upload rule. A newer partial or
        // oversized unbounded storage write cannot inherit an older full proof.
        if (WholeBufferProducer(address, size, it->address, it->size, it->exact))
            p.uploaded[identity] = {address, size, it->version, image, count};
        return;
    }
}
u64 Gt7DlssPass::UploadedVersion(const void *identity, u64 frame, u32 count, u64 image) const {
    const auto &p = *impl;
    const auto found = p.uploaded.find(identity);
    return p.buffer_frame == frame && found != p.uploaded.end() && found->second.count == count && found->second.image == image
               ? found->second.version
               : 0;
}
void Gt7DlssPass::RevokeUpload(const void *identity) {
    impl->uploaded.erase(identity);
    if (identity == impl->scene_identity)
        impl->scene_dirty = true;
}
void Gt7DlssPass::Capture(u64 frame, u64 ps, const Metadata& m, const std::array<Input, 4>& sources,
                          Input history_output, Input scene_output) {
    auto& p = *impl;
    if (!Config().enabled)
        return;
    Poll(frame); // retire asynchronous validation before this frame's reconstruction
    if (p.captured == frame && p.carried_frame != frame) {
        Fallback(frame, "multiple-main-TAA-draws");
        return;
    }
    if (!p.supported || !ProvenSchema(ps) ||
        !Constants(m.words, p.input.width, p.input.height)) {
        Fallback(frame, "unsupported-display-schema-or-region");
        return;
    }
    if (p.display_hdr && sources[2].format != vk::Format::eB10G11R11UfloatPack32 &&
        sources[2].format != vk::Format::eR16G16B16A16Sfloat) {
        Fallback(frame, "HDR-requires-linear-float-scene");
        return;
    }
    const auto blit = vk::FormatFeatureFlagBits2::eBlitSrc | vk::FormatFeatureFlagBits2::eBlitDst |
                      vk::FormatFeatureFlagBits2::eSampledImageFilterLinear;
    if (!p.gpu.IsFormatSupported(scene_output.format, blit) ||
        !p.gpu.IsFormatSupported(vk::Format::eR16G16B16A16Sfloat, blit)) {
        Fallback(frame, "unsupported-HDR-scene-blit-format");
        return;
    }
    // TAA MRT0 is history, MRT1 is resolved scene. Only this schema may accumulate.
    if (!scene_output.image || scene_output.format != sources[2].format ||
        history_output.format != vk::Format::eR16G16B16A16Sfloat ||
        sources[3].format != vk::Format::eR16G16B16A16Sfloat || scene_output.size != p.input ||
        history_output.size != p.input || sources[1].format != vk::Format::eR16G16Sfloat ||
        (sources[0].format != vk::Format::eD32Sfloat &&
         sources[0].format != vk::Format::eD32SfloatS8Uint &&
         sources[0].format != vk::Format::eR32Sfloat)) {
        Fallback(frame, "missing-MRT1-or-input-format");
        return;
    }
    const bool depth_aspect = bool(sources[0].aspects & vk::ImageAspectFlagBits::eDepth);
    if (depth_aspect != (sources[0].format != vk::Format::eR32Sfloat)) {
        Fallback(frame, "depth-format-aspect-mismatch");
        return;
    }
    for (const auto& s : sources) {
        if (!s.image || !s.version || s.size != p.input || s.image == scene_output.image ||
            s.image == history_output.image) {
            Fallback(frame, "input-alias-stale-or-extent");
            return;
        }
    }
    const auto camera = Config().enabled ? MatchCamera(m.words, p.camera_sequence) : std::nullopt;
    p.camera_valid = camera.has_value();
    if (!camera) {
        Fallback(frame, "camera-snapshot-missing-or-ambiguous");
        return;
    }
    p.camera_sequence = camera->sequence;
    const bool pair_changed =
        p.history_pair[0] && !((p.history_pair[0] == sources[3].identity &&
                                p.history_pair[1] == history_output.identity) ||
                               (p.history_pair[1] == sources[3].identity &&
                                p.history_pair[0] == history_output.identity));
    if (pair_changed)
        SceneEpoch("game-history-pair-change");
    Impl::Slot* slot{};
    for (auto& candidate : p.slots)
        if (candidate.get() != p.current && p.scheduler.IsFree(candidate->tick)) {
            slot = candidate.get();
            break;
        }
    const auto color_bytes = sources[2].format == vk::Format::eR16G16B16A16Sfloat ? 8ull : 4ull;
    const auto depth_bytes = sources[0].format == vk::Format::eD32SfloatS8Uint ? 8ull : 4ull;
    const u64 estimate = u64(p.input.width) * p.input.height * (2 * color_bytes + depth_bytes + 4) +
                         u64(p.output.width) * p.output.height * (color_bytes + 12);
    const u64 required = estimate + estimate / 16; // allow for allocation alignment
    const bool grows = !slot || slot->color.format != sources[2].format ||
        slot->depth.format != sources[0].format || slot->color.size != p.input ||
        slot->output.size != p.output;
    if (grows) {
        const auto now = Common::DlssMemory::Milliseconds();
        if (!RytyDlssG::Budget::Retry(now, p.retry_after)) {
            ++p.memory_refusals;
            Fallback(frame, "owned-image-budget", false);
            return;
        }
        const auto decide = [&] {
            const u64 old_bytes = slot ? slot->Bytes() : 0;
            const u64 total = p.OwnedBytes() - old_bytes + required;
            const u64 extra = required > old_bytes ? required - old_bytes : 0;
            const u64 ui = Common::Sharp::ui_image_bytes.load(std::memory_order_relaxed);
            const auto heap = p.gpu.CanReportMemoryUsage() ? p.gpu.GetCurrentMemoryBudget()
                                                           : Instance::MemoryBudget{};
            return RytyDlssG::Budget::Check(total, ui, extra, heap.usage, heap.budget);
        };
        auto decision = decide();
        if (!decision.allowed) {
            Common::DlssMemory::RequestTrim();
            // Preserve the displayed scene and this retry candidate. Spare slots
            // and recycled images are destroyed only after their GPU ticks have
            // retired.
            p.TrimSlots(true, slot);
            VideoCore::TrimUnusedRecycledImages(p.gpu.GetAllocator(), true);
            Vulkan::TrimEmptyVmaBlocks(p.gpu.GetAllocator());
            ++p.budget_trims;
            decision = decide();
        }
        if (!decision.allowed) {
            ++p.memory_refusals;
            p.budget_reason = decision.reason;
            p.budget_was_refused = true;
            p.retry_after = now + 250;
            Fallback(frame, "owned-image-budget", false);
            return;
        }
        p.retry_after = 0;
        p.budget_reason = "ready";
    }
    if (!slot && p.slots.size() < 8) {
        p.slots.push_back(std::make_unique<Impl::Slot>());
        slot = p.slots.back().get();
    }
    if (!slot) {
        Fallback(frame, "owned-frame-ring-busy");
        return;
    }
    const auto allocation_failed = [&](const char* reason) {
        ++p.memory_refusals;
        p.budget_was_refused = true;
        p.budget_reason = reason;
        p.retry_after = Common::DlssMemory::Milliseconds() + 250;
        Common::DlssMemory::RequestTrim();
        // No commands have referenced this candidate during this allocation
        // attempt.
        std::erase_if(p.slots, [&](const auto& candidate) {
            return candidate.get() == slot && candidate.get() != p.current &&
                   p.scheduler.IsFree(candidate->tick);
        });
        Fallback(frame, reason, false);
    };
    slot->scene_read.reset();
    slot->color_read.reset(); // the slot's previous timeline tick has retired
    slot->merge_views.clear();
    if (!slot->color.Create(p.gpu, sources[2].format, p.input, sources[2].aspects) ||
        !slot->depth.Create(p.gpu, sources[0].format, p.input, sources[0].aspects) ||
        !slot->motion.Create(p.gpu, sources[1].format, p.input, sources[1].aspects,
                             vk::ImageUsageFlagBits::eStorage) ||
        !slot->output.Create(p.gpu, vk::Format::eR16G16B16A16Sfloat, p.output,
                             vk::ImageAspectFlagBits::eColor, vk::ImageUsageFlagBits::eStorage)) {
        allocation_failed("owned-image-allocation");
        return;
    }
    vk::ImageUsageFlags scene_usage = vk::ImageUsageFlagBits::eColorAttachment;
    if (Common::Sharp::Get().resync_merge &&
        (scene_output.format == vk::Format::eR16G16B16A16Sfloat ||
         scene_output.format == vk::Format::eB10G11R11UfloatPack32) &&
        p.gpu.IsFormatSupported(scene_output.format, vk::FormatFeatureFlagBits2::eStorageImage))
        scene_usage |= vk::ImageUsageFlagBits::eStorage; // MergeScene reads and writes it
    if (!slot->scene.Create(p.gpu, scene_output.format, p.output,
                            vk::ImageAspectFlagBits::eColor, scene_usage)) {
        allocation_failed("owned-HDR-scene-allocation");
        return;
    }
    // DLSS sees semantic RGB, while the scene proxy keeps the game's MRT storage mapping.
    auto input_view = p.gpu.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
        .image = slot->color.image, .viewType = vk::ImageViewType::e2D,
        .format = slot->color.format, .components = sources[2].mapping,
        .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}});
    if (input_view.result != vk::Result::eSuccess) {
        allocation_failed("HDR-input-view-allocation");
        return;
    }
    slot->color_read = std::move(input_view.value);
    if (p.current && (p.current->color.format != sources[2].format ||
                      p.current->depth.format != sources[0].format)) {
        p.history.Invalidate("input-format-change");
        SceneEpoch("input-format-change");
    }
    auto* previous_slot = p.current;
    const auto previous_mapping = p.native_scene.mapping;
    p.current = slot;
    if (p.budget_was_refused) {
        ++p.budget_recoveries;
        p.budget_was_refused = false;
    }
    p.budget_reason = "ready";
    slot->frame = frame;
    slot->epoch = p.epoch;
    slot->used = std::chrono::steady_clock::now();
    slot->tick = p.scheduler.CurrentTick();
    Common::Sharp::temporal_image_bytes.store(p.OwnedBytes(), std::memory_order_relaxed);
    p.constants = m.words;
    p.camera = camera.value_or(Camera{});
    p.captured = frame;
    p.branch_ready = false;
    p.attempted = false;
    p.reconstructed = p.scene_dirty = false;
    p.native_scene = scene_output;
    p.native_scene.mapping = sources[2].mapping;
    p.scene_identity = scene_output.identity;
    p.scene_version = scene_output.version;
    p.history.GamePair(reinterpret_cast<uint64_t>(sources[3].identity),
                       reinterpret_cast<uint64_t>(history_output.identity));
    p.history_pair = {sources[3].identity, history_output.identity};
    const auto cmd = p.scheduler.DirectCommandBuffer();
    slot->color.Copy(cmd, sources[2]);
    slot->depth.Copy(cmd, sources[0]);
    slot->motion.Copy(cmd, sources[1]);
    ++p.captures;
    DiagnosticImage(cmd, frame, sources[2], "color-in");
    DiagnosticImage(cmd, frame, sources[0], "depth");
    DiagnosticImage(cmd, frame, sources[1], "motion-raw");
    DiagnosticImage(cmd, frame, scene_output, "taa-native");
    if (p.DumpActive(frame) && previous_slot && previous_slot->frame + 1 == frame &&
        previous_slot->color.size == p.input && previous_slot->color.format == slot->color.format &&
        previous_mapping == sources[2].mapping) {
        p.dump_previous_color_frame = previous_slot->frame;
        p.QueueDump(cmd, frame, "previous-depth", previous_slot->depth.image,
            previous_slot->depth.aspects, previous_slot->depth.layout,
            previous_slot->depth.format, previous_slot->depth.size,
            "adjacent captured frame; same as previous-color-in");
        p.QueueDump(cmd, frame, "previous-motion-corrected", previous_slot->motion.image,
            previous_slot->motion.aspects, previous_slot->motion.layout,
            previous_slot->motion.format, previous_slot->motion.size,
            "adjacent frame vectors after its normalization");
        p.QueueDump(cmd, frame, "previous-color-in", previous_slot->color.image,
            previous_slot->color.aspects, previous_slot->color.layout, previous_slot->color.format,
            p.input, "adjacent previous frame, identical extent/format/view mapping");
        previous_slot->tick = std::max(previous_slot->tick, p.scheduler.CurrentTick());
    }
    if (p.numeric_valid)
        ++p.qualified_draws;
    p.qualification_reason = p.numeric_valid ? "qualified" : "waiting-for-input-probe";
    p.ProbeInputs(cmd, frame); // independent of the downstream graph
    const auto reconstructed = Reconstruct(cmd, frame);
    if (p.DumpActive(frame) && reconstructed == Result::Success) {
        p.QueueDump(cmd, frame, "dlss-out", p.current->output.image,
            p.current->output.aspects, p.current->output.layout,
            p.current->output.format, p.output, "semantic RGB before native storage mapping");
    }
    if (reconstructed != Result::Success)
        return;
    // Swizzle the owned RGBA output back to the proven MRT1 storage convention.
    // The SDK has already consumed it; its internal history is not this surface.
    if (sources[2].mapping.r == vk::ComponentSwizzle::eB) {
        if (!p.swap_pipeline) {
            const auto device = p.gpu.GetDevice();
            auto module = device.createShaderModuleUnique(
                {.codeSize = sizeof(GT7_TDLSS_SWAP_SPV), .pCode = GT7_TDLSS_SWAP_SPV});
            if (module.result != vk::Result::eSuccess) {
                Fallback(frame, "HDR-swizzle-module");
                return;
            }
            auto pipeline = device.createComputePipelineUnique({}, vk::ComputePipelineCreateInfo{
                .stage = {.stage = vk::ShaderStageFlagBits::eCompute,
                          .module = *module.value, .pName = "main"},
                .layout = *p.pipeline_layout});
            if (pipeline.result != vk::Result::eSuccess) {
                Fallback(frame, "HDR-swizzle-pipeline");
                return;
            }
            p.swap_pipeline = std::move(pipeline.value);
        }
        p.current->output.Transit(cmd, vk::ImageLayout::eGeneral);
        const vk::DescriptorImageInfo image{.imageView = *p.current->output.view,
                                             .imageLayout = vk::ImageLayout::eGeneral};
        const vk::WriteDescriptorSet write{.dstBinding = 0, .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageImage, .pImageInfo = &image};
        cmd.bindPipeline(vk::PipelineBindPoint::eCompute, *p.swap_pipeline);
        cmd.pushDescriptorSetKHR(vk::PipelineBindPoint::eCompute, *p.pipeline_layout, 0, write);
        cmd.dispatch((p.output.width + 7) / 8, (p.output.height + 7) / 8, 1);
    }
    p.current->output.Transit(cmd, vk::ImageLayout::eTransferSrcOptimal);
    p.current->scene.Transit(cmd, vk::ImageLayout::eTransferDstOptimal);
    Blit(cmd, p.current->output.image, p.output, p.current->scene.image, p.output);
    p.current->scene.Transit(cmd, vk::ImageLayout::eGeneral);
    p.reconstructed = true;
    FinishSceneWrite(cmd, frame, scene_output);
    ++p.injected;
    if (Common::Sharp::Get().dlss_log && (p.injected <= 16 || frame % 240 == 0))
        LOG_INFO(Render_Vulkan, "GT7-DLSS scene-inject frame={} native={}x{} owned={}x{}", frame,
                 p.input.width, p.input.height, p.output.width, p.output.height);
}
bool Gt7DlssPass::HasScene(u64 frame, const void* identity, u64 version) const {
    return Reconstructed(frame) && !impl->scene_dirty && impl->scene_identity == identity &&
           impl->scene_version == version;
}
bool Gt7DlssPass::Reconstructed(u64 frame) const {
    return impl->current && impl->captured == frame && impl->reconstructed;
}
bool Gt7DlssPass::OwnsScene(u64 frame, const void* identity) const {
    return Reconstructed(frame) && impl->scene_identity == identity;
}
vk::Extent2D Gt7DlssPass::SceneExtent() const { return impl->output; }
void Gt7DlssPass::SceneVersion(u64 version) {
    impl->scene_version = version;
    impl->native_scene.version = version;
    impl->scene_dirty = false;
}
void Gt7DlssPass::SyncScene(vk::CommandBuffer cmd, u64 frame, Input native) {
    auto& p = *impl;
    if (!Reconstructed(frame) || native.identity != p.scene_identity || !native.image ||
        native.image != p.native_scene.image ||
        native.format != p.native_scene.format || native.size != p.input)
        return;
    if (!p.scene_dirty && native.version == p.scene_version)
        return;
    // An un-replayed native writer changed the 1080p scene. Never reuse an earlier HDR version.
    // Merge (default): keep the reconstructed detail and apply only what the writer changed.
    // Replace: the old behaviour, the upscaled native scene wins everywhere (loses detail).
    const bool dump = p.DumpActive(frame) && p.dump_resync != frame;
    if (dump) {
        p.dump_resync = frame;
        p.QueueDump(cmd, frame, "resync-owned-before", p.current->scene.image,
            p.current->scene.aspects, p.current->scene.layout, p.current->scene.format,
            p.output, "high-resolution scene before the unreplayed writer");
        DiagnosticImage(cmd, frame, native, "resync-native");
        LOG_WARNING(Render_Vulkan, "GT7-DLSS resync-dump: frame={} ps={:#x} previous={} current={}",
                    frame, p.last_dirty_ps, p.scene_version, native.version);
    }
    ++p.dirty_writers[p.last_dirty_ps];
    const bool merged = Common::Sharp::Get().resync_merge && p.MergeScene(cmd, native);
    if (!merged) {
        Barrier(cmd, native.image, native.aspects, vk::ImageLayout::eGeneral,
                vk::ImageLayout::eTransferSrcOptimal);
        p.current->scene.Transit(cmd, vk::ImageLayout::eTransferDstOptimal);
        Blit(cmd, native.image, native.size, p.current->scene.image, p.output);
        Barrier(cmd, native.image, native.aspects, vk::ImageLayout::eTransferSrcOptimal,
                vk::ImageLayout::eGeneral);
        p.current->scene.Transit(cmd, vk::ImageLayout::eGeneral);
        ++p.resync_replaced;
    } else {
        ++p.resync_merged;
    }
    if (dump) {
        p.QueueDump(cmd, frame, "resync-owned-after", p.current->scene.image,
            p.current->scene.aspects, p.current->scene.layout, p.current->scene.format,
            p.output, "scene after resync");
        if (merged) p.QueueDump(cmd, frame, "resync-reference", p.current->ref.image,
            p.current->ref.aspects, p.current->ref.layout, p.current->ref.format,
            p.input, "downsampled reconstruction used by the delta merge");
    }
    ++p.scene_resyncs;
    if (Common::Sharp::Get().dlss_log && (p.scene_resyncs <= 16 || frame % 240 == 0))
        LOG_INFO(Render_Vulkan, "GT7-DLSS scene-resync frame={} lastWriterPs={:#x} mode={}", frame,
                 p.last_dirty_ps, merged ? "merge" : "replace");
    SceneVersion(native.version);
}
vk::ImageView Gt7DlssPass::SceneTarget(u64 frame, const void* identity, u64 version) const {
    return HasScene(frame, identity, version) ? *impl->current->scene.view : vk::ImageView{};
}
void Gt7DlssPass::FinishSceneWrite(vk::CommandBuffer cmd, u64 frame, Input native, bool replay) {
    auto& p = *impl;
    if (!Reconstructed(frame) || native.identity != p.scene_identity ||
        native.image != p.native_scene.image || native.format != p.native_scene.format ||
        native.size != p.input)
        return;
    p.current->scene.Transit(cmd, vk::ImageLayout::eTransferSrcOptimal);
    Barrier(cmd, native.image, native.aspects, vk::ImageLayout::eGeneral,
            vk::ImageLayout::eTransferDstOptimal);
    Blit(cmd, p.current->scene.image, p.output, native.image, native.size);
    Barrier(cmd, native.image, native.aspects, vk::ImageLayout::eTransferDstOptimal,
            vk::ImageLayout::eGeneral);
    p.current->scene.Transit(cmd, vk::ImageLayout::eGeneral);
    SceneVersion(native.version);
    p.scene_replays += replay;
    p.current->tick = p.scheduler.CurrentTick();
}
vk::ImageView Gt7DlssPass::BeginBranch(vk::CommandBuffer cmd, u64 frame, VAddr display, u64 version,
                                       vk::Format format) {
    auto& p = *impl;
    if (!Reconstructed(frame) || p.scene_dirty || p.branch_ready ||
        (format != vk::Format::eA2B10G10R10UnormPack32 && format != vk::Format::eR8G8B8A8Unorm &&
         format != vk::Format::eB8G8R8A8Unorm)) {
        // A second composite after the first one worked is not a failure; the rest are.
        if (!p.branch_ready)
            NoteCompositeReject(frame, !Reconstructed(frame) ? "branch:not-reconstructed"
                                       : p.scene_dirty       ? "branch:scene-dirty"
                                                             : "branch:target-format",
                                0);
        return {};
    }
    if (!p.current->branch.Create(p.gpu, format, p.output, vk::ImageAspectFlagBits::eColor,
                                  vk::ImageUsageFlagBits::eColorAttachment)) {
        Fallback(frame, "scene-branch-allocation");
        return {};
    }
    auto view = p.gpu.GetDevice().createImageViewUnique(vk::ImageViewCreateInfo{
        .image = p.current->scene.image,
        .viewType = vk::ImageViewType::e2D,
        .format = p.current->scene.format,
        .components = p.consumer_mapping,
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eColor, .levelCount = 1, .layerCount = 1}});
    if (view.result != vk::Result::eSuccess) {
        Fallback(frame, "owned-consumer-view-allocation");
        return {};
    }
    p.current->scene_read = std::move(view.value);
    p.current->branch.Transit(cmd, vk::ImageLayout::eGeneral);
    p.current->scene.Transit(cmd, vk::ImageLayout::eGeneral);
    p.display_address = display;
    p.display_version = version;
    p.current->tick = p.scheduler.CurrentTick();
    return *p.current->branch.view;
}
vk::ImageView Gt7DlssPass::ColorView() const {
    return impl->current ? *impl->current->scene_read : vk::ImageView{};
}
void Gt7DlssPass::FinishBranch(vk::CommandBuffer cmd, u64 frame) {
    if (impl->captured == frame) {
        impl->branch_ready = true;
        if (impl->DumpActive(frame)) impl->QueueDump(impl->scheduler.DirectCommandBuffer(), frame,
            "post-chain", impl->current->branch.image, impl->current->branch.aspects,
            impl->current->branch.layout, impl->current->branch.format,
            impl->output, "game post chain, display encoded, before sharpening/UI");
        ++impl->composites;
        if (Common::Sharp::Get().dlss_log && (impl->composites <= 16 || frame % 240 == 0))
            LOG_INFO(Render_Vulkan, "GT7-DLSS scene-composite frame={} output={}x{}", frame,
                     impl->output.width, impl->output.height);
    }
}
Gt7DlssPass::Result Gt7DlssPass::Evaluate(vk::CommandBuffer cmd, u64 frame, VAddr display,
                                          u64 version, vk::Extent2D output) {
    auto& p = *impl;
    if (p.branch_ready && p.captured == frame && p.display_address == display &&
        p.display_version == version && p.output == output && p.reconstructed) {
        p.current->branch.Transit(cmd, vk::ImageLayout::eGeneral);
        return Result::Success; // SR already ran before post-FX. This is only presentation.
    }
    // UI can flip without rendering another 3D scene. Reuse only the exact last
    // display version and clean owned scene, never a stale pose after a write/cut.
    static const bool reuse_unchanged = [] {
        const char* value = std::getenv("RYGT_DLSS_REUSE_UNCHANGED");
        return !(value && value[0] == '0');
    }();
    if (reuse_unchanged && p.current && p.branch_ready && p.reconstructed &&
        p.captured < frame && p.scene_identity && !p.scene_dirty && !p.history.pending &&
        p.display_address == display && p.display_version == version && p.output == output) {
        p.current->branch.Transit(cmd, vk::ImageLayout::eGeneral);
        p.current->tick = std::max(p.current->tick, p.scheduler.CurrentTick());
        p.history.Commit(frame, p.camera.identity); // no intervening scene update was lost
        ++p.unchanged_reuse;
        return Result::Success;
    }
    if (p.captured != frame) {
        // A frame without a scene of its own that could not reuse the last one: the first term of
        // the reuse predicate above that failed.
        const char* why = !reuse_unchanged                ? "reuse-switched-off"
                          : !p.current                    ? "no-owned-slot"
                          : !p.branch_ready               ? "no-branch-image"
                          : !p.reconstructed              ? "not-reconstructed"
                          : p.captured == ~u64{} || p.captured >= frame ? "no-older-scene"
                          : !p.scene_identity             ? "no-scene-identity"
                          : p.scene_dirty                 ? "scene-dirty"
                          : p.history.pending             ? "history-pending"
                          : p.display_address != display  ? "display-address-differs"
                          : p.display_version != version  ? "display-version-changed"
                          : p.output != output            ? "output-differs"
                                                          : "other";
        const auto last = p.reuse_reject_last_frame.find(why);
        if (last == p.reuse_reject_last_frame.end() || last->second != frame) {
            p.reuse_reject_last_frame[why] = frame;
            ++p.reuse_reject_frames[why];
        }
    }
    // Presentation failure cannot invalidate a successfully evaluated history:
    // its reconstructed scene was already consumed by the native post graph.
    Fallback(frame, p.attempted && p.captured == frame ? "post-scene-composite-unavailable"
                                                     : "no-reconstructed-scene-this-frame",
             !Reconstructed(frame));
    // Viewport 1 can reconstruct the final native scene when viewport 2 succeeded
    // but its high-resolution composite was unavailable. A failed SDK evaluation
    // may have recorded partial commands: retain the native fallback in that case.
    // (A carried frame did not evaluate this frame: its last evaluation's result stands.)
    const bool sdk_failed = p.attempted && p.captured == frame &&
                            ((p.last_eval.frame != frame && p.carried_frame != frame) ||
                             !p.last_eval.ok);
    if (!sdk_failed && p.attempted && p.captured == frame && p.last_eval.ok) {
        if (RytyDlss::PermitSpatialFallback(frame)) ++p.spatial_grants;
        else ++p.spatial_denials;
    }
    return sdk_failed ? Result::Failed : Result::Unavailable;
}
Gt7DlssPass::Result Gt7DlssPass::Reconstruct(vk::CommandBuffer cmd, u64 frame) {
    auto& p = *impl;
    if (p.evaluated == frame)
        return p.attempted ? Result::Failed : Result::Unavailable;
    if (!RytyDlss::TemporalSupports(p.input, p.output)) {
        Fallback(frame, "backend-ratio-or-FG-unavailable");
        return Result::Unavailable;
    }
    if (!p.numeric_valid) {
        Fallback(frame, "waiting-for-asynchronous-input-validation");
        return Result::Unavailable;
    }
    p.evaluated = frame;
    if (!p.Normalize(cmd)) {
        Fallback(frame, "motion-normalization-pipeline");
        return Result::Unavailable;
    }
    if (p.DumpActive(frame)) p.QueueDump(cmd, frame, "motion-corrected", p.current->motion.image,
        p.current->motion.aspects, p.current->motion.layout, p.current->motion.format,
        p.input, "exact motion vectors tagged to DLSS after sentinel/jitter normalization");
    const auto now = std::chrono::steady_clock::now();
    const float ms = p.last_success.time_since_epoch().count()
                         ? std::chrono::duration<float, std::milli>(now - p.last_success).count()
                         : 16.667f;
    const char* reset = p.history.Reason(frame, p.camera.identity, ms);
    if (!reset)
        reset = CameraCut(p.camera.projection, p.camera.view, p.previous.projection, p.previous.view);
    p.current->color.Transit(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    p.current->output.Transit(cmd, vk::ImageLayout::eGeneral);
    const auto& cfg = Config();
    const auto jitter = RasterJitter(p.constants, cfg.jitter_sign_x, cfg.jitter_sign_y);
    RytyDlssG::TemporalFrame f{.frame = frame,
                               .tick = p.scheduler.CurrentTick(),
                               .color = p.current->color.Descriptor(),
                               .depth = p.current->depth.Descriptor(),
                               .motion = p.current->motion.Descriptor(),
                               .output = p.current->output.Descriptor(),
                               .camera = p.camera,
                               .previous_projection =
                                   reset ? p.camera.projection : p.previous.projection,
                               .previous_view = reset ? p.camera.view : p.previous.view,
                               .jitter_x = jitter[0],
                               .jitter_y = jitter[1],
                               .mvec_scale = cfg.mvec_scale,
                               .frame_ms = std::max(ms, .01f),
                               .reset = reset != nullptr,
                               .hdr = true};
    f.color.view = *p.current->color_read;
    if (reset) {
        ++p.resets[reset];
        ++p.reset_requests;
    }
    ++p.evaluations;
    p.attempted = true;
    p.current->tick = p.scheduler.CurrentTick();
    const int time_slot = p.BeginTime(cmd);
    const bool ok = RytyDlss::EvaluateTemporal(cmd, f);
    p.EndTime(cmd, time_slot);
    if (!ok && time_slot >= 0)
        p.time_slots[time_slot].pending = false; // nothing was recorded between the stamps
    p.last_eval = {f.jitter_x, f.jitter_y, p.constants[20], p.constants[21], f.reset, ok,
                   reset ? reset : "none", p.current->depth.format, p.current->color.format,
                   bool(p.current->depth.aspects & vk::ImageAspectFlagBits::eDepth),
                   p.camera.lens.inverted, p.camera.lens.near_plane, p.camera.lens.far_plane,
                   p.camera.lens.fov, f.mvec_scale, frame};
    p.scheduler.GetDynamicState().Invalidate();
    if (Common::Sharp::Get().dlss_log && (p.evaluations <= 16 || !ok) &&
        now - p.last_evaluate_log >= std::chrono::seconds(1)) {
        p.last_evaluate_log = now;
        LOG_INFO(Render_Vulkan,
                 "GT7-DLSS evaluate frame={} input={}x{} output={}x{} jitter=({}, {}) reset={} "
                 "reason={} success={}",
                 frame, p.input.width, p.input.height, p.output.width, p.output.height,
                 f.jitter_x, f.jitter_y, f.reset, reset ? reset : "none", ok);
    }
    if (!ok) {
        Fallback(frame, "failed-evaluation");
        return Result::Failed;
    }
    if (++p.successes == 1)
        LOG_INFO(Render_Vulkan, "GT7-DLSS: temporal reconstruction running, {}x{} to {}x{}",
                 p.input.width, p.input.height, p.output.width, p.output.height);
    Common::Sharp::temporal_mip_latched.store(true, std::memory_order_relaxed);
    p.qualification_reason = "qualified";
    p.history.Commit(frame, p.camera.identity);
    p.previous = p.camera;
    p.last_success = now;
    return Result::Success;
}
RytyDlssG::TemporalImage Gt7DlssPass::Output() const {
    return impl->current->branch.Descriptor();
}
RytyDlssG::TemporalImage Gt7DlssPass::OutputForPresent(vk::CommandBuffer cmd) {
    impl->current->branch.Transit(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
    impl->current->tick = impl->scheduler.CurrentTick();
    return Output();
}
void Gt7DlssPass::NoteWrite(const void* identity, u64 ps) {
    auto& p = *impl;
    RevokeUpload(identity);
    if (!TaaHash(ps) && !InitializeHash(ps) &&
        (identity == p.history_pair[0] || identity == p.history_pair[1]))
        Invalidate("game-history-replaced-or-cleared");
    if (identity == p.scene_identity && !TaaHash(ps)) {
        p.scene_dirty = true;
        p.last_dirty_ps = ps;
    }
}

void Gt7DlssPass::NoteSpatialSource(u64 frame, vk::Extent2D source, const char* filter) {
    auto& p = *impl;
    p.spatial_source = source;
    p.spatial_filter = filter;
    if (p.spatial_source_frame != frame) {
        ++p.spatial_filters[std::string_view(filter) == "DLSS-reset" ? 1 : 0];
        p.spatial_source_frame = frame;
    }
}

void Gt7DlssPass::NotePresentation(u64 frame, bool temporal, bool spatial) {
    auto& p = *impl;
    if (p.present_frame == frame) return; // first eligible scene handoff, once per frame
    p.present_frame = frame;
    const int mode = temporal ? 0 : spatial ? 1 : 2;
    if (p.present_mode >= 0 && mode != p.present_mode) ++p.present_flips;
    p.present_mode = mode;
    ++p.present_frames[mode];
}

bool Gt7DlssPass::WantsDiagnosticDraws(u64 frame) const {
    return Common::Sharp::Diagnostics() && impl->dump_io &&
           (impl->dump_request.Ready(frame) || impl->DumpActive(frame));
}

void Gt7DlssPass::AbortDiagnostic(const char* reason) noexcept {
    if (impl && Common::Sharp::Diagnostics())
        impl->AbortDump(reason ? reason : "diagnostic error");
}

void Gt7DlssPass::DiagnosticDraw(u64 frame, const std::string& state, bool geometry,
                               bool motion, bool motion_export) {
    if (!WantsDiagnosticDraws(frame))
        return;
    try {
        auto& p = *impl;
        if (p.dump_trace_frame != frame) {
            p.dump_trace_frame = frame;
            p.dump_draw_state.clear();
            p.dump_draws = p.dump_geometry = p.dump_missing_motion = 0;
            p.dump_before_evaluation = p.dump_geometry_before = p.dump_no_motion_export = 0;
            p.dump_trace_dropped = 0;
            p.dump_trace_starts_before = !Reconstructed(frame);
        }
        ++p.dump_draws;
        const bool before = !Reconstructed(frame);
        p.dump_before_evaluation += before;
        if (geometry) {
            p.dump_geometry_before += before;
            p.dump_no_motion_export += !motion_export;
            ++p.dump_geometry;
            p.dump_missing_motion += !motion;
        }
        // A whole frame (shadow cascades, probes, post chain) can exceed 2 MiB before the main
        // pass. Full-resolution depth-tested geometry, which holds the cockpit hands and wheel,
        // keeps its own headroom so the decisive draws are never the ones dropped.
        const std::size_t limit = geometry ? 6 * 1024 * 1024 : 2 * 1024 * 1024;
        if (p.dump_draw_state.size() + state.size() + 40 <= limit) {
            p.dump_draw_state += fmt::format("draw={} {}\n", p.dump_draws, state);
        } else {
            ++p.dump_trace_dropped;
        }
    } catch (...) {
        impl->AbortDump("draw metadata");
    }
}

void Gt7DlssPass::DiagnosticImage(vk::CommandBuffer cmd, u64 frame, Input image, const char* role) {
    if (!Common::Sharp::Diagnostics())
        return;
    if (!impl->dump_io)
        return;
    try {
        auto& p = *impl;
        const bool starting = p.dump_request.Take(frame);
        if (starting) {
            p.dump_frame = frame;
            p.dump_bytes = 0;
            p.dump_skipped = 0;
            p.dump_previous_color_frame = ~u64{};
            p.dump_previous_projection = p.previous.projection;
            p.dump_previous_view = p.previous.view;
            ++p.dump_id;
            LOG_WARNING(Render_Vulkan, "GT7-DLSS dump: bundle={} frame={} dir={}", p.dump_id, frame,
                        DiagnosticDirectory().string());
        }
        if (!p.DumpActive(frame))
            return;
        if (starting || (role && std::string_view(role) == "final-scene")) {
            const nlohmann::json metadata{
                {"bundle", p.dump_id},
                {"frame", frame},
                {"capturedFrame", p.captured},
                {"evaluatedFrame", p.evaluated},
                {"lastTaa", p.last_taa},
                {"reason", p.qualification_reason},
                {"numericValid", p.numeric_valid},
                {"reconstructed", Reconstructed(frame)},
                {"sdkSuccess", p.last_eval.frame == frame && p.last_eval.ok},
                {"temporalCapturedThisFrame", p.captured == frame},
                {"temporalEvaluatedThisFrame", p.last_eval.frame == frame},
                {"presentationPath", p.present_frame != frame ? "pending"
                                     : p.present_mode == 0    ? "temporal"
                                     : p.present_mode == 1    ? "spatial"
                                                              : "native"},
                {"sdkFrame", p.last_eval.frame},
                {"constants", p.constants},
                {"jitterPixels", RasterJitter(p.constants)},
                {"mvecScale", {-.5f * Config().mvec_scale, .5f * Config().mvec_scale}},
                {"projection", p.camera.projection},
                {"view", p.camera.view},
                {"mip", Common::Sharp::ResolutionBias()},
                {"displayPQ", p.display_hdr},
                {"historyPending", p.history.pending != nullptr},
                {"historyResetReason",
                 p.history.pending ? nlohmann::json(p.history.pending) : nlohmann::json(nullptr)},
                {"depthInverted", p.camera.lens.inverted},
                {"preExposure", Common::Sharp::Get().pre_exposure},
                {"exposureScale", Common::Sharp::Get().exposure_scale},
                {"autoExposure", Common::Sharp::Get().auto_exposure},
                {"colorViewSwizzle",
                 {int(p.native_scene.mapping.r), int(p.native_scene.mapping.g),
                  int(p.native_scene.mapping.b), int(p.native_scene.mapping.a)}},
                {"draws", p.dump_draws},
                {"geometryDepthDraws", p.dump_geometry},
                {"geometryWithoutMotionTarget", p.dump_missing_motion},
                {"geometryWithoutMotionExport", p.dump_no_motion_export},
                {"traceFrame", p.dump_trace_frame},
                {"traceStartsBeforeEvaluation", p.dump_trace_starts_before},
                {"traceDrawsBeforeEvaluation", p.dump_before_evaluation},
                {"traceGeometryBeforeEvaluation", p.dump_geometry_before},
                {"traceDroppedDraws", p.dump_trace_dropped},
                {"earliestDiagnosticFrame", p.dump_request.earliest_frame},
                {"previousColorFrame", p.dump_previous_color_frame == ~u64{}
                    ? nlohmann::json(nullptr) : nlohmann::json(p.dump_previous_color_frame)},
                {"previousProjection", p.dump_previous_projection},
                {"previousView", p.dump_previous_view},
                {"normalizationCorrection", {p.constants[20] * Config().motion_jitter,
                                              p.constants[21] * Config().motion_jitter}},
                {"cameraMotionIncluded", true},
                {"motionVectorsJittered", false}};
            if (!p.QueueText(fmt::format("dlss-{:02}-f{}-frame.json", p.dump_id, frame),
                             metadata.dump(2)))
                return;
            if (p.dump_trace_frame == frame &&
                !p.QueueText(fmt::format("dlss-{:02}-f{}-draws.txt", p.dump_id, frame),
                             p.dump_draw_state))
                return;
            if (!starting)
                LOG_WARNING(Render_Vulkan,
                            "GT7-DLSS cockpit-evidence: frame={} draws={} geometry={} "
                            "withoutMotionTarget={} withoutMotionExport={} beforeEvaluation={} "
                            "geometryBefore={} traceDropped={} traceStartBefore={} "
                            "(attachment/export contract, fragment velocity requires the planes)",
                            frame, p.dump_draws, p.dump_geometry, p.dump_missing_motion,
                            p.dump_no_motion_export, p.dump_before_evaluation,
                            p.dump_geometry_before, p.dump_trace_dropped,
                            p.dump_trace_starts_before);
        }
        p.QueueDump(cmd, frame, role, image.image, image.aspects, vk::ImageLayout::eGeneral,
                    image.format, image.size, "raw storage channels; scene before UI");
    } catch (...) {
        impl->AbortDump("frame metadata");
    }
}

void Gt7DlssPass::Poll(u64 frame) {
    auto& p = *impl;
    p.CollectTime();
    p.TrimSlots();
    const auto memory_now = std::chrono::steady_clock::now();
    if (!p.scene_identity && !p.branch_ready) {
        if (p.inactive_since == std::chrono::steady_clock::time_point{})
            p.inactive_since = memory_now;
    } else {
        p.inactive_since = {};
    }
    if (Config().enabled && memory_now - p.housekeeping_last >= std::chrono::seconds{1}) {
        p.housekeeping_last = memory_now;
        if (p.inactive_since != std::chrono::steady_clock::time_point{} &&
            memory_now - p.inactive_since >= std::chrono::seconds{2} &&
            RytyDlssG::ReleaseIdleTemporal(p.scheduler.GetWorkSemaphore()->KnownGpuTick())) {
            ++p.sdk_releases;
            p.history.Invalidate("SDK-idle-release");
        }
        const auto heap = p.gpu.CanReportMemoryUsage() ? p.gpu.GetCurrentMemoryBudget()
                                                       : Instance::MemoryBudget{};
        const auto threshold =
            heap.budget ? std::min(p.gpu.GetTotalMemoryBudget(), heap.budget - heap.budget / 8) : 0;
        if (threshold && heap.usage >= threshold) {
            Common::DlssMemory::RequestTrim();
            VideoCore::TrimUnusedRecycledImages(p.gpu.GetAllocator(), true);
            Vulkan::TrimEmptyVmaBlocks(p.gpu.GetAllocator());
        }
    }
    if (Common::Sharp::Diagnostics()) {
        try {
            if (!p.dump_io_tried) {
                p.dump_io_tried = true;
                p.dump_io = std::make_unique<Gt7Diagnostic::Writer>(
                    DiagnosticDirectory(),
                    Common::FS::GetUserPath(Common::FS::PathType::LogDir) / "dlss.now",
                    &ReportDiagnosticIo);
            }
            if (p.dump_io && !p.dump_request.armed && !p.DumpActive(frame) &&
                p.dump_io->TakeRequest()) {
                p.dump_request.Arm(frame);
                LOG_WARNING(Render_Vulkan,
                            "GT7-DLSS dump-armed: observedFrame={} earliestFrame={}",
                            frame, p.dump_request.earliest_frame);
            }
            for (auto it = p.sharp_readbacks.begin(); it != p.sharp_readbacks.end();) {
                if (!*it) {
                    it = p.sharp_readbacks.erase(it);
                    continue;
                }
                if (!p.scheduler.IsFree((*it)->tick)) {
                    ++it;
                    continue;
                }
                auto retired = std::move(*it);
                it = p.sharp_readbacks.erase(it);
                if (!p.dump_io || p.dump_io->Failed(retired->bundle))
                    continue;
                if (!Gt7Diagnostic::MappedSpan(retired->mapped, retired->bytes,
                                               retired->allocated_bytes) ||
                    vmaInvalidateAllocation(retired->allocator, retired->allocation, 0,
                                            VK_WHOLE_SIZE) != VK_SUCCESS) {
                    p.dump_io->Fail(retired->bundle, "retired readback mapping or invalidation");
                    continue;
                }
                // Shared ownership hands a retired mapping to the file thread without copying
                // a whole image on the recorder. VMA uses internal synchronization here.
                auto owner = std::shared_ptr<SharpReadback>(std::move(retired));
                Gt7Diagnostic::Write binary{owner->bundle, owner->name + ".bin", {},
                                            owner,         owner->mapped,        owner->bytes};
                Gt7Diagnostic::Write metadata{owner->bundle, owner->name + ".json",
                                              owner->metadata};
                if (!p.dump_io->Enqueue(std::move(metadata)) ||
                    !p.dump_io->Enqueue(std::move(binary))) {
                    p.dump_io->Fail(owner->bundle, "retired output queue");
                }
            }
        } catch (...) {
            if (p.dump_io)
                p.AbortDump("diagnostic polling");
            else
                ReportDiagnosticIo(false, 0, "writer initialization", 0);
        }
    }
    for (auto it = p.probes.begin(); it != p.probes.end();) {
        auto& probe = **it;
        if (!p.scheduler.IsFree(probe.tick)) {
            ++it;
            continue;
        }
        vmaInvalidateAllocation(probe.allocator, probe.allocation, 0, VK_WHOLE_SIZE);
        const auto* bytes = static_cast<const u8*>(probe.mapped);
        unsigned geometry = 0, usable_motion = 0;
        bool sane = true;
        for (unsigned tile = 0; tile < ProbeTiles; ++tile) {
            float lo = 1e30f, hi = -1e30f;
            double sum = 0, mx = 0, my = 0;
            unsigned finite = 0, sentinel = 0, clear = 0, motion_finite = 0;
            float min_x = 1e30f, max_x = -1e30f, min_y = 1e30f, max_y = -1e30f;
            for (unsigned i = 0; i < 64; ++i) {
                float depth;
                std::memcpy(&depth, bytes + tile * 512 + i * 4, 4);
                if (std::isfinite(depth)) {
                    lo = std::min(lo, depth);
                    hi = std::max(hi, depth);
                    sum += depth;
                    ++finite;
                    clear += depth == (probe.inverted ? 0.f : 1.f);
                }
                half_float::half v[2];
                std::memcpy(v, bytes + tile * 512 + 256 + i * 4, 4);
                const float x = float(v[0]), y = float(v[1]);
                sentinel += x <= -1000 || y <= -1000;
                if (std::isfinite(x) && std::isfinite(y) && std::abs(x) < 4 && std::abs(y) < 4) {
                    mx += x;
                    my += y;
                    ++motion_finite;
                    if (std::isfinite(depth) && depth > 0 && depth < 1) {
                        ++usable_motion;
                    }
                    min_x = std::min(min_x, x);
                    max_x = std::max(max_x, x);
                    min_y = std::min(min_y, y);
                    max_y = std::max(max_y, y);
                }
            }
            sane &= finite == 64 && lo >= 0 && hi <= 1;
            geometry += finite - clear;
            // Sky/clear samples may legitimately have no vectors. Require motion
            // coverage on actual geometry across the grid, not in every 8x8 tile.
        }
        if (probe.size == p.input && probe.epoch == p.epoch && !p.epoch_pending &&
            p.last_taa != ~u64{} && probe.frame <= p.last_taa && p.last_taa - probe.frame <= 120) {
            const auto verdict = QualifyProbe(sane, geometry, usable_motion);
            if (verdict == ProbeResult::InvalidDepth || verdict == ProbeResult::InvalidMotion) {
                p.numeric_valid = false;
                Invalidate(verdict == ProbeResult::InvalidDepth ? "invalid-depth-or-motion-statistics"
                                                               : "insufficient-geometry-motion");
            } else if (verdict == ProbeResult::Qualified) {
                p.numeric_valid = true;
            } // all-clear / tiny geometry: inconclusive; keep prior qualification
            if (p.numeric_valid)
                ++p.numeric_passes;
            else if (!sane || geometry >= 16)
                ++p.numeric_failures;
            p.qualification_reason = p.numeric_valid ? "qualified"
                                     : !sane ? "invalid-depth-or-motion-statistics"
                                     : geometry < 16 ? "empty-probe-grid-retry"
                                                     : "insufficient-geometry-motion";
        }
        it = p.probes.erase(it);
    }
    // One status line a second, only with GPU.dlss_log (default off).
    const auto now = std::chrono::steady_clock::now();
    if ((Common::Sharp::Get().dlss_log || Common::Sharp::Diagnostics()) && Config().enabled &&
        now - p.window.start >= std::chrono::seconds(1)) {
        auto& w = p.window;
        const u64 evaluations = p.evaluations - w.evaluations, successes = p.successes - w.successes,
                  captures = p.captures - w.captures, merged = p.resync_merged - w.merged,
                  replaced = p.resync_replaced - w.replaced;
        u64 fallbacks = 0;
        for (const auto& [reason, count] : p.fallbacks)
            fallbacks += count;
        const u64 falls = fallbacks - w.fallbacks;
        w.start = now;
        w.evaluations = p.evaluations;
        w.successes = p.successes;
        w.captures = p.captures;
        w.merged = p.resync_merged;
        w.replaced = p.resync_replaced;
        w.fallbacks = fallbacks;
        const auto report = RytyDlssG::LastTemporalReport();
        const double mean_ms = p.eval_ms_count ? p.eval_ms_sum / double(p.eval_ms_count) : 0.0;
        std::array<u64, 3> paths{};
        for (u32 i = 0; i < paths.size(); ++i) {
            paths[i] = p.present_frames[i] - p.logged_present_frames[i];
            p.logged_present_frames[i] = p.present_frames[i];
        }
        std::string reasons;
        for (const auto& [reason, count] : p.fallback_frames) {
            const auto delta = count - p.logged_fallback_frames[reason];
            if (delta) reasons += fmt::format("{}{}:{}", reasons.empty() ? "" : ",", reason, delta);
            p.logged_fallback_frames[reason] = count;
        }
        std::string rejects;
        for (const auto& [key, count] : p.reject_frames) {
            const auto delta = count - p.logged_reject_frames[key];
            if (delta) rejects += fmt::format("{}{}:{}", rejects.empty() ? "" : ",", key, delta);
            p.logged_reject_frames[key] = count;
        }
        std::string reuse_rejects;
        for (const auto& [key, count] : p.reuse_reject_frames) {
            const auto delta = count - p.logged_reuse_rejects[key];
            if (delta)
                reuse_rejects += fmt::format("{}{}:{}", reuse_rejects.empty() ? "" : ",", key, delta);
            p.logged_reuse_rejects[key] = count;
        }
        const u64 carried_now = p.carried - p.logged_carried;
        p.logged_carried = p.carried;
        VmaBudget heaps[VK_MAX_MEMORY_HEAPS]{};
        vmaGetHeapBudgets(p.gpu.GetAllocator(), heaps);
        u64 device_used = 0, device_budget = 0;
        const auto& memory = p.gpu.GetMemoryProperties();
        for (u32 i = 0; i < memory.memoryHeapCount; ++i) {
            if (memory.memoryHeaps[i].flags & vk::MemoryHeapFlagBits::eDeviceLocal) {
                device_used += heaps[i].usage;
                device_budget += heaps[i].budget;
            }
        }
        LOG_WARNING(
            Render_Vulkan,
            "GT7-DLSS 1s: eval={}/{} captures={} fallbacks={} reason={} "
            "render={}x{} "
            "out={}x{} mode={} preset={}->{} gpuMs={:.2f} (max {:.2f}) "
            "mip={:.2f} "
            "sharpen={:.0f}% resync merge={} replace={} present temporal={} "
            "spatial={} "
            "native={} reuse={} TAAseen={} epoch={} resets={} flips={} "
            "fallbackFrames=[{}] "
            "compositeRejects=[{}] reuseRejects=[{}] sceneCarried={} "
            "warn39={}/{} privateMiB={} uiEstMiB={} recycleMiB={} "
            "deviceMiB={}/{} "
            "gameBudgetMiB={} memExt={} slots={} trimmed={} allocRefused={} "
            "budgetReason={} recovered={} budgetTrims={} sdkReleased={} "
            "spatialSource={}x{} spatialFilter={} spatialFrames={}/{}",
            successes, evaluations, captures, falls, p.qualification_reason, p.input.width,
            p.input.height, p.output.width, p.output.height, report.mode_name,
            report.preset_requested, report.preset_sent, mean_ms, p.eval_ms_max,
            Common::Sharp::ResolutionBias(), Common::Sharp::sharpen_percent.load(), merged,
            replaced, paths[0], paths[1], paths[2], p.unchanged_reuse - p.logged_unchanged_reuse,
            p.taa_draws, p.epoch, p.reset_requests - p.logged_reset_requests,
            p.present_flips - p.logged_present_flips, reasons.empty() ? "none" : reasons,
            rejects.empty() ? "none" : rejects, reuse_rejects.empty() ? "none" : reuse_rejects,
            carried_now, report.budget_warnings - p.logged_budget_warnings, report.budget_warnings,
            p.OwnedBytes() >> 20, Common::Sharp::ui_image_bytes.load() >> 20,
            VideoCore::RecycledImageBytes() >> 20, device_used >> 20, device_budget >> 20,
            p.gpu.GetTotalMemoryBudget() >> 20, p.gpu.CanReportMemoryUsage(), p.slots.size(),
            p.slots_trimmed, p.memory_refusals, p.budget_reason, p.budget_recoveries,
            p.budget_trims, p.sdk_releases, p.spatial_source.width, p.spatial_source.height,
            p.spatial_filter, p.spatial_filters[0] - p.logged_spatial_filters[0],
            p.spatial_filters[1] - p.logged_spatial_filters[1]);
        p.logged_spatial_filters = p.spatial_filters;
        p.logged_budget_warnings = report.budget_warnings;
        p.logged_reset_requests = p.reset_requests;
        p.logged_present_flips = p.present_flips;
        p.logged_unchanged_reuse = p.unchanged_reuse;
        p.eval_ms_sum = p.eval_ms_max = 0.0;
        p.eval_ms_count = 0;
    }
}
} // namespace Vulkan
