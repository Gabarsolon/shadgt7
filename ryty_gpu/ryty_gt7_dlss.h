// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <memory>
#include <string_view>
#include <vector>
#include "ryty_gpu/ryty_dlssg.h"

namespace Vulkan {
class Instance;
class Scheduler;
class Gt7DlssPass {
public:
    struct Metadata {
        std::vector<u8> bytes;
        std::array<float, 29> words{};
        VAddr address{};
    };
    struct Input {
        vk::Image image{};
        vk::Format format{};
        vk::Extent2D size{};
        vk::ImageAspectFlags aspects{vk::ImageAspectFlagBits::eColor};
        VAddr address{};
        const void* identity{};
        u64 version{}, generation{};
        vk::ComponentMapping mapping{};
    };
    enum class Result { Unavailable, Failed, Success };
    Gt7DlssPass(const Instance&, Scheduler&);
    ~Gt7DlssPass();
    void SetDisplay(vk::Extent2D input, vk::Extent2D output, bool hdr);
    void Capture(u64 frame, u64 ps, const Metadata&, const std::array<Input, 4>&,
                 Input history_output, Input scene_output);
    bool HasScene(u64 frame, const void* identity, u64 version) const;
    vk::ImageView BeginBranch(vk::CommandBuffer, u64 frame, VAddr display, u64 version, vk::Format);
    vk::ImageView ColorView() const;
    void FinishBranch(vk::CommandBuffer, u64 frame);
    bool Reconstructed(u64 frame) const;
    bool OwnsScene(u64 frame, const void* identity) const;
    vk::Extent2D SceneExtent() const;
    void SceneVersion(u64 version);
    void SyncScene(vk::CommandBuffer, u64 frame, Input native);
    vk::ImageView SceneTarget(u64 frame, const void* identity, u64 version) const;
    void FinishSceneWrite(vk::CommandBuffer, u64 frame, Input native, bool replay = false);
    Result Evaluate(vk::CommandBuffer, u64 frame, VAddr display, u64 version, vk::Extent2D output);
    RytyDlssG::TemporalImage Output() const;
    RytyDlssG::TemporalImage OutputForPresent(vk::CommandBuffer);
    void Invalidate(const char* reason);
    void NoteWrite(const void* identity, u64 ps = 0);
    void Poll(u64 frame);
    // RYGT_DLSS_DIAG=1 + user/log/dlss.now: one scene, even without an admitted TAA pass.
    void DiagnosticImage(vk::CommandBuffer, u64 frame, Input image, const char* role);
    bool WantsDiagnosticDraws(u64 frame) const;
    void AbortDiagnostic(const char* reason) noexcept;
    void DiagnosticDraw(u64 frame, const std::string& state, bool geometry, bool motion,
                        bool motion_export);


    void NotePresentation(u64 frame, bool temporal, bool spatial);
    void NoteSpatialSource(u64 frame, vk::Extent2D source, const char* filter);
    /// Why a draw that samples the reconstructed scene at a display target was not taken as the
    /// final scene composite (once per frame and reason; `ps` 0 when no shader applies). Shown in
    /// the once-a-second line as compositeRejects=[reason:count] and rejectPs=[0xhash:reason:count].
    void NoteCompositeReject(u64 frame, const char* reason, u64 ps);
    /// A frame in which the game drew no new 3D scene (a paused replay, photo mode, a menu over a
    /// frozen scene) still draws its final composite from the same scene image. When that image
    /// is the one the last reconstruction owns, nobody wrote it since (same version, not dirty)
    /// and the last evaluation worked, the owned scene is current for this frame too: the frame
    /// then takes the ordinary composite path (the game's post chain replayed at the output size)
    /// instead of the spatial upscale. False for any frame with a scene of its own.
    bool CarryScene(u64 frame, const void* identity, u64 version);
    void Fallback(u64 frame, const char* reason, bool invalidate = true);
    std::string_view QualificationReason() const;
    // Scene epochs: a new full-scene TAA draw, a changed resource set or a gap starts a new epoch.
    void NoteDraw(u64 frame, bool taa, bool full_scene, u64 scene_key);
    void SceneEpoch(const char *reason);
    void ConsumerMapping(vk::ComponentMapping mapping);
    void NoteBufferWrite(u64 frame, VAddr address, u64 size, u64 version, bool exact);
    void NoteUpload(const void *identity, VAddr address, u64 size, u64 frame, u32 kind, u32 count, u64 image);
    u64 UploadedVersion(const void *identity, u64 frame, u32 count, u64 image) const;
    void RevokeUpload(const void *identity);

  private:
    Result Reconstruct(vk::CommandBuffer, u64 frame);
    struct Impl;
    std::unique_ptr<Impl> impl;
};
} // namespace Vulkan
