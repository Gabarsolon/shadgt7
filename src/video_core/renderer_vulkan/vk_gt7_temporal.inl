// SPDX-License-Identifier: GPL-2.0-or-later
// Included inside namespace Vulkan in vk_rasterizer_native.cpp.
static Gt7DlssPass::Input TemporalNativeImage(const Native::ImageEntry& e, u64 generation) {
    vk::ImageAspectFlags aspects = vk::ImageAspectFlagBits::eColor;
    if (e.info.props.is_depth) {
        aspects = vk::ImageAspectFlagBits::eDepth;
        if (e.info.props.has_stencil)
            aspects |= vk::ImageAspectFlagBits::eStencil;
    }
    return {vk::Image{Handle<VkImage_T>(e.image->NativeHandle())}, e.info.pixel_format,
            {e.info.size.width, e.info.size.height}, aspects, e.info.guest_address, &e,
            e.write_version.load(), generation};
}

// A draw the reconstruction has to know about: the game's TAA resolve decides the scene epoch,
// and the clear of its history ends the accumulated history.
void NativeRenderer::NoteTemporalDraw(const AmdGpu::Regs& regs, const GraphicsPipeline* pipeline,
                                      const Targets* targets, const DrawBindings* b,
                                      const char* status) {
    const auto* fs = pipeline ? pipeline->GetStages()[u32(Shader::SwStage::Fragment)] : nullptr;
    u64 ps = fs ? fs->pgm_hash : 0;
    if (!pipeline) {
        std::scoped_lock lookup{r.pipeline_cache.LookupMutex()};
        const auto& selected = r.pipeline_cache.SelectedInfos();
        if (const auto* f = selected[u32(Shader::SwStage::Fragment)])
            ps = f->pgm_hash;
    }
    if (temporal_dlss->WantsDiagnosticDraws(frame)) {
        try {
            const auto* vs =
                pipeline ? pipeline->GetStages()[u32(Shader::SwStage::Vertex)] : nullptr;
            const auto* color = targets ? targets->colors[0] : nullptr;
            const auto* motion = targets ? targets->colors[1] : nullptr;
            const auto* depth = targets ? targets->depth : nullptr;
            const bool geometry = depth && color && targets->state.width == 1920 &&
                                  targets->state.height == 1080 &&
                                  std::string_view(status) == "executed";
            const bool has_motion = motion &&
                                    motion->info.pixel_format == vk::Format::eR16G16Sfloat &&
                                    (regs.color_target_mask.raw & 0xf0) != 0;
            const u32 shader_mask = pipeline ? pipeline->GetGraphicsKey().mrt_mask : 0;
            const bool motion_export = has_motion && (shader_mask & 2) &&
                                       (regs.color_target_mask.raw & 0x30) == 0x30;
            std::string images;
            if (b)
                for (const auto* image : b->sampled) {
                    if (image && images.size() < 2048)
                        images += fmt::format(" {:#x}:{}x{}:{}", image->info.guest_address,
                                              image->info.size.width, image->info.size.height,
                                              vk::to_string(image->info.pixel_format));
                }
            temporal_dlss->DiagnosticDraw(
                frame,
                fmt::format(
                    "status={} ps={:#x} vs={:#x} raster={}x{} viewport={}x{} indices={} "
                    "instances={} "
                    "color={:#x}:{} MRT1={:#x}:{} depth={:#x}:{} depthWrite={} zscale={} "
                    "zoffset={} guestMask={:#x} "
                    "shaderMask={:#x} afterReconstruction={} TAA={} depthTest={} depthCompare={} "
                    "stencil={} viewportOffset=({}, {}) motionXYExport={} sampled=[{}]",
                    status, ps, vs ? vs->pgm_hash : 0, targets ? targets->state.width : 0,
                    targets ? targets->state.height : 0, std::abs(regs.viewports[0].xscale * 2),
                    std::abs(regs.viewports[0].yscale * 2), regs.num_indices,
                    regs.num_instances.NumInstances(), color ? color->info.guest_address : 0,
                    vk::to_string(color ? color->info.pixel_format : vk::Format::eUndefined),
                    motion ? motion->info.guest_address : 0,
                    vk::to_string(motion ? motion->info.pixel_format : vk::Format::eUndefined),
                    depth ? depth->info.guest_address : 0,
                    vk::to_string(depth ? depth->info.pixel_format : vk::Format::eUndefined),
                    bool(regs.depth_control.depth_write_enable), regs.viewports[0].zscale,
                    regs.viewports[0].zoffset, regs.color_target_mask.raw,
                    pipeline ? pipeline->GetGraphicsKey().mrt_mask : 0,
                    temporal_dlss->Reconstructed(frame), Gt7Temporal::TaaHash(ps),
                    bool(regs.depth_control.depth_enable), u32(regs.depth_control.depth_func),
                    bool(regs.depth_control.stencil_enable), regs.viewports[0].xoffset,
                    regs.viewports[0].yoffset, motion_export, images),
                geometry, has_motion, motion_export);
        } catch (...) {
            temporal_dlss->AbortDiagnostic("draw state formatting");
        }
    }
    const bool taa = Gt7Temporal::TaaHash(ps), initialize = Gt7Temporal::InitializeHash(ps);
    if (initialize && std::string_view(status) == "executed" && targets &&
        targets->state.width == 1920 && targets->state.height == 1080 &&
        std::abs(regs.viewports[0].xscale * 2) == 1920 &&
        std::abs(regs.viewports[0].yscale * 2) == 1080)
        temporal_dlss->Invalidate("TAA-initialize");
    const bool full_scene = taa && targets && b && b->temporal &&
        targets->state.width == 1920 && targets->state.height == 1080 &&
        std::abs(regs.viewports[0].xscale * 2) == 1920 &&
        std::abs(regs.viewports[0].yscale * 2) == 1080 && regs.viewports[0].xoffset == 960 &&
        regs.viewports[0].yoffset == 540;
    u64 scene_key = 0;
    if (full_scene && fs && fs->images.size() == 4 && targets->colors[0]) {
        const auto history_address = fs->images[3].GetSharp(*fs).Address();
        // Unordered history pair: ordinary ping-pong does not create an epoch.
        scene_key = reinterpret_cast<uintptr_t>(targets->colors[0]);
        for (const auto* sampled : b->sampled)
            if (sampled->info.guest_address == history_address)
                scene_key ^= reinterpret_cast<uintptr_t>(sampled);
        scene_key ^= std::rotl(u64(reinterpret_cast<uintptr_t>(targets->colors[1])), 17);
    }
    temporal_dlss->NoteDraw(frame, taa && std::string_view(status) == "executed", full_scene,
                            scene_key);
}

// RefreshSources keeps copied versions in held_versions, not in the alias's
// write_version/written_frame. Require a complete, already recorded copy from a
// current-frame producer. Do not infer freshness from address overlap or v==0.
u64 NativeRenderer::TemporalInputVersion(const Native::ImageEntry &entry, u64 scene_frame) {
    const auto prove = [&](auto &&self, const Native::ImageEntry &e, unsigned depth) -> u64 {
        if (depth > 4 || !e.image || e.reused.load() || e.info.resources.levels != 1 ||
            e.info.resources.layers != 1 || e.info.num_samples != 1)
            return 0;
        const u64 own = e.write_version.load();
        const u64 foreign = e.foreign_clock.load();
        u64 result = own && own >= foreign && e.written_frame.load() == scene_frame ? own : 0;
        if (!e.needs_upload.load() && !e.force_upload.load()) {
            const u64 uploaded =
                temporal_dlss->UploadedVersion(&e, scene_frame, e.total_uploads.load(), e.image->NativeHandle());
            if (uploaded >= std::max(own, foreign))
                result = std::max(result, uploaded);
        }
        std::vector<Native::SourceLink> sources;
        {
            std::scoped_lock lock{const_cast<Native::ImageEntry &>(e).sources_mutex};
            sources = e.sources;
        }
        const auto held = held_versions.find(&e);
        if (held == held_versions.end())
            return result;
        const auto base = held->second.find(0);
        if (base == held->second.end() || base->second < std::max(own, foreign))
            return result;
        for (const auto &link : sources) {
            const auto &at = link.place;
            if (!link.source || at.mip != 0 || at.layer != 0 || at.src_mip != 0 ||
                at.src_layer != 0 || at.width < e.info.size.width ||
                at.height < e.info.size.height ||
                link.source->info.size.width < e.info.size.width ||
                link.source->info.size.height < e.info.size.height)
                continue;
            const u64 producer = self(self, *link.source, depth + 1);
            if (Gt7Temporal::FreshInputCopy(own, foreign, base->second, producer, true))
                result = std::max(result, producer);
        }
        return result;
    };
    return prove(prove, entry, 0);
}

void NativeRenderer::CaptureTemporalInputs(const AmdGpu::Regs& regs,
                                           const GraphicsPipeline& pipeline, const Targets& targets,
                                           const DrawBindings& b) {
    const auto* fs = pipeline.GetStages()[u32(Shader::SwStage::Fragment)];
    const auto* vs = pipeline.GetStages()[u32(Shader::SwStage::Vertex)];
    if (!fs || !Gt7Temporal::TaaHash(fs->pgm_hash))
        return;
    // Qualification failures below describe the main scene only. GT7 also runs TAA
    // for small views; its attachment/constant contract must not invalidate our main history.
    const auto& scene_viewport = regs.viewports[0];
    if (!Gt7Temporal::MainScene(targets.state.width, targets.state.height,
            std::abs(scene_viewport.xscale * 2), std::abs(scene_viewport.yscale * 2),
            scene_viewport.xoffset, scene_viewport.yoffset)) {
        if (Common::Sharp::Diagnostics()) {
            static std::set<std::tuple<u64, u32, u32>> seen;
            if (seen.size() < 16 && seen.emplace(fs->pgm_hash, targets.state.width,
                                               targets.state.height).second) {
                LOG_WARNING(Render_Vulkan, "GT7-DLSS subview: ps={:#x} raster={}x{} "
                            "MRT={} mainHistoryKept=1", fs->pgm_hash, targets.state.width,
                            targets.state.height, targets.state.num_color_attachments);
            }
        }
        return;
    }
    const auto fail = [&](const char* reason) {
        temporal_dlss->Fallback(frame, reason);
        if (Common::Sharp::Get().dlss_log || Common::Sharp::Diagnostics()) {
            const auto& vp = regs.viewports[0];
            const std::string key = fmt::format(
                "ps={:#x} vs={:#x} raster={}x{} viewport=({},{} {}x{}) MRT={} "
                "depth={} images={} indices={} instances={} cb={} reason={}",
                fs->pgm_hash, vs ? vs->pgm_hash : 0, targets.state.width, targets.state.height,
                vp.xoffset-std::abs(vp.xscale), vp.yoffset-std::abs(vp.yscale),
                std::abs(vp.xscale*2), std::abs(vp.yscale*2),
                targets.state.num_color_attachments, bool(targets.depth), fs->images.size(),
                regs.num_indices, regs.num_instances.NumInstances(),
                b.temporal ? b.temporal->bytes.size() : 0, reason);
            static std::set<std::string> reported;
            if (reported.size() < 64 && reported.emplace(key).second) {
                std::string inputs;
                for (const auto* image : b.sampled) {
                    if (!image) continue;
                    inputs += fmt::format(" {:#x}:{}x{}:{}:writeFrame{}:version{}",
                        image->info.guest_address, image->info.size.width, image->info.size.height,
                        vk::to_string(image->info.pixel_format), image->written_frame.load(),
                        image->write_version.load());
                }
                LOG_WARNING(Render_Vulkan, "GT7-DLSS shape: frame={} {} inputs=[{}]",
                            frame, key, inputs);
            }
        }
    };
    if (!vs || vs->pgm_hash != 0x3863962b || !Gt7Temporal::ProvenSchema(fs->pgm_hash)) {
        fail("unverified-TAA-variant");
        return;
    }
    if (!b.temporal || b.temporal->bytes.size() < 116) {
        fail("immutable-constants-missing");
        return;
    }
    if (regs.num_indices > 6 || regs.num_instances.NumInstances() != 1 || targets.depth ||
        targets.state.num_color_attachments != 2 || !targets.colors[0] || !targets.colors[1] ||
        fs->images.size() != 4 || !b.stored.empty() || b.writes_buffers) {
        fail("TAA-attachments-or-draw-contract");
        return;
    }
    const auto& c = b.temporal->words;
    const bool scene_extent = c[16] == 1920 && c[17] == 1080 && targets.state.width == 1920 &&
                              targets.state.height == 1080;
    if (!std::isfinite(c[16]) || !std::isfinite(c[17]) || !scene_extent) {
        // Other views/thumbnails do not invalidate a main scene captured earlier
        // in this frame. No main capture still falls back at presentation.
        return;
    }
    const auto& vp = regs.viewports[0];
    if (std::abs(vp.xscale * 2) != c[16] || std::abs(vp.yscale * 2) != c[17] ||
        vp.xoffset != c[16] * .5f || vp.yoffset != c[17] * .5f) {
        fail("unsupported-TAA-viewport");
        return;
    }
    const auto make = [&](const Native::ImageEntry& e) { return TemporalNativeImage(e, registry->Generation()); };
    std::array<Gt7DlssPass::Input, 4> inputs{};
    unsigned i = 0;
    for (const auto& binding : fs->images) {
        const auto sharp = binding.GetSharp(*fs);
        const auto mapping = sharp.DstSelect();
        const auto rgb = [](const AmdGpu::CompMapping &m) {
            using S = AmdGpu::CompSwizzle;
            return m.g == S::Green &&
                   ((m.r == S::Red && m.b == S::Blue) || (m.r == S::Blue && m.b == S::Red));
        };
        const auto mrt = regs.color_buffers[1].Swizzle();
        const bool supported_mapping =
            i == 0 ? mapping.r == AmdGpu::CompSwizzle::Red
            : i == 1
                ? mapping.r == AmdGpu::CompSwizzle::Red && mapping.g == AmdGpu::CompSwizzle::Green
            : i == 2 ? rgb(mapping) && rgb(mrt) && mapping.r == mrt.r && mapping.g == mrt.g &&
                           mapping.b == mrt.b
                     : true; // history is never copied or sampled by DLSS
        if (!supported_mapping) {
            fail("unsupported-input-swizzle");
            return;
        }
        if (binding.is_written || binding.NumBindings(*fs) != 1 || sharp.base_level != 0 ||
            sharp.base_array != 0 || sharp.width + 1 != c[16] || sharp.height + 1 != c[17]) {
            fail("unsupported-input-subresource");
            return;
        }
        const auto it = std::ranges::find_if(
            b.sampled, [&](const auto* e) { return e->info.guest_address == sharp.Address(); });
        if (it == b.sampled.end() || !(*it)->image || (*it)->reused.load() ||
            (*it)->info.num_samples != 1 || (*it)->info.resources.layers != 1 ||
            (*it)->info.resources.levels != 1) {
            fail("missing-or-reused-input");
            return;
        }
        const Native::ImageEntry* selected = *it;
        u64 effective_version = i < 3 ? TemporalInputVersion(*selected, frame)
                                      : selected->write_version.load();
        if (i == 0 && selected->info.pixel_format == vk::Format::eR32Sfloat && !effective_version) {
            // The native depth-twin path excludes stencil, so a stale R32 view
            // cannot import D32S8. Select only a unique current real depth producer
            // at this exact binding's base, pitch and full geometry. Never guess
            // from a nearby address or accept an old/clear-only allocation.
            const Native::ImageEntry* depth = nullptr;
            bool ambiguous = false;
            registry->FindExisting(sharp.Address(), [&](const Native::ImageEntry& candidate) {
                const auto& info = candidate.info;
                if (!candidate.image || candidate.reused.load() || !info.props.is_depth ||
                    (info.pixel_format != vk::Format::eD32Sfloat &&
                     info.pixel_format != vk::Format::eD32SfloatS8Uint) ||
                    info.guest_address != sharp.Address() || info.pitch != selected->info.pitch ||
                    info.size.width != c[16] || info.size.height != c[17] ||
                    info.num_samples != 1 || info.resources.levels != 1 || info.resources.layers != 1 ||
                    !TemporalInputVersion(candidate, frame))
                    return false;
                if (depth && depth->image->NativeHandle() != candidate.image->NativeHandle())
                    ambiguous = true;
                depth = &candidate;
                return true;
            });
            if (depth && !ambiguous) {
                Splice();
                r.scheduler.EndRendering();
                selected = depth;
                effective_version = TemporalInputVersion(*depth, frame);
            }
        }
        if (i < 3 && !effective_version) {
            fail("input-not-written-this-scene-frame");
            return;
        }
        inputs[i] = make(*selected);
        inputs[i].version = effective_version;
        inputs[i].mapping = LiverpoolToVK::ComponentMapping(mapping);
        ++i;
    }
    for (unsigned n = 0; n < 2; ++n) {
        const auto* e = targets.colors[n];
        if (e->info.num_samples != 1 || e->info.resources.layers != 1 ||
            e->info.resources.levels != 1 || targets.color_ranges[n].mip != 0 ||
            targets.color_ranges[n].layer != 0) {
            fail("unsupported-MRT-subresource");
            return;
        }
    }
    // GPU time is after this successful TAA draw; no later guest memory reads.
    const auto history = make(*targets.colors[0]), scene = make(*targets.colors[1]);
    Splice();
    r.scheduler.EndRendering();
    temporal_dlss->Capture(frame, fs->pgm_hash, *b.temporal, inputs, history, scene);
    if (temporal_dlss->Reconstructed(frame)) {
        // Substitution writes the native scene too. Publish its version so every
        // native alias/readback sees it; history MRT0 is deliberately untouched.
        temporal_write_ps = fs->pgm_hash;
        Written(targets.colors[1]);
        temporal_write_ps = 0;
        temporal_dlss->SceneVersion(targets.colors[1]->write_version.load());
    }
}

vk::DescriptorImageInfo* NativeRenderer::TemporalSceneBinding(const AmdGpu::Regs& regs,
                                                              const GraphicsPipeline& pipeline,
                                                              const Targets& targets,
                                                              DrawBindings& b) {
    if (!Gt7Temporal::Config().enabled || !targets.colors[0] ||
        !IsDisplayAddress(targets.colors[0]->info.guest_address))
        return nullptr;
    const auto* fs = pipeline.GetStages()[u32(Shader::SwStage::Fragment)];
    const auto* vs = pipeline.GetStages()[u32(Shader::SwStage::Vertex)];
    // A draw that samples the reconstructed scene into a display image is a candidate for the
    // final scene composite; when it is turned down the reason goes to the once-a-second line
    // (compositeRejects), with the pixel shader hash for the shader whitelist below.
    bool candidate = false;
    if (temporal_dlss->Reconstructed(frame)) {
        for (const auto* sampled : b.sampled) {
            candidate |= sampled && temporal_dlss->OwnsScene(frame, sampled);
        }
    }
    const u64 ps_hash = fs ? fs->pgm_hash : 0;
    const auto reject = [&](const char* reason) -> vk::DescriptorImageInfo* {
        if (candidate) {
            temporal_dlss->NoteCompositeReject(frame, reason, ps_hash);
        }
        return nullptr;
    };
    if (regs.num_indices > 6 || regs.num_instances.NumInstances() != 1 ||
        targets.state.num_color_attachments != 1 || targets.depth || regs.blend_control[0].enable ||
        (regs.color_target_mask.raw & 7) != 7 || b.writes_buffers || !b.stored.empty())
        return reject("draw-shape");
    if (!fs || !vs || fs->loads.GetAny(Shader::IR::Attribute::FragCoord) ||
        (vs->pgm_hash != 0x3863962b && vs->pgm_hash != 0xc830abbf)) {
        if (candidate) {
            // The vertex shader's hash is the one to look at when it is the one turned down.
            const bool vs_only = fs && vs && !fs->loads.GetAny(Shader::IR::Attribute::FragCoord);
            temporal_dlss->NoteCompositeReject(
                frame, !fs || !vs ? "no-stage"
                       : vs_only  ? "vertex-shader-not-listed"
                                  : "ps-reads-fragcoord",
                vs_only ? vs->pgm_hash : ps_hash);
        }
        return nullptr;
    }
    // Executed final scene composites. Their ordinary draw keeps the native
    // scene route current; the replay samples the reconstructed post-FX scene.
    switch (fs->pgm_hash) {
    case 0xf97953f5:
    case 0xf8a8c4e7:
    case 0xd30f5ed6:
    case 0x9468c599:
    case 0xfdca40b6:
    case 0xde32d99a:
    case 0x0b6b416e:
    case 0x2b572aba:
    case 0x1db6f4cb:
    case 0x15d4951c:
    case 0x8dc11d41:
    case 0xdf75f23b:
    case 0x48b1d92f: // executed race vignetting variant, run3 frame1230
    case 0xb0d47c36:
    case 0x4f794c7e:
        break;
    default:
        return reject("pixel-shader-not-listed");
    }
    const auto& e = *targets.colors[0];
    const auto& vp = regs.viewports[0];
    if (std::abs(vp.xscale * 2) != e.info.size.width ||
        std::abs(vp.yscale * 2) != e.info.size.height || vp.xoffset != e.info.size.width * .5f ||
        vp.yoffset != e.info.size.height * .5f)
        return reject("viewport-not-full-target");
    const auto plan =
        PlanStageResourceBindings(*fs, Shader::Backend::StageStart(u32(fs->sw_stage)));
    u32 unified = plan.textures.unified;
    vk::DescriptorImageInfo* found{};
    for (const auto& binding : fs->images) {
        const auto sharp = binding.GetSharp(*fs);
        if (!binding.is_written && binding.NumBindings(*fs) == 1 && sharp.base_level == 0 &&
            sharp.base_array == 0) {
            for (const auto* sampled : b.sampled) {
                if (sampled->info.guest_address != sharp.Address())
                    continue;
                // A frame without a 3D scene of its own (paused replay, photo mode, a menu over a
                // frozen scene) composes the same, unwritten scene image again: it keeps the last
                // reconstruction (CarryScene); a frame with a scene of its own never does.
                if (!temporal_dlss->HasScene(frame, sampled, sampled->write_version.load()) &&
                    !(temporal_dlss->CarryScene(frame, sampled, sampled->write_version.load()) &&
                      temporal_dlss->HasScene(frame, sampled, sampled->write_version.load())))
                    continue;
                if (sharp.width + 1 != e.info.size.width || sharp.height + 1 != e.info.size.height)
                    return reject("scene-size-differs-from-target");
                for (size_t i = 0; i < b.writes.size(); ++i) {
                    auto& write = b.writes[i];
                    if (b.write_sets[i] == Shader::Backend::StageSet(u32(fs->sw_stage)) &&
                        write.dstBinding == unified &&
                        write.descriptorType == vk::DescriptorType::eSampledImage &&
                        write.descriptorCount == 1) {
                        if (found)
                            return reject("scene-bound-twice");
                        const auto m = sharp.DstSelect();
                        using S = AmdGpu::CompSwizzle;
                        if (m.g != S::Green || !((m.r == S::Red && m.b == S::Blue) ||
                                                 (m.r == S::Blue && m.b == S::Red)))
                            return reject("scene-channel-selector");
                        // Raw pre-TAA storage equals MRT1 storage (proved at capture).
                        // Preserve this consumer's view, including its alpha selector.
                        temporal_dlss->ConsumerMapping(LiverpoolToVK::ComponentMapping(m));
                        found = const_cast<vk::DescriptorImageInfo*>(write.pImageInfo);
                    }
                }
            }
        }
        unified += binding.NumBindings(*fs);
    }
    if (!found) {
        return reject("scene-not-bound-as-one-sampled-image");
    }
    return found;
}
