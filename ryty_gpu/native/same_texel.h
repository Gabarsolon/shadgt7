// SPDX-FileCopyrightText: Copyright 2026 RyGT Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <utility>
#include <vector>

#include "common/types.h"

namespace Native {

/// PERF-GT7-057, FIX-GT7-080: an image descriptor of a same-texel dispatch as it was bound. The
/// picture's guest range (size 0 when bound empty), the view its T# asks for (ImageViewInfo,
/// before NativeRenderer::View keeps it inside the picture), and the picture's own mips and layers.
struct SameTexelImage {
    VAddr address = 0;
    u64 size = 0;
    bool written = false;
    u32 base_level = 0;
    u32 levels = 1; ///< the view's levels, or its bindings when each level is bound alone
    u32 base_layer = 0;
    u32 layers = 1;
    u32 image_mips = 1;
    u32 image_layers = 1;
    bool level_per_binding = false; ///< MipStorageFallbackMode::DynamicIndex
};

/// The mip and layer of the picture a dispatch stores into, as the GPU sees them (mip << 32 |
/// layer), when its `fetch` descriptor reads exactly that texel of the same picture and no other
/// descriptor reads or writes any of it. `mip` and `layer` are the indices the code gives both
/// views. Nothing otherwise: the barrier stays.
///
/// FIX-GT7-080: two views of one picture that start at other mips name the same level with other
/// indices (base 0 at mip 1, base 1 at mip 0), so the key holds the level each view reaches
/// inside the picture, as View bounds it. Only the descriptor the code reads with texelFetch at
/// the texel it stores may meet the stored picture: GT7's probe filter samples three others with
/// textureLod at other coordinates.
inline std::optional<u64> SameTexelKey(std::span<const SameTexelImage> images, u32 fetch, u32 store,
                                       u32 mip, u32 layer) {
    if (fetch >= images.size() || store >= images.size() || fetch == store) {
        return std::nullopt;
    }
    const SameTexelImage& stored = images[store];
    const SameTexelImage& read = images[fetch];
    if (!stored.written || read.written || stored.size == 0 || read.size == 0 ||
        read.address != stored.address || read.size != stored.size) {
        return std::nullopt;
    }
    for (size_t i = 0; i < images.size(); ++i) {
        if (i == fetch || i == store) {
            continue;
        }
        const SameTexelImage& other = images[i];
        if (other.written) {
            return std::nullopt; // one stored picture only
        }
        if (other.size != 0 && other.address < stored.address + stored.size &&
            stored.address < other.address + other.size) {
            return std::nullopt; // another read meets the stored picture
        }
    }
    // The first level or layer of a view and how many it holds, as View keeps it in the picture.
    const auto bounded = [](u32 base, u32 count, u32 total) {
        total = std::max<u32>(total, 1);
        base = std::min<u32>(base, total - 1);
        return std::pair<u32, u32>{base, std::clamp<u32>(count, 1, total - base)};
    };
    u32 stored_level;
    if (stored.level_per_binding) {
        // A view of one level per binding, each kept in the picture on its own.
        if (mip >= stored.levels) {
            return std::nullopt;
        }
        stored_level =
            std::min<u32>(stored.base_level + mip, std::max<u32>(stored.image_mips, 1) - 1);
    } else {
        const auto [base, count] = bounded(stored.base_level, stored.levels, stored.image_mips);
        if (mip >= count) {
            return std::nullopt;
        }
        stored_level = base + mip;
    }
    const auto [layer_base, layer_count] =
        bounded(stored.base_layer, stored.layers, stored.image_layers);
    if (layer >= layer_count) {
        return std::nullopt;
    }
    const u32 stored_layer = layer_base + layer;
    const auto [read_base, read_levels] = bounded(read.base_level, read.levels, read.image_mips);
    const auto [read_layer_base, read_layers] =
        bounded(read.base_layer, read.layers, read.image_layers);
    if (mip >= read_levels || layer >= read_layers || read_base + mip != stored_level ||
        read_layer_base + layer != stored_layer) {
        return std::nullopt; // the read is not the texel stored
    }
    return (u64(stored_level) << 32) | stored_layer;
}

/// PERF-GT7-057: what a run's same-texel dispatches stored since its last barrier.
/// FIX-GT7-081: a dispatch that stores without a key leaves the run's writes unknown, so no later
/// dispatch of the run counts as reading or storing a texel not yet stored: the barrier taken
/// before that dispatch doesn't order its writes with what comes after.
/// FIX-GT7-087: and what the run's dispatches sampled other than at the texels they store (GT7's
/// cs_img0 to 2, filtered at other coordinates, or anything a dispatch without a key read). A
/// destination that meets one of those is not new: an earlier dispatch may still be reading it
/// where this one writes. Only the reads of a dispatch's own destination at its texel are left out.
struct SameTexelRun {
    std::vector<std::pair<VAddr, u64>> stored;
    std::vector<std::pair<VAddr, u64>> other_reads; ///< (address, size)
    bool unknown = false;

    void Clear() {
        stored.clear();
        other_reads.clear();
        unknown = false;
    }
    bool IsNew(VAddr picture, u64 size, u64 key) const {
        const VAddr end = picture + std::max<u64>(size, 1);
        return !unknown &&
               std::ranges::none_of(other_reads,
                                    [&](const std::pair<VAddr, u64>& read) {
                                        return read.first < end && picture < read.first + read.second;
                                    }) &&
               std::ranges::none_of(stored, [&](const std::pair<VAddr, u64>& texel) {
                   return texel.first == picture && texel.second == key;
               });
    }
    void Note(VAddr picture, std::optional<u64> key, bool stores) {
        if (key) {
            stored.emplace_back(picture, *key);
        } else if (stores) {
            unknown = true;
        }
    }
    void NoteRead(VAddr address, u64 size) {
        size = std::max<u64>(size, 1);
        const std::pair<VAddr, u64> read{address, size};
        if (std::ranges::find(other_reads, read) == other_reads.end()) {
            other_reads.push_back(read); // GT7: the same three pictures for a whole run
        }
    }
};

} // namespace Native
