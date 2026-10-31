// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once

#include <mutex>

#include "common/types.h"

namespace Vulkan {

/// FIX-GT7-025: which save the driver pipeline cache file holds. A save is numbered by the count
/// of new pipelines when it was asked for (its data, read after, holds at least those). It takes
/// the file's place only with a later count than the file holds, so a save asked for earlier
/// that finishes later never replaces a newer one, and a save still being written doesn't count
/// as kept (the exit save then writes its own instead of trusting it).
class DriverCacheFile {
public:
    /// The count of the save the file holds (0: the one loaded at start).
    u32 Kept() const {
        std::scoped_lock lk{mutex};
        return kept;
    }

    /// Runs `replace`, which puts the save's temporary file in place (true when done), if
    /// `count` is later than the file's. True when the file then holds this save.
    template <typename Replace>
    bool Commit(u32 count, Replace&& replace) {
        std::scoped_lock lk{mutex};
        if (count <= kept || !replace()) {
            return false;
        }
        kept = count;
        return true;
    }

private:
    mutable std::mutex mutex;
    u32 kept{};
};

} // namespace Vulkan
