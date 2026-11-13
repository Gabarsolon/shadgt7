// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ryty_gpu/ryty_dlss_budget.h"
#include "ryty_gpu/ryty_gt7_spatial_policy.h"
#include "ryty_gpu/ryty_gt7_dlss_policy.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
    using namespace Vulkan::RytyDlssG::Budget;
    unsigned checks = 0;
    const auto check = [&](bool valid) {
        ++checks;
        if (!valid)
            std::abort();
    };
    check(Check(100 * MiB, 253 * MiB, 100 * MiB, 6300 * MiB, 7200 * MiB).allowed);
    check(!Check(100 * MiB, 253 * MiB, 100 * MiB, 6300 * MiB, 6200 * MiB).allowed);
    check(!Check(300 * MiB, 253 * MiB, 133 * MiB, 6200 * MiB, 6319 * MiB).allowed);
    check(Check(205 * MiB, 253 * MiB, 133 * MiB, 6105 * MiB, 6319 * MiB).allowed);
    check(Check(512 * MiB, 512 * MiB, 0, 0, 0).allowed);
    check(!Check(512 * MiB + 1, 0, 0, 0, 0).allowed);
    check(!Check(1, 1024 * MiB, 0, 0, 0).allowed);
    check(!Check(0, 1024 * MiB + 1, 0, 0, 0).allowed);
    check(!Check(0, 0, 1, ~Bytes{0}, ~Bytes{0}).allowed);
    check(!Check(0, 0, 2, ~Bytes{0} - 1, ~Bytes{0}).allowed);
    check(Check(0, 0, 1, ~Bytes{0} - 1, ~Bytes{0}).allowed);
    check(!Retry(1249, 1250));
    check(Retry(1250, 1250));
    // Refusal cannot latch permanently: every later retry admits when headroom returns.
    for (int i = 0; i < 5000; ++i) {
        check(Retry(1250 + i, 1250));
        check(Check(205 * MiB, 253 * MiB, 133 * MiB, 6000 * MiB, 6319 * MiB).allowed);
    }
    check(!CanRelease(false, 4, 10, 3000));
    check(!CanRelease(true, 0, 10, 3000));
    check(!CanRelease(true, 4, 3, 3000));
    check(!CanRelease(true, 4, 4, 1999));
    check(CanRelease(true, 4, 4, 2000));
    check(CanRelease(true, 4, 100, 3000));
    for (int bits = 0; bits < 8; ++bits) {
        const bool temporal = bits & 1, scene = bits & 2, reconstructed = bits & 4;
        check(Vulkan::Gt7Spatial::PreferNativeFilter(temporal, scene, reconstructed) ==
              (temporal && scene && !reconstructed));
    }
    using Vulkan::Gt7Temporal::CoversScene;
    check(CoversScene(0, 0, 1920, 1080, 1920, 1080));
    check(CoversScene(0, 0, 8192, 8192, 1920, 1080)); // same rendered pixels as exact clip
    check(CoversScene(0, 0, 2560, 1440, 1920, 1080));
    check(!CoversScene(1, 0, 8192, 8192, 1920, 1080)); // cropped left edge
    check(!CoversScene(0, 1, 8192, 8192, 1920, 1080));
    check(!CoversScene(0, 0, 1919, 1080, 1920, 1080));
    check(!CoversScene(0, 0, 1920, 1079, 1920, 1080));
    check(!CoversScene(0, 0, 0, 0, 0, 0));
    for (unsigned width : {0u, 1u, 1919u, 1920u, 2560u, 8192u, ~0u}) {
        for (unsigned height : {0u, 1u, 1079u, 1080u, 1440u, 8192u, ~0u}) {
            for (int x : {0, 1, 128}) {
                for (int y : {0, 1, 128}) {
                    // The four attachment corners must all survive the scissor intersection.
                    const auto inside = [&](unsigned px, unsigned py) {
                        return int64_t(px) >= x && int64_t(py) >= y &&
                            int64_t(px) < int64_t(x) + width &&
                            int64_t(py) < int64_t(y) + height;
                    };
                    const bool covers = inside(0, 0) && inside(1919, 0) &&
                                        inside(0, 1079) && inside(1919, 1079);
                    check(CoversScene(x, y, width, height, 1920, 1080) == covers);
                }
            }
        }
    }
    struct Clip {
        struct { int32_t x{}, y{}; } offset;
        struct { uint32_t width{}, height{}; } extent;
    };
    using Vulkan::Gt7Temporal::CoversSceneViewportZero;
    const Clip full{{0, 0}, {1920, 1080}}, empty{}, partial{{0, 0}, {960, 1080}};
    check(!CoversSceneViewportZero(std::vector<Clip>{}, true, 1920, 1080));
    check(CoversSceneViewportZero(std::vector{full}, true, 1920, 1080));
    for (const auto extra : {empty, partial, full, Clip{{128, 64}, {12, 20}}}) {
        check(CoversSceneViewportZero(std::vector{full, extra}, true, 1920, 1080));
        check(!CoversSceneViewportZero(std::vector{empty, extra}, true, 1920, 1080));
        check(!CoversSceneViewportZero(std::vector{partial, extra}, true, 1920, 1080));
        check(!CoversSceneViewportZero(std::vector{full, extra}, false, 1920, 1080));
    }
    std::vector<Clip> bound(16, empty);
    bound[0] = full;
    check(CoversSceneViewportZero(bound, true, 1920, 1080));
    bound[0].offset.x = 1;
    check(!CoversSceneViewportZero(bound, true, 1920, 1080));
    bound[0] = {{0, 0}, {8192, 8192}};
    check(CoversSceneViewportZero(bound, true, 1920, 1080));
    check(!CoversSceneViewportZero(bound, false, 1920, 1080));
    std::printf("PASS %u production policy checks\n", checks);
}
