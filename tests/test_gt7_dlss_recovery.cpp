// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include "ryty_gpu/ryty_dlss_budget.h"
#include "ryty_gpu/ryty_gt7_spatial_policy.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

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
    std::printf("PASS %u production policy checks\n", checks);
}
