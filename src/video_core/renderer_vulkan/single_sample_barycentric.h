// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <cstdlib>
namespace Vulkan {
// FIX-220: opt-in again (as upstream): on AMD GPUs only with RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC
// set to 1, true, on or yes (any case, spaces ignored; the launcher's "AMD graphics fix" box). On
// by default it made AMD shader loading much slower than 27b.
inline bool SingleSampleBarycentricRequested() {
    const char* value = std::getenv("RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC");
    if (!value) {
        return false;
    }
    while (*value == ' ' || *value == '"') {
        ++value;
    }
    const char c = static_cast<char>(*value | 0x20);
    return *value == '1' || c == 't' || c == 'y' || (c == 'o' && (value[1] | 0x20) == 'n');
}
constexpr bool UseSingleSampleBarycentric(std::uint32_t vendor, bool supports_khr,
                                         unsigned raster_samples, bool requested) {
    return requested && vendor == 0x1002 && supports_khr && raster_samples == 1;
}
} // namespace Vulkan
