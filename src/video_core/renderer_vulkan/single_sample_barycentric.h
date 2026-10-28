// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <cstdlib>
namespace Vulkan {
// On for AMD GPUs; RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC=0 turns it off for diagnosis.
// FIX-217: "0", "false", "off" or "no" (any case, spaces around it ignored) turn it off.
inline bool SingleSampleBarycentricRequested() {
    const char* value = std::getenv("RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC");
    if (!value) {
        return true;
    }
    while (*value == ' ' || *value == '"') {
        ++value;
    }
    const char c = static_cast<char>(*value | 0x20);
    return !(*value == '0' || c == 'f' || c == 'n' || (c == 'o' && (value[1] | 0x20) == 'f'));
}
constexpr bool UseSingleSampleBarycentric(std::uint32_t vendor, bool supports_khr,
                                         unsigned raster_samples, bool requested) {
    return requested && vendor == 0x1002 && supports_khr && raster_samples == 1;
}
} // namespace Vulkan
