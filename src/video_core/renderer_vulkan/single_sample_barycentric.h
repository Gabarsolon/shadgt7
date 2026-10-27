// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <cstdlib>
namespace Vulkan {
// On for AMD GPUs; RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC=0 turns it off for diagnosis.
inline bool SingleSampleBarycentricRequested() {
    const char* value = std::getenv("RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC");
    return !(value && value[0] == '0');
}
constexpr bool UseSingleSampleBarycentric(std::uint32_t vendor, bool supports_khr,
                                         unsigned raster_samples, bool requested) {
    return requested && vendor == 0x1002 && supports_khr && raster_samples == 1;
}
} // namespace Vulkan
