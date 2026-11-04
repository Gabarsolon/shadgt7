// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <cstdlib>
namespace Vulkan {
// FIX-220: on AMD GPUs, RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC set to 1, true, on or yes (any case,
// spaces ignored; the launcher's "AMD graphics fix" box) turns it on, any other value off.
// FIX-GT7-061: on when unset. Only the fragment shaders that read a vertex's own values take it
// (FIX-217), so loading stays as fast; without it AMD's explicit-vertex path left GT7's Trial
// Mountain rock walls flat grey with dark triangles on an RX 7600 (driver 26.10.1, LLPC).
inline bool SingleSampleBarycentricRequested() {
    const char* value = std::getenv("RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC");
    if (!value || !*value) {
        return true;
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
