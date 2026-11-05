// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <cstdlib>
namespace Vulkan {
// FIX-220: KHR barycentrics on AMD GPUs for the fragment shaders that read a vertex's own values
// (FIX-217). FIX-GT7-061: with AMD's explicit-vertex path GT7's Trial Mountain rock walls came out
// flat grey with dark triangles on an RX 7600 (driver 26.10.1, LLPC). The launcher's "AMD Graphics
// Fix" option (off by default; an RX 6600 XT failed to build shaders with it on) passes
// RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC: 1, true, on or yes turns it on, any other value off. Unset,
// it is off (the Vulkan instance sets the default). RYGT_AMD_EXPLICIT_VERTEX=1 turns it off whatever else is set.
inline bool& AmdBarycentricDefault() {
    static bool on = false;
    return on;
}
inline bool SingleSampleBarycentricRequested() {
    if (const char* off = std::getenv("RYGT_AMD_EXPLICIT_VERTEX"); off && off[0] == '1') {
        return false;
    }
    const char* value = std::getenv("RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC");
    if (!value || !*value) {
        return AmdBarycentricDefault();
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
