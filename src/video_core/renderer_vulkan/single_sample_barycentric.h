// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
#include <cstdlib>
namespace Vulkan {
// FIX-220: KHR barycentrics on AMD GPUs for the fragment shaders that read a vertex's own values
// (FIX-217). FIX-GT7-061: on by default: with AMD's explicit-vertex path GT7's Trial Mountain rock
// walls came out flat grey with dark triangles on an RX 7600 (driver 26.10.1, LLPC). The
// launcher's "AMD graphics fix" box sets RYGT_AMD_SINGLE_SAMPLE_BARYCENTRIC itself (off when unticked), so
// that variable no longer turns it off; RYGT_AMD_EXPLICIT_VERTEX=1 does (AMD's path, as before).
inline bool SingleSampleBarycentricRequested() {
    const char* value = std::getenv("RYGT_AMD_EXPLICIT_VERTEX");
    return !(value && value[0] == '1');
}
constexpr bool UseSingleSampleBarycentric(std::uint32_t vendor, bool supports_khr,
                                         unsigned raster_samples, bool requested) {
    return requested && vendor == 0x1002 && supports_khr && raster_samples == 1;
}
} // namespace Vulkan
