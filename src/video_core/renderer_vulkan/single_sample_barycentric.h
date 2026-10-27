// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <cstdint>
namespace Vulkan {
constexpr bool UseSingleSampleBarycentric(std::uint32_t vendor, bool supports_khr,
                                         unsigned raster_samples, bool requested) {
    return requested && vendor == 0x1002 && supports_khr && raster_samples == 1;
}
} // namespace Vulkan
