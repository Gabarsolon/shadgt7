// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#pragma once
#include <cstdint>

namespace Vulkan {
// FIX-227 aliases the scratch address with these push words. A non-full mask
// occupies them itself, so neither a writer pass nor scratch reads are available.
constexpr bool TessScratchAvailable(std::uint32_t mask0, std::uint32_t mask1) {
    return mask0 == ~std::uint32_t{0} && mask1 == ~std::uint32_t{0};
}
} // namespace Vulkan
