// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#include <array>
#include <bit>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "common/assert.h"
#include "common/logging/log.h"
#include "common/ryty_config.h"
#include "core/emulator_settings.h"
#include "video_core/renderer_vulkan/vk_shader_util.h"

namespace Vulkan {

namespace {
std::mutex code_mutex;
std::unordered_map<VkShaderModule, std::vector<u32>> module_code;

// FIX-GT7-143: the brightness of GT7's own interface on an HDR output ("paper white").
//
// GT7's interface and HUD pixel shaders have an HDR variant each (menu_shaders.bin, techniques
// named ...HD1...). It decodes sRGB, goes to BT.2020, multiplies by the literal 0.025 and encodes
// PQ: white comes out at 0.025 x 10,000 = 250 nits, whatever display the game was told of. No
// variable of the game feeds that number (its HDR settings were changed while it ran, and its
// memory searched: the interface never moved), so the literal is replaced here, in the translated
// shader, as its module is made: the shader store keeps the game's own code.
//
// The shaders that read an interface picture back (blurred or faded panels, pictures drawn in two
// steps) decode PQ and return to BT.709 through a matrix with the inverse built in, 40 times the
// standard one. Its nine constants are rescaled the other way in the same shader, so what is
// decoded and encoded again keeps its luminance and only what is drawn from sRGB takes the new
// white (with the literal alone, the title screen's picture was scaled twice: measured 0.36 of
// its luminance for 150 nits asked, 0.6 expected).
//
// GPU.hdr_paper_white_nits in the launcher's settings (50 to 1000, with HDR output allowed;
// RYGT_HDR_PAPER_WHITE_NITS overrides it) gives the nits of the interface's white. Unset, or 250,
// nothing is touched.
//
// A shader is taken for one of those when its float constants hold the literal together with the
// sRGB threshold and the five PQ constants: 27 shaders of a 15,722-shader store, all of them the
// interface's, each with that literal once; no shader has the literal with part of the rest.
//
// Seven other shaders hold the x40 matrix with an sRGB encode and no literal: the SDR mode's
// readers of a PQ picture (stored before any HDR session). They are left alone, as SDR needs.
// Not proven: that none of them runs in HDR mode on a picture holding interface, where it would
// be off by the new white over 250.
//
// A new white is new code for these shaders: the driver's cache, the shader-object binaries and
// the compile pack know them by their code, so they are compiled again at their first use.
constexpr u32 InterfaceWhite = 0x3ccccccc; // 0.025 as the game holds it, one step under the float
constexpr u32 SrgbThreshold = 0x3d25aed5;  // 0.0404499, 12.92 x 0.0031308
constexpr u32 PqM1 = 0x3e232000;           // 0.1593017578125
constexpr u32 PqM2 = 0x429db000;           // 78.84375
constexpr u32 PqC1 = 0x3f560000;           // 0.8359375
constexpr u32 PqC2 = 0x4196d000;           // 18.8515625
constexpr u32 PqC3 = 0x41958000;           // 18.6875
// BT.2020 to BT.709 times 40 (1 / 0.025), as the shaders hold it.
constexpr std::array<float, 9> DecodeMatrix{
    66.41986846923828125f,     -23.50626373291015625f, -2.913595676422119140625f,
    -4.981883525848388671875f, 45.31580352783203125f,  -0.3339198529720306396484375f,
    -0.72614729404449462890625f, -4.023891448974609375f, 44.75003814697265625f,
};

/// The constant to put in place of 0.025 (nits / 10000 as a float's bits), or 0 for no change.
u32 InterfaceWhiteWanted() {
    static const u32 wanted = [] {
        double nits = 0.0;
        const char* env = std::getenv("RYGT_HDR_PAPER_WHITE_NITS");
        if (env != nullptr && env[0] != 0) {
            nits = std::atof(env);
        } else {
            // HDR output allowed as the swapchain reads it (the game's settings, else the
            // global ones).
            const auto set = Common::RytyGameConfigValue("hdr_paper_white_nits");
            if (!EmulatorSettings.IsHdrAllowed() || !set) {
                return 0u;
            }
            nits = std::atof(set->c_str());
        }
        if (!(nits >= 50.0 && nits <= 1000.0) || nits == 250.0) {
            return 0u;
        }
        const float scale = static_cast<float>(nits / 10000.0);
        u32 bits;
        std::memcpy(&bits, &scale, sizeof(bits));
        LOG_INFO(Render_Vulkan, "FIX-GT7-143: the game's interface white at {} nits on an HDR output",
                 nits);
        return bits;
    }();
    return wanted;
}

struct InterfaceConstants {
    std::vector<size_t> white;                     // words holding 0.025
    std::vector<std::pair<size_t, float>> decode;  // words holding a decode matrix constant
};

/// The words to change in an HDR interface shader's constants; none for any other shader.
InterfaceConstants FindInterfaceWhite(std::span<const u32> code) {
    constexpr u32 OpConstant = 43;
    constexpr u32 OpFunction = 54;
    InterfaceConstants places;
    if (code.size() < 5 || code[0] != 0x07230203) {
        return places;
    }
    u32 seen = 0;
    for (size_t at = 5; at < code.size();) {
        const u32 count = code[at] >> 16;
        const u32 op = code[at] & 0xffff;
        if (count == 0 || at + count > code.size() || op == OpFunction) {
            break; // constants come before the first function
        }
        if (op == OpConstant && count == 4) {
            switch (code[at + 3]) {
            case InterfaceWhite:
                places.white.push_back(at + 3);
                break;
            case SrgbThreshold:
                seen |= 1;
                break;
            case PqM1:
                seen |= 2;
                break;
            case PqM2:
                seen |= 4;
                break;
            case PqC1:
                seen |= 8;
                break;
            case PqC2:
                seen |= 16;
                break;
            case PqC3:
                seen |= 32;
                break;
            default:
                for (const float value : DecodeMatrix) {
                    if (code[at + 3] == std::bit_cast<u32>(value)) {
                        places.decode.emplace_back(at + 3, value);
                    }
                }
                break;
            }
        }
        at += count;
    }
    if (seen != 63) {
        places.white.clear();
        places.decode.clear();
    }
    return places;
}
} // namespace

std::vector<u32> ModuleCode(vk::ShaderModule module) {
    std::scoped_lock lock{code_mutex};
    const auto it = module_code.find(static_cast<VkShaderModule>(module));
    return it != module_code.end() ? it->second : std::vector<u32>{};
}

vk::ShaderModule CompileSPV(std::span<const u32> code, vk::Device device, bool keep_code) {
    // FIX-GT7-143: GT7's HDR interface shaders with the wanted white.
    std::vector<u32> patched;
    if (const u32 wanted = InterfaceWhiteWanted(); wanted != 0) {
        if (const auto places = FindInterfaceWhite(code); !places.white.empty()) {
            patched.assign(code.begin(), code.end());
            for (const size_t place : places.white) {
                patched[place] = wanted;
            }
            const float back = 0.025f / std::bit_cast<float>(wanted);
            for (const auto& [place, value] : places.decode) {
                patched[place] = std::bit_cast<u32>(value * back);
            }
            code = patched;
        }
    }
    const vk::ShaderModuleCreateInfo shader_info = {
        .codeSize = code.size() * sizeof(u32),
        .pCode = code.data(),
    };

    auto [module_result, module] = device.createShaderModule(shader_info);
    ASSERT_MSG(module_result == vk::Result::eSuccess, "Failed to compile SPIR-V shader: {}",
               vk::to_string(module_result));
    if (keep_code) {
        std::scoped_lock lock{code_mutex};
        module_code[static_cast<VkShaderModule>(module)].assign(code.begin(), code.end());
    }
    return module;
}

} // namespace Vulkan
