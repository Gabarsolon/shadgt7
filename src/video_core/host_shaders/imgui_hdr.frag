// SPDX-FileCopyrightText: Copyright 2026 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450 core

// FIX-GT7-142: the emulator's own drawings (frame counter, shader messages, notifications,
// menus) on an HDR10 output. Their colours are sRGB; written as they are into a PQ picture, white
// came out at several thousand nits (measured: the frame counter's text at 2,700 to 3,300 nits
// over a game interface at 250). Here they are placed at uWhiteNits for white, in BT.2020 and PQ
// as the output expects. The game's picture goes through the same pipeline already encoded for
// the display: it is drawn with uWhiteNits 0 and left alone.
// Same inputs as imgui's stock fragment shader (imgui_impl_vulkan.cpp).

layout(location = 0) out vec4 fColor;
layout(set = 0, binding = 0) uniform sampler2D sTexture;
layout(location = 0) in struct { vec4 Color; vec2 UV; } In;

layout(push_constant) uniform uPushConstant {
    // After the vertex stage's scale and translation.
    layout(offset = 16) float uWhiteNits;
} pc;

vec3 srgb_to_linear(vec3 c) {
    return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, lessThanEqual(c, vec3(0.04045)));
}

// SMPTE ST 2084
vec3 nits_to_pq(vec3 nits) {
    const float m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    vec3 y = pow(clamp(nits / 10000.0, 0.0, 1.0), vec3(m1));
    return pow((c1 + c2 * y) / (1.0 + c3 * y), vec3(m2));
}

void main() {
    vec4 color = In.Color * texture(sTexture, In.UV.st);
    if (pc.uWhiteNits > 0.0) {
        // BT.709 to BT.2020 primaries, on linear light.
        const mat3 to_bt2020 = mat3(0.6274, 0.0691, 0.0164,
                                    0.3293, 0.9195, 0.0880,
                                    0.0433, 0.0114, 0.8956);
        vec3 linear = srgb_to_linear(clamp(color.rgb, 0.0, 1.0));
        color.rgb = nits_to_pq(to_bt2020 * linear * pc.uWhiteNits);
    }
    fColor = color;
}
