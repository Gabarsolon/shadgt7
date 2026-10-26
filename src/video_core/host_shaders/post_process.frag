// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D texSampler;
// GT7-UW: the HUD drawn over black and over white, laid over the picture at 16:9.
layout (binding = 1) uniform sampler2D hudBlack;
layout (binding = 2) uniform sampler2D hudWhite;

layout (push_constant) uniform settings {
    float gamma;
    bool hdr;
    bool srgb_input;
    bool hud;
    float hud_x0;
    float hud_x1;
} pp;

const float cutoff = 0.0031308, a = 1.055, b = 0.055, d = 12.92;
vec3 gamma(vec3 rgb) {
    return mix(
        a * pow(rgb, vec3(1.0 / (2.4 + 1.0 - pp.gamma))) - b,
        d * rgb / pp.gamma,
        lessThan(rgb, vec3(cutoff))
    );
}

// Exact inverse of gamma() at unit gamma, for buffers that are sRGB encoded but must be sampled
// through a UNORM view because Vulkan has no sRGB variant of the 10-bit format.
vec3 degamma(vec3 rgb) {
    return mix(
        pow(max(rgb + b, 0.0) / a, vec3(2.4)),
        rgb / d,
        lessThan(rgb, vec3(d * cutoff))
    );
}

void main() {
    vec4 color_linear = texture(texSampler, uv);
    if (pp.hud && uv.x >= pp.hud_x0 && uv.x <= pp.hud_x1) {
        // Over black the HUD left its colour times its coverage; over white also 1 - coverage.
        // Blended as the game blends it, in the display buffer's encoding.
        const vec2 hud_uv = vec2((uv.x - pp.hud_x0) / (pp.hud_x1 - pp.hud_x0), uv.y);
        const vec3 black = texture(hudBlack, hud_uv).rgb;
        const vec3 white = texture(hudWhite, hud_uv).rgb;
        const float coverage = clamp(1.0 - dot(white - black, vec3(1.0 / 3.0)), 0.0, 1.0);
        color_linear.rgb = color_linear.rgb * (1.0 - coverage) + black;
    }
    if (pp.hdr) {
        color = color_linear;
    } else {
        if (pp.srgb_input) color_linear.rgb = degamma(color_linear.rgb);
        color = vec4(gamma(color_linear.rgb), color_linear.a);
    }
}
