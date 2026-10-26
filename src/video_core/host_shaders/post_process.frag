// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450

layout (location = 0) in vec2 uv;
layout (location = 0) out vec4 color;

layout (binding = 0) uniform sampler2D texSampler;
// GT7-UW: the HUD drawn over black and over white, laid over the picture at 16:9.
layout (binding = 1) uniform sampler2D hudBlack;
layout (binding = 2) uniform sampler2D hudWhite;
// GT7-UW: HUD pieces placed elsewhere: src (x, y, w, h of the 16:9 HUD), dst (of the output).
struct HudWidget {
    vec4 src;
    vec4 dst;
};
layout (std140, binding = 3) uniform HudLayout {
    uvec4 count;
    HudWidget widgets[16];
} layout_;

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

bool inside(vec2 p, vec4 rect) {
    return p.x >= rect.x && p.y >= rect.y && p.x < rect.x + rect.z && p.y < rect.y + rect.w;
}

void main() {
    vec4 color_linear = texture(texSampler, uv);
    if (pp.hud) {
        // Where on the 16:9 HUD this pixel shows: a placed piece's, else the centred HUD's
        // (but not what a piece shows elsewhere).
        bool shown = false;
        vec2 hud_uv = vec2(0.0);
        const uint count = min(layout_.count.x, 16u);
        for (uint i = 0u; i < count; ++i) {
            const HudWidget w = layout_.widgets[i];
            if (inside(uv, w.dst)) {
                hud_uv = w.src.xy + (uv - w.dst.xy) / w.dst.zw * w.src.zw;
                shown = true;
                break;
            }
        }
        if (!shown && uv.x >= pp.hud_x0 && uv.x <= pp.hud_x1) {
            hud_uv = vec2((uv.x - pp.hud_x0) / (pp.hud_x1 - pp.hud_x0), uv.y);
            shown = true;
            for (uint i = 0u; i < count; ++i) {
                if (inside(hud_uv, layout_.widgets[i].src)) {
                    shown = false;
                    break;
                }
            }
        }
        if (shown) {
            // Over black the HUD left its colour times its coverage; over white also
            // 1 - coverage. Blended as the game blends it, in the display buffer's encoding.
            const vec3 black = texture(hudBlack, hud_uv).rgb;
            const vec3 white = texture(hudWhite, hud_uv).rgb;
            const float coverage = clamp(1.0 - dot(white - black, vec3(1.0 / 3.0)), 0.0, 1.0);
            color_linear.rgb = color_linear.rgb * (1.0 - coverage) + black;
        }
    }
    if (pp.hdr) {
        color = color_linear;
    } else {
        if (pp.srgb_input) color_linear.rgb = degamma(color_linear.rgb);
        color = vec4(gamma(color_linear.rgb), color_linear.a);
    }
}
