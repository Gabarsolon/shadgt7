// SPDX-FileCopyrightText: Copyright 2024 shadPS4 Emulator Project
// SPDX-License-Identifier: GPL-2.0-or-later

#version 450
#extension GL_GOOGLE_include_directive : require

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
    float dlss_sharpening; // the RCAS gain (0..1) of the scene pass
    bool scene_output;     // the scene pass: RCAS only, stored as it was read
    bool scene_srgb;       // the scene was read through an sRGB view: linear samples
} pp;

// AMD RCAS after DLSS (GPU.dlss_sharpening), the function FSR 1 uses. It works on perceptual
// values: PQ and gamma-encoded buffers are, an sRGB view decodes on fetch, so encode for the
// filter and decode its result.
#define A_GPU 1
#define A_GLSL 1
#include "fsr/ffx_a.h"
#define FSR_RCAS_F 1
vec3 rcas_encode(vec3 v) {
    return mix(1.055 * pow(max(v, vec3(0)), vec3(1.0 / 2.4)) - 0.055, 12.92 * v,
               lessThanEqual(v, vec3(0.0031308)));
}
vec3 rcas_decode(vec3 v) {
    return mix(pow(max((v + 0.055) / 1.055, vec3(0)), vec3(2.4)), v / 12.92,
               lessThanEqual(v, vec3(0.04045)));
}
AF4 FsrRcasLoadF(ASU2 p) {
    vec4 v = texelFetch(texSampler, clamp(p, ivec2(0), textureSize(texSampler, 0) - 1), 0);
    if (pp.scene_srgb) v.rgb = rcas_encode(v.rgb);
    return v;
}
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) { }
#include "fsr/ffx_fsr1.h"

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
    if (pp.scene_output) {
        ivec2 pixel = clamp(ivec2(uv * textureSize(texSampler, 0)), ivec2(0),
                            textureSize(texSampler, 0) - 1);
        vec4 scene = texelFetch(texSampler, pixel, 0);
        vec3 sharpened;
        FsrRcasF(sharpened.r, sharpened.g, sharpened.b, uvec2(pixel),
                 uvec4(floatBitsToUint(pp.dlss_sharpening), 0u, 0u, 0u));
        sharpened = max(sharpened, vec3(0));
        color = vec4(pp.scene_srgb ? rcas_decode(sharpened) : sharpened, scene.a);
        return;
    }
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
