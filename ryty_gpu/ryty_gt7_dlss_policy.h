// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace Vulkan::Gt7Temporal {
using Matrix = std::array<float, 16>;
inline constexpr Matrix Identity() {
    return {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
}
inline constexpr Matrix Multiply(const Matrix& a, const Matrix& b) {
    Matrix r{};
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x)
            for (int k = 0; k < 4; ++k)
                r[y * 4 + x] += a[y * 4 + k] * b[k * 4 + x];
    return r;
}
inline bool Invert(const Matrix& a, Matrix& result) {
    double v[4][8]{};
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            if (!std::isfinite(a[y * 4 + x]))
                return false;
            v[y][x] = a[y * 4 + x];
            v[y][x + 4] = x == y;
        }
    for (int x = 0; x < 4; ++x) {
        int p = x;
        for (int y = x + 1; y < 4; ++y)
            if (std::abs(v[y][x]) > std::abs(v[p][x]))
                p = y;
        if (std::abs(v[p][x]) < 1e-10)
            return false;
        for (int k = 0; k < 8; ++k)
            std::swap(v[p][k], v[x][k]);
        const double scale = v[x][x];
        for (int k = 0; k < 8; ++k)
            v[x][k] /= scale;
        for (int y = 0; y < 4; ++y)
            if (y != x) {
                const double f = v[y][x];
                for (int k = 0; k < 8; ++k)
                    v[y][k] -= f * v[x][k];
            }
    }
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) {
            result[y * 4 + x] = float(v[y][x + 4]);
            if (!std::isfinite(result[y * 4 + x]))
                return false;
        }
    return true;
}
// Only this view is reconstructed. A thumbnail or split viewport must not discard a
// main scene that was already reconstructed earlier in the same guest frame.
inline bool MainScene(unsigned width, unsigned height, float viewport_width,
                      float viewport_height, float xoffset, float yoffset) {
    return width == 1920 && height == 1080 && viewport_width == 1920 &&
           viewport_height == 1080 && xoffset == 960 && yoffset == 540;
}

inline bool TaaHash(uint64_t hash) {
    switch (hash) {
    case 0xcfa125dd:
    case 0x0fd5ae97:
    case 0x5ab75ff4:
    case 0x5e2664aa:
    case 0xae2195b7:
    case 0xc964fc93:
    case 0x9f7462a4:
    case 0x902e1ddb:
        return true;
    default:
        return false;
    }
}
inline bool InitializeHash(uint64_t h) {
    return h == 0x423924ee || h == 0x6f703f02;
}
// Only Depth_NewMode's binding schema is statically closed. Other variants are traced.
inline bool ProvenSchema(uint64_t h) {
    return h == 0xcfa125dd || h == 0x0fd5ae97;
}
inline constexpr float Abs(float v) {
    return v < 0 ? -v : v;
}
inline constexpr bool Finite(float v) {
    return v == v && v <= 3.402823466e38f && v >= -3.402823466e38f;
}
// A Vulkan scissor can extend beyond the attachment. Such a rectangle still covers
// the whole scene; exact extent equality incorrectly rejects the native post composite.
inline constexpr bool CoversScene(int32_t x, int32_t y, uint32_t width, uint32_t height,
                                  uint32_t scene_width, uint32_t scene_height) {
    return scene_width && scene_height && x == 0 && y == 0 && width >= scene_width &&
           height >= scene_height;
}
// Without a pre-rasterization ViewportIndex output, Vulkan uses viewport/scissor 0.
// Additional bound rectangles do not crop that draw. Unknown indexed selection stays native.
template <typename Scissors>
inline bool CoversSceneViewportZero(const Scissors& scissors, bool fixed_viewport_zero,
                                    uint32_t scene_width, uint32_t scene_height) {
    if (!fixed_viewport_zero || scissors.empty())
        return false;
    const auto& clip = scissors[0];
    return CoversScene(clip.offset.x, clip.offset.y, clip.extent.width, clip.extent.height,
                       scene_width, scene_height);
}
// The alias need not have an own write. The recorded complete copy must hold
// exactly the version proved fresh at its producer, newer than foreign writes.
inline constexpr bool FreshInputCopy(uint64_t own, uint64_t foreign, uint64_t held,
                                     uint64_t producer, bool complete) {
    return complete && producer != 0 && held == producer && held >= own && held >= foreign;
}
inline constexpr bool WholeBufferProducer(uint64_t image, uint64_t image_size, uint64_t writer,
                                          uint64_t writer_size, bool exact) {
    constexpr auto max = std::numeric_limits<uint64_t>::max();
    return image_size && writer_size && image <= max - image_size && writer <= max - writer_size &&
           writer <= image && writer + writer_size >= image + image_size &&
           (exact || writer_size - image_size <= image_size);
}
inline constexpr bool Constants(const std::array<float, 29>& c, unsigned w, unsigned h) {
    for (float v : c)
        if (!Finite(v))
            return false;
    return w >= 960 && h >= 540 && w <= 8192 && h <= 8192 && c[16] == float(w) &&
           c[17] == float(h) && Abs(c[14] - w * .5f) < .01f && Abs(c[15] + h * .5f) < .01f &&
           Abs(c[20] * c[14]) <= 2 && Abs(c[21] * c[15]) <= 2 && c[18] == 1 && c[19] == 1 &&
           c[24] == 0 && c[25] == 0;
}
inline constexpr std::array<float, 2> RasterJitter(const std::array<float, 29>& c, float sx = -1,
                                                   float sy = -1) {
    return {sx * c[20] * c[14], sy * c[21] * c[15]};
}
struct Projection {
    float near_plane{}, far_plane{}, fov{}, aspect{};
    bool inverted{};
};
// Row-vector perspective, Vulkan/D3D depth [0,1]. Reject orthographic, oblique,
// infinite far, and unknown conventions instead of inventing near/far/FOV.
inline bool Perspective(const Matrix& p, Projection& result) {
    for (float v : p)
        if (!std::isfinite(v))
            return false;
    for (int i : {1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 15})
        if (std::abs(p[i]) > 1e-5f)
            return false;
    if (std::abs(std::abs(p[11]) - 1) > 1e-4f || std::abs(p[0]) < .01f || std::abs(p[5]) < .01f)
        return false;
    const float a = p[10] / p[11], b = p[14] / p[11];
    if (std::abs(a) < 1e-9f || std::abs(1 - a) < 1e-9f)
        return false;
    const float z0 = std::abs(-b / a), z1 = std::abs(b / (1 - a));
    result = {std::min(z0, z1), std::max(z0, z1), 2 * std::atan(1 / std::abs(p[5])),
              std::abs(p[5] / p[0]), z1 < z0};
    return result.near_plane >= .001f && result.near_plane < 100 &&
           result.far_plane > result.near_plane && result.far_plane <= 1e8f && result.fov > .1f &&
           result.fov < 3 && result.aspect > .5f && result.aspect < 4;
}
enum class ProbeResult { InvalidDepth, Empty, InvalidMotion, Qualified };
inline constexpr ProbeResult QualifyProbe(bool sane, unsigned geometry, unsigned motion) {
    return !sane ? ProbeResult::InvalidDepth
           : geometry < 16 ? ProbeResult::Empty
           : motion >= 16 && motion * 2 >= geometry ? ProbeResult::Qualified
                                                   : ProbeResult::InvalidMotion;
}
inline constexpr bool LensCut(const Projection& lens, const Projection& old) {
    const auto changed = [](float a, float b, float relative) {
        return Abs(a - b) > relative * std::max(Abs(a), Abs(b));
    };
    return lens.inverted != old.inverted || changed(lens.near_plane, old.near_plane, .02f) ||
           changed(lens.far_plane, old.far_plane, .02f) ||
           Abs(lens.fov - old.fov) > .01f || Abs(lens.aspect - old.aspect) > .005f;
}
inline constexpr const char* ViewCut(const Matrix& inverse, const Matrix& old) {
    float distance = 0;
    for (int i = 0; i < 3; ++i) {
        const float delta = inverse[12 + i] - old[12 + i];
        distance += delta * delta;
    }
    if (distance > 2500)
        return "camera-position-cut";
    // All three axes detect roll as well as yaw/pitch, normalized for small drift.
    for (int axis : {0, 4, 8}) {
        float dot = 0, a = 0, b = 0;
        for (int i = 0; i < 3; ++i) {
            dot += inverse[axis + i] * old[axis + i];
            a += inverse[axis + i] * inverse[axis + i];
            b += old[axis + i] * old[axis + i];
        }
        if (a < .5f || b < .5f || dot < 0 || dot * dot < .25f * a * b)
            return "camera-rotation-cut";
    }
    return nullptr;
}
// Original (unjittered) matrices, never arena addresses or snapshot sequence IDs.
// Allow racing motion; abrupt lens changes / >50 m / >60 degrees start history.
inline const char* CameraCut(const Matrix& projection, const Matrix& view,
                             const Matrix& old_projection, const Matrix& old_view) {
    Projection lens{}, old_lens{};
    Matrix inverse{}, old_inverse{};
    if (!Perspective(projection, lens) || !Perspective(old_projection, old_lens) ||
        !Invert(view, inverse) || !Invert(old_view, old_inverse))
        return "invalid-camera";
    if (LensCut(lens, old_lens))
        return "camera-lens-cut";
    for (unsigned i = 0; i < 16; ++i)
        if (std::abs(projection[i] - old_projection[i]) > .01f)
            return "projection-change";
    return ViewCut(inverse, old_inverse);
}
struct History {
    uint64_t last = ~uint64_t{}, view{}, sampled{}, written{};
    const char* pending = "first-frame";
    constexpr void Invalidate(const char* reason) {
        pending = reason;
    }
    constexpr const char* Reason(uint64_t frame, uint64_t /*diagnostic_address*/, float elapsed_ms) const {
        if (pending)
            return pending;
        if (frame != last + 1)
            return "missed-scene-frame";
        if (elapsed_ms > 250)
            return "long-gap";
        return nullptr;
    }
    constexpr void GamePair(uint64_t read, uint64_t write) {
        if (sampled &&
            !((read == sampled && write == written) || (read == written && write == sampled)))
            Invalidate("game-history-pair-change");
        sampled = read;
        written = write;
    }
    constexpr void Commit(uint64_t frame, uint64_t camera) {
        last = frame;
        view = camera;
        pending = nullptr;
    }
};
// These compile-time regressions run in the prescribed ryty_gpu build, without
// loading the renderer or launching a game. Exercise actual validation/history.
static_assert([] {
    std::array<float, 29> c{};
    c[14] = 960;
    c[15] = -540;
    c[16] = 1920;
    c[17] = 1080;
    c[18] = c[19] = 1;
    c[20] = .25f / 960;
    c[21] = -.125f / 540;
    if (!Constants(c, 1920, 1080) || RasterJitter(c)[0] != -.25f || RasterJitter(c)[1] != -.125f)
        return false;
    c[28] = std::numeric_limits<float>::quiet_NaN();
    if (Constants(c, 1920, 1080))
        return false;
    c[28] = 0;
    c[28] = std::numeric_limits<float>::max(); // observed GT7 C28, not an invalid infinity
    if (!Constants(c, 1920, 1080))
        return false;
    c[24] = 1;
    if (Constants(c, 1920, 1080))
        return false;
    c[24] = 0;
    c[16] = 1600;
    return !Constants(c, 1920, 1080);
}());
static_assert([] {
    History h;
    if (!h.Reason(10, 1, 16))
        return false;
    h.GamePair(100, 200);
    h.Commit(10, 1);
    h.GamePair(200, 100);
    if (h.Reason(11, 1, 16))
        return false; // normal ping-pong must accumulate
    if (!h.Reason(12, 1, 16) || h.Reason(11, 2, 16) || !h.Reason(11, 1, 300))
        return false;
    h.GamePair(300, 400);
    if (!h.Reason(11, 1, 16))
        return false;
    h.Commit(11, 1);
    h.Invalidate("failed-evaluation");
    return h.Reason(12, 1, 16) != nullptr;
}());
static_assert([] {
    const Projection lens{.1f, 5000.f, 1.0472f, 1.77778f, true};
    auto other = lens;
    if (LensCut(lens, other)) return false;
    other.fov += .1f;
    if (!LensCut(lens, other)) return false;
    other = lens; other.aspect = 1;
    if (!LensCut(lens, other)) return false;
    other = lens; other.near_plane = .2f;
    if (!LensCut(lens, other)) return false;
    other = lens; other.far_plane = 10000;
    if (!LensCut(lens, other)) return false;
    auto view = Identity();
    view[12] = 5;
    if (ViewCut(view, Identity())) return false; // ordinary racing translation
    view[12] = 100;
    if (!ViewCut(view, Identity())) return false;
    view = Identity(); view[0] = view[5] = -1; // abrupt roll
    return ViewCut(view, Identity()) != nullptr;
}());
static_assert(QualifyProbe(true, 0, 0) == ProbeResult::Empty); // sky is inconclusive
static_assert(QualifyProbe(true, 64, 64) == ProbeResult::Qualified); // even a flat wall
static_assert(QualifyProbe(true, 64, 3) == ProbeResult::InvalidMotion);
static_assert(QualifyProbe(false, 64, 64) == ProbeResult::InvalidDepth);
static_assert(Multiply(Identity(), Identity()) == Identity());
static_assert(FreshInputCopy(0, 0, 42, 42, true));   // observed zero-own-version alias
static_assert(!FreshInputCopy(0, 0, 41, 42, true));  // producer changed after copy
static_assert(!FreshInputCopy(0, 43, 42, 42, true)); // overwritten by a buffer write
static_assert(!FreshInputCopy(43, 0, 42, 42, true)); // newer local write wins
static_assert(!FreshInputCopy(0, 0, 42, 42, false)); // crop/partial copy cannot qualify
static_assert(!FreshInputCopy(0, 0, 0, 0, true));    // no untracked-writer exemption
static_assert(WholeBufferProducer(100, 100, 100, 100, false));
static_assert(!WholeBufferProducer(100, 100, 150, 100, false)); // partial overlap
static_assert(!WholeBufferProducer(100, 100, 0, 1000, false));  // unbounded storage region
static_assert(WholeBufferProducer(100, 100, 0, 1000, true));    // exact copy/fill
static_assert(!WholeBufferProducer(UINT64_MAX - 9, 100, 0, 1000, true));
} // namespace Vulkan::Gt7Temporal
