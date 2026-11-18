// smear_render.h — Smear's motion blur renderer (CPU reference).
//
// Blur is gathered from the current frame only, along per-pixel motion vectors
// derived from the optical flow to the previous and next frames. Neighbour
// frames are used to measure motion and to find occlusions, never blended in,
// so they cannot leave ghost copies. Passes (the GPU kernels in
// plugin/gpu/smear_kernels.inc implement the same ones):
//
//   1. motion:     per pixel, the blur vector (half the shutter-long streak) and
//                  an occlusion score. With both neighbours, the two flows of a
//                  pixel visible in all three frames agree (to_next ≈ −to_prev:
//                  forward-backward consistency under locally constant
//                  velocity), and both land on the pixel's colour. Where they
//                  disagree, the flow whose target does not match is the one
//                  that lost the pixel behind something: the pixel is occluded
//                  on that side, and its motion comes from the other flow.
//      refine:     a pixel adopts a neighbour's vector when that single centred
//                  velocity explains it better in both neighbour frames; three
//                  frames constrain far more than two, so vectors that bled
//                  across an object's edge are corrected.
//   2. tiles:      per 16×16 tile, the longest blur vector, and the
//                  occlusion-weighted sum of blur vectors (occluded pixels
//                  belong to the layer that is behind).
//   3. neighbours: per tile, the dominant vector among all tiles whose streaks
//                  can reach it, and the velocity of the background layer
//                  around it (from the occluded pixels). Passes 2–3 run three
//                  times: each round down-weights occluded pixels far from the
//                  previous background estimate, so a few misjudged pixels of
//                  the front layer cannot drag it.
//   4. gather:     a reconstruction filter after McGuire et al. 2012 ("A
//                  Reconstruction Filter for Plausible Motion Blur") with the
//                  tile/neighbour-max scheme of Jimenez 2014. Pixels moving
//                  like the local background are "behind", others "in front".
//                  A pixel's colour is its own layer blurred along its own
//                  vector (front samples excluded), covered for the fraction of
//                  the shutter during which front samples' streaks pass over it.
//                  Background therefore never smears over the foreground, and
//                  the foreground's blur still spills over the background.
#pragma once

#include "smear_core.h"

namespace smear {

enum class Track { Previous = 1, Next = 2, Both = 3 };

struct BlurSettings {
    double shutter = 0.5;   // Blur Amount: fraction of a frame the shutter is open (0.5 = 180°), centred
    Track track = Track::Both;
    int max_samples = 32;   // samples per pixel along the streaks
    bool show_vectors = false;  // debug view: motion vectors as colours instead of the blur
};

// Fixed tuning shared by the CPU and GPU paths (the GPU path copies these into
// its parameter blocks). Velocities are in pixels per frame, errors in 0–1 units.
struct BlurTuning {
    int tile = 16;                // tile side for the max/neighbour passes
    float max_radius = 128.0f;    // longest half-streak, pixels
    float photo_lo = 0.03f;       // colour error at which a flow starts to be doubted…
    float photo_hi = 0.10f;       // …and at which it is rejected
    float refine_margin = 0.01f;  // motion refinement: error gain needed to adopt a neighbour's vector
    int refine_steps[2] = {3, 1}; // motion refinement passes: neighbour distances (0 = off)
    float cons_abs = 0.75f;       // flows disagreeing by less than this (px) are consistent…
    float cons_rel = 0.10f;       // …plus this fraction of the speed
    float layer_abs = 1.0f;       // speed difference from the background (px/frame) that starts "in front"…
    float layer_rel = 0.10f;      // …plus this fraction of the background speed
    float mass_lo = 2.0f;         // occluded pixels (sum of scores) needed nearby before layering kicks in…
    float mass_hi = 8.0f;         // …and for full layering
    int robust_rounds = 2;        // reweighting rounds of the background estimate
    int bg_tiles = 3;             // background estimate radius: this many tiles…
    float bg_scale = 2.0f;        // …plus this many dominant streak lengths
};

// --- Pass 1: motion ---------------------------------------------------------------

// Whether pixel (x, y) moved by `u` lands inside `img`.
inline bool inside(const Image& img, int x, int y, Vec2 u) {
    const float px = float(x) + u.x, py = float(y) + u.y;
    return px > -0.5f && py > -0.5f && px < float(img.width) - 0.5f && py < float(img.height) - 0.5f;
}

// Colour error between `c` and image `img` at (x, y); 1 outside the image.
inline float match_error(const Pixel& c, const Image& img, float x, float y) {
    if (!inside(img, 0, 0, {x, y})) return 1.0f;
    const Pixel s = sample_image(img, x, y);
    return 0.25f * (std::fabs(s.r - c.r) + std::fabs(s.g - c.g) + std::fabs(s.b - c.b) + std::fabs(s.a - c.a));
}

// Cell = (half-streak x, half-streak y, occlusion 0–1, 0).
inline Vec4 motion_cell(const Image& cur, const Image* prev, const Image* next, const FlowField* to_prev,
                        const FlowField* to_next, float half_shutter, float max_radius, const BlurTuning& t, int x,
                        int y) {
    const Pixel c = cur.at(x, y);
    const bool hp = prev && to_prev, hn = next && to_next;
    Vec2 v;
    float occ = 0.0f;
    if (hp && hn) {
        const Vec2 un = to_next->at(x, y), up = to_prev->at(x, y);
        const float en = match_error(c, *next, float(x) + un.x, float(y) + un.y);
        const float ep = match_error(c, *prev, float(x) + up.x, float(y) + up.y);
        const float wn = 1.0f - smoothstepf(t.photo_lo, t.photo_hi, en);
        const float wp = 1.0f - smoothstepf(t.photo_lo, t.photo_hi, ep);
        const float dx = un.x + up.x, dy = un.y + up.y;
        const float d = std::sqrt(dx * dx + dy * dy);
        const float speed = 0.5f * (std::sqrt(un.x * un.x + un.y * un.y) + std::sqrt(up.x * up.x + up.y * up.y));
        const float lo = t.cons_abs + t.cons_rel * speed;
        const float incons = smoothstepf(lo, 2.0f * lo, d);
        const float kn = (1.0f - incons) * 0.5f + incons * (wn + 1e-3f) / (wn + wp + 2e-3f);
        v = {kn * un.x - (1.0f - kn) * up.x, kn * un.y - (1.0f - kn) * up.y};
        // Occluded on one side: one flow matches, the other does not. A target
        // outside the frame is a pixel leaving the picture, not an occlusion.
        occ = (inside(*next, x, y, un) && inside(*prev, x, y, up)) ? std::fabs(wn - wp) : 0.0f;
    } else if (hn) {
        const Vec2 un = to_next->at(x, y);
        v = un;
        occ = inside(*next, x, y, un) ? smoothstepf(t.photo_lo, t.photo_hi, match_error(c, *next, float(x) + un.x, float(y) + un.y)) : 0.0f;
    } else if (hp) {
        const Vec2 up = to_prev->at(x, y);
        v = {-up.x, -up.y};
        occ = inside(*prev, x, y, up) ? smoothstepf(t.photo_lo, t.photo_hi, match_error(c, *prev, float(x) + up.x, float(y) + up.y)) : 0.0f;
    }
    float hx = v.x * half_shutter, hy = v.y * half_shutter;
    const float len = std::sqrt(hx * hx + hy * hy);
    if (len > max_radius) {
        hx *= max_radius / len;
        hy *= max_radius / len;
    }
    return {hx, hy, occ, 0.0f};
}

// Pass 1b: each pixel keeps its vector or takes that of a neighbour `step`
// pixels away (8 directions), whichever explains it best in BOTH neighbour
// frames (a single centred velocity must match the next frame at +v and the
// previous at −v). Three frames constrain far more than two: vectors bleeding
// across an object's edge fail one of them. Mixed edge pixels snap to the
// layer they belong to. Occlusion scores are recomputed for adopted vectors.
inline float motion_cost(const Pixel& c, const Image* prev, const Image* next, int x, int y, float vx, float vy,
                         const BlurTuning& t, float& occ) {
    float en = 0.0f, ep = 0.0f;
    bool in_n = true, in_p = true;
    if (next) {
        in_n = inside(*next, x, y, {vx, vy});
        en = match_error(c, *next, float(x) + vx, float(y) + vy);
    }
    if (prev) {
        in_p = inside(*prev, x, y, {-vx, -vy});
        ep = match_error(c, *prev, float(x) - vx, float(y) - vy);
    }
    if (next && prev) {
        const float wn = 1.0f - smoothstepf(t.photo_lo, t.photo_hi, en);
        const float wp = 1.0f - smoothstepf(t.photo_lo, t.photo_hi, ep);
        occ = (in_n && in_p) ? std::fabs(wn - wp) : 0.0f;
        // Visible on both sides: both errors count; occluded on one side: the
        // better side counts fully, the other only up to the rejection level.
        return std::fmin(en, ep) + std::fmin(std::fmax(en, ep), t.photo_hi);
    }
    const float e = next ? en : ep;
    occ = (next ? in_n : in_p) ? smoothstepf(t.photo_lo, t.photo_hi, e) : 0.0f;
    return e;
}

inline Vec4 refine_motion_cell(const Image& cur, const Image* prev, const Image* next, const Grid<Vec4>& mot,
                               float inv_half, int step, const BlurTuning& t, int x, int y) {
    const Pixel c = cur.at(x, y);
    const Vec4 m0 = mot.at(x, y);
    float occ0 = 0.0f;
    const float e0 = motion_cost(c, prev, next, x, y, m0.x * inv_half, m0.y * inv_half, t, occ0);
    Vec4 best = m0;
    float best_e = e0 - t.refine_margin;  // a neighbour must be clearly better
    for (int k = 0; k < 8; ++k) {
        const int px = clampi(x + kOffX[k] * step, 0, mot.width - 1), py = clampi(y + kOffY[k] * step, 0, mot.height - 1);
        const Vec4 m = mot.at(px, py);
        if (std::fabs(m.x - best.x) + std::fabs(m.y - best.y) < 0.01f) continue;  // same vector: nothing to gain
        float occ = 0.0f;
        const float e = motion_cost(c, prev, next, x, y, m.x * inv_half, m.y * inv_half, t, occ);
        if (e < best_e) {
            best_e = e;
            best = {m.x, m.y, occ, 0.0f};
        }
    }
    return best;
}

// --- Pass 2: tiles ---------------------------------------------------------------

struct TileCells {
    Grid<Vec2> vmax;  // longest half-streak in the tile
    Grid<Vec4> occ;   // (Σ occ · h.x, Σ occ · h.y, Σ occ, 0)
};

// `prior`: the previous estimate of the neighbourhood sums (pass 3 output), or
// null. With a prior, occluded pixels far from the prior background velocity
// weigh less (Cauchy weights), so a few misjudged pixels of the front layer
// cannot drag the background estimate: two such rounds converge on the
// dominant occluded velocity.
inline void tile_cell(const Grid<Vec4>& mot, const Grid<Vec4>* prior, int tile, float sigma_abs, float sigma_rel,
                      int tx, int ty, Vec2& vmax, Vec4& occ) {
    const int x0 = tx * tile, y0 = ty * tile;
    const int x1 = std::min(x0 + tile, mot.width), y1 = std::min(y0 + tile, mot.height);
    float px = 0, py = 0, inv_s2 = 0;
    const bool robust = prior != nullptr;
    if (robust) {
        const Vec4 pr = prior->at(tx, ty);
        const float k = 1.0f / std::fmax(pr.z, 1e-6f);
        px = pr.x * k;
        py = pr.y * k;
        const float sg = sigma_abs + sigma_rel * std::sqrt(px * px + py * py);
        inv_s2 = 1.0f / (sg * sg);
    }
    float best = -1.0f, sx = 0, sy = 0, so = 0;
    vmax = {};
    for (int y = y0; y < y1; ++y)
        for (int x = x0; x < x1; ++x) {
            const Vec4 m = mot.at(x, y);
            const float l2 = m.x * m.x + m.y * m.y;
            if (l2 > best) {
                best = l2;
                vmax = {m.x, m.y};
            }
            float wgt = m.z;
            if (robust) {
                const float dx = m.x - px, dy = m.y - py;
                wgt = wgt / (1.0f + (dx * dx + dy * dy) * inv_s2);
            }
            sx += wgt * m.x;
            sy += wgt * m.y;
            so += wgt;
        }
    occ = {sx, sy, so, 0.0f};
}

// --- Pass 3: neighbours ------------------------------------------------------------

// Dominant half-streak among the tiles that can reach tile (tx, ty), and the
// summed occlusion data of the tiles around it (radius grows with the motion).
inline void neighbour_cell(const Grid<Vec2>& vmax, const Grid<Vec4>& occ, int tile, int reach, int t_bg_base,
                           float t_bg_scale, int tx, int ty, Vec2& nmax, Vec4& nocc) {
    const int tw = vmax.width, th = vmax.height;
    float best = -1.0f;
    nmax = {};
    for (int dy = -reach; dy <= reach; ++dy) {
        const int y = ty + dy;
        if (y < 0 || y >= th) continue;
        for (int dx = -reach; dx <= reach; ++dx) {
            const int x = tx + dx;
            if (x < 0 || x >= tw) continue;
            const Vec2 v = vmax.at(x, y);
            const float l2 = v.x * v.x + v.y * v.y;
            if (l2 <= best) continue;
            const int ring = std::max(std::abs(dx), std::abs(dy));
            if (ring > 1) {
                // Gap between the two tiles, and whether the streak points at us.
                const float gap = float(ring - 1) * float(tile);
                const float len = std::sqrt(l2);
                if (len <= gap) continue;
                const float ddx = float(-dx), ddy = float(-dy);
                const float cosang = std::fabs(v.x * ddx + v.y * ddy) / (len * std::sqrt(ddx * ddx + ddy * ddy));
                if (cosang < 0.5f) continue;
            }
            best = l2;
            nmax = v;
        }
    }
    const float lmax = std::sqrt(std::fmax(best, 0.0f));
    const int r = std::min(reach, t_bg_base + int(std::ceil(t_bg_scale * lmax / float(tile))));
    float sx = 0, sy = 0, so = 0;
    for (int y = std::max(0, ty - r); y <= std::min(th - 1, ty + r); ++y)
        for (int x = std::max(0, tx - r); x <= std::min(tw - 1, tx + r); ++x) {
            const Vec4 o = occ.at(x, y);
            sx += o.x;
            sy += o.y;
            so += o.z;
        }
    nocc = {sx, sy, so, 0.0f};
}

// --- Pass 4: gather -------------------------------------------------------------------

// How much a pixel's layer is in front, 0–1: its speed difference from the
// local background, once there are enough occlusions around to tell layers apart.
inline float front_score(float hx, float hy, float bx, float by, float to_velocity, float layer_gate,
                         const BlurTuning& t) {
    const float dx = (hx - bx) * to_velocity, dy = (hy - by) * to_velocity;
    const float dv = std::sqrt(dx * dx + dy * dy);
    const float bs = std::sqrt(bx * bx + by * by) * to_velocity;
    const float lo = t.layer_abs + t.layer_rel * bs;
    return layer_gate * smoothstepf(lo, 2.0f * lo + 1.0f, dv);
}

struct Background {
    float x = 0, y = 0;  // half-streak of the local background
    float gate = 0;      // 0: no occlusions around, layers cannot be told apart
};

// Local background at pixel (x, y): occlusion-weighted mean streak of the
// neighbour tiles, interpolated between tile centres.
inline Background background_at(const Grid<Vec4>& nocc, int tile, const BlurTuning& t, int x, int y) {
    const float qx = (float(x) + 0.5f) / float(tile) - 0.5f, qy = (float(y) + 0.5f) / float(tile) - 0.5f;
    const Bilinear bl = bilinear(qx, qy, nocc.width, nocc.height);
    const Vec4 o00 = nocc.at(bl.x0, bl.y0), o10 = nocc.at(bl.x1, bl.y0);
    const Vec4 o01 = nocc.at(bl.x0, bl.y1), o11 = nocc.at(bl.x1, bl.y1);
    auto lerp2 = [&](float a, float b, float c, float d) {
        const float top = a + (b - a) * bl.fx, bottom = c + (d - c) * bl.fx;
        return top + (bottom - top) * bl.fy;
    };
    const float sx = lerp2(o00.x, o10.x, o01.x, o11.x), sy = lerp2(o00.y, o10.y, o01.y, o11.y);
    const float mass = lerp2(o00.z, o10.z, o01.z, o11.z);
    const float inv_mass = 1.0f / std::fmax(mass, 1e-6f);
    Background b;
    b.x = sx * inv_mass;
    b.y = sy * inv_mass;
    b.gate = smoothstepf(t.mass_lo, t.mass_hi, mass);
    return b;
}

// Fraction of a pixel covered by a streak of half-length `len` at distance `d`.
inline float cover(float d, float len) { return clampf(len + 0.5f - d, 0.0f, 1.0f); }

// Hue wheel for the vector view (direction = hue, speed = saturation, front = brightness).
inline Pixel vector_colour(float hx, float hy, float to_velocity, float front, float occ) {
    const float vx = hx * to_velocity, vy = hy * to_velocity;
    const float speed = std::sqrt(vx * vx + vy * vy);
    const float hue = (std::atan2(vy, vx) * 0.15915494f + 0.5f) * 6.0f;  // 0..6
    const float r = clampf(std::fabs(hue - 3.0f) - 1.0f, 0.0f, 1.0f);
    const float g = clampf(2.0f - std::fabs(hue - 2.0f), 0.0f, 1.0f);
    const float b = clampf(2.0f - std::fabs(hue - 4.0f), 0.0f, 1.0f);
    const float sat = 1.0f - std::exp(-speed * 0.0625f);
    const float val = 0.5f + 0.5f * front;
    Pixel p{val * (1.0f - sat + sat * r), val * (1.0f - sat + sat * g), val * (1.0f - sat + sat * b), 1.0f};
    const float o = 0.6f * occ;  // occluded pixels tinted magenta
    p.r += (1.0f - p.r) * o;
    p.g *= 1.0f - o;
    p.b += (1.0f - p.b) * o;
    return p;
}

struct GatherParams {
    int tile = 16;
    int max_samples = 32;
    float to_velocity = 4.0f;  // half-streak → pixels per frame (2 / shutter)
    bool show_vectors = false;
};

inline Pixel gather_cell(const Image& cur, const Grid<Vec4>& mot, const Grid<Vec2>& nmax, const Grid<Vec4>& nocc,
                         const GatherParams& gp, const BlurTuning& t, int x, int y) {
    const int w = cur.width, h = cur.height;
    const Pixel c = cur.at(x, y);
    const Vec4 m = mot.at(x, y);
    const int tx = x / gp.tile, ty = y / gp.tile;

    const Background bk = background_at(nocc, gp.tile, t, x, y);
    const float bgx = bk.x, bgy = bk.y, gate = bk.gate;
    const float zx = front_score(m.x, m.y, bgx, bgy, gp.to_velocity, gate, t);

    if (gp.show_vectors) return vector_colour(m.x, m.y, gp.to_velocity, zx, m.z);

    const Vec2 nm = nmax.at(tx, ty);
    const float ln = std::sqrt(nm.x * nm.x + nm.y * nm.y);
    const float lx = std::sqrt(m.x * m.x + m.y * m.y);
    if (ln < 0.5f) return c;

    // Sample counts: up to one per pixel of streak, split between the dominant
    // direction (front layers covering us) and our own (our layer's blur).
    // When our own vector is the dominant one, nothing nearby moves faster
    // over us: all samples go to our own streak.
    const int budget = std::max(4, gp.max_samples);
    const float dnx = nm.x - m.x, dny = nm.y - m.y;
    const bool own_dominant = dnx * dnx + dny * dny < 0.25f;
    int nb = 0;
    if (lx >= 0.5f) nb = std::min(own_dominant ? budget : budget / 2, std::max(2, 2 * int(std::ceil(lx))));
    nb &= ~1;
    int na = 0;
    if (!own_dominant) {
        na = std::min(budget - nb, std::max(2, 2 * int(std::ceil(ln))));
        na = std::max(2, na & ~1);
    }

    const float ndx = nm.x / ln, ndy = nm.y / ln;
    const float lxn = std::fabs(m.x * ndx + m.y * ndy);  // our streak along the dominant direction

    // Front coverage (layers whose streaks pass over this pixel).
    float fw = 0, fr = 0, fg = 0, fb = 0, fa = 0;
    const float da = na > 0 ? 2.0f * ln / float(na) : 0.0f;
    for (int i = 0; i < na; ++i) {
        const float s = -1.0f + (2.0f * float(i) + 1.0f) / float(na);
        const float px = float(x) + s * nm.x, py = float(y) + s * nm.y;  // outside: edge pixels extend
        const int ix = clampi(int(std::floor(px + 0.5f)), 0, w - 1), iy = clampi(int(std::floor(py + 0.5f)), 0, h - 1);
        const Vec4 my = mot.at(ix, iy);
        const float zy = front_score(my.x, my.y, bgx, bgy, gp.to_velocity, gate, t);
        const float front = clampf((zy - zx) * 2.0f, 0.0f, 1.0f);
        const float back = clampf((zx - zy) * 2.0f, 0.0f, 1.0f);
        const float same = 1.0f - front - back;
        const float ly = std::fabs(my.x * ndx + my.y * ndy);
        const float excess = clampf((ly - lxn) / (ly + 1.0f), 0.0f, 1.0f);
        const float d = std::fabs(s) * ln;
        const float wgt = cover(d, ly) / (2.0f * ly + 1.0f) * da * (front + same * excess);
        if (wgt <= 0.0f) continue;
        const Pixel cy = sample_image(cur, px, py);
        fw += wgt;
        fr += wgt * cy.r;
        fg += wgt * cy.g;
        fb += wgt * cy.b;
        fa += wgt * cy.a;
    }

    // Own layer (and whatever is behind it) blurred along our own streak.
    const float wc = (nb > 0 ? 0.01f : 1.0f) / (2.0f * lx + 1.0f);  // the streak samples already cover this pixel
    float bw = wc, br = wc * c.r, bg = wc * c.g, bb = wc * c.b, ba = wc * c.a;
    if (nb > 0) {
        const float db = 2.0f * lx / float(nb);
        for (int i = 0; i < nb; ++i) {
            const float s = -1.0f + (2.0f * float(i) + 1.0f) / float(nb);
            const float px = float(x) + s * m.x, py = float(y) + s * m.y;
            const int ix = clampi(int(std::floor(px + 0.5f)), 0, w - 1),
                      iy = clampi(int(std::floor(py + 0.5f)), 0, h - 1);
            const Vec4 my = mot.at(ix, iy);
            const float zy = front_score(my.x, my.y, bgx, bgy, gp.to_velocity, gate, t);
            const float front = clampf((zy - zx) * 2.0f, 0.0f, 1.0f);
            const float d = std::fabs(s) * lx;
            const float wgt = cover(d, lx) / (2.0f * lx + 1.0f) * db * (1.0f - front);
            if (wgt <= 0.0f) continue;
            const Pixel cy = sample_image(cur, px, py);
            bw += wgt;
            br += wgt * cy.r;
            bg += wgt * cy.g;
            bb += wgt * cy.b;
            ba += wgt * cy.a;
        }
    }
    const float ib = 1.0f / bw;
    Pixel out{br * ib, bg * ib, bb * ib, ba * ib};
    if (fw > 0.0f) {
        const float alpha = clampf(fw, 0.0f, 1.0f), inf = alpha / fw, keep = 1.0f - alpha;
        out = {fr * inf + out.r * keep, fg * inf + out.g * keep, fb * inf + out.b * keep, fa * inf + out.a * keep};
    }
    return out;
}

// --- Frame -------------------------------------------------------------------------

// Number of tiles covering `size` pixels.
inline int tile_count(int size, int tile) { return (size + tile - 1) / tile; }

// Tiles a streak of the longest allowed length can cross.
inline int tile_reach(const BlurTuning& t) { return 1 + int(std::ceil(t.max_radius / float(t.tile))); }

// Intermediate buffers of one frame (exposed for tests and the benchmark).
struct BlurBuffers {
    Grid<Vec4> motion;  // pass 1
    Grid<Vec2> tile_max;
    Grid<Vec4> tile_occ;  // pass 2
    Grid<Vec2> neighbour_max;
    Grid<Vec4> neighbour_occ;  // pass 3
    GatherParams gather;
};

// Which inputs `render_motion_blur` uses, by Track and availability.
struct Inputs {
    const Image* prev = nullptr;
    const Image* next = nullptr;
    const FlowField* to_prev = nullptr;
    const FlowField* to_next = nullptr;
};

inline Inputs select_inputs(const Image& current, const Image* previous, const Image* next, const FlowField* to_prev,
                            const FlowField* to_next, Track track) {
    const int w = current.width, h = current.height;
    auto usable = [&](const Image* img, const FlowField* f) {
        return img && f && img->width == w && img->height == h && f->width == w && f->height == h;
    };
    Inputs in;
    if (track != Track::Next && usable(previous, to_prev)) {
        in.prev = previous;
        in.to_prev = to_prev;
    }
    if (track != Track::Previous && usable(next, to_next)) {
        in.next = next;
        in.to_next = to_next;
    }
    return in;
}

// Half of the shutter, in frames; the vector view uses 180° when the shutter is closed.
inline float half_shutter(const BlurSettings& s) {
    const float shutter = float(std::max(0.0, s.shutter));
    return shutter > 0.0f ? shutter * 0.5f : 0.25f;
}

// Passes 1–3.
inline void prepare_blur(const Image& current, const Inputs& in, const BlurSettings& s, const BlurTuning& t,
                         BlurBuffers& b) {
    const int w = current.width, h = current.height;
    const float half = half_shutter(s);
    b.motion = Grid<Vec4>(w, h);
    parallel_for(h, [&](int y) {
        for (int x = 0; x < w; ++x)
            b.motion.at(x, y) = motion_cell(current, in.prev, in.next, in.to_prev, in.to_next, half, t.max_radius, t, x, y);
    });
    for (int step : t.refine_steps) {
        if (step <= 0) continue;
        Grid<Vec4> refined(w, h);
        parallel_for(h, [&](int y) {
            for (int x = 0; x < w; ++x)
                refined.at(x, y) = refine_motion_cell(current, in.prev, in.next, b.motion, 1.0f / half, step, t, x, y);
        });
        b.motion = std::move(refined);
    }
    const int tw = tile_count(w, t.tile), th = tile_count(h, t.tile);
    b.tile_max = Grid<Vec2>(tw, th);
    b.tile_occ = Grid<Vec4>(tw, th);
    b.neighbour_max = Grid<Vec2>(tw, th);
    b.neighbour_occ = Grid<Vec4>(tw, th);
    const int reach = tile_reach(t);
    const float sigma_abs = t.layer_abs * half, sigma_rel = t.layer_rel;
    for (int round = 0; round <= t.robust_rounds; ++round) {
        const Grid<Vec4> prior = round > 0 ? b.neighbour_occ : Grid<Vec4>();
        parallel_for(th, [&](int ty) {
            for (int tx = 0; tx < tw; ++tx)
                tile_cell(b.motion, round > 0 ? &prior : nullptr, t.tile, sigma_abs, sigma_rel, tx, ty,
                          b.tile_max.at(tx, ty), b.tile_occ.at(tx, ty));
        });
        parallel_for(th, [&](int ty) {
            for (int tx = 0; tx < tw; ++tx)
                neighbour_cell(b.tile_max, b.tile_occ, t.tile, reach, t.bg_tiles, t.bg_scale, tx, ty, b.neighbour_max.at(tx, ty),
                               b.neighbour_occ.at(tx, ty));
        });
    }
    b.gather.tile = t.tile;
    b.gather.max_samples = s.max_samples;
    b.gather.to_velocity = 1.0f / half;
    b.gather.show_vectors = s.show_vectors;
}

// `current`: the frame to blur. `previous`/`next` and the flows from `current`
// to them may be null (first/last frame, or not tracked); `track` picks which
// ones measure the motion. `out` is resized to the size of `current`.
inline void render_motion_blur(const Image& current, const Image* previous, const Image* next,
                               const FlowField* to_prev, const FlowField* to_next, Image& out,
                               const BlurSettings& s, const BlurTuning& t = BlurTuning{}) {
    const int w = current.width, h = current.height;
    out = Image(w, h);
    if (w <= 0 || h <= 0) return;
    const Inputs in = select_inputs(current, previous, next, to_prev, to_next, s.track);
    if ((!in.prev && !in.next) || (s.shutter <= 0.0 && !s.show_vectors)) {
        out = current;
        return;
    }
    BlurBuffers b;
    prepare_blur(current, in, s, t, b);
    parallel_for(h, [&](int y) {
        for (int x = 0; x < w; ++x)
            out.at(x, y) = gather_cell(current, b.motion, b.neighbour_max, b.neighbour_occ, b.gather, t, x, y);
    });
}

}  // namespace smear
