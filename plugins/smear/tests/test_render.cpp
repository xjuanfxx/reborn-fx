// Tests for the motion blur renderer (smear_render.h), fed with exact flows so
// they test the reconstruction itself.
#include "test_util.h"

namespace {

struct Frames {
    smear::Image prev, cur, next;
    smear::FlowField to_prev, to_next;
};

Frames frames(const test::Scene& s) {
    return {s.frame(-1), s.frame(0), s.frame(1), s.flow(-1), s.flow(1)};
}

smear::Image blur(const Frames& f, const smear::BlurSettings& bs) {
    smear::Image out;
    smear::render_motion_blur(f.cur, &f.prev, &f.next, &f.to_prev, &f.to_next, out, bs);
    return out;
}

void test_still() {
    test::Scene s;
    s.square = true;
    const Frames f = frames(s);
    const smear::Image out = blur(f, smear::BlurSettings{});
    CHECK(out.width == f.cur.width && out.height == f.cur.height);
    CHECK(test::max_abs_diff(out, f.cur) == 0.0);
}

void test_closed_shutter_and_missing_frames() {
    test::Scene s;
    s.bvx = 6;
    const Frames f = frames(s);
    smear::BlurSettings bs;
    bs.shutter = 0;
    CHECK(test::max_abs_diff(blur(f, bs), f.cur) == 0.0);
    smear::Image out;
    smear::render_motion_blur(f.cur, nullptr, nullptr, nullptr, nullptr, out, smear::BlurSettings{});
    CHECK(test::max_abs_diff(out, f.cur) == 0.0);
    // Track Next without a next frame: nothing to measure motion with.
    bs.shutter = 0.5;
    bs.track = smear::Track::Next;
    smear::render_motion_blur(f.cur, &f.prev, nullptr, &f.to_prev, nullptr, out, bs);
    CHECK(test::max_abs_diff(out, f.cur) == 0.0);
}

// Uniform motion: the result is the box average of the frame along the streak.
void test_uniform_motion() {
    test::Scene s;
    s.bvx = 8;
    const Frames f = frames(s);
    smear::BlurSettings bs;  // 180°: streak = 4 px, ±2 px
    bs.max_samples = 64;
    const smear::Image out = blur(f, bs);
    double err = 0;
    int n = 0;
    for (int y = 10; y < s.h - 10; ++y)
        for (int x = 10; x < s.w - 10; ++x) {
            // Exact streak average from the analytic texture (fine sampling).
            smear::Pixel ref{};
            const int k = 64;
            for (int i = 0; i < k; ++i) {
                const smear::Pixel p = s.colour(float(x) + (-1.0f + (2.0f * float(i) + 1.0f) / float(k)) * 2.0f, float(y), 0);
                ref.g += p.g / float(k);
            }
            err += std::fabs(out.at(x, y).g - ref.g);
            ++n;
        }
    std::printf("uniform motion: mean error vs streak average %.4f\n", err / n);
    CHECK_LT(err / n, 0.01);
    // Every track mode agrees for constant velocity.
    for (smear::Track t : {smear::Track::Next, smear::Track::Previous}) {
        smear::BlurSettings b2 = bs;
        b2.track = t;
        CHECK_LT(test::mean_abs_diff(blur(f, b2), out), 1e-4);
    }
}

// Still square over a fast background (camera following a subject): the
// background must not smear over the square, and the square stays sharp.
void test_background_never_covers_foreground() {
    test::Scene s;
    s.w = 128;
    s.h = 96;
    s.bvx = 20;
    s.square = true;
    s.cx = 64;
    s.cy = 48;
    s.half = 20;
    const Frames f = frames(s);
    smear::BlurSettings bs;
    bs.shutter = 1.0;  // ±10 px streaks
    const smear::Image out = blur(f, bs);
    double inside = 0;
    for (int y = 48 - 18; y <= 48 + 18; ++y)
        for (int x = 64 - 18; x <= 64 + 18; ++x) {
            const smear::Pixel a = out.at(x, y), b = f.cur.at(x, y);
            inside = std::fmax(inside, std::fabs(a.r - b.r) + std::fabs(a.g - b.g) + std::fabs(a.b - b.b));
        }
    std::printf("still square over fast background: max change inside the square %.4f\n", inside);
    CHECK_LT(inside, 0.02);
    // The background did blur.
    CHECK(std::fabs(out.at(10, 10).g - f.cur.at(10, 10).g) + std::fabs(out.at(20, 80).r - f.cur.at(20, 80).r) > 0.01);
}

// Moving square over a still background: its blur spills over the background,
// but the background next to it does not take the square's motion.
void test_foreground_spills_over_background() {
    test::Scene s;
    s.w = 128;
    s.h = 96;
    s.square = true;
    s.cx = 64;
    s.cy = 48;
    s.half = 16;
    s.fvx = 16;
    const Frames f = frames(s);
    smear::BlurSettings bs;
    bs.shutter = 1.0;  // ±8 px
    const smear::Image out = blur(f, bs);
    // Just right of the square (inside its streak): mixed with the square.
    double spill = 0;
    for (int y = 36; y <= 60; ++y)
        for (int x = 64 + 16 + 1; x <= 64 + 16 + 6; ++x) spill += std::fabs(out.at(x, y).g - f.cur.at(x, y).g) / (25.0 * 6.0);
    std::printf("moving square: mean change just past its edge %.4f\n", spill);
    CHECK(spill > 0.01);
    // Far from it: untouched; above it (motion is horizontal): untouched.
    double far = 0;
    for (int y = 0; y < 96; ++y)
        for (int x = 0; x < 30; ++x) far = std::fmax(far, std::fabs(out.at(x, y).g - f.cur.at(x, y).g));
    for (int x = 40; x < 90; ++x)
        for (int y = 0; y < 48 - 16 - 3; ++y) far = std::fmax(far, std::fabs(out.at(x, y).g - f.cur.at(x, y).g));
    std::printf("moving square: max change on still background away from its streaks %.5f\n", far);
    CHECK_LT(far, 1e-4);
}

void test_transparency() {
    // A transparent frame with an opaque moving square: blur keeps premultiplied consistency.
    test::Scene s;
    s.square = true;
    s.fvx = 10;
    Frames f = frames(s);
    for (smear::Image* img : {&f.prev, &f.cur, &f.next})
        for (int y = 0; y < s.h; ++y)
            for (int x = 0; x < s.w; ++x) {
                const float t = img == &f.prev ? -1.0f : (img == &f.cur ? 0.0f : 1.0f);
                if (!s.in_square(float(x), float(y), t)) img->at(x, y) = {};
            }
    const smear::Image out = blur(f, smear::BlurSettings{});
    bool premultiplied = true;
    for (const smear::Pixel& p : out.px)
        premultiplied = premultiplied && p.a >= -1e-6f && p.a <= 1.0f + 1e-6f && p.r <= p.a + 1e-5f &&
                        p.g <= p.a + 1e-5f && p.b <= p.a + 1e-5f;
    CHECK(premultiplied);
    CHECK(out.at(2, 2).a == 0.0f);
    // The edge along the motion became partially transparent.
    const float a_edge = out.at(int(s.cx + s.half), int(s.cy)).a;
    CHECK(a_edge > 0.2f && a_edge < 0.9f);
}

void test_vectors_view() {
    test::Scene s;
    s.bvx = 6;
    const Frames f = frames(s);
    smear::BlurSettings bs;
    bs.show_vectors = true;
    const smear::Image out = blur(f, bs);
    const smear::Pixel p = out.at(40, 30);
    CHECK(p.a == 1.0f);
    CHECK(p.b > p.r && p.g > p.r);  // rightward motion: cyan
    // Still: neutral grey.
    test::Scene still;
    const smear::Image o2 = blur(frames(still), bs);
    CHECK(std::fabs(o2.at(40, 30).r - 0.5f) < 1e-5f && std::fabs(o2.at(40, 30).g - 0.5f) < 1e-5f);
}

}  // namespace

int main() {
    test_still();
    test_closed_shutter_and_missing_frames();
    test_uniform_motion();
    test_background_never_covers_foreground();
    test_foreground_spills_over_background();
    test_transparency();
    test_vectors_view();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("All render tests passed\n");
    return 0;
}
