// Tests for the CPU renderer (tremor_render.h).
#include <cmath>
#include <cstdio>

#include "tremor_render.h"

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

static bool near(float a, float b, float eps = 1e-4f) { return std::fabs(a - b) <= eps; }
static bool same(const tremor::Pixel& a, const tremor::Pixel& b, float eps = 1e-4f) {
    return near(a.r, b.r, eps) && near(a.g, b.g, eps) && near(a.b, b.b, eps) && near(a.a, b.a, eps);
}

static tremor::Image gradient(int w, int h) {
    tremor::Image img(w, h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) img.at(x, y) = {x / float(w), y / float(h), 0.25f, 1.0f};
    return img;
}

static tremor::FrameState still(int w, int h) {
    tremor::FrameState st;
    st.center_x = (w - 1) * 0.5;
    st.center_y = (h - 1) * 0.5;
    return st;
}

static void test_still_frame() {
    // No motion: the output is the source, edges included.
    const tremor::Image src = gradient(32, 24);
    tremor::Image out(32, 24);
    tremor::render_frame(src, out, 0, 0, still(32, 24), tremor::Look{});
    CHECK(same(out.at(10, 10), src.at(10, 10)));
    CHECK(same(out.at(31, 23), src.at(31, 23)));
    CHECK(same(out.at(0, 0), src.at(0, 0)));
}

static void test_blend() {
    // Blend cross-fades between the moved picture and the source.
    const tremor::Image src = gradient(40, 30);
    tremor::Image out(40, 30);
    tremor::FrameState st = still(40, 30);
    st.shift_x = 100;  // moved out entirely: transparent
    tremor::Look l;
    l.blend = 0.25f;
    tremor::render_frame(src, out, 0, 0, st, l);
    const tremor::Pixel s = src.at(10, 10);
    CHECK(same(out.at(10, 10), {s.r * 0.25f, s.g * 0.25f, s.b * 0.25f, 0.25f}));
    l.blend = 1.0f;
    tremor::render_frame(src, out, 0, 0, st, l);
    CHECK(same(out.at(10, 10), s));
}

static void test_shift_and_edges() {
    const tremor::Image src = gradient(40, 30);
    tremor::Image out(40, 30);
    tremor::FrameState st = still(40, 30);
    st.shift_x = 3, st.shift_y = 2;  // the picture moves right and down
    tremor::Look l;
    tremor::render_frame(src, out, 0, 0, st, l);
    CHECK(same(out.at(13, 12), src.at(10, 10)));
    CHECK(same(out.at(1, 1), {}));  // Transparent

    l.edges = tremor::Edges::Black;
    tremor::render_frame(src, out, 0, 0, st, l);
    CHECK(same(out.at(1, 1), {0, 0, 0, 1}));

    l.edges = tremor::Edges::Mirror;
    tremor::render_frame(src, out, 0, 0, st, l);
    CHECK(same(out.at(1, 0), src.at(1, 1)));  // (-2, -2) mirrors to (1, 1)

    l.edges = tremor::Edges::Tile;
    tremor::render_frame(src, out, 0, 0, st, l);
    CHECK(same(out.at(1, 0), src.at(38, 28)));
}

static void test_composite_and_adjust() {
    const tremor::Image src = gradient(40, 30);
    tremor::Image out(40, 30);
    tremor::FrameState st = still(40, 30);
    st.shift_x = 5;
    tremor::Look l;
    l.alpha_composite = true;
    tremor::render_frame(src, out, 0, 0, st, l);
    CHECK(same(out.at(2, 10), src.at(2, 10)));    // revealed edge shows the source
    CHECK(same(out.at(20, 10), src.at(15, 10)));  // moved picture on top

    // Color offsets apply without the composite, and not to transparent areas.
    l.alpha_composite = false;
    l.adjust[0] = 0.1f;
    tremor::render_frame(src, out, 0, 0, st, l);
    CHECK(near(out.at(20, 10).r, src.at(15, 10).r + 0.1f));
    CHECK(same(out.at(2, 10), {}));
}

static void test_zoom_and_rotation() {
    const tremor::Image src = gradient(41, 41);
    tremor::Image out(41, 41);
    tremor::Look l;

    // Zoom 1 (2x magnification): the center stays, (30,20) shows (25,20).
    tremor::FrameState z = still(41, 41);
    z.zoom = 1.0;
    tremor::render_frame(src, out, 0, 0, z, l);
    CHECK(same(out.at(20, 20), src.at(20, 20)));
    CHECK(same(out.at(30, 20), src.at(25, 20)));

    // 90 degrees: (30,20) shows (20,30); the center stays.
    tremor::FrameState r = still(41, 41);
    r.rotation_deg = 90;
    tremor::render_frame(src, out, 0, 0, r, l);
    CHECK(same(out.at(20, 20), src.at(20, 20)));
    CHECK(same(out.at(30, 20), src.at(20, 30)));

    // Channel separation: red stays, green comes from elsewhere off-center.
    r.rotation_deg = 0;
    r.separation_deg = 20;
    tremor::render_frame(src, out, 0, 0, r, l);
    CHECK(near(out.at(30, 20).r, src.at(30, 20).r));
    CHECK(!near(out.at(30, 20).g, src.at(30, 20).g, 1e-2f));
}

static void test_deinterlace() {
    tremor::Image stripes(2, 6);
    for (int y = 0; y < 6; ++y)
        for (int x = 0; x < 2; ++x) stripes.at(x, y) = {float(y % 2), 0, 0, 1};
    tremor::Image out(2, 6);
    tremor::deinterlace(stripes, out, tremor::Deinterlace::UpperOnly);  // rebuilds odd rows
    CHECK(near(out.at(0, 1).r, 0.0f) && near(out.at(0, 3).r, 0.0f));
    tremor::deinterlace(stripes, out, tremor::Deinterlace::LowerOnly);  // rebuilds even rows
    CHECK(near(out.at(0, 2).r, 1.0f) && near(out.at(0, 1).r, 1.0f));
    tremor::deinterlace(stripes, out, tremor::Deinterlace::Merge);
    CHECK(near(out.at(0, 1).r, 0.5f));  // .25*0 + .5*1 + .25*0
    CHECK(near(out.at(0, 2).r, 0.5f));  // .25*1 + .5*0 + .25*1
    CHECK(near(out.at(0, 1).a, 1.0f));
}

int main() {
    test_still_frame();
    test_blend();
    test_shift_and_edges();
    test_composite_and_adjust();
    test_zoom_and_rotation();
    test_deinterlace();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("render: all tests passed\n");
    return 0;
}
