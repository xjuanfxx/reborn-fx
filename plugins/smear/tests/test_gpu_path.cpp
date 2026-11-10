// Runs the GPU passes (smear_kernels.inc orchestrated by SmearGpuRender.h) on
// the CPU through tests/cpu_prelude.h and compares them with the CPU renderer
// (smear_core.h / smear_render.h). This validates the kernels and the pass
// sequence without a GPU.
#include <set>
#include <string>

#include "cpu_prelude.h"
#include "test_util.h"

using smear_cpu::CpuBackend;
using smear_cpu::CpuScratch;
using smear_cpu::GpuFlow;
using smear_cpu::GpuImage;

namespace {

test::Scene moving_square() {
    test::Scene s;
    s.w = 112;
    s.h = 80;
    s.bvx = -2.5f;
    s.bvy = 1.0f;
    s.square = true;
    s.cx = 50;
    s.cy = 40;
    s.half = 15;
    s.fvx = 7.0f;
    s.fvy = -1.5f;
    return s;
}

void test_flow_matches(smear::Quality q, double sensitivity) {
    const test::Scene s = moving_square();
    const smear::Image a = s.frame(0), b = s.frame(1);
    smear::FlowSettings fs;
    fs.quality = q;
    fs.sensitivity = sensitivity;
    smear::FlowField cpu;
    smear::compute_flow(a, b, cpu, fs);

    GpuImage ga(a), gb(b);
    GpuFlow out(a.width, a.height);
    CpuBackend gpu;
    CpuScratch scratch;
    CHECK(smear::gpu::compute_flow(gpu, scratch, ga.buf, gb.buf, fs, out.buf));
    CHECK(gpu.missing.empty());
    const double d = test::mean_flow_diff(cpu, out.field());
    std::printf("flow  quality %d sensitivity %3.0f: mean |gpu - cpu| = %.2e px (%d passes)\n", int(q), sensitivity, d,
                gpu.passes);
    CHECK_LT(d, 1e-3);
}

struct Case {
    const char* name;
    smear::BlurSettings bs;
    bool drop_prev = false, drop_next = false;
    int crop = 0;  // output = current minus this margin on every side
};

void test_render_matches(const Case& c, smear::Quality q) {
    const test::Scene s = moving_square();
    const smear::Image prev = s.frame(-1), cur = s.frame(0), next = s.frame(1);
    smear::FlowSettings fs;
    fs.quality = q;

    // CPU reference: flows then blur.
    smear::FlowField fp, fn;
    smear::compute_flow(cur, prev, fp, fs);
    smear::compute_flow(cur, next, fn, fs);
    smear::Image ref;
    smear::render_motion_blur(cur, c.drop_prev ? nullptr : &prev, c.drop_next ? nullptr : &next, &fp, &fn, ref, c.bs);

    // GPU path on the CPU.
    GpuImage gp(prev), gc(cur), gn(next);
    GpuImage out(cur.width - 2 * c.crop, cur.height - 2 * c.crop);
    smear::gpu::FrameInputs in;
    in.current = gc.buf;
    if (!c.drop_prev) in.previous = gp.buf;
    if (!c.drop_next) in.next = gn.buf;
    in.out_ox = c.crop;
    in.out_oy = c.crop;
    CpuBackend gpu;
    CpuScratch scratch;
    CHECK(smear::gpu::render(gpu, scratch, in, c.bs, fs, out.buf));
    CHECK(gpu.missing.empty());

    smear::Image crop(out.buf.width, out.buf.height);
    for (int y = 0; y < crop.height; ++y)
        for (int x = 0; x < crop.width; ++x) crop.at(x, y) = ref.at(x + c.crop, y + c.crop);
    const smear::Image got = out.image();
    const double d = test::mean_abs_diff(crop, got), m = test::max_abs_diff(crop, got);
    const double changed = test::mean_abs_diff(cur, ref);  // how much the blur did, for scale
    std::printf("render %-14s quality %d: mean |gpu - cpu| = %.2e (max %.2e; blur changed %.2e), %d passes\n", c.name,
                int(q), d, m, changed, gpu.passes);
    CHECK_LT(d, 1e-3);
}

void test_all_kernels_exist() {
    const char* names[] = {"smear_flow_gray", "smear_flow_downsample", "smear_flow_patches", "smear_flow_densify",
                           "smear_flow_upsample", "smear_flow_refine", "smear_motion", "smear_motion_refine",
                           "smear_tiles", "smear_neighbours", "smear_gather", "smear_copy"};
    for (const char* n : names) CHECK(smear_cpu::registry().count(n) == 1);
    CHECK(smear_cpu::registry().size() == sizeof(names) / sizeof(names[0]));
    CpuBackend gpu;
    int dummy = 0;
    CHECK(!gpu.run("no_such_kernel", 1, 1, {&dummy}, &dummy, sizeof(dummy)));
}

}  // namespace

int main() {
    test_all_kernels_exist();
    test_flow_matches(smear::Quality::Draft, 50);
    test_flow_matches(smear::Quality::Normal, 50);
    test_flow_matches(smear::Quality::Best, 50);
    test_flow_matches(smear::Quality::Normal, 0);
    test_flow_matches(smear::Quality::Normal, 100);

    smear::BlurSettings both;
    smear::BlurSettings wide = both;
    wide.shutter = 1.0;
    wide.max_samples = 16;
    smear::BlurSettings next_only = both;
    next_only.track = smear::Track::Next;
    smear::BlurSettings prev_only = both;
    prev_only.track = smear::Track::Previous;
    smear::BlurSettings vectors = both;
    vectors.show_vectors = true;
    smear::BlurSettings closed = both;
    closed.shutter = 0.0;

    const Case cases[] = {
        {"both", both},
        {"shutter 360", wide},
        {"next only", next_only},
        {"previous only", prev_only},
        {"vectors", vectors},
        {"closed shutter", closed},
        {"no previous", both, true, false},
        {"cropped", both, false, false, 9},
    };
    for (const Case& c : cases) test_render_matches(c, smear::Quality::Normal);
    test_render_matches(cases[0], smear::Quality::Draft);
    test_render_matches(cases[0], smear::Quality::Best);

    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("GPU path matches the CPU renderer\n");
    return 0;
}
