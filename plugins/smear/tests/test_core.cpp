// Tests for the optical flow (smear_core.h).
#include <atomic>
#include <thread>
#include <vector>

#include "test_util.h"

namespace {

// Mean end-point error over pixels at least `margin` from the border.
double interior_epe(const smear::FlowField& f, float tx, float ty, int margin) {
    double s = 0;
    int n = 0;
    for (int y = margin; y < f.height - margin; ++y)
        for (int x = margin; x < f.width - margin; ++x) {
            const smear::Vec2 v = f.at(x, y);
            s += std::sqrt(double(v.x - tx) * (v.x - tx) + double(v.y - ty) * (v.y - ty));
            ++n;
        }
    return s / n;
}

void test_parallel_for() {
    for (int count : {0, 1, 3, 17, 1000}) {
        std::vector<std::atomic<int>> hits(size_t(count) + 1);
        smear::parallel_for(count, [&](int i) { hits[size_t(i)]++; });
        bool once = true;
        for (int i = 0; i < count; ++i) once = once && hits[size_t(i)] == 1;
        CHECK(once);
        CHECK(hits[size_t(count)] == 0);
    }
}

void test_plan() {
    smear::FlowSettings fs;
    const smear::FlowPlan normal = smear::plan_flow(fs, 1920, 1080);
    CHECK(normal.finest == 1);
    CHECK(normal.coarsest >= 4);
    CHECK((1080 >> normal.coarsest) >= 2 * normal.patch);
    fs.sensitivity = 0;
    const smear::FlowPlan steady = smear::plan_flow(fs, 1920, 1080);
    fs.sensitivity = 100;
    const smear::FlowPlan detailed = smear::plan_flow(fs, 1920, 1080);
    CHECK(steady.patch > normal.patch && normal.patch > detailed.patch);
    fs.quality = smear::Quality::Draft;
    CHECK(smear::plan_flow(fs, 1920, 1080).finest == 2);
    fs.quality = smear::Quality::Best;
    CHECK(smear::plan_flow(fs, 1920, 1080).finest == 0);
    // Tiny frames: one level, still valid.
    const smear::FlowPlan tiny = smear::plan_flow(fs, 8, 6);
    CHECK(tiny.coarsest == 0 && tiny.finest == 0);
    CHECK(smear::level_size(1080, 1) == 540 && smear::level_size(1081, 1) == 541 && smear::level_size(5, 9) == 1);
}

void test_static() {
    test::Scene s;
    const smear::Image a = s.frame(0);
    smear::FlowField f;
    smear::compute_flow(a, a, f, smear::FlowSettings{});
    CHECK(f.width == a.width && f.height == a.height);
    CHECK_LT(interior_epe(f, 0, 0, 0), 1e-3);
}

void test_translation(smear::Quality q, float vx, float vy) {
    test::Scene s;
    s.w = 192;
    s.h = 144;
    s.bvx = vx;
    s.bvy = vy;
    smear::FlowSettings fs;
    fs.quality = q;
    smear::FlowField f;
    smear::compute_flow(s.frame(0), s.frame(1), f, fs);
    const double e = interior_epe(f, vx, vy, int(std::ceil(std::fabs(vx) + std::fabs(vy))) + 4);
    std::printf("translation (%5.2f, %5.2f), quality %d: EPE %.3f px\n", vx, vy, int(q), e);
    CHECK_LT(e, 0.15);
}

void test_two_motions() {
    // A square moving over a differently moving background: each side gets its own motion.
    test::Scene s;
    s.w = 128;
    s.h = 96;
    s.bvx = -3;
    s.square = true;
    s.cx = 60;
    s.cy = 48;
    s.half = 18;
    s.fvx = 9;
    s.fvy = 2;
    smear::FlowField f;
    smear::compute_flow(s.frame(0), s.frame(1), f, smear::FlowSettings{});
    const smear::Vec2 in = f.at(60, 48), out = f.at(15, 15);
    CHECK_LT(std::fabs(in.x - 9) + std::fabs(in.y - 2), 0.3);
    CHECK_LT(std::fabs(out.x + 3) + std::fabs(out.y), 0.3);
    // Boundaries stay sharp: pixels 3 px inside/outside the square's left edge.
    const smear::Vec2 l_in = f.at(60 - 18 + 3, 48), l_out = f.at(60 - 18 - 3, 48);
    CHECK_LT(std::fabs(l_in.x - 9), 1.0);
    CHECK_LT(std::fabs(l_out.x + 3), 1.0);
}

void test_concurrent() {
    // Stateless: concurrent renders give identical results.
    test::Scene s;
    s.bvx = 2.5f;
    const smear::Image a = s.frame(0), b = s.frame(1);
    smear::FlowField ref;
    smear::compute_flow(a, b, ref, smear::FlowSettings{});
    std::vector<smear::FlowField> out(4);
    std::vector<std::thread> threads;
    for (auto& o : out) threads.emplace_back([&] { smear::compute_flow(a, b, o, smear::FlowSettings{}); });
    for (auto& t : threads) t.join();
    for (const auto& o : out) CHECK(test::mean_flow_diff(o, ref) == 0.0);
}

void test_mismatched_sizes() {
    smear::FlowField f;
    smear::compute_flow(smear::Image(10, 10), smear::Image(12, 10), f, smear::FlowSettings{});
    CHECK(f.width == 10 && f.height == 10);
    CHECK(f.at(5, 5).x == 0.0f && f.at(5, 5).y == 0.0f);
}

}  // namespace

int main() {
    test_parallel_for();
    test_plan();
    test_static();
    for (smear::Quality q : {smear::Quality::Draft, smear::Quality::Normal, smear::Quality::Best}) {
        test_translation(q, 3.0f, 0.0f);
        test_translation(q, -1.25f, 2.5f);
        test_translation(q, 14.0f, -6.0f);
    }
    test_two_motions();
    test_concurrent();
    test_mismatched_sizes();
    if (failures) {
        std::printf("%d check(s) failed\n", failures);
        return 1;
    }
    std::printf("All core tests passed\n");
    return 0;
}
