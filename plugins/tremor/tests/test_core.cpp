// Tests for the motion model (tremor_core.h).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#include "tremor_core.h"

static int failures = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        if (!(cond)) {                                                  \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++failures;                                                 \
        }                                                               \
    } while (0)

static bool near(double a, double b, double eps = 1e-9) { return std::fabs(a - b) <= eps; }

// Every smooth motion off and no jumps: a still frame.
static tremor::Params still() {
    tremor::Params p;
    p.h_max = p.v_max = 0;
    p.h_erratic = p.v_erratic = p.zoom_erratic = 0;
    p.zoom_base = p.zoom_factor = 0;
    p.rot_max = p.ch_sep = 0;
    return p;
}

static void test_random_sequence() {
    // Known values of the Microsoft C runtime rand() after srand(1).
    tremor::LegacyRandom r(1);
    CHECK(r.next() == 41);
    CHECK(r.next() == 18467);
    CHECK(r.next() == 6334);
#ifdef _MSC_VER
    for (unsigned seed : {0u, 7u, 12345u, 4000000000u}) {
        std::srand(seed);
        tremor::LegacyRandom t(seed);
        for (int i = 0; i < 100; ++i) CHECK(t.next() == std::rand());
    }
#endif
}

static void test_still() {
    const tremor::FrameState s = tremor::evaluate(still(), 1234, 30, 1920, 1080);
    CHECK(s.zoom == 0 && s.rotation_deg == 0 && s.separation_deg == 0);
    CHECK(s.shift_x == 0 && s.shift_y == 0);
    CHECK(s.center_x == 959.5 && s.center_y == 539.5);
}

static void test_waves() {
    // At time 0 every wave is cos(phase), with the phase in degrees.
    tremor::Params p = still();
    p.h_max = 16, p.h_amp = 0.1, p.h_phase = 30;
    p.v_max = 4, p.v_amp = 0.1, p.v_phase = 30;
    p.zoom_base = 0.1, p.zoom_factor = 0.1, p.zoom_amp = 0.1, p.zoom_phase = 90;
    p.rot_max = 2, p.rot_amp = 1, p.rot_phase = 15;
    p.ch_sep = 4, p.ch_amp = 0.4, p.ch_phase = 30;
    const double d = tremor::kPi / 180.0;
    const tremor::FrameState s = tremor::evaluate(p, 0, 30, 1000, 500);
    CHECK(near(s.shift_x, std::cos(30 * d) * 0.1 * 16 * 0.01 * 1000));
    CHECK(near(s.shift_y, std::cos(30 * d) * 0.1 * 4 * 0.01 * 500));
    CHECK(near(s.zoom, 0.1));  // cos(90 degrees) = 0
    CHECK(near(s.rotation_deg, std::cos(15 * d) * 2));
    CHECK(near(s.separation_deg, std::cos(30 * d) * 1.6));

    // One second later the horizontal wave has turned Freq. times.
    p.h_freq = 2;
    const tremor::FrameState t = tremor::evaluate(p, 30, 30, 1000, 500);
    CHECK(near(t.shift_x, std::cos(2 * 2 * tremor::kPi + 30 * d) * 0.1 * 16 * 0.01 * 1000, 1e-6));
}

static void test_jumps() {
    // 100% erratic: every frame jumps, and the same seed and time give the same jump.
    tremor::Params p = still();
    p.h_erratic = p.v_erratic = 100;
    int moved = 0;
    for (int f = 0; f < 50; ++f) {
        const tremor::FrameState a = tremor::evaluate(p, f * 1001, 30000, 640, 480);
        const tremor::FrameState b = tremor::evaluate(p, f * 1001, 30000, 640, 480);
        CHECK(a.shift_x == b.shift_x && a.shift_y == b.shift_y);
        CHECK(std::fabs(a.shift_x) <= 640 && std::fabs(a.shift_y) <= 480);
        if (a.shift_x != 0 || a.shift_y != 0) ++moved;
    }
    CHECK(moved > 40);

    // "Erratic(H) Adj." scales the horizontal jump only.
    p.erratic_h_adj = 0;
    const tremor::FrameState s = tremor::evaluate(p, 77, 30, 640, 480);
    CHECK(s.shift_x == 0);
    CHECK(s.shift_y != 0);

    // A different seed changes the pattern.
    tremor::Params q = still();
    q.h_erratic = 50;
    int differ = 0;
    for (int f = 0; f < 50; ++f) {
        q.seed = 0;
        const double a = tremor::evaluate(q, f, 30, 640, 480).shift_x;
        q.seed = 9;
        const double b = tremor::evaluate(q, f, 30, 640, 480).shift_x;
        if (a != b) ++differ;
    }
    CHECK(differ > 10);
}

int main() {
    test_random_sequence();
    test_still();
    test_waves();
    test_jumps();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("core: all tests passed\n");
    return 0;
}
