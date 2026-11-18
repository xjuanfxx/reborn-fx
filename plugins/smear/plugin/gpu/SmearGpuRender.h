// SmearGpuRender.h — turns a frame into GPU passes.
//
// Independent of After Effects and of any GPU API: a Backend enqueues the
// kernels of smear_kernels.inc by name, a Scratch hands out temporary buffers.
// The plugin implements both for CUDA, OpenCL and Metal; the tests implement
// them on the CPU (tests/cpu_prelude.h) and compare against the CPU renderer.
// Mirrors smear::compute_flow (smear_core.h) and smear::render_motion_blur
// (smear_render.h) pass for pass.
//
// Requirements on the backend: run() calls execute in the order they are made
// (one in-order queue/stream), so a buffer written by one pass can be read by
// the next and reused afterwards. Scratch buffers live until the Scratch is
// destroyed (after the queue has finished with them).
#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <vector>

#include "smear_gpu_types.h"
#include "smear_render.h"

namespace smear::gpu {
struct Buffer { void* mem; int width, height, pitch; };            // pitch in elements
class Scratch { public: virtual ~Scratch()=default; virtual bool alloc(int w,int h,int channels,Buffer& out)=0; }; // channels 2 (flow) or 4 (rgba), float
class Backend { public: virtual ~Backend()=default;
  // Enqueue kernel `name` over a w×h grid with these buffers (in order) and one param block.
  virtual bool run(const char* name, int w, int h, std::initializer_list<void*> buffers, const void* params, size_t params_size)=0; };
}

namespace smear::gpu {

// The frames of one render, float4 (BGRA, premultiplied), all the same size.
// `previous`/`next`: mem == nullptr when unavailable (first/last frame).
struct FrameInputs {
    Buffer current{nullptr, 0, 0, 0};
    Buffer previous{nullptr, 0, 0, 0};
    Buffer next{nullptr, 0, 0, 0};
    int out_ox = 0, out_oy = 0;  // output pixel (0, 0) is pixel (out_ox, out_oy) of `current`
};

template <class P>
bool run(Backend& gpu, const char* name, int w, int h, std::initializer_list<void*> buffers, const P& params) {
    return gpu.run(name, w, h, buffers, &params, sizeof(P));
}

// Buffers of the flow passes, allocated once and reused by both flows of a frame.
struct FlowWorkspace {
    FlowPlan plan;
    std::vector<Buffer> levels;  // float4 pyramid
    std::vector<Buffer> flows;   // float2 dense flow per searched level
    Buffer patches{nullptr, 0, 0, 0};
    Buffer temp[2] = {{nullptr, 0, 0, 0}, {nullptr, 0, 0, 0}};  // full resolution float2

    bool init(Scratch& scratch, const FlowSettings& fs, int w, int h) {
        plan = plan_flow(fs, w, h);
        levels.assign(size_t(plan.coarsest) + 1, Buffer{nullptr, 0, 0, 0});
        flows.assign(size_t(plan.coarsest) + 1, Buffer{nullptr, 0, 0, 0});
        int max_nx = 1, max_ny = 1;
        for (int l = 0; l <= plan.coarsest; ++l) {
            const int lw = level_size(w, l), lh = level_size(h, l);
            if (!scratch.alloc(lw, lh, 4, levels[size_t(l)])) return false;
            if (l >= plan.finest) {
                if (!scratch.alloc(lw, lh, 2, flows[size_t(l)])) return false;
                const PatchGrid g = patch_grid(plan, lw, lh);
                max_nx = std::max(max_nx, g.nx);
                max_ny = std::max(max_ny, g.ny);
            }
        }
        if (!scratch.alloc(max_nx, max_ny, 2, patches)) return false;
        return scratch.alloc(w, h, 2, temp[0]) && scratch.alloc(w, h, 2, temp[1]);
    }
};

// Dense flow from `from` to `to` into `out` (float2, same size). `ws` must have
// been initialised for these settings and this size.
inline bool compute_flow(Backend& gpu, FlowWorkspace& ws, const Buffer& from, const Buffer& to, const Buffer& out) {
    const FlowPlan& plan = ws.plan;
    const int w = from.width, h = from.height;

    SmearGrayParams gp{};
    gp.w = w;
    gp.h = h;
    gp.from_pitch = from.pitch;
    gp.to_pitch = to.pitch;
    gp.dst_pitch = ws.levels[0].pitch;
    if (!run(gpu, "smear_flow_gray", w, h, {from.mem, to.mem, ws.levels[0].mem}, gp)) return false;
    for (int l = 1; l <= plan.coarsest; ++l) {
        const Buffer &src = ws.levels[size_t(l) - 1], &dst = ws.levels[size_t(l)];
        SmearDownParams dp{};
        dp.sw = src.width;
        dp.sh = src.height;
        dp.src_pitch = src.pitch;
        dp.w = dst.width;
        dp.h = dst.height;
        dp.dst_pitch = dst.pitch;
        if (!run(gpu, "smear_flow_downsample", dst.width, dst.height, {src.mem, dst.mem}, dp)) return false;
    }

    // Full-resolution passes after the search: optional upsampling, then the refinements.
    std::vector<int> final_passes;  // 0 = upsample, >0 = refine distance
    if (plan.finest > 0) final_passes.push_back(0);
    for (int s : refine_steps(plan))
        if (s > 0) final_passes.push_back(s);

    for (int l = plan.coarsest; l >= plan.finest; --l) {
        const Buffer& L = ws.levels[size_t(l)];
        const bool has_coarse = l < plan.coarsest;
        const Buffer& coarse = has_coarse ? ws.flows[size_t(l) + 1] : ws.flows[size_t(l)];  // not read without coarse
        const PatchGrid g = patch_grid(plan, L.width, L.height);
        SmearPatchParams pp{};
        pp.w = L.width;
        pp.h = L.height;
        pp.level_pitch = L.pitch;
        pp.cw = coarse.width;
        pp.ch = coarse.height;
        pp.coarse_pitch = coarse.pitch;
        pp.has_coarse = has_coarse ? 1 : 0;
        pp.patch = g.patch;
        pp.stride = g.stride;
        pp.nx = g.nx;
        pp.ny = g.ny;
        pp.patch_pitch = ws.patches.pitch;
        pp.iterations = plan.iterations;
        pp.radius = search_radius(plan, l);
        pp.rescue = rescue_radius(plan, l);
        pp.rescue_ratio = plan.rescue_ratio;
        pp.rescue_min_var = plan.rescue_min_var;
        pp.max_step = plan.max_step;
        pp.min_det = plan.min_det;
        if (!run(gpu, "smear_flow_patches", g.nx, g.ny, {L.mem, coarse.mem, ws.patches.mem}, pp)) return false;

        const Buffer& dense = (l == 0 && final_passes.empty()) ? out : ws.flows[size_t(l)];
        SmearDensifyParams dp{};
        dp.w = L.width;
        dp.h = L.height;
        dp.level_pitch = L.pitch;
        dp.patch = g.patch;
        dp.stride = g.stride;
        dp.nx = g.nx;
        dp.ny = g.ny;
        dp.patch_pitch = ws.patches.pitch;
        dp.flow_pitch = dense.pitch;
        dp.photo_eps = plan.photo_eps;
        if (!run(gpu, "smear_flow_densify", L.width, L.height, {L.mem, ws.patches.mem, dense.mem}, dp)) return false;
    }

    const Buffer* src = &ws.flows[size_t(plan.finest)];
    for (size_t i = 0; i < final_passes.size(); ++i) {
        const Buffer& dst = i + 1 == final_passes.size() ? out : ws.temp[i % 2];
        const Buffer& L0 = ws.levels[0];
        if (final_passes[i] == 0) {
            SmearUpsampleParams up{};
            up.w = w;
            up.h = h;
            up.level_pitch = L0.pitch;
            up.cw = src->width;
            up.ch = src->height;
            up.coarse_pitch = src->pitch;
            up.scale = 1 << plan.finest;
            up.flow_pitch = dst.pitch;
            up.photo_eps = plan.photo_eps;
            if (!run(gpu, "smear_flow_upsample", w, h, {L0.mem, src->mem, dst.mem}, up)) return false;
        } else {
            SmearRefineParams rp{};
            rp.w = w;
            rp.h = h;
            rp.level_pitch = L0.pitch;
            rp.src_pitch = src->pitch;
            rp.dst_pitch = dst.pitch;
            rp.dist = final_passes[i];
            if (!run(gpu, "smear_flow_refine", w, h, {L0.mem, src->mem, dst.mem}, rp)) return false;
        }
        src = &dst;
    }
    return true;
}

// One-call flow: allocates its own workspace from `scratch`.
inline bool compute_flow(Backend& gpu, Scratch& scratch, const Buffer& from, const Buffer& to,
                         const FlowSettings& fs, const Buffer& out) {
    FlowWorkspace ws;
    return ws.init(scratch, fs, from.width, from.height) && compute_flow(gpu, ws, from, to, out);
}

// Renders the motion blur of `in.current` into `out` (float4, any size; see
// FrameInputs::out_ox/out_oy). Mirrors smear::render_motion_blur.
inline bool render(Backend& gpu, Scratch& scratch, const FrameInputs& in, const BlurSettings& bs,
                   const FlowSettings& fs, const Buffer& out, const BlurTuning& t = BlurTuning{}) {
    const Buffer& cur = in.current;
    const int w = cur.width, h = cur.height;
    auto usable = [&](const Buffer& b) { return b.mem && b.width == w && b.height == h; };
    const bool use_prev = bs.track != Track::Next && usable(in.previous);
    const bool use_next = bs.track != Track::Previous && usable(in.next);
    if ((!use_prev && !use_next) || (bs.shutter <= 0.0 && !bs.show_vectors)) {
        SmearCopyParams cp{};
        cp.w = w;
        cp.h = h;
        cp.src_pitch = cur.pitch;
        cp.out_w = out.width;
        cp.out_h = out.height;
        cp.out_pitch = out.pitch;
        cp.out_ox = in.out_ox;
        cp.out_oy = in.out_oy;
        return run(gpu, "smear_copy", out.width, out.height, {cur.mem, out.mem}, cp);
    }

    // Optical flow to the neighbours.
    FlowWorkspace ws;
    if (!ws.init(scratch, fs, w, h)) return false;
    Buffer to_prev{cur.mem, w, h, 0}, to_next{cur.mem, w, h, 0};  // placeholders, not read when unused
    if (use_prev && (!scratch.alloc(w, h, 2, to_prev) || !compute_flow(gpu, ws, cur, in.previous, to_prev))) return false;
    if (use_next && (!scratch.alloc(w, h, 2, to_next) || !compute_flow(gpu, ws, cur, in.next, to_next))) return false;
    const Buffer& prev = use_prev ? in.previous : cur;
    const Buffer& next = use_next ? in.next : cur;

    // Pass 1: motion vectors, then their refinement.
    Buffer mot[2];
    if (!scratch.alloc(w, h, 4, mot[0]) || !scratch.alloc(w, h, 4, mot[1])) return false;
    const float half = half_shutter(bs);
    SmearMotionParams mp{};
    mp.w = w;
    mp.h = h;
    mp.cur_pitch = cur.pitch;
    mp.prev_pitch = prev.pitch;
    mp.next_pitch = next.pitch;
    mp.to_prev_pitch = to_prev.pitch;
    mp.to_next_pitch = to_next.pitch;
    mp.mot_pitch = mot[0].pitch;
    mp.has_prev = use_prev ? 1 : 0;
    mp.has_next = use_next ? 1 : 0;
    mp.half_shutter = half;
    mp.max_radius = t.max_radius;
    mp.photo_lo = t.photo_lo;
    mp.photo_hi = t.photo_hi;
    mp.cons_abs = t.cons_abs;
    mp.cons_rel = t.cons_rel;
    if (!run(gpu, "smear_motion", w, h, {cur.mem, prev.mem, next.mem, to_prev.mem, to_next.mem, mot[0].mem}, mp))
        return false;
    int m = 0;
    for (int step : t.refine_steps) {
        if (step <= 0) continue;
        SmearMotionRefineParams rp{};
        rp.w = w;
        rp.h = h;
        rp.cur_pitch = cur.pitch;
        rp.prev_pitch = prev.pitch;
        rp.next_pitch = next.pitch;
        rp.src_pitch = mot[m].pitch;
        rp.dst_pitch = mot[1 - m].pitch;
        rp.has_prev = mp.has_prev;
        rp.has_next = mp.has_next;
        rp.dist = step;
        rp.inv_half = 1.0f / half;
        rp.margin = t.refine_margin;
        rp.photo_lo = t.photo_lo;
        rp.photo_hi = t.photo_hi;
        if (!run(gpu, "smear_motion_refine", w, h, {cur.mem, prev.mem, next.mem, mot[m].mem, mot[1 - m].mem}, rp))
            return false;
        m = 1 - m;
    }
    const Buffer& motion = mot[m];

    // Passes 2–3: tiles and neighbourhoods, with the robust background rounds.
    const int tw = tile_count(w, t.tile), th = tile_count(h, t.tile);
    Buffer tmax, tocc, nmax, nocc[2];
    if (!scratch.alloc(tw, th, 2, tmax) || !scratch.alloc(tw, th, 4, tocc) || !scratch.alloc(tw, th, 2, nmax) ||
        !scratch.alloc(tw, th, 4, nocc[0]) || !scratch.alloc(tw, th, 4, nocc[1]))
        return false;
    int n = 0;
    for (int round = 0; round <= t.robust_rounds; ++round) {
        const Buffer& prior = nocc[1 - n];  // previous round's output (not read on round 0)
        SmearTileParams tp{};
        tp.w = w;
        tp.h = h;
        tp.mot_pitch = motion.pitch;
        tp.tile = t.tile;
        tp.tw = tw;
        tp.th = th;
        tp.prior_pitch = prior.pitch;
        tp.tmax_pitch = tmax.pitch;
        tp.tocc_pitch = tocc.pitch;
        tp.has_prior = round > 0 ? 1 : 0;
        tp.sigma_abs = t.layer_abs * half;
        tp.sigma_rel = t.layer_rel;
        if (!run(gpu, "smear_tiles", tw, th, {motion.mem, prior.mem, tmax.mem, tocc.mem}, tp)) return false;
        SmearNeighbourParams np{};
        np.tw = tw;
        np.th = th;
        np.tile = t.tile;
        np.reach = tile_reach(t);
        np.tmax_pitch = tmax.pitch;
        np.tocc_pitch = tocc.pitch;
        np.nmax_pitch = nmax.pitch;
        np.nocc_pitch = nocc[n].pitch;
        np.bg_tiles = t.bg_tiles;
        np.bg_scale = t.bg_scale;
        if (!run(gpu, "smear_neighbours", tw, th, {tmax.mem, tocc.mem, nmax.mem, nocc[n].mem}, np)) return false;
        n = 1 - n;
    }
    const Buffer& neighbour_occ = nocc[1 - n];

    // Pass 4: gather.
    SmearGatherParams gp{};
    gp.w = w;
    gp.h = h;
    gp.cur_pitch = cur.pitch;
    gp.mot_pitch = motion.pitch;
    gp.tw = tw;
    gp.th = th;
    gp.nmax_pitch = nmax.pitch;
    gp.nocc_pitch = neighbour_occ.pitch;
    gp.out_w = out.width;
    gp.out_h = out.height;
    gp.out_pitch = out.pitch;
    gp.out_ox = in.out_ox;
    gp.out_oy = in.out_oy;
    gp.tile = t.tile;
    gp.max_samples = bs.max_samples;
    gp.show_vectors = bs.show_vectors ? 1 : 0;
    gp.to_velocity = 1.0f / half;
    gp.layer_abs = t.layer_abs;
    gp.layer_rel = t.layer_rel;
    gp.mass_lo = t.mass_lo;
    gp.mass_hi = t.mass_hi;
    return run(gpu, "smear_gather", out.width, out.height,
               {cur.mem, motion.mem, nmax.mem, neighbour_occ.mem, out.mem}, gp);
}

}  // namespace smear::gpu
