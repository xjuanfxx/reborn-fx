/* smear_gpu_types.h — parameter blocks shared by the host and the GPU kernels.
 *
 * Compiled as C++ on the host and embedded verbatim in the CUDA, OpenCL and
 * Metal kernel sources, so it must stay plain C: only `int`/`float` fields
 * (4 bytes each, no padding), no includes, no host-only types.
 *
 * Images are float4 in After Effects' GPU layout: x = B, y = G, z = R, w = A,
 * premultiplied. Flow fields are float2 (pixels). Pitches are in elements
 * (pixels or cells), not bytes. Each kernel runs over the grid given in its
 * comment and returns early outside it (grids may be rounded up).
 *
 * Passes, in order (mirroring core/include/smear_core.h and smear_render.h):
 *   flow (once per neighbour frame): gray → downsample × levels →
 *       per level coarse→fine: patches → densify → [upsample] → refine × n
 *   blur: motion → motion_refine × n → (tiles → neighbours) × (1 + rounds) → gather
 */
#ifndef SMEAR_GPU_TYPES_H
#define SMEAR_GPU_TYPES_H

/* smear_flow_gray, grid w × h: level 0 of the pyramid from two frames.
 * Cell = (brightness of `from`, brightness of `to`, gradient of `from`). */
typedef struct SmearGrayParams {
    int w, h;
    int from_pitch, to_pitch, dst_pitch;
} SmearGrayParams;

/* smear_flow_downsample, grid w × h: the next coarser level (half size). */
typedef struct SmearDownParams {
    int sw, sh, src_pitch;  /* finer level */
    int w, h, dst_pitch;    /* this level */
} SmearDownParams;

/* smear_flow_patches, grid nx × ny: one flow per patch. */
typedef struct SmearPatchParams {
    int w, h, level_pitch;
    int cw, ch, coarse_pitch;  /* dense flow of the coarser level */
    int has_coarse;            /* 0 on the coarsest level (coarse buffer not read) */
    int patch, stride, nx, ny, patch_pitch;
    int iterations;
    int radius;                /* integer search radius */
    float max_step, min_det;
} SmearPatchParams;

/* smear_flow_densify, grid w × h: dense flow of one level from its patches. */
typedef struct SmearDensifyParams {
    int w, h, level_pitch;
    int patch, stride, nx, ny, patch_pitch;
    int flow_pitch;
    float photo_eps;
} SmearDensifyParams;

/* smear_flow_upsample, grid w × h (full resolution). */
typedef struct SmearUpsampleParams {
    int w, h, level_pitch;
    int cw, ch, coarse_pitch;
    int scale;  /* 2^level of the coarse flow */
    int flow_pitch;
    float photo_eps;
} SmearUpsampleParams;

/* smear_flow_refine, grid w × h (full resolution). */
typedef struct SmearRefineParams {
    int w, h, level_pitch;
    int src_pitch, dst_pitch;
    int dist;  /* neighbour distance */
} SmearRefineParams;

/* smear_motion, grid w × h: blur half-vector and occlusion per pixel.
 * Motion cell = (half-streak x, half-streak y, occlusion 0–1, 0). */
typedef struct SmearMotionParams {
    int w, h;
    int cur_pitch, prev_pitch, next_pitch;
    int to_prev_pitch, to_next_pitch, mot_pitch;
    int has_prev, has_next;  /* 0: that frame/flow is not used (buffer not read) */
    float half_shutter, max_radius;
    float photo_lo, photo_hi, cons_abs, cons_rel;
} SmearMotionParams;

/* smear_motion_refine, grid w × h: adopt a neighbour's vector if it explains the pixel better. */
typedef struct SmearMotionRefineParams {
    int w, h;
    int cur_pitch, prev_pitch, next_pitch;
    int src_pitch, dst_pitch;
    int has_prev, has_next;
    int dist;
    float inv_half, margin, photo_lo, photo_hi;
} SmearMotionRefineParams;

/* smear_tiles, grid tw × th. */
typedef struct SmearTileParams {
    int w, h, mot_pitch;
    int tile, tw, th;
    int prior_pitch, tmax_pitch, tocc_pitch;
    int has_prior;  /* 0 on the first round (prior buffer not read) */
    float sigma_abs, sigma_rel;
} SmearTileParams;

/* smear_neighbours, grid tw × th. */
typedef struct SmearNeighbourParams {
    int tw, th, tile, reach;
    int tmax_pitch, tocc_pitch, nmax_pitch, nocc_pitch;
    int bg_tiles;
    float bg_scale;
} SmearNeighbourParams;

/* smear_gather, grid out_w × out_h: the blurred output (or the vector view).
 * Output pixel (x, y) is pixel (x + out_ox, y + out_oy) of the current frame. */
typedef struct SmearGatherParams {
    int w, h, cur_pitch, mot_pitch;
    int tw, th, nmax_pitch, nocc_pitch;
    int out_w, out_h, out_pitch, out_ox, out_oy;
    int tile, max_samples, show_vectors;
    float to_velocity;
    float layer_abs, layer_rel, mass_lo, mass_hi;
} SmearGatherParams;

/* smear_copy, grid out_w × out_h: output = current frame (nothing to blur). */
typedef struct SmearCopyParams {
    int w, h, src_pitch;
    int out_w, out_h, out_pitch, out_ox, out_oy;
} SmearCopyParams;

#endif
