// Reception.cpp — After Effects effect (SmartFX, 8/16/32 bpc, GPU, multi-frame rendering).
//
// Thin adapter: read parameters → reception::make_params (core) → reception::render_cpu
// or the GPU passes. All of the effect's logic lives in core/include.
//
// Unlike the plugin it replaces, nothing here keeps state between renders or
// takes a global lock (frames render in parallel), every checkout is checked in
// exactly once, and every buffer access is bounded by the size of the buffer
// actually received.
#include "Reception.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "AE_EffectGPUSuites.h"
#include "ReceptionGpuRender.h"
#include "reception_cpu.h"

namespace {

constexpr A_long kInputCheckout = 0;

// What PreRender computes and SmartRender consumes.
struct RenderInfo {
    reception::Params controls;
    long long time = 0, time_step = 1, time_scale = 1;
    double ds_x = 1, ds_y = 1;
    int full_w = 0, full_h = 0;  // layer size, full resolution
    int bits = 8;
    PF_LRect src_rect{};  // input buffer, in layer coords (downsampled)
    PF_LRect canvas{};    // the whole layer, in layer coords (downsampled)
    PF_LRect out_rect{};  // output rect returned to AE
};

void delete_render_info(void* p) { delete static_cast<RenderInfo*>(p); }

bool empty_rect(const PF_LRect& r) { return r.right <= r.left || r.bottom <= r.top; }

PF_LRect union_rect(const PF_LRect& a, const PF_LRect& b) {
    if (empty_rect(a)) return b;
    if (empty_rect(b)) return a;
    return {std::min(a.left, b.left), std::min(a.top, b.top), std::max(a.right, b.right),
            std::max(a.bottom, b.bottom)};
}

PF_LRect intersect_rect(const PF_LRect& a, const PF_LRect& b) {
    PF_LRect r{std::max(a.left, b.left), std::max(a.top, b.top), std::min(a.right, b.right),
               std::min(a.bottom, b.bottom)};
    if (r.right < r.left) r.right = r.left;
    if (r.bottom < r.top) r.bottom = r.top;
    return r;
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

PF_Err add_float(PF_InData* in_data, const char* name, double valid_min, double valid_max, double slider_min,
                 double slider_max, double dflt, int index, A_short precision = PF_Precision_HUNDREDTHS) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_ADD_FLOAT_SLIDERX(name, valid_min, valid_max, slider_min, slider_max, dflt, precision,
                         PF_ValueDisplayFlag_NONE, 0, disk_id(index));
    return PF_Err_NONE;
}

PF_Err add_int(PF_InData* in_data, const char* name, int lo, int hi, int dflt, int index) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER(name, lo, hi, lo, hi, dflt, disk_id(index));
    return PF_Err_NONE;
}

PF_Err add_checkbox(PF_InData* in_data, const char* name, bool dflt, int index) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_ADD_CHECKBOXX(name, dflt ? TRUE : FALSE, 0, disk_id(index));
    return PF_Err_NONE;
}

PF_Err add_popup(PF_InData* in_data, const char* name, int choices, int dflt, const char* items, int index) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_ADD_POPUP(name, choices, A_short(dflt), items, disk_id(index));
    return PF_Err_NONE;
}

PF_Err begin_group(PF_InData* in_data, const char* name, int index) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_ADD_TOPIC(name, disk_id(index));
    return PF_Err_NONE;
}

PF_Err end_group(PF_InData* in_data, int index) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_END_TOPIC(disk_id(index));
    return PF_Err_NONE;
}

PF_Err add_color(PF_InData* in_data, const char* name, const reception::Rgb& c, int index) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR(name, A_u_char(std::lround(c.r * 255)), A_u_char(std::lround(c.g * 255)),
                 A_u_char(std::lround(c.b * 255)), disk_id(index));
    return PF_Err_NONE;
}

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data) {
    PF_Err err = PF_Err_NONE;
    const reception::Params d;

    // Must follow the order of RECEPTION_PARAMS in Reception.h.
    ERR(add_popup(in_data, "Render Using", 2, RENDER_GPU_OR_CPU, "GPU (CPU fallback)|CPU Only", P_RENDER_USING));
    ERR(add_float(in_data, "Blend With Original", 0, 100, 0, 100, d.blend, P_BLEND));
    ERR(add_int(in_data, "Random Seed", 0, 32000, d.seed, P_SEED));
    ERR(add_popup(in_data, "Edges", 4, int(d.edges), "Transparent|Mirror|Tile|Black", P_EDGES));

    ERR(begin_group(in_data, "Color", P_COLOR_START));
    ERR(add_color(in_data, "Tint Color", d.tint, P_TINT));
    ERR(add_float(in_data, "Brightness Lift", -100, 100, -100, 100, d.lift, P_LIFT));
    ERR(add_float(in_data, "Bright Variance", -100, 100, -100, 100, d.bright_var, P_BRIGHT_VAR));
    ERR(add_float(in_data, "Contrast", -50, 50, -50, 50, d.contrast, P_CONTRAST));
    ERR(add_float(in_data, "Contrast Variance", -100, 100, -100, 100, d.contrast_var, P_CONTRAST_VAR));
    ERR(end_group(in_data, P_COLOR_END));

    ERR(begin_group(in_data, "Noise", P_NOISE_START));
    ERR(add_float(in_data, "Noise Max.", 0, 100, 0, 100, d.noise_max, P_NOISE_MAX));
    ERR(add_float(in_data, "Noise Variance", -30, 100, 0, 30, d.noise_var, P_NOISE_VAR));
    ERR(add_float(in_data, "Micro Noise", 0, 100, 0, 50, d.micro, P_MICRO));
    ERR(add_float(in_data, "Noise Width", 0, 200, 0, 50, d.noise_width, P_NOISE_WIDTH));
    ERR(add_float(in_data, "Noise Tempo", 0, 200, 0, 100, d.noise_tempo, P_NOISE_TEMPO));
    ERR(end_group(in_data, P_NOISE_END));

    ERR(begin_group(in_data, "Distortion", P_DIST_START));
    ERR(add_float(in_data, "Distortion Min.", 0, 100, 0, 100, d.dist_min, P_DIST_MIN));
    ERR(add_float(in_data, "Distortion Max.", 0, 100, 0, 100, d.dist_max, P_DIST_MAX));
    ERR(add_float(in_data, "Tempo Min.", 0, 100, 0, 100, d.tempo_min, P_TEMPO_MIN));
    ERR(add_float(in_data, "Tempo Max.", 0, 100, 0, 100, d.tempo_max, P_TEMPO_MAX));
    ERR(add_float(in_data, "Amplitude 1", 0, 100, 0, 100, d.amp1, P_AMP1));
    ERR(add_float(in_data, "Amplitude 2", 0, 100, 0, 100, d.amp2, P_AMP2));
    ERR(add_float(in_data, "Frequency 1", 0, 100, 0, 20, d.freq1, P_FREQ1));
    ERR(add_float(in_data, "Frequency 2", 0, 100, 0, 20, d.freq2, P_FREQ2));
    ERR(add_float(in_data, "Speed 1", 0, 99, 0, 20, d.speed1, P_SPEED1));
    ERR(add_float(in_data, "Speed 2", 0, 99, 0, 20, d.speed2, P_SPEED2));
    ERR(add_float(in_data, "Offset Variance", 0, 100, 0, 20, d.offset_var, P_OFFSET_VAR));
    ERR(end_group(in_data, P_DIST_END));

    ERR(begin_group(in_data, "Glow", P_GLOW_START));
    ERR(add_float(in_data, "Glow Size", 0, 64, 0, 32, d.glow_size, P_GLOW_SIZE, PF_Precision_INTEGER));
    ERR(add_float(in_data, "Glow Intensity", 0, 100, 0, 100, d.glow_intensity, P_GLOW_INTENSITY));
    ERR(add_float(in_data, "Glow Threshold", 0, 100, 0, 100, d.glow_threshold, P_GLOW_THRESHOLD));
    ERR(end_group(in_data, P_GLOW_END));

    ERR(begin_group(in_data, "Ghosting", P_GHOST_START));
    ERR(add_float(in_data, "Ghost Offset 1", 0, 100, 0, 100, d.ghost_off1, P_GHOST_OFF1));
    ERR(add_float(in_data, "Ghost Offset 2", 0, 100, 0, 100, d.ghost_off2, P_GHOST_OFF2));
    ERR(add_float(in_data, "Ghost Offset 3", 0, 100, 0, 100, d.ghost_off3, P_GHOST_OFF3));
    ERR(add_float(in_data, "Primary Weight", 0, 100, 0, 100, d.w_primary, P_W_PRIMARY));
    ERR(add_float(in_data, "Ghost 1 Weight", 0, 100, 0, 100, d.w_ghost1, P_W_GHOST1));
    ERR(add_float(in_data, "Ghost 2 Weight", 0, 100, 0, 100, d.w_ghost2, P_W_GHOST2));
    ERR(add_float(in_data, "Ghost 3 Weight", 0, 100, 0, 100, d.w_ghost3, P_W_GHOST3));
    ERR(end_group(in_data, P_GHOST_END));

    ERR(begin_group(in_data, "Bars", P_BARS_START));
    ERR(add_float(in_data, "Bar Height", 0, 100, 0, 100, d.bar_height, P_BAR_HEIGHT));
    ERR(add_float(in_data, "Bar Ripple Freq.", 0, 200, 0, 100, d.bar_ripple, P_BAR_RIPPLE));
    ERR(add_float(in_data, "Bar Darkness", 0, 100, 0, 100, d.bar_dark, P_BAR_DARK));
    ERR(end_group(in_data, P_BARS_END));

    ERR(begin_group(in_data, "Vertical Hold", P_HOLD_START));
    ERR(add_float(in_data, "Roll Probability", 0, 100, 0, 100, d.roll_prob, P_ROLL_PROB));
    ERR(add_float(in_data, "Roll Length", 0, 5000, 0, 1000, d.roll_length, P_ROLL_LENGTH));
    ERR(add_float(in_data, "Roll Speed", 0, 200, 0, 100, d.roll_speed, P_ROLL_SPEED));
    ERR(end_group(in_data, P_HOLD_END));

    out_data->num_params = P_NUM_PARAMS;
    return err;
}

// ---------------------------------------------------------------------------
// Reading parameters at a given time (each checkout is checked in right away)
// ---------------------------------------------------------------------------

class ParamReader {
public:
    ParamReader(PF_InData* in_data, A_long time) : in_data_(in_data), time_(time) {}

    double fval(int i, PF_Err& err) {
        PF_ParamDef d;
        if (!checkout(i, d, err)) return 0;
        const double v = d.u.fs_d.value;
        err |= PF_CHECKIN_PARAM(in_data_, &d);
        return v;
    }
    int ival(int i, PF_Err& err) {
        PF_ParamDef d;
        if (!checkout(i, d, err)) return 0;
        const int v = d.u.sd.value;
        err |= PF_CHECKIN_PARAM(in_data_, &d);
        return v;
    }
    bool bval(int i, PF_Err& err) {
        PF_ParamDef d;
        if (!checkout(i, d, err)) return false;
        const bool v = d.u.bd.value != 0;
        err |= PF_CHECKIN_PARAM(in_data_, &d);
        return v;
    }
    reception::Rgb color(int i, PF_Err& err) {
        PF_ParamDef d;
        if (!checkout(i, d, err)) return {};
        const PF_Pixel v = d.u.cd.value;
        err |= PF_CHECKIN_PARAM(in_data_, &d);
        return {v.red / 255.0, v.green / 255.0, v.blue / 255.0};
    }
    int popup(int i, PF_Err& err) {
        PF_ParamDef d;
        if (!checkout(i, d, err)) return 1;
        const int v = d.u.pd.value;
        err |= PF_CHECKIN_PARAM(in_data_, &d);
        return v;
    }

private:
    bool checkout(int index, PF_ParamDef& def, PF_Err& err) {
        AEFX_CLR_STRUCT(def);
        const PF_Err e = PF_CHECKOUT_PARAM(in_data_, index, time_, in_data_->time_step, in_data_->time_scale, &def);
        err |= e;
        return e == PF_Err_NONE;
    }

    PF_InData* in_data_;
    A_long time_;
};

reception::Params read_params(PF_InData* in_data, A_long time, PF_Err& err) {
    ParamReader r(in_data, time);
    reception::Params p;
    p.blend = r.fval(P_BLEND, err);
    p.seed = r.ival(P_SEED, err);
    p.edges = static_cast<reception::Edges>(std::clamp(r.popup(P_EDGES, err), 1, 4));
    p.tint = r.color(P_TINT, err);
    p.lift = r.fval(P_LIFT, err);
    p.bright_var = r.fval(P_BRIGHT_VAR, err);
    p.contrast = r.fval(P_CONTRAST, err);
    p.contrast_var = r.fval(P_CONTRAST_VAR, err);
    p.noise_max = r.fval(P_NOISE_MAX, err);
    p.noise_var = r.fval(P_NOISE_VAR, err);
    p.micro = r.fval(P_MICRO, err);
    p.noise_width = r.fval(P_NOISE_WIDTH, err);
    p.noise_tempo = r.fval(P_NOISE_TEMPO, err);
    p.dist_min = r.fval(P_DIST_MIN, err);
    p.dist_max = r.fval(P_DIST_MAX, err);
    p.tempo_min = r.fval(P_TEMPO_MIN, err);
    p.tempo_max = r.fval(P_TEMPO_MAX, err);
    p.amp1 = r.fval(P_AMP1, err);
    p.amp2 = r.fval(P_AMP2, err);
    p.freq1 = r.fval(P_FREQ1, err);
    p.freq2 = r.fval(P_FREQ2, err);
    p.speed1 = r.fval(P_SPEED1, err);
    p.speed2 = r.fval(P_SPEED2, err);
    p.offset_var = r.fval(P_OFFSET_VAR, err);
    p.glow_size = r.fval(P_GLOW_SIZE, err);
    p.glow_intensity = r.fval(P_GLOW_INTENSITY, err);
    p.glow_threshold = r.fval(P_GLOW_THRESHOLD, err);
    p.ghost_off1 = r.fval(P_GHOST_OFF1, err);
    p.ghost_off2 = r.fval(P_GHOST_OFF2, err);
    p.ghost_off3 = r.fval(P_GHOST_OFF3, err);
    p.w_primary = r.fval(P_W_PRIMARY, err);
    p.w_ghost1 = r.fval(P_W_GHOST1, err);
    p.w_ghost2 = r.fval(P_W_GHOST2, err);
    p.w_ghost3 = r.fval(P_W_GHOST3, err);
    p.bar_height = r.fval(P_BAR_HEIGHT, err);
    p.bar_ripple = r.fval(P_BAR_RIPPLE, err);
    p.bar_dark = r.fval(P_BAR_DARK, err);
    p.roll_prob = r.fval(P_ROLL_PROB, err);
    p.roll_length = r.fval(P_ROLL_LENGTH, err);
    p.roll_speed = r.fval(P_ROLL_SPEED, err);
    return p;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

PF_Err About(PF_InData* in_data, PF_OutData* out_data) {
    AEGP_SuiteHandler suites(in_data->pica_basicP);
    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
                                          "%s %d.%d.%d\r"
                                          "Bad TV reception: noise, wavy distortion, glow, ghosts, rolling bars and vertical roll.",
                                          RECEPTION_NAME, RECEPTION_MAJOR_VERSION, RECEPTION_MINOR_VERSION,
                                          RECEPTION_BUG_VERSION);
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_OutData* out_data) {
    out_data->my_version =
        PF_VERSION(RECEPTION_MAJOR_VERSION, RECEPTION_MINOR_VERSION, RECEPTION_BUG_VERSION, PF_Stage_DEVELOP, 1);
    out_data->out_flags = RECEPTION_OUT_FLAGS;
    out_data->out_flags2 = RECEPTION_OUT_FLAGS2;
    return PF_Err_NONE;
}

PF_Err PreRender(PF_InData* in_data, PF_OutData* /*out_data*/, PF_PreRenderExtra* extra) {
    PF_Err err = PF_Err_NONE;
    auto info = std::make_unique<RenderInfo>();

    const A_long now = in_data->current_time;
    info->controls = read_params(in_data, now, err);
    ParamReader r(in_data, now);
    const int render_using = r.popup(P_RENDER_USING, err);
    if (err) return err;

    const bool gpu_ready = extra->input->what_gpu != PF_GPU_Framework_NONE && extra->input->gpu_data;
    if (gpu_ready && render_using == RENDER_GPU_OR_CPU) {
        extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    }

    // Rows shift sideways, ghosts repeat them and the picture rolls vertically,
    // so any output pixel can need any input pixel: request the whole layer
    // (bounded, so effects before us that expand the buffer still work).
    info->ds_x = double(in_data->downsample_x.num) / in_data->downsample_x.den;
    info->ds_y = double(in_data->downsample_y.num) / in_data->downsample_y.den;
    const PF_LRect layer{0, 0, A_long(std::ceil(in_data->width * info->ds_x)),
                         A_long(std::ceil(in_data->height * info->ds_y))};
    PF_RenderRequest req = extra->input->output_request;
    req.rect = union_rect(layer, extra->input->output_request.rect);
    req.preserve_rgb_of_zero_alpha = TRUE;
    req.field = PF_Field_FRAME;
    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);
    ERR(extra->cb->checkout_layer(in_data->effect_ref, P_INPUT, kInputCheckout, &req, now, in_data->time_step,
                                  in_data->time_scale, &in_result));
    if (err) return err;

    info->canvas = layer;
    info->full_w = in_data->width;
    info->full_h = in_data->height;
    info->src_rect = in_result.result_rect;
    info->time = now;
    info->time_step = in_data->time_step;
    info->time_scale = in_data->time_scale;
    info->bits = extra->input->bitdepth;
    info->out_rect = intersect_rect(extra->input->output_request.rect, layer);
    extra->output->result_rect = info->out_rect;
    extra->output->max_result_rect = layer;
    extra->output->solid = FALSE;
    extra->output->pre_render_data = info.release();
    extra->output->delete_pre_render_data_func = delete_render_info;
    return err;
}

// Output pixel (0,0) on the canvas. The output buffer starts at the corner of the
// rect returned in PreRender; if AE hands over another size, fall back to the SDK
// convention (-output_origin).
void output_corner(const RenderInfo& info, const PF_EffectWorld* out, const PF_InData* in_data, int& ox, int& oy) {
    const PF_LRect& o = info.out_rect;
    if (out->width == o.right - o.left && out->height == o.bottom - o.top) {
        ox = o.left - info.canvas.left;
        oy = o.top - info.canvas.top;
    } else {
        ox = -in_data->output_origin_x - info.canvas.left;
        oy = -in_data->output_origin_y - info.canvas.top;
    }
}

// The frame as the renderers see it, from the buffers actually received.
reception::Frame make_frame(const RenderInfo& info, const PF_EffectWorld* in, const PF_EffectWorld* out,
                         const PF_InData* in_data) {
    reception::Frame f;
    f.time = info.time;
    f.time_step = info.time_step;
    f.time_scale = info.time_scale;
    f.canvas_w = info.canvas.right - info.canvas.left;
    f.canvas_h = info.canvas.bottom - info.canvas.top;
    f.full_w = info.full_w;
    f.full_h = info.full_h;
    f.ds_x = info.ds_x;
    f.ds_y = info.ds_y;
    f.bits = info.bits;
    f.in_w = in->width;
    f.in_h = in->height;
    f.in_ox = info.src_rect.left - info.canvas.left;
    f.in_oy = info.src_rect.top - info.canvas.top;
    f.out_w = out->width;
    f.out_h = out->height;
    output_corner(info, out, in_data, f.out_ox, f.out_oy);
    return f;
}

// AE pixels <-> float4 (B, G, R, A), 0–1 at 8/16 bpc.
template <class PixelT>
struct Channel;
template <>
struct Channel<PF_Pixel8> {
    static constexpr float kMax = PF_MAX_CHAN8;
    static A_u_char store(float v) { return A_u_char(std::clamp(v, 0.0f, 1.0f) * kMax + 0.5f); }
};
template <>
struct Channel<PF_Pixel16> {
    static constexpr float kMax = PF_MAX_CHAN16;
    static A_u_short store(float v) { return A_u_short(std::clamp(v, 0.0f, 1.0f) * kMax + 0.5f); }
};
template <>
struct Channel<PF_PixelFloat> {
    static constexpr float kMax = 1.0f;
    static PF_FpShort store(float v) { return v; }
};

template <class PixelT>
PixelT* row_ptr(PF_EffectWorld* w, int y) {
    return reinterpret_cast<PixelT*>(reinterpret_cast<char*>(w->data) + size_t(y) * w->rowbytes);
}

template <class PixelT>
void render_world(PF_EffectWorld* in, PF_EffectWorld* out, const RenderInfo& info, const PF_InData* in_data) {
    const ReceptionParams p = reception::make_params(info.controls, make_frame(info, in, out, in_data));
    const float inv = 1.0f / Channel<PixelT>::kMax;
    std::vector<reception::float4> src(size_t(in->width) * size_t(in->height));
    reception::parallel_for(in->height, [&](int y) {
        const PixelT* row = row_ptr<PixelT>(in, y);
        reception::float4* d = &src[size_t(y) * in->width];
        for (int x = 0; x < in->width; ++x)
            d[x] = {row[x].blue * inv, row[x].green * inv, row[x].red * inv, row[x].alpha * inv};
    });
    std::vector<reception::float4> dst(size_t(out->width) * size_t(out->height));
    reception::render_cpu(p, src.data(), dst.data());
    reception::parallel_for(out->height, [&](int y) {
        PixelT* row = row_ptr<PixelT>(out, y);
        const reception::float4* s = &dst[size_t(y) * out->width];
        for (int x = 0; x < out->width; ++x) {
            row[x].blue = Channel<PixelT>::store(s[x].x);
            row[x].green = Channel<PixelT>::store(s[x].y);
            row[x].red = Channel<PixelT>::store(s[x].z);
            row[x].alpha = Channel<PixelT>::store(s[x].w);
        }
    });
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const auto* info = static_cast<const RenderInfo*>(extra->input->pre_render_data);
    if (!info) return PF_Err_BAD_CALLBACK_PARAM;

    PF_EffectWorld *input = nullptr, *output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, kInputCheckout, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (!err && input && output && output->width > 0 && output->height > 0) {
        AEFX_SuiteScoper<PF_WorldSuite2> world_suite(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        PF_PixelFormat format = PF_PixelFormat_INVALID;
        ERR(world_suite->PF_GetPixelFormat(input, &format));
        if (!err) {
            switch (format) {
                case PF_PixelFormat_ARGB128: render_world<PF_PixelFloat>(input, output, *info, in_data); break;
                case PF_PixelFormat_ARGB64: render_world<PF_Pixel16>(input, output, *info, in_data); break;
                case PF_PixelFormat_ARGB32: render_world<PF_Pixel8>(input, output, *info, in_data); break;
                default: err = PF_Err_BAD_CALLBACK_PARAM; break;
            }
        }
    }
    ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, kInputCheckout));
    return err;
}

// ---------------------------------------------------------------------------
// GPU
// ---------------------------------------------------------------------------

using GpuSuite = AEFX_SuiteScoper<PF_GPUDeviceSuite1>;

// Loads the kernels for the device AE is about to render on. On failure no
// gpu_data is returned and AE keeps rendering Reception on the CPU.
PF_Err GpuDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra) {
    PF_Err err = PF_Err_NONE;
    GpuSuite gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
    PF_GPUDeviceInfo device;
    AEFX_CLR_STRUCT(device);
    ERR(gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &device));
    if (err) return err;

    std::unique_ptr<reception::gpu::Backend> backend;
    switch (extra->input->what_gpu) {
        case PF_GPU_Framework_CUDA: backend = reception::gpu::make_cuda(device.contextPV, device.command_queuePV); break;
        case PF_GPU_Framework_OPENCL:
            backend = reception::gpu::make_opencl(device.contextPV, device.devicePV, device.command_queuePV);
            break;
        case PF_GPU_Framework_METAL: backend = reception::gpu::make_metal(device.devicePV, device.command_queuePV); break;
        default: break;
    }
    if (backend) {
        extra->output->gpu_data = backend.release();
        out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
    return err;
}

PF_Err GpuDeviceSetdown(PF_GPUDeviceSetdownExtra* extra) {
    delete static_cast<reception::gpu::Backend*>(extra->input->gpu_data);
    return PF_Err_NONE;
}

// Temporary GPU worlds for one render; disposed when the render ends.
class AeScratch final : public reception::gpu::Scratch {
public:
    AeScratch(PF_InData* in_data, const PF_GPUDeviceSuite1* suite, A_u_long device_index)
        : in_data_(in_data), suite_(suite), device_index_(device_index) {}
    ~AeScratch() override {
        for (PF_EffectWorld* w : worlds_) suite_->DisposeGPUWorld(in_data_->effect_ref, w);
    }

    bool alloc(int width, int height, reception::gpu::Buffer& out) override {
        PF_EffectWorld* world = nullptr;
        const PF_RationalScale square{1, 1};
        if (suite_->CreateGPUWorld(in_data_->effect_ref, device_index_, width, height, square, in_data_->field,
                                   PF_PixelFormat_GPU_BGRA128, false, &world) != PF_Err_NONE)
            return false;
        worlds_.push_back(world);
        void* mem = nullptr;
        if (suite_->GetGPUWorldData(in_data_->effect_ref, world, &mem) != PF_Err_NONE) return false;
        out = {mem, width, height, int(world->rowbytes / sizeof(PF_PixelFloat))};
        return true;
    }

private:
    PF_InData* in_data_;
    const PF_GPUDeviceSuite1* suite_;
    A_u_long device_index_;
    std::vector<PF_EffectWorld*> worlds_;
};

PF_Err SmartRenderGpu(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const auto* info = static_cast<const RenderInfo*>(extra->input->pre_render_data);
    auto* backend = static_cast<reception::gpu::Backend*>(const_cast<void*>(extra->input->gpu_data));
    if (!info || !backend) return PF_Err_BAD_CALLBACK_PARAM;

    PF_EffectWorld *input = nullptr, *output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, kInputCheckout, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (!err && input && output && output->width > 0 && output->height > 0) {
        AEFX_SuiteScoper<PF_WorldSuite2> world_suite(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        GpuSuite gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
        PF_PixelFormat format = PF_PixelFormat_INVALID;
        ERR(world_suite->PF_GetPixelFormat(input, &format));
        if (!err && format != PF_PixelFormat_GPU_BGRA128) err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

        void *in_mem = nullptr, *out_mem = nullptr;
        ERR(gpu->GetGPUWorldData(in_data->effect_ref, input, &in_mem));
        ERR(gpu->GetGPUWorldData(in_data->effect_ref, output, &out_mem));
        if (!err) {
            const ReceptionParams p = reception::make_params(info->controls, make_frame(*info, input, output, in_data));
            const int px = int(sizeof(PF_PixelFloat));
            const reception::gpu::Buffer in_buf{in_mem, input->width, input->height, int(input->rowbytes / px)};
            const reception::gpu::Buffer out_buf{out_mem, output->width, output->height, int(output->rowbytes / px)};
            AeScratch scratch(in_data, gpu.get(), extra->input->device_index);
            if (!reception::gpu::render(*backend, scratch, p, in_buf, out_buf)) err = PF_Err_INTERNAL_STRUCT_DAMAGED;
        }
    }
    ERR2(extra->cb->checkin_layer_pixels(in_data->effect_ref, kInputCheckout));
    return err;
}

}  // namespace

extern "C" DllExport PF_Err PluginDataEntryFunction2(PF_PluginDataPtr inPtr, PF_PluginDataCB2 inPluginDataCallBackPtr,
                                                     SPBasicSuite* /*inSPBasicSuitePtr*/,
                                                     const char* /*inHostName*/, const char* /*inHostVersion*/) {
    PF_Err result = PF_Err_INVALID_CALLBACK;  // the SDK macro writes to `result`
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, RECEPTION_NAME, RECEPTION_MATCH_NAME, RECEPTION_CATEGORY,
                                     AE_RESERVED_INFO, "EffectMain", RECEPTION_SUPPORT_URL);
    return result;
}

PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data, PF_ParamDef* /*params*/[],
                  PF_LayerDef* /*output*/, void* extra) {
    PF_Err err = PF_Err_NONE;
    try {
        switch (cmd) {
            case PF_Cmd_ABOUT: err = About(in_data, out_data); break;
            case PF_Cmd_GLOBAL_SETUP: err = GlobalSetup(out_data); break;
            case PF_Cmd_PARAMS_SETUP: err = ParamsSetup(in_data, out_data); break;
            case PF_Cmd_SMART_PRE_RENDER:
                err = PreRender(in_data, out_data, static_cast<PF_PreRenderExtra*>(extra));
                break;
            case PF_Cmd_SMART_RENDER:
                err = SmartRender(in_data, out_data, static_cast<PF_SmartRenderExtra*>(extra));
                break;
            case PF_Cmd_SMART_RENDER_GPU:
                err = SmartRenderGpu(in_data, out_data, static_cast<PF_SmartRenderExtra*>(extra));
                break;
            case PF_Cmd_GPU_DEVICE_SETUP:
                err = GpuDeviceSetup(in_data, out_data, static_cast<PF_GPUDeviceSetupExtra*>(extra));
                break;
            case PF_Cmd_GPU_DEVICE_SETDOWN:
                err = GpuDeviceSetdown(static_cast<PF_GPUDeviceSetdownExtra*>(extra));
                break;
            default: break;
        }
    } catch (PF_Err& thrown) {
        err = thrown;
    } catch (...) {
        err = PF_Err_INTERNAL_STRUCT_DAMAGED;
    }
    return err;
}
