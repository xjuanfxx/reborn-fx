// Crosstalk.cpp — After Effects effect (SmartFX, 8/16/32 bpc, GPU, multi-frame rendering).
//
// Thin adapter: read parameters → crosstalk::make_params (core) → crosstalk::render_cpu
// or the GPU passes. All of the effect's logic lives in core/include.
//
// Unlike the plugin it replaces, nothing here keeps state between renders or
// takes a global lock (frames render in parallel), every checkout is checked in
// exactly once, and every buffer access is bounded by the size of the buffer
// actually received.
#include "Crosstalk.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "AE_EffectGPUSuites.h"
#include "CrosstalkGpuRender.h"
#include "crosstalk_cpu.h"

namespace {

constexpr A_long kInputCheckout = 0;

// What PreRender computes and SmartRender consumes.
struct RenderInfo {
    crosstalk::Params controls;
    long long time = 0, time_scale = 1;
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

PF_Err add_int_range(PF_InData* in_data, const char* name, int lo, int hi, int slider_lo, int slider_hi, int dflt,
                     int index) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_ADD_SLIDER(name, lo, hi, slider_lo, slider_hi, dflt, disk_id(index));
    return PF_Err_NONE;
}

PF_Err add_color(PF_InData* in_data, const char* name, const crosstalk::Rgb& c, int index) {
    PF_ParamDef def;
    AEFX_CLR_STRUCT(def);
    PF_ADD_COLOR(name, A_u_char(std::lround(c.r * 255)), A_u_char(std::lround(c.g * 255)),
                 A_u_char(std::lround(c.b * 255)), disk_id(index));
    return PF_Err_NONE;
}

const char* const kMethods = "Shift|Linear|Noisy|Sawtooth|Wave|Wave Motion";

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data) {
    PF_Err err = PF_Err_NONE;
    const crosstalk::Params d;
    const crosstalk::Field &e = d.even, &o = d.odd;

    // Must follow the order of CROSSTALK_PARAMS in Crosstalk.h.
    ERR(add_popup(in_data, "Render Using", 2, RENDER_GPU_OR_CPU, "GPU (CPU fallback)|CPU Only", P_RENDER_USING));
    ERR(add_float(in_data, "Blend With Original", 0, 100, 0, 100, d.blend, P_BLEND));
    ERR(add_int(in_data, "Random Seed", 0, 32000, d.seed, P_SEED));
    ERR(add_popup(in_data, "Edges", 4, int(d.edges), "Transparent|Mirror|Tile|Black", P_EDGES));

    ERR(begin_group(in_data, "Fields Setup", P_FIELDS_START));
    ERR(add_float(in_data, "(E) Size", 1, 400, 1, 40, e.size, P_E_SIZE, PF_Precision_INTEGER));
    ERR(add_float(in_data, "(O) Size", 1, 400, 1, 40, o.size, P_O_SIZE, PF_Precision_INTEGER));
    ERR(add_float(in_data, "(E) Process Input", 0, 100, 0, 100, e.process, P_E_PROCESS));
    ERR(add_float(in_data, "(O) Process Input", 0, 100, 0, 100, o.process, P_O_PROCESS));
    ERR(add_float(in_data, "(E) Luminance", -100, 100, -100, 100, e.luminance, P_E_LUM));
    ERR(add_float(in_data, "(O) Luminance", -100, 100, -100, 100, o.luminance, P_O_LUM));
    ERR(add_checkbox(in_data, "(E) Tint", e.tint, P_E_TINT));
    ERR(add_checkbox(in_data, "(O) Tint", o.tint, P_O_TINT));
    ERR(add_color(in_data, "(E) Tint Color", e.tint_color, P_E_TINT_COLOR));
    ERR(add_color(in_data, "(O) Tint Color", o.tint_color, P_O_TINT_COLOR));
    ERR(end_group(in_data, P_FIELDS_END));

    ERR(begin_group(in_data, "Noise", P_NOISE_START));
    ERR(add_float(in_data, "(E) Noise Level", 0, 100, 0, 100, e.noise, P_E_NOISE));
    ERR(add_float(in_data, "(O) Noise Level", 0, 100, 0, 100, o.noise, P_O_NOISE));
    ERR(add_checkbox(in_data, "(E) Chroma Noise", e.chroma, P_E_CHROMA));
    ERR(add_checkbox(in_data, "(O) Chroma Noise", o.chroma, P_O_CHROMA));
    ERR(add_int_range(in_data, "(E) Decay Amount", 1, 32, 1, 16, e.decay, P_E_DECAY));
    ERR(add_int_range(in_data, "(O) Decay Amount", 1, 32, 1, 16, o.decay, P_O_DECAY));
    ERR(add_checkbox(in_data, "(E) Smear", e.smear, P_E_SMEAR));
    ERR(add_checkbox(in_data, "(O) Smear", o.smear, P_O_SMEAR));
    ERR(end_group(in_data, P_NOISE_END));

    ERR(begin_group(in_data, "Modulation", P_MOD_START));
    ERR(add_popup(in_data, "(E) Distortion Method", 6, int(e.method), kMethods, P_E_METHOD));
    ERR(add_popup(in_data, "(O) Distortion Method", 6, int(o.method), kMethods, P_O_METHOD));
    ERR(add_float(in_data, "(E) Horiz. Adj.", -200, 300, -200, 200, e.hadj, P_E_HADJ));
    ERR(add_float(in_data, "(O) Horiz. Adj.", -200, 300, -200, 200, o.hadj, P_O_HADJ));
    ERR(add_float(in_data, "Influence", -5000, 32000, -5000, 32000, d.influence, P_INFLUENCE, PF_Precision_INTEGER));
    ERR(add_int_range(in_data, "(E) Comb Filt. Level", 0, 20, 0, 8, e.comb, P_E_COMB));
    ERR(add_int_range(in_data, "(O) Comb Filt. Level", 0, 20, 0, 8, o.comb, P_O_COMB));
    ERR(end_group(in_data, P_MOD_END));

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
    crosstalk::Rgb color(int i, PF_Err& err) {
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

crosstalk::Field read_field(ParamReader& r, int odd, PF_Err& err) {
    // Even and odd controls alternate in the table: (E) at P_E_*, (O) right after.
    crosstalk::Field f;
    f.size = r.fval(P_E_SIZE + odd, err);
    f.process = r.fval(P_E_PROCESS + odd, err);
    f.luminance = r.fval(P_E_LUM + odd, err);
    f.tint = r.bval(P_E_TINT + odd, err);
    f.tint_color = r.color(P_E_TINT_COLOR + odd, err);
    f.noise = r.fval(P_E_NOISE + odd, err);
    f.chroma = r.bval(P_E_CHROMA + odd, err);
    f.decay = r.ival(P_E_DECAY + odd, err);
    f.smear = r.bval(P_E_SMEAR + odd, err);
    f.method = static_cast<crosstalk::Method>(std::clamp(r.popup(P_E_METHOD + odd, err), 1, 6));
    f.hadj = r.fval(P_E_HADJ + odd, err);
    f.comb = r.ival(P_E_COMB + odd, err);
    return f;
}

crosstalk::Params read_params(PF_InData* in_data, A_long time, PF_Err& err) {
    ParamReader r(in_data, time);
    crosstalk::Params p;
    p.blend = r.fval(P_BLEND, err);
    p.seed = r.ival(P_SEED, err);
    p.edges = static_cast<crosstalk::Edges>(std::clamp(r.popup(P_EDGES, err), 1, 4));
    p.influence = r.fval(P_INFLUENCE, err);
    p.even = read_field(r, 0, err);
    p.odd = read_field(r, 1, err);
    return p;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

PF_Err About(PF_InData* in_data, PF_OutData* out_data) {
    AEGP_SuiteHandler suites(in_data->pica_basicP);
    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
                                          "%s %d.%d.%d\r"
                                          "Interlaced field interference: tinted, noisy, displaced bands of rows.",
                                          CROSSTALK_NAME, CROSSTALK_MAJOR_VERSION, CROSSTALK_MINOR_VERSION,
                                          CROSSTALK_BUG_VERSION);
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_OutData* out_data) {
    out_data->my_version =
        PF_VERSION(CROSSTALK_MAJOR_VERSION, CROSSTALK_MINOR_VERSION, CROSSTALK_BUG_VERSION, PF_Stage_DEVELOP, 1);
    out_data->out_flags = CROSSTALK_OUT_FLAGS;
    out_data->out_flags2 = CROSSTALK_OUT_FLAGS2;
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

    // Rows are displaced by up to several layer widths and the edges wrap or
    // mirror, so any output pixel can need any pixel of its row: request the
    // whole layer (bounded, so effects before us that expand the buffer still work).
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
crosstalk::Frame make_frame(const RenderInfo& info, const PF_EffectWorld* in, const PF_EffectWorld* out,
                         const PF_InData* in_data) {
    crosstalk::Frame f;
    f.time = info.time;
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
    const CrosstalkParams p = crosstalk::make_params(info.controls, make_frame(info, in, out, in_data));
    const float inv = 1.0f / Channel<PixelT>::kMax;
    std::vector<crosstalk::float4> src(size_t(in->width) * size_t(in->height));
    crosstalk::parallel_for(in->height, [&](int y) {
        const PixelT* row = row_ptr<PixelT>(in, y);
        crosstalk::float4* d = &src[size_t(y) * in->width];
        for (int x = 0; x < in->width; ++x)
            d[x] = {row[x].blue * inv, row[x].green * inv, row[x].red * inv, row[x].alpha * inv};
    });
    std::vector<crosstalk::float4> dst(size_t(out->width) * size_t(out->height));
    crosstalk::render_cpu(p, src.data(), dst.data());
    crosstalk::parallel_for(out->height, [&](int y) {
        PixelT* row = row_ptr<PixelT>(out, y);
        const crosstalk::float4* s = &dst[size_t(y) * out->width];
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
// gpu_data is returned and AE keeps rendering Crosstalk on the CPU.
PF_Err GpuDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra) {
    PF_Err err = PF_Err_NONE;
    GpuSuite gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
    PF_GPUDeviceInfo device;
    AEFX_CLR_STRUCT(device);
    ERR(gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &device));
    if (err) return err;

    std::unique_ptr<crosstalk::gpu::Backend> backend;
    switch (extra->input->what_gpu) {
        case PF_GPU_Framework_CUDA: backend = crosstalk::gpu::make_cuda(device.contextPV, device.command_queuePV); break;
        case PF_GPU_Framework_OPENCL:
            backend = crosstalk::gpu::make_opencl(device.contextPV, device.devicePV, device.command_queuePV);
            break;
        case PF_GPU_Framework_METAL: backend = crosstalk::gpu::make_metal(device.devicePV, device.command_queuePV); break;
        default: break;
    }
    if (backend) {
        extra->output->gpu_data = backend.release();
        out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
    return err;
}

PF_Err GpuDeviceSetdown(PF_GPUDeviceSetdownExtra* extra) {
    delete static_cast<crosstalk::gpu::Backend*>(extra->input->gpu_data);
    return PF_Err_NONE;
}

PF_Err SmartRenderGpu(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const auto* info = static_cast<const RenderInfo*>(extra->input->pre_render_data);
    auto* backend = static_cast<crosstalk::gpu::Backend*>(const_cast<void*>(extra->input->gpu_data));
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
            const CrosstalkParams p = crosstalk::make_params(info->controls, make_frame(*info, input, output, in_data));
            const int px = int(sizeof(PF_PixelFloat));
            const crosstalk::gpu::Buffer in_buf{in_mem, input->width, input->height, int(input->rowbytes / px)};
            const crosstalk::gpu::Buffer out_buf{out_mem, output->width, output->height, int(output->rowbytes / px)};
            if (!crosstalk::gpu::render(*backend, p, in_buf, out_buf)) err = PF_Err_INTERNAL_STRUCT_DAMAGED;
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
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, CROSSTALK_NAME, CROSSTALK_MATCH_NAME, CROSSTALK_CATEGORY,
                                     AE_RESERVED_INFO, "EffectMain", CROSSTALK_SUPPORT_URL);
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
