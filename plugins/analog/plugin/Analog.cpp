// Analog.cpp — After Effects effect (SmartFX, 8/16/32 bpc, GPU, multi-frame rendering).
//
// Thin adapter: read parameters → analog::make_params (core) → analog::render_cpu
// or the GPU passes. All of the effect's logic lives in core/include.
//
// Unlike the plugin it replaces, nothing here keeps state between renders, it
// never checks out its own input layer as a parameter, every checkout is
// checked in exactly once, and every buffer access is bounded by the size of
// the buffer actually received.
#include "Analog.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "AE_EffectGPUSuites.h"
#include "AnalogGpuRender.h"
#include "analog_cpu.h"

namespace {

constexpr A_long kInputCheckout = 0;

// What PreRender computes and SmartRender consumes.
struct RenderInfo {
    analog::Params controls;
    long long time = 0, time_scale = 1;
    double ds_x = 1, ds_y = 1;
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

PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data) {
    PF_Err err = PF_Err_NONE;
    const analog::Params d;

    // Must follow the order of ANALOG_PARAMS in Analog.h.
    ERR(add_popup(in_data, "Render Using", 2, RENDER_GPU_OR_CPU, "GPU (CPU fallback)|CPU Only", P_RENDER_USING));
    ERR(add_float(in_data, "Random Seed", 0, 1000000, 0, 10000, d.seed, P_SEED, PF_Precision_INTEGER));
    ERR(add_float(in_data, "Signal Strength", 0, 200, 0, 200, d.strength, P_STRENGTH));
    ERR(add_float(in_data, "Signal Amplification", 0, 100, 0, 10, d.amplification, P_AMPLIFICATION));
    ERR(add_float(in_data, "Cutoff Filter", 1, 100000, 1, 1000, d.cutoff, P_CUTOFF));
    ERR(add_float(in_data, "Luma Noise", 0, 400, 0, 100, d.luma_noise, P_LUMA_NOISE));
    ERR(add_float(in_data, "Chroma Noise", 0, 400, 0, 100, d.chroma_noise, P_CHROMA_NOISE));
    ERR(add_float(in_data, "Hue Noise", 0, 200, 0, 100, d.hue_noise, P_HUE_NOISE));
    ERR(add_float(in_data, "Chroma Loss", 0, 1, 0, 1, d.chroma_loss, P_CHROMA_LOSS));
    ERR(add_float(in_data, "Sharpen", 0, 100, 0, 10, d.sharpen, P_SHARPEN));
    ERR(add_int(in_data, "Device-to-Device Connections", 0, 100, int(d.generations), P_GENERATIONS));
    ERR(add_popup(in_data, "Emulating VHS", 4, int(d.vhs), "Off|SP|LP|EP", P_VHS));
    ERR(add_checkbox(in_data, "Destruction", d.destruction, P_DESTRUCTION));

    ERR(begin_group(in_data, "Luma Modulation", P_FM_START));
    ERR(add_checkbox(in_data, "Enable", d.fm, P_FM_ON));
    ERR(add_float(in_data, "Frequency", 0.001, 40, 0.001, 4, d.fm_freq, P_FM_FREQ, PF_Precision_THOUSANDTHS));
    ERR(add_float(in_data, "Phase", 0.001, 40, 0.001, 10, d.fm_dev, P_FM_PHASE, PF_Precision_THOUSANDTHS));
    ERR(add_float(in_data, "Lightness", 0, 1, 0, 1, d.fm_light, P_FM_LIGHT, PF_Precision_THOUSANDTHS));
    ERR(add_int(in_data, "Quantize", 0, 100, int(d.fm_quantize), P_FM_QUANTIZE));
    ERR(add_checkbox(in_data, "Tape Errors", d.tape_errors, P_TAPE_ERRORS));
    ERR(add_float(in_data, "Tape Errors Start", 0, 1, 0, 1, d.te_start, P_TE_START));
    ERR(add_float(in_data, "Tape Errors End", 0, 1, 0, 1, d.te_end, P_TE_END));
    ERR(add_float(in_data, "Tape Errors Probability", 0, 1, 0, 1, d.te_prob, P_TE_PROB));
    ERR(add_float(in_data, "Tape Errors Fade", 0.01, 5, 0.01, 5, d.te_fade, P_TE_FADE));
    ERR(add_float(in_data, "Tape Errors Gamma", 0.01, 14, 0.01, 14, d.te_gamma, P_TE_GAMMA));
    ERR(end_group(in_data, P_FM_END));

    ERR(begin_group(in_data, "TV Distortion", P_TV_START));
    ERR(add_float(in_data, "Thick Distortion", 0, 100, 0, 100, d.thick, P_THICK));
    ERR(add_float(in_data, "Fine Distortion", 0, 10, 0, 10, d.fine, P_FINE));
    ERR(add_float(in_data, "Distortion Speed", 0, 200, 0, 200, d.dist_speed, P_DIST_SPEED));
    ERR(add_float(in_data, "Roll Speed", 0, 2000, 0, 200, d.roll_speed, P_ROLL));
    ERR(end_group(in_data, P_TV_END));

    ERR(begin_group(in_data, "Experimental", P_EXP_START));
    ERR(add_checkbox(in_data, "Low-Pass Filter In", d.lp_in, P_LP_IN));
    ERR(add_popup(in_data, "Low-Pass Filter Out", 3, int(d.lp_out), "Off|Lite|Strong", P_LP_OUT));
    ERR(add_popup(in_data, "Phase Shift", 4, 1, "0|90|180|270", P_PHASE_SHIFT));
    ERR(add_int(in_data, "Phase Shift Offset", 0, 3, d.phase_offset, P_PHASE_OFFSET));
    ERR(add_checkbox(in_data, "S-Video Out", d.svideo, P_SVIDEO));
    ERR(add_checkbox(in_data, "Chroma Vertical Blend", d.vblend, P_VBLEND));
    ERR(add_checkbox(in_data, "Sharpen On/Off", d.sharpen_on, P_SHARPEN_ON));
    ERR(add_float(in_data, "Scanlines Blur", 0, 100, 0, 10, d.scan_blur, P_SCAN_BLUR));
    ERR(end_group(in_data, P_EXP_END));

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

analog::Params read_params(PF_InData* in_data, A_long time, PF_Err& err) {
    ParamReader r(in_data, time);
    analog::Params p;
    p.seed = r.fval(P_SEED, err);
    p.strength = r.fval(P_STRENGTH, err);
    p.amplification = r.fval(P_AMPLIFICATION, err);
    p.cutoff = r.fval(P_CUTOFF, err);
    p.luma_noise = r.fval(P_LUMA_NOISE, err);
    p.chroma_noise = r.fval(P_CHROMA_NOISE, err);
    p.hue_noise = r.fval(P_HUE_NOISE, err);
    p.chroma_loss = r.fval(P_CHROMA_LOSS, err);
    p.sharpen = r.fval(P_SHARPEN, err);
    p.generations = r.ival(P_GENERATIONS, err);
    p.vhs = static_cast<analog::Vhs>(std::clamp(r.popup(P_VHS, err), 1, 4));
    p.destruction = r.bval(P_DESTRUCTION, err);
    p.fm = r.bval(P_FM_ON, err);
    p.fm_freq = r.fval(P_FM_FREQ, err);
    p.fm_dev = r.fval(P_FM_PHASE, err);
    p.fm_light = r.fval(P_FM_LIGHT, err);
    p.fm_quantize = r.ival(P_FM_QUANTIZE, err);
    p.tape_errors = r.bval(P_TAPE_ERRORS, err);
    p.te_start = r.fval(P_TE_START, err);
    p.te_end = r.fval(P_TE_END, err);
    p.te_prob = r.fval(P_TE_PROB, err);
    p.te_fade = r.fval(P_TE_FADE, err);
    p.te_gamma = r.fval(P_TE_GAMMA, err);
    p.thick = r.fval(P_THICK, err);
    p.fine = r.fval(P_FINE, err);
    p.dist_speed = r.fval(P_DIST_SPEED, err);
    p.roll_speed = r.fval(P_ROLL, err);
    p.lp_in = r.bval(P_LP_IN, err);
    p.lp_out = static_cast<analog::LowPassOut>(std::clamp(r.popup(P_LP_OUT, err), 1, 3));
    p.phase_shift = (std::clamp(r.popup(P_PHASE_SHIFT, err), 1, 4) - 1) * 90;
    p.phase_offset = r.ival(P_PHASE_OFFSET, err);
    p.svideo = r.bval(P_SVIDEO, err);
    p.vblend = r.bval(P_VBLEND, err);
    p.sharpen_on = r.bval(P_SHARPEN_ON, err);
    p.scan_blur = r.fval(P_SCAN_BLUR, err);
    return p;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

PF_Err About(PF_InData* in_data, PF_OutData* out_data) {
    AEGP_SuiteHandler suites(in_data->pica_basicP);
    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
                                          "%s %d.%d.%d\r"
                                          "Composite video and VHS tape emulation.",
                                          ANALOG_NAME, ANALOG_MAJOR_VERSION, ANALOG_MINOR_VERSION,
                                          ANALOG_BUG_VERSION);
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_OutData* out_data) {
    out_data->my_version =
        PF_VERSION(ANALOG_MAJOR_VERSION, ANALOG_MINOR_VERSION, ANALOG_BUG_VERSION, PF_Stage_DEVELOP, 1);
    out_data->out_flags = ANALOG_OUT_FLAGS;
    out_data->out_flags2 = ANALOG_OUT_FLAGS2;
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

    // Every line is filtered from its left edge and the picture rolls and
    // jitters, so any output pixel can need any input pixel: request the whole
    // layer (bounded, so effects before us that expand the buffer still work).
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
analog::Frame make_frame(const RenderInfo& info, const PF_EffectWorld* in, const PF_EffectWorld* out,
                         const PF_InData* in_data) {
    analog::Frame f;
    f.time = info.time;
    f.time_scale = info.time_scale;
    f.canvas_w = info.canvas.right - info.canvas.left;
    f.rows = info.canvas.bottom - info.canvas.top;
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
    const AnalogParams p = analog::make_params(info.controls, make_frame(info, in, out, in_data));
    const float inv = 1.0f / Channel<PixelT>::kMax;
    std::vector<analog::float4> src(size_t(in->width) * size_t(in->height));
    analog::parallel_for(in->height, [&](int y) {
        const PixelT* row = row_ptr<PixelT>(in, y);
        analog::float4* d = &src[size_t(y) * in->width];
        for (int x = 0; x < in->width; ++x)
            d[x] = {row[x].blue * inv, row[x].green * inv, row[x].red * inv, row[x].alpha * inv};
    });
    std::vector<analog::float4> dst(size_t(out->width) * size_t(out->height));
    analog::render_cpu(p, src.data(), dst.data());
    analog::parallel_for(out->height, [&](int y) {
        PixelT* row = row_ptr<PixelT>(out, y);
        const analog::float4* s = &dst[size_t(y) * out->width];
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
// gpu_data is returned and AE keeps rendering Analog on the CPU.
PF_Err GpuDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra) {
    PF_Err err = PF_Err_NONE;
    GpuSuite gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
    PF_GPUDeviceInfo device;
    AEFX_CLR_STRUCT(device);
    ERR(gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &device));
    if (err) return err;

    std::unique_ptr<analog::gpu::Backend> backend;
    switch (extra->input->what_gpu) {
        case PF_GPU_Framework_CUDA: backend = analog::gpu::make_cuda(device.contextPV, device.command_queuePV); break;
        case PF_GPU_Framework_OPENCL:
            backend = analog::gpu::make_opencl(device.contextPV, device.devicePV, device.command_queuePV);
            break;
        case PF_GPU_Framework_METAL: backend = analog::gpu::make_metal(device.devicePV, device.command_queuePV); break;
        default: break;
    }
    if (backend) {
        extra->output->gpu_data = backend.release();
        out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
    return err;
}

PF_Err GpuDeviceSetdown(PF_GPUDeviceSetdownExtra* extra) {
    delete static_cast<analog::gpu::Backend*>(extra->input->gpu_data);
    return PF_Err_NONE;
}

// Temporary GPU worlds for one render; disposed when the render ends.
class AeScratch final : public analog::gpu::Scratch {
public:
    AeScratch(PF_InData* in_data, const PF_GPUDeviceSuite1* suite, A_u_long device_index)
        : in_data_(in_data), suite_(suite), device_index_(device_index) {}
    ~AeScratch() override {
        for (PF_EffectWorld* w : worlds_) suite_->DisposeGPUWorld(in_data_->effect_ref, w);
    }

    bool alloc(int width, int height, analog::gpu::Buffer& out) override {
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
    auto* backend = static_cast<analog::gpu::Backend*>(const_cast<void*>(extra->input->gpu_data));
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
            const AnalogParams p = analog::make_params(info->controls, make_frame(*info, input, output, in_data));
            const int px = int(sizeof(PF_PixelFloat));
            const analog::gpu::Buffer in_buf{in_mem, input->width, input->height, int(input->rowbytes / px)};
            const analog::gpu::Buffer out_buf{out_mem, output->width, output->height, int(output->rowbytes / px)};
            AeScratch scratch(in_data, gpu.get(), extra->input->device_index);
            if (!analog::gpu::render(*backend, scratch, p, in_buf, out_buf)) err = PF_Err_INTERNAL_STRUCT_DAMAGED;
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
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, ANALOG_NAME, ANALOG_MATCH_NAME, ANALOG_CATEGORY,
                                     AE_RESERVED_INFO, "EffectMain", ANALOG_SUPPORT_URL);
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
