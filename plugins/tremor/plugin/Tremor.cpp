// Tremor.cpp — After Effects effect (SmartFX, 8/16/32 bpc, GPU, multi-frame rendering).
//
// Thin adapter: read parameters → tremor::evaluate (core) → tremor::render_frame
// or the GPU passes. All of the effect's logic lives in core/include.
#include "Tremor.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

#include "AE_EffectGPUSuites.h"
#include "TremorGpuRender.h"
#include "tremor_core.h"
#include "tremor_render.h"

namespace {

constexpr A_long kInputCheckout = 0;

// What PreRender computes and SmartRender consumes.
struct RenderInfo {
    tremor::FrameState state;
    tremor::Look look;
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

// A wave: "<name> Max/Freq./Amp./Phase" style controls share these ranges.
PF_Err add_freq(PF_InData* in_data, const char* name, double dflt, int index) {
    return add_float(in_data, name, 0, 100, 0, 100, dflt, index);
}
PF_Err add_amp(PF_InData* in_data, const char* name, double dflt, int index) {
    return add_float(in_data, name, 0, 10, 0, 10, dflt, index);
}
PF_Err add_phase(PF_InData* in_data, const char* name, double dflt, int index) {
    return add_float(in_data, name, -360, 360, -360, 360, dflt, index);
}

// Names, order and ranges follow the classic layout of this kind of effect.
PF_Err ParamsSetup(PF_InData* in_data, PF_OutData* out_data) {
    PF_Err err = PF_Err_NONE;

    // Must follow the order of TREMOR_PARAMS in Tremor.h.
    ERR(add_popup(in_data, "Render Using", 2, RENDER_GPU_OR_CPU, "GPU (CPU fallback)|CPU Only", P_RENDER_USING));
    ERR(add_float(in_data, "Blend With Original", 0, 100, 0, 100, 0, P_BLEND));
    ERR(add_int(in_data, "Random Seed", 0, 32000, 0, P_SEED));
    ERR(add_popup(in_data, "Deinterlace", 4, 1, "None|Lower Only|Upper Only|Merge", P_DEINTERLACE));
    ERR(add_popup(in_data, "Edges", 4, 1, "Transparent|Mirror|Tile|Black", P_EDGES));

    ERR(begin_group(in_data, "Color Control", P_COLOR_START));
    ERR(add_checkbox(in_data, "Enable Alpha Composite", false, P_ALPHA_COMPOSITE));
    ERR(add_float(in_data, "R Adjust", -100, 100, -100, 100, 0, P_R_ADJUST));
    ERR(add_float(in_data, "G Adjust", -100, 100, -100, 100, 0, P_G_ADJUST));
    ERR(add_float(in_data, "B Adjust", -100, 100, -100, 100, 0, P_B_ADJUST));
    ERR(end_group(in_data, P_COLOR_END));

    ERR(begin_group(in_data, "Motion Control", P_MOTION_START));
    ERR(add_float(in_data, "Horizontal Max", -100, 200, -100, 100, 16, P_H_MAX));
    ERR(add_freq(in_data, "Horiz. Freq.", 2, P_H_FREQ));
    ERR(add_amp(in_data, "Horiz. Amp.", 0.1, P_H_AMP));
    ERR(add_phase(in_data, "Horiz. Phase", 30, P_H_PHASE));
    ERR(add_float(in_data, "Vertical Max", -100, 200, -100, 100, 4, P_V_MAX));
    ERR(add_freq(in_data, "Vert. Freq.", 2, P_V_FREQ));
    ERR(add_amp(in_data, "Vert. Amp.", 0.1, P_V_AMP));
    ERR(add_phase(in_data, "Vert. Phase", 30, P_V_PHASE));
    ERR(add_float(in_data, "(H) Erratic", 0, 100, 0, 100, 0.6, P_H_ERRATIC));
    ERR(add_float(in_data, "(V) Erratic", 0, 100, 0, 100, 0.3, P_V_ERRATIC));
    ERR(add_float(in_data, "Zoom Base", -32, 32, -16, 16, 0.1, P_ZOOM_BASE));
    ERR(add_float(in_data, "Zoom Factor", -32, 32, -16, 16, 0.1, P_ZOOM_FACTOR));
    ERR(add_freq(in_data, "Zoom Freq.", 7, P_ZOOM_FREQ));
    ERR(add_amp(in_data, "Zoom Amp.", 0.1, P_ZOOM_AMP));
    ERR(add_phase(in_data, "Zoom Phase", 90, P_ZOOM_PHASE));
    ERR(add_float(in_data, "Zoom Erratic", 0, 100, 0, 100, 2, P_ZOOM_ERRATIC));
    ERR(add_float(in_data, "Rotation Max", -720, 720, -720, 720, 2, P_ROT_MAX));
    ERR(add_freq(in_data, "Rotation Freq.", 16, P_ROT_FREQ));
    ERR(add_amp(in_data, "Rotation Amp.", 1, P_ROT_AMP));
    ERR(add_phase(in_data, "Rotation Phase", 15, P_ROT_PHASE));
    ERR(add_float(in_data, "Channel Separate", -360, 360, -360, 360, 4, P_CH_SEP));
    ERR(add_freq(in_data, "Ch.Separation Freq.", 18, P_CH_FREQ));
    ERR(add_amp(in_data, "Ch.Separation Amp.", 0.4, P_CH_AMP));
    ERR(add_phase(in_data, "Ch.Separation Phase", 30, P_CH_PHASE));
    ERR(end_group(in_data, P_MOTION_END));

    ERR(begin_group(in_data, "Advanced", P_ADVANCED_START));
    ERR(add_float(in_data, "Erratic(H) Adj.", -100, 100, -100, 100, 100, P_ERRATIC_H_ADJ));
    ERR(add_float(in_data, "Erratic(V) Adj.", -100, 100, -100, 100, 100, P_ERRATIC_V_ADJ));
    ERR(add_float(in_data, "Erratic(Zoom) Adj.", -100, 100, -100, 100, 100, P_ERRATIC_ZOOM_ADJ));
    ERR(end_group(in_data, P_ADVANCED_END));

    out_data->num_params = P_NUM_PARAMS;
    return err;
}

// ---------------------------------------------------------------------------
// Reading parameters at a given time
// ---------------------------------------------------------------------------

class ParamReader {
public:
    ParamReader(PF_InData* in_data, A_long time) : in_data_(in_data), time_(time) {}

    double fval(int i, PF_Err& err) {
        PF_ParamDef d;
        err |= checkout(i, d);
        const double v = d.u.fs_d.value;
        err |= checkin(d);
        return v;
    }
    int ival(int i, PF_Err& err) {
        PF_ParamDef d;
        err |= checkout(i, d);
        const int v = d.u.sd.value;
        err |= checkin(d);
        return v;
    }
    bool bval(int i, PF_Err& err) {
        PF_ParamDef d;
        err |= checkout(i, d);
        const bool v = d.u.bd.value != 0;
        err |= checkin(d);
        return v;
    }
    int popup(int i, PF_Err& err) {
        PF_ParamDef d;
        err |= checkout(i, d);
        const int v = d.u.pd.value;
        err |= checkin(d);
        return v;
    }

private:
    PF_Err checkout(int index, PF_ParamDef& def) {
        AEFX_CLR_STRUCT(def);
        return PF_CHECKOUT_PARAM(in_data_, index, time_, in_data_->time_step, in_data_->time_scale, &def);
    }
    PF_Err checkin(PF_ParamDef& def) { return PF_CHECKIN_PARAM(in_data_, &def); }

    PF_InData* in_data_;
    A_long time_;
};

tremor::Params read_params(PF_InData* in_data, A_long time, PF_Err& err) {
    ParamReader r(in_data, time);
    tremor::Params p;
    p.blend = r.fval(P_BLEND, err);
    p.seed = r.ival(P_SEED, err);
    p.deinterlace = static_cast<tremor::Deinterlace>(std::clamp(r.popup(P_DEINTERLACE, err), 1, 4));
    p.edges = static_cast<tremor::Edges>(std::clamp(r.popup(P_EDGES, err), 1, 4));
    p.alpha_composite = r.bval(P_ALPHA_COMPOSITE, err);
    p.r_adjust = r.fval(P_R_ADJUST, err);
    p.g_adjust = r.fval(P_G_ADJUST, err);
    p.b_adjust = r.fval(P_B_ADJUST, err);
    p.h_max = r.fval(P_H_MAX, err);
    p.h_freq = r.fval(P_H_FREQ, err);
    p.h_amp = r.fval(P_H_AMP, err);
    p.h_phase = r.fval(P_H_PHASE, err);
    p.v_max = r.fval(P_V_MAX, err);
    p.v_freq = r.fval(P_V_FREQ, err);
    p.v_amp = r.fval(P_V_AMP, err);
    p.v_phase = r.fval(P_V_PHASE, err);
    p.h_erratic = r.fval(P_H_ERRATIC, err);
    p.v_erratic = r.fval(P_V_ERRATIC, err);
    p.zoom_base = r.fval(P_ZOOM_BASE, err);
    p.zoom_factor = r.fval(P_ZOOM_FACTOR, err);
    p.zoom_freq = r.fval(P_ZOOM_FREQ, err);
    p.zoom_amp = r.fval(P_ZOOM_AMP, err);
    p.zoom_phase = r.fval(P_ZOOM_PHASE, err);
    p.zoom_erratic = r.fval(P_ZOOM_ERRATIC, err);
    p.rot_max = r.fval(P_ROT_MAX, err);
    p.rot_freq = r.fval(P_ROT_FREQ, err);
    p.rot_amp = r.fval(P_ROT_AMP, err);
    p.rot_phase = r.fval(P_ROT_PHASE, err);
    p.ch_sep = r.fval(P_CH_SEP, err);
    p.ch_freq = r.fval(P_CH_FREQ, err);
    p.ch_amp = r.fval(P_CH_AMP, err);
    p.ch_phase = r.fval(P_CH_PHASE, err);
    p.erratic_h_adj = r.fval(P_ERRATIC_H_ADJ, err);
    p.erratic_v_adj = r.fval(P_ERRATIC_V_ADJ, err);
    p.erratic_zoom_adj = r.fval(P_ERRATIC_ZOOM_ADJ, err);
    return p;
}

// ---------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------

// The About box is plain text, at most PF_MAX_EFFECT_MSG_LEN (255) chars; \r breaks lines.
PF_Err About(PF_InData* in_data, PF_OutData* out_data) {
    AEGP_SuiteHandler suites(in_data->pica_basicP);
    suites.ANSICallbacksSuite1()->sprintf(out_data->return_msg,
                                          "%s %d.%d.%d\r"
                                          "Unsteady camera: shake, zoom, sway and color fringing.",
                                          TREMOR_NAME, TREMOR_MAJOR_VERSION, TREMOR_MINOR_VERSION,
                                          TREMOR_BUG_VERSION);
    return PF_Err_NONE;
}

PF_Err GlobalSetup(PF_OutData* out_data) {
    out_data->my_version =
        PF_VERSION(TREMOR_MAJOR_VERSION, TREMOR_MINOR_VERSION, TREMOR_BUG_VERSION, PF_Stage_DEVELOP, 1);
    out_data->out_flags = TREMOR_OUT_FLAGS;
    out_data->out_flags2 = TREMOR_OUT_FLAGS2;
    return PF_Err_NONE;
}

PF_Err PreRender(PF_InData* in_data, PF_OutData* /*out_data*/, PF_PreRenderExtra* extra) {
    PF_Err err = PF_Err_NONE;
    auto info = std::make_unique<RenderInfo>();

    const A_long now = in_data->current_time;
    const tremor::Params p = read_params(in_data, now, err);
    ParamReader r(in_data, now);
    const int render_using = r.popup(P_RENDER_USING, err);
    if (err) return err;

    // Offer the GPU render only when AE renders on the GPU, our kernels loaded on that
    // device (gpu_data) and the user did not pick CPU. Otherwise AE calls SmartRender.
    const bool gpu_ready = extra->input->what_gpu != PF_GPU_Framework_NONE && extra->input->gpu_data;
    if (gpu_ready && render_using == RENDER_GPU_OR_CPU) {
        extra->output->flags |= PF_RenderOutputFlag_GPU_RENDER_POSSIBLE;
    }

    // The effect rotates and zooms the whole layer around its center, so any output
    // pixel can need any input pixel: request the whole layer (bounded, so effects
    // before us that expand the buffer still return pixels).
    const double ds_x = double(in_data->downsample_x.num) / in_data->downsample_x.den;
    const double ds_y = double(in_data->downsample_y.num) / in_data->downsample_y.den;
    const PF_LRect layer{0, 0, A_long(std::ceil(in_data->width * ds_x)), A_long(std::ceil(in_data->height * ds_y))};
    PF_RenderRequest req = extra->input->output_request;
    req.rect = union_rect(layer, extra->input->output_request.rect);
    req.preserve_rgb_of_zero_alpha = TRUE;
    req.field = PF_Field_FRAME;
    PF_CheckoutResult in_result;
    AEFX_CLR_STRUCT(in_result);
    ERR(extra->cb->checkout_layer(in_data->effect_ref, P_INPUT, kInputCheckout, &req, now, in_data->time_step,
                                  in_data->time_scale, &in_result));
    if (err) return err;

    // Like the original, the output keeps the layer's size.
    info->canvas = layer;
    info->src_rect = in_result.result_rect;
    info->state = tremor::evaluate(p, now, in_data->time_scale, layer.right, layer.bottom);
    info->look = tremor::make_look(p, extra->input->bitdepth);

    info->out_rect = intersect_rect(extra->input->output_request.rect, layer);
    extra->output->result_rect = info->out_rect;
    extra->output->max_result_rect = layer;
    extra->output->solid = FALSE;
    extra->output->pre_render_data = info.release();
    extra->output->delete_pre_render_data_func = delete_render_info;
    return err;
}

// AE buffer ↔ tremor::Image conversion (float RGBA, 0–1 at 8/16 bpc).
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

// Copies buffer `w` into `img` with its corner at (ox, oy); anything outside is dropped.
template <class PixelT>
void blit_to_image(PF_EffectWorld* w, tremor::Image& img, int ox, int oy) {
    const float inv = 1.0f / Channel<PixelT>::kMax;
    tremor::parallel_for(w->height, [&](int y) {
        const int iy = y + oy;
        if (iy < 0 || iy >= img.height) return;
        const PixelT* row = row_ptr<PixelT>(w, y);
        for (int x = 0; x < w->width; ++x) {
            const int ix = x + ox;
            if (ix < 0 || ix >= img.width) continue;
            img.at(ix, iy) = {row[x].red * inv, row[x].green * inv, row[x].blue * inv, row[x].alpha * inv};
        }
    });
}

template <class PixelT>
void from_image(const tremor::Image& img, PF_EffectWorld* w) {
    tremor::parallel_for(std::min<int>(w->height, img.height), [&](int y) {
        PixelT* row = row_ptr<PixelT>(w, y);
        for (int x = 0; x < std::min<int>(w->width, img.width); ++x) {
            const tremor::Pixel& p = img.at(x, y);
            row[x].red = Channel<PixelT>::store(p.r);
            row[x].green = Channel<PixelT>::store(p.g);
            row[x].blue = Channel<PixelT>::store(p.b);
            row[x].alpha = Channel<PixelT>::store(p.a);
        }
    });
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

template <class PixelT>
void render_world(PF_EffectWorld* in, PF_EffectWorld* out, const RenderInfo& info, const PF_InData* in_data) {
    const PF_LRect& cv = info.canvas;
    tremor::Image canvas(cv.right - cv.left, cv.bottom - cv.top);
    blit_to_image<PixelT>(in, canvas, info.src_rect.left - cv.left, info.src_rect.top - cv.top);
    int ox = 0, oy = 0;
    output_corner(info, out, in_data, ox, oy);
    tremor::Image dst(out->width, out->height);
    tremor::render_frame(canvas, dst, ox, oy, info.state, info.look);
    from_image<PixelT>(dst, out);
}

PF_Err SmartRender(PF_InData* in_data, PF_OutData* out_data, PF_SmartRenderExtra* extra) {
    PF_Err err = PF_Err_NONE, err2 = PF_Err_NONE;
    const auto* info = static_cast<const RenderInfo*>(extra->input->pre_render_data);
    if (!info) return PF_Err_BAD_CALLBACK_PARAM;

    PF_EffectWorld *input = nullptr, *output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, kInputCheckout, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (!err && input && output) {
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
// gpu_data is returned and AE keeps rendering Tremor on the CPU.
PF_Err GpuDeviceSetup(PF_InData* in_data, PF_OutData* out_data, PF_GPUDeviceSetupExtra* extra) {
    PF_Err err = PF_Err_NONE;
    GpuSuite gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
    PF_GPUDeviceInfo device;
    AEFX_CLR_STRUCT(device);
    ERR(gpu->GetDeviceInfo(in_data->effect_ref, extra->input->device_index, &device));
    if (err) return err;

    std::unique_ptr<tremor::gpu::Backend> backend;
    switch (extra->input->what_gpu) {
        case PF_GPU_Framework_CUDA: backend = tremor::gpu::make_cuda(device.contextPV, device.command_queuePV); break;
        case PF_GPU_Framework_OPENCL:
            backend = tremor::gpu::make_opencl(device.contextPV, device.devicePV, device.command_queuePV);
            break;
        case PF_GPU_Framework_METAL: backend = tremor::gpu::make_metal(device.devicePV, device.command_queuePV); break;
        default: break;
    }
    if (backend) {
        extra->output->gpu_data = backend.release();
        out_data->out_flags2 = PF_OutFlag2_SUPPORTS_GPU_RENDER_F32;
    }
    return err;
}

PF_Err GpuDeviceSetdown(PF_GPUDeviceSetdownExtra* extra) {
    delete static_cast<tremor::gpu::Backend*>(extra->input->gpu_data);
    return PF_Err_NONE;
}

// Temporary GPU worlds for one render; disposed when the render ends.
class AeScratch final : public tremor::gpu::Scratch {
public:
    AeScratch(PF_InData* in_data, const PF_GPUDeviceSuite1* suite, A_u_long device_index)
        : in_data_(in_data), suite_(suite), device_index_(device_index) {}
    ~AeScratch() override {
        for (PF_EffectWorld* w : worlds_) suite_->DisposeGPUWorld(in_data_->effect_ref, w);
    }

    bool alloc(int width, int height, tremor::gpu::Buffer& out) override {
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
    auto* backend = static_cast<tremor::gpu::Backend*>(const_cast<void*>(extra->input->gpu_data));
    if (!info || !backend) return PF_Err_BAD_CALLBACK_PARAM;

    PF_EffectWorld *input = nullptr, *output = nullptr;
    ERR(extra->cb->checkout_layer_pixels(in_data->effect_ref, kInputCheckout, &input));
    ERR(extra->cb->checkout_output(in_data->effect_ref, &output));
    if (!err && input && output) {
        AEFX_SuiteScoper<PF_WorldSuite2> world_suite(in_data, kPFWorldSuite, kPFWorldSuiteVersion2, out_data);
        GpuSuite gpu(in_data, kPFGPUDeviceSuite, kPFGPUDeviceSuiteVersion1, out_data);
        PF_PixelFormat format = PF_PixelFormat_INVALID;
        ERR(world_suite->PF_GetPixelFormat(input, &format));
        if (!err && format != PF_PixelFormat_GPU_BGRA128) err = PF_Err_UNRECOGNIZED_PARAM_TYPE;

        void *in_mem = nullptr, *out_mem = nullptr;
        ERR(gpu->GetGPUWorldData(in_data->effect_ref, input, &in_mem));
        ERR(gpu->GetGPUWorldData(in_data->effect_ref, output, &out_mem));
        if (!err) {
            const PF_LRect& cv = info->canvas;
            tremor::gpu::Layout layout;
            layout.canvas_w = cv.right - cv.left;
            layout.canvas_h = cv.bottom - cv.top;
            layout.in_ox = info->src_rect.left - cv.left;
            layout.in_oy = info->src_rect.top - cv.top;
            output_corner(*info, output, in_data, layout.out_ox, layout.out_oy);

            const int px = int(sizeof(PF_PixelFloat));
            const tremor::gpu::Buffer in_buf{in_mem, input->width, input->height, int(input->rowbytes / px)};
            const tremor::gpu::Buffer out_buf{out_mem, output->width, output->height, int(output->rowbytes / px)};
            AeScratch scratch(in_data, gpu.get(), extra->input->device_index);
            if (!tremor::gpu::render(*backend, scratch, info->state, info->look, layout, in_buf, out_buf)) {
                err = PF_Err_INTERNAL_STRUCT_DAMAGED;
            }
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
    result = PF_REGISTER_EFFECT_EXT2(inPtr, inPluginDataCallBackPtr, TREMOR_NAME, TREMOR_MATCH_NAME, TREMOR_CATEGORY,
                                     AE_RESERVED_INFO, "EffectMain", TREMOR_SUPPORT_URL);
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
