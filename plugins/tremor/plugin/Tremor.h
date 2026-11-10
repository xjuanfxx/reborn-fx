// Tremor.h — declarations of the After Effects effect.
#pragma once

#define PF_DEEP_COLOR_AWARE 1

#include "AEConfig.h"
#ifdef AE_OS_WIN
    #include <Windows.h>
#endif
#include "entry.h"
#include "AE_Effect.h"
#include "AE_EffectCB.h"
#include "AE_EffectCBSuites.h"
#include "AE_Macros.h"
#include "AEFX_SuiteHandlerTemplate.h"
#include "AEGP_SuiteHandler.h"
#include "Param_Utils.h"

#define TREMOR_NAME "Tremor"
#define TREMOR_MATCH_NAME "XJFX Tremor"
#define TREMOR_CATEGORY "Tremor"
#define TREMOR_SUPPORT_URL "https://github.com/"

#define TREMOR_MAJOR_VERSION 0
#define TREMOR_MINOR_VERSION 1
#define TREMOR_BUG_VERSION 0

// Must match AE_Effect_Global_OutFlags(_2) in TremorPiPL.r.
#define TREMOR_OUT_FLAGS (PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_DEEP_COLOR_AWARE)
#define TREMOR_OUT_FLAGS2                                                                         \
    (PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |                        \
     PF_OutFlag2_SUPPORTS_THREADED_RENDERING | PF_OutFlag2_SUPPORTS_GPU_RENDER_F32)

// Every parameter as X(index name, disk ID), in the order ParamsSetup adds them
// (the order shown in the Effect Controls panel).
//
// Disk IDs identify each parameter in saved projects: never change or reuse one
// once released. They are independent of the order. New parameters use 1000+.
#define TREMOR_PARAMS(X)          \
    X(P_INPUT, 0)                 \
    X(P_RENDER_USING, 1000)       \
    X(P_BLEND, 2)                 \
    X(P_SEED, 3)                  \
    X(P_DEINTERLACE, 4)           \
    X(P_EDGES, 1001)              \
                                  \
    X(P_COLOR_START, 5)           \
    X(P_ALPHA_COMPOSITE, 6)       \
    X(P_R_ADJUST, 7)              \
    X(P_G_ADJUST, 8)              \
    X(P_B_ADJUST, 9)              \
    X(P_COLOR_END, 10)            \
                                  \
    X(P_MOTION_START, 11)         \
    X(P_H_MAX, 12)                \
    X(P_H_FREQ, 13)               \
    X(P_H_AMP, 14)                \
    X(P_H_PHASE, 15)              \
    X(P_V_MAX, 16)                \
    X(P_V_FREQ, 17)               \
    X(P_V_AMP, 18)                \
    X(P_V_PHASE, 19)              \
    X(P_H_ERRATIC, 20)            \
    X(P_V_ERRATIC, 21)            \
    X(P_ZOOM_BASE, 22)            \
    X(P_ZOOM_FACTOR, 23)          \
    X(P_ZOOM_FREQ, 24)            \
    X(P_ZOOM_AMP, 25)             \
    X(P_ZOOM_PHASE, 26)           \
    X(P_ZOOM_ERRATIC, 27)         \
    X(P_ROT_MAX, 28)              \
    X(P_ROT_FREQ, 29)             \
    X(P_ROT_AMP, 30)              \
    X(P_ROT_PHASE, 31)            \
    X(P_CH_SEP, 32)               \
    X(P_CH_FREQ, 33)              \
    X(P_CH_AMP, 34)               \
    X(P_CH_PHASE, 35)             \
    X(P_MOTION_END, 36)           \
                                  \
    X(P_ADVANCED_START, 37)       \
    X(P_ERRATIC_H_ADJ, 38)        \
    X(P_ERRATIC_V_ADJ, 39)        \
    X(P_ERRATIC_ZOOM_ADJ, 40)     \
    X(P_ADVANCED_END, 41)

// Parameter indices, in display order.
enum {
#define TREMOR_PARAM_INDEX(name, id) name,
    TREMOR_PARAMS(TREMOR_PARAM_INDEX)
#undef TREMOR_PARAM_INDEX
    P_NUM_PARAMS
};

// "Render Using" choices.
enum { RENDER_GPU_OR_CPU = 1, RENDER_CPU = 2 };

inline A_long disk_id(int param_index) {
    static constexpr A_long kIds[] = {
#define TREMOR_PARAM_ID(name, id) id,
        TREMOR_PARAMS(TREMOR_PARAM_ID)
#undef TREMOR_PARAM_ID
    };
    static_assert(sizeof(kIds) / sizeof(kIds[0]) == P_NUM_PARAMS, "one disk ID per parameter");
    return kIds[param_index];
}

extern "C" {
DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data,
                            PF_ParamDef* params[], PF_LayerDef* output, void* extra);
}
