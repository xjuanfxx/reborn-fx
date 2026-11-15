// Reception.h — declarations of the After Effects effect.
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

#define RECEPTION_NAME "Reception"
#define RECEPTION_MATCH_NAME "XJFX Reception"
#define RECEPTION_CATEGORY "Reception"
#define RECEPTION_SUPPORT_URL "https://github.com/"

#define RECEPTION_MAJOR_VERSION 0
#define RECEPTION_MINOR_VERSION 1
#define RECEPTION_BUG_VERSION 0

// Must match AE_Effect_Global_OutFlags(_2) in ReceptionPiPL.r.
#define RECEPTION_OUT_FLAGS (PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_DEEP_COLOR_AWARE)
#define RECEPTION_OUT_FLAGS2                                                                         \
    (PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |                        \
     PF_OutFlag2_SUPPORTS_THREADED_RENDERING | PF_OutFlag2_SUPPORTS_GPU_RENDER_F32)

// Every parameter as X(index name, disk ID), in the order ParamsSetup adds them
// (the order shown in the Effect Controls panel).
//
// Disk IDs identify each parameter in saved projects: never change or reuse one
// once released. They follow the parameter indices of the plugin this effect
// replaces, so its controls keep their identity. New parameters use 1000+.
#define RECEPTION_PARAMS(X)     \
    X(P_INPUT, 0)               \
    X(P_RENDER_USING, 1000)     \
    X(P_BLEND, 2)               \
    X(P_SEED, 3)                \
    X(P_EDGES, 1001)            \
                                \
    X(P_COLOR_START, 4)         \
    X(P_TINT, 5)                \
    X(P_LIFT, 6)                \
    X(P_BRIGHT_VAR, 7)          \
    X(P_CONTRAST, 8)            \
    X(P_CONTRAST_VAR, 9)        \
    X(P_COLOR_END, 10)          \
                                \
    X(P_NOISE_START, 11)        \
    X(P_NOISE_MAX, 12)          \
    X(P_NOISE_VAR, 13)          \
    X(P_MICRO, 14)              \
    X(P_NOISE_WIDTH, 15)        \
    X(P_NOISE_TEMPO, 16)        \
    X(P_NOISE_END, 17)          \
                                \
    X(P_DIST_START, 18)         \
    X(P_DIST_MIN, 19)           \
    X(P_DIST_MAX, 20)           \
    X(P_TEMPO_MIN, 21)          \
    X(P_TEMPO_MAX, 22)          \
    X(P_AMP1, 23)               \
    X(P_AMP2, 24)               \
    X(P_FREQ1, 25)              \
    X(P_FREQ2, 26)              \
    X(P_SPEED1, 27)             \
    X(P_SPEED2, 28)             \
    X(P_OFFSET_VAR, 29)         \
    X(P_DIST_END, 30)           \
                                \
    X(P_GLOW_START, 31)         \
    X(P_GLOW_SIZE, 32)          \
    X(P_GLOW_INTENSITY, 33)     \
    X(P_GLOW_THRESHOLD, 34)     \
    X(P_GLOW_END, 35)           \
                                \
    X(P_GHOST_START, 36)        \
    X(P_GHOST_OFF1, 37)         \
    X(P_GHOST_OFF2, 38)         \
    X(P_GHOST_OFF3, 39)         \
    X(P_W_PRIMARY, 40)          \
    X(P_W_GHOST1, 41)           \
    X(P_W_GHOST2, 42)           \
    X(P_W_GHOST3, 43)           \
    X(P_GHOST_END, 44)          \
                                \
    X(P_BARS_START, 45)         \
    X(P_BAR_HEIGHT, 46)         \
    X(P_BAR_RIPPLE, 47)         \
    X(P_BAR_DARK, 48)           \
    X(P_BARS_END, 49)           \
                                \
    X(P_HOLD_START, 50)         \
    X(P_ROLL_PROB, 51)          \
    X(P_ROLL_LENGTH, 52)        \
    X(P_ROLL_SPEED, 53)         \
    X(P_HOLD_END, 54)

// Parameter indices, in display order.
enum {
#define RECEPTION_PARAM_INDEX(name, id) name,
    RECEPTION_PARAMS(RECEPTION_PARAM_INDEX)
#undef RECEPTION_PARAM_INDEX
    P_NUM_PARAMS
};

// "Render Using" choices.
enum { RENDER_GPU_OR_CPU = 1, RENDER_CPU = 2 };

inline A_long disk_id(int param_index) {
    static constexpr A_long kIds[] = {
#define RECEPTION_PARAM_ID(name, id) id,
        RECEPTION_PARAMS(RECEPTION_PARAM_ID)
#undef RECEPTION_PARAM_ID
    };
    static_assert(sizeof(kIds) / sizeof(kIds[0]) == P_NUM_PARAMS, "one disk ID per parameter");
    return kIds[param_index];
}

extern "C" {
DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data,
                            PF_ParamDef* params[], PF_LayerDef* output, void* extra);
}
