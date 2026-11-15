// Analog.h — declarations of the After Effects effect.
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

#define ANALOG_NAME "Analog"
#define ANALOG_MATCH_NAME "XJFX Analog"
#define ANALOG_CATEGORY "Analog"
#define ANALOG_SUPPORT_URL "https://github.com/"

#define ANALOG_MAJOR_VERSION 0
#define ANALOG_MINOR_VERSION 1
#define ANALOG_BUG_VERSION 0

// Must match AE_Effect_Global_OutFlags(_2) in AnalogPiPL.r.
#define ANALOG_OUT_FLAGS (PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_DEEP_COLOR_AWARE)
#define ANALOG_OUT_FLAGS2                                                                         \
    (PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |                        \
     PF_OutFlag2_SUPPORTS_THREADED_RENDERING | PF_OutFlag2_SUPPORTS_GPU_RENDER_F32)

// Every parameter as X(index name, disk ID), in the order ParamsSetup adds them
// (the order shown in the Effect Controls panel).
//
// Disk IDs identify each parameter in saved projects: never change or reuse one
// once released. New parameters use 1000+.
#define ANALOG_PARAMS(X)        \
    X(P_INPUT, 0)               \
    X(P_RENDER_USING, 1000)     \
    X(P_SEED, 1)                \
    X(P_STRENGTH, 2)            \
    X(P_AMPLIFICATION, 3)       \
    X(P_CUTOFF, 4)              \
    X(P_LUMA_NOISE, 5)          \
    X(P_CHROMA_NOISE, 6)        \
    X(P_HUE_NOISE, 7)           \
    X(P_CHROMA_LOSS, 8)         \
    X(P_SHARPEN, 9)             \
    X(P_GENERATIONS, 10)        \
    X(P_VHS, 11)                \
    X(P_DESTRUCTION, 12)        \
                                \
    X(P_FM_START, 13)           \
    X(P_FM_ON, 14)              \
    X(P_FM_FREQ, 15)            \
    X(P_FM_PHASE, 16)           \
    X(P_FM_LIGHT, 17)           \
    X(P_FM_QUANTIZE, 18)        \
    X(P_TAPE_ERRORS, 19)        \
    X(P_TE_START, 20)           \
    X(P_TE_END, 21)             \
    X(P_TE_PROB, 22)            \
    X(P_TE_FADE, 23)            \
    X(P_TE_GAMMA, 24)           \
    X(P_FM_END, 25)             \
                                \
    X(P_TV_START, 26)           \
    X(P_THICK, 27)              \
    X(P_FINE, 28)               \
    X(P_DIST_SPEED, 29)         \
    X(P_ROLL, 30)               \
    X(P_TV_END, 31)             \
                                \
    X(P_EXP_START, 32)          \
    X(P_LP_IN, 33)              \
    X(P_LP_OUT, 34)             \
    X(P_PHASE_SHIFT, 35)        \
    X(P_PHASE_OFFSET, 36)       \
    X(P_SVIDEO, 37)             \
    X(P_VBLEND, 38)             \
    X(P_SHARPEN_ON, 39)         \
    X(P_SCAN_BLUR, 40)          \
    X(P_EXP_END, 41)

// Parameter indices, in display order.
enum {
#define ANALOG_PARAM_INDEX(name, id) name,
    ANALOG_PARAMS(ANALOG_PARAM_INDEX)
#undef ANALOG_PARAM_INDEX
    P_NUM_PARAMS
};

// "Render Using" choices.
enum { RENDER_GPU_OR_CPU = 1, RENDER_CPU = 2 };

inline A_long disk_id(int param_index) {
    static constexpr A_long kIds[] = {
#define ANALOG_PARAM_ID(name, id) id,
        ANALOG_PARAMS(ANALOG_PARAM_ID)
#undef ANALOG_PARAM_ID
    };
    static_assert(sizeof(kIds) / sizeof(kIds[0]) == P_NUM_PARAMS, "one disk ID per parameter");
    return kIds[param_index];
}

extern "C" {
DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data,
                            PF_ParamDef* params[], PF_LayerDef* output, void* extra);
}
