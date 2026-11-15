// Crosstalk.h — declarations of the After Effects effect.
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

#define CROSSTALK_NAME "Crosstalk"
#define CROSSTALK_MATCH_NAME "XJFX Crosstalk"
#define CROSSTALK_CATEGORY "Crosstalk"
#define CROSSTALK_SUPPORT_URL "https://github.com/"

#define CROSSTALK_MAJOR_VERSION 0
#define CROSSTALK_MINOR_VERSION 1
#define CROSSTALK_BUG_VERSION 0

// Must match AE_Effect_Global_OutFlags(_2) in CrosstalkPiPL.r.
#define CROSSTALK_OUT_FLAGS (PF_OutFlag_NON_PARAM_VARY | PF_OutFlag_DEEP_COLOR_AWARE)
#define CROSSTALK_OUT_FLAGS2                                                                         \
    (PF_OutFlag2_SUPPORTS_SMART_RENDER | PF_OutFlag2_FLOAT_COLOR_AWARE |                        \
     PF_OutFlag2_SUPPORTS_THREADED_RENDERING | PF_OutFlag2_SUPPORTS_GPU_RENDER_F32)

// Every parameter as X(index name, disk ID), in the order ParamsSetup adds them
// (the order shown in the Effect Controls panel).
//
// Disk IDs identify each parameter in saved projects: never change or reuse one
// once released. They follow the parameter indices of the plugin this effect
// replaces, so its controls keep their identity. New parameters use 1000+.
#define CROSSTALK_PARAMS(X)         X(P_INPUT, 0)                   X(P_RENDER_USING, 1000)         X(P_BLEND, 2)                   X(P_SEED, 3)                    X(P_EDGES, 1001)                                                X(P_FIELDS_START, 4)            X(P_E_SIZE, 5)                  X(P_O_SIZE, 6)                  X(P_E_PROCESS, 7)               X(P_O_PROCESS, 8)               X(P_E_LUM, 9)                   X(P_O_LUM, 10)                  X(P_E_TINT, 11)                 X(P_O_TINT, 12)                 X(P_E_TINT_COLOR, 13)           X(P_O_TINT_COLOR, 14)           X(P_FIELDS_END, 15)                                             X(P_NOISE_START, 16)            X(P_E_NOISE, 17)                X(P_O_NOISE, 18)                X(P_E_CHROMA, 19)               X(P_O_CHROMA, 20)               X(P_E_DECAY, 21)                X(P_O_DECAY, 22)                X(P_E_SMEAR, 23)                X(P_O_SMEAR, 24)                X(P_NOISE_END, 25)                                              X(P_MOD_START, 26)              X(P_E_METHOD, 27)               X(P_O_METHOD, 28)               X(P_E_HADJ, 29)                 X(P_O_HADJ, 30)                 X(P_INFLUENCE, 31)              X(P_E_COMB, 32)                 X(P_O_COMB, 33)                 X(P_MOD_END, 34)

// Parameter indices, in display order.
enum {
#define CROSSTALK_PARAM_INDEX(name, id) name,
    CROSSTALK_PARAMS(CROSSTALK_PARAM_INDEX)
#undef CROSSTALK_PARAM_INDEX
    P_NUM_PARAMS
};

// "Render Using" choices.
enum { RENDER_GPU_OR_CPU = 1, RENDER_CPU = 2 };

inline A_long disk_id(int param_index) {
    static constexpr A_long kIds[] = {
#define CROSSTALK_PARAM_ID(name, id) id,
        CROSSTALK_PARAMS(CROSSTALK_PARAM_ID)
#undef CROSSTALK_PARAM_ID
    };
    static_assert(sizeof(kIds) / sizeof(kIds[0]) == P_NUM_PARAMS, "one disk ID per parameter");
    return kIds[param_index];
}

extern "C" {
DllExport PF_Err EffectMain(PF_Cmd cmd, PF_InData* in_data, PF_OutData* out_data,
                            PF_ParamDef* params[], PF_LayerDef* output, void* extra);
}
