# Reception

An After Effects effect that makes a layer look like a television with poor
reception: flickering brightness and contrast, snow, wavy horizontal
distortion, glow, ghost images, a dark rolling bar and lost vertical hold.

- Exact, documented behaviour: [docs/behaviour.md](docs/behaviour.md).
- **GPU rendering**: CUDA and OpenCL on Windows, Metal on macOS, with CPU fallback. The CPU and the GPU run
  the same kernel source (`core/include/reception_kernels.inc`).
- **Multi-frame rendering**: every value is a pure function of the controls and
  the frame; no global state.
- 8, 16 and 32 bpc. Half/quarter resolution previews look like the final render.
- Windows and macOS (Apple Silicon and Intel).

## Controls

| Group | Controls |
| --- | --- |
| — | Render Using, Blend With Original, Random Seed, Edges |
| Color | Tint Color, Brightness Lift, Bright Variance, Contrast, Contrast Variance |
| Noise | Noise Max., Noise Variance, Micro Noise, Noise Width, Noise Tempo |
| Distortion | Distortion Min./Max., Tempo Min./Max., Amplitude 1/2, Frequency 1/2, Speed 1/2, Offset Variance |
| Glow | Glow Size, Glow Intensity, Glow Threshold |
| Ghosting | Ghost Offset 1/2/3, Primary Weight, Ghost 1/2/3 Weight |
| Bars | Bar Height, Bar Ripple Freq., Bar Darkness |
| Vertical Hold | Roll Probability, Roll Length, Roll Speed |

## Building

Reception builds with the rest of the collection: see the [main README](../../README.md#building).

`build/bin/Release/reception_gpu_check` runs the kernels on the local GPU (CUDA
and OpenCL), compares them with the CPU renderer and times a 1080p frame.

## Installing

Windows: `scripts/install.ps1 -AeVersion 2026` (asks for administrator rights).
macOS: `sh scripts/install-mac.sh /path/to/Reception.plugin`.
