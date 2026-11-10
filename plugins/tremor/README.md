# Tremor

An After Effects effect that makes a layer look shot on an unsteady camera:
sway, zoom pulses, small turns with color fringing, and random jumps.

- Exact, documented behaviour: [docs/behaviour.md](docs/behaviour.md).
- **GPU rendering**: CUDA and OpenCL on Windows, Metal on macOS, with CPU fallback.
- **Multi-frame rendering**: stateless and thread-safe.
- 8, 16 and 32 bpc.

## Controls

| Group | Controls |
| --- | --- |
| — | Render Using, Blend With Original, Random Seed, Deinterlace, Edges |
| Color Control | Enable Alpha Composite, R/G/B Adjust |
| Motion Control | Horizontal/Vertical Max, Freq., Amp., Phase; (H)/(V) Erratic; Zoom Base, Factor, Freq., Amp., Phase, Erratic; Rotation Max, Freq., Amp., Phase; Channel Separate, Freq., Amp., Phase |
| Advanced | Erratic(H), Erratic(V), Erratic(Zoom) Adj. |

## Building

Tremor builds with the rest of the collection: see the [main README](../../README.md#building).

`build/bin/Release/gpu_check` runs the GPU kernels on the local GPU (CUDA and
OpenCL) and compares them with the CPU renderer. `render_frames <dir>` writes
frames of a test pattern as PPM images.

## Installing

Windows: `scripts/install.ps1 -AeVersion 2026` (asks for administrator rights).
macOS: `sh scripts/install-mac.sh /path/to/Tremor.plugin`.
