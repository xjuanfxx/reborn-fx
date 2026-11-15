# Crosstalk

An After Effects effect that splits a layer into alternating bands of rows —
even and odd "fields" — and gives each its own interference: tint, brightness,
noise, echo, horizontal displacement and a column comb.

- Exact, documented behaviour: [docs/behaviour.md](docs/behaviour.md).
- **GPU rendering**: CUDA and OpenCL on Windows, Metal on macOS, with CPU fallback. The CPU and the GPU run
  the same kernel source (`core/include/crosstalk_kernels.inc`).
- **Multi-frame rendering**: stateless and thread-safe; every random value comes
  from a hash of seed, frame, row and column.
- 8, 16 and 32 bpc. Half/quarter resolution previews look like the final render.
- Windows and macOS (Apple Silicon and Intel).

## Controls

| Group | Controls |
| --- | --- |
| — | Render Using, Blend With Original, Random Seed, Edges |
| Fields Setup | (E)/(O) Size, Process Input, Luminance, Tint, Tint Color |
| Noise | (E)/(O) Noise Level, Chroma Noise, Decay Amount, Smear |
| Modulation | (E)/(O) Distortion Method, Horiz. Adj.; Influence; (E)/(O) Comb Filt. Level |

## Building

Crosstalk builds with the rest of the collection: see the [main README](../../README.md#building).

`build/bin/Release/crosstalk_gpu_check` runs the kernels on the local GPU (CUDA
and OpenCL), compares them with the CPU renderer and times a 1080p frame.

## Installing

Windows: `scripts/install.ps1 -AeVersion 2026` (asks for administrator rights).
macOS: `sh scripts/install-mac.sh /path/to/Crosstalk.plugin`.
