# Analog

An After Effects effect that sends a layer through a simulated composite video
signal and a VHS tape: chroma bleed and delay, dot crawl, frequency-modulated
luma with its ringing, noise, color loss, tape errors, line jitter and roll.

- Exact, documented behaviour: [docs/behaviour.md](docs/behaviour.md).
- **GPU rendering**: CUDA and OpenCL on Windows, Metal on macOS, with CPU fallback. The CPU and the GPU run
  the same kernel source (`core/include/analog_kernels.inc`).
- **Multi-frame rendering**: stateless and thread-safe; every random value comes
  from a hash of seed, frame, line and sample.
- 8, 16 and 32 bpc. Half/quarter resolution previews look like the final render.
- Windows and macOS (Apple Silicon and Intel).

## Controls

| Group | Controls |
| --- | --- |
| — | Render Using, Random Seed, Signal Strength, Signal Amplification, Cutoff Filter, Luma/Chroma/Hue Noise, Chroma Loss, Sharpen, Device-to-Device Connections, Emulating VHS, Destruction |
| Luma Modulation | Enable, Frequency, Phase, Lightness, Quantize, Tape Errors (Start, End, Probability, Fade, Gamma) |
| TV Distortion | Thick Distortion, Fine Distortion, Distortion Speed, Roll Speed |
| Experimental | Low-Pass Filter In/Out, Phase Shift, Phase Shift Offset, S-Video Out, Chroma Vertical Blend, Sharpen On/Off, Scanlines Blur |

## Building

Analog builds with the rest of the collection: see the [main README](../../README.md#building).

`build/bin/Release/analog_gpu_check` runs the kernels on the local GPU (CUDA and
OpenCL), compares them with the CPU renderer and times a 1080p frame.
`analog_render_frames <dir> [input.ppm] [frames]` writes frames as PPM images.

## Installing

Windows: `scripts/install.ps1 -AeVersion 2026` (asks for administrator rights).
macOS: `sh scripts/install-mac.sh /path/to/Analog.plugin`.
