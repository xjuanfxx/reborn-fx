# reborn-fx

Classic After Effects plugins, reborn. Effects whose makers are gone, or that
stopped evolving, rebuilt from scratch with new code and a modern feature set:

- GPU rendering (CUDA and OpenCL on Windows, Metal on macOS) with CPU fallback
- Multi-frame rendering (stateless, thread-safe)
- 8, 16 and 32 bits per channel
- Windows and macOS (Apple Silicon and Intel)

## Plugins

| Plugin | What it does | Docs |
| --- | --- | --- |
| [Tremor](plugins/tremor) | Unsteady camera: sway, breathing zoom, small turns with RGB fringing and random jolts. | [behaviour](plugins/tremor/docs/behaviour.md) |

## Downloads

Each plugin is released on its own: see [Releases](../../releases)
(tags `<plugin>-v<version>`, e.g. `tremor-v0.1.0`), with a Windows `.aex` and a
macOS `.plugin` per release.

## Building

Requirements: CMake 3.20+, Visual Studio 2022 (Windows) or Xcode (macOS), and
the Adobe After Effects SDK. Optional: nvcc 12.8 for CUDA kernels (without it,
CUDA projects render on the CPU; OpenCL and Metal need nothing extra).

The SDK is not redistributable, so it is not in the repository. Point CMake at
its `Examples` folder (or copy `Headers/`, `Util/` and `Resources/` into
`third_party/ae_sdk`):

```bash
cmake -S . -B build -DAE_SDK_DIR=/path/to/AfterEffectsSDK/Examples
```

```bash
cmake --build build --config Release
```

```bash
ctest --test-dir build -C Release
```

Plugins and tools land in `build/bin` (`build/bin/Release` with Visual Studio).
Without the SDK, only the plugin cores and their tests are built.

## Layout

Each plugin lives in `plugins/<name>/` and is self-contained: `core/` (renderer,
independent of After Effects), `plugin/` (the After Effects adapter and GPU
kernels), `tests/`, `tools/`, `scripts/` (installers) and `docs/behaviour.md`.
