# Reception — behaviour

Reception makes a layer look like a television with poor reception: flickering
brightness and contrast, snow, wavy horizontal distortion, glow, ghost images,
a dark rolling bar and a picture that loses vertical hold.

All rows, widths and distances are full-resolution layer pixels, and `f` is the
layer frame number (layer time / frame duration). At reduced preview resolution
every pixel uses the values of the full-resolution row and column it covers, so
previews match the final render. Colours are handled in 8-bit levels (white =
255) on unpremultiplied values; 32 bpc keeps values above white.

`U` is a random value in [0, 1) from the Microsoft C runtime `rand()` sequence
(`(rand() % 10000) / 10000`); "seeded with s" means the sequence starts from `s`.
`H` is the layer height.

## Pipeline

1. **Color** (per row; seeded with *Random Seed* + `f`) — two random walks run
   down the rows, each holding its value for `U·H/8` rows before stepping:
   contrast by `±Contrast Variance`, brightness by `±Bright Variance` (four
   warm-up steps, then the brightness walk is halved). Per row:
   `bright = clamp(walk + Brightness Lift, −200, 200)` levels,
   `contrast = clamp(walk + 2.55·Contrast, −127, 127)`.
   Each pixel: `c = c × Tint Color`, `c = clip(c + bright)`, then the S-curve
   `m = min(c, 255 − c)`, `y = 127·(m/127)^e` with `e = 127/(127 − k)` for
   `k ≥ 0` or `(k + 127)/127` for `k < 0`, `c = c < 128 ? y : 255 − y`.
2. **Noise** — a line-noise amplitude walks down the rows (seeded with seed + `f`,
   after a warm-up over one frame height): it steps by `±Noise Variance` every
   *Noise Tempo*/2 … *Noise Tempo* rows and stays within ±2.55·*Noise Max.*
   Along each row, segments of *Noise Width*/2 pixels (random phase per row)
   take a value in ±amplitude; every pixel adds ±2.55·*Micro Noise* on top.
   The same value is added to R, G and B.
3. **Distortion** (when *Distortion Min.* or *Max.* is above 0) — a strength
   `d` in [Min, Max] % changes every *Tempo Min.* … *Tempo Max.* frames
   (seeded with the seed). Per frame:
   `A1 = |cos(f/(100 − Speed 1) + 50)|·Amplitude 1·d`,
   `A2 = cos((f + 50)/(100 − Speed 2) + 20)·Amplitude 2·d`. Row `y` reads the
   source `shift` pixels to the right (whole pixels):
   `shift = sin(f/12 + 20 + Frequency 1·y/D)·A1 + sin(10000 − f/12 − 90 + Frequency 2·y/D)·A2 + o(y)`,
   `D = 0.1·H + 1`, `o` a walk down the rows stepping by `±Offset Variance`
   (seeded with `f/2`: it changes every two frames). Steps 1–2 are applied to
   the source pixel, so the noise moves with the picture.
4. **Glow** (when *Glow Size*·2 ≥ 1, *Intensity* ≠ 0 and *Threshold* < 100) —
   pixels whose luma `0.3R + 0.59G + 0.11B` is above *Glow Threshold* % are
   tent-blurred (radius 2·*Glow Size*), multiplied by *Glow Intensity* (clipped
   to white) and screened over the picture.
5. **Ghosting** (when a ghost weight is above 0) — three copies offset
   horizontally by `o1, o2, o3` pixels: per frame (seeded with seed + `f`) four
   steps of `o1 += int(2U − 1)`, `o2 += int(10U − 5)`, `o3 += int(20U − 10)`,
   each clamped to ±*Ghost Offset 1/2/3*. Colour =
   `Primary·c + Ghost 1·c(x+o1) + Ghost 2·c(x+o2) + Ghost 3·c(x+o3)` (weights /100),
   clipped; alpha from the primary.
6. **Bars** — a band whose position `v = v0 + y·s` (`s = Bar Ripple Freq. + 0.0001`,
   one full cycle `L = H·s`, band length `len = L·Bar Height/100`) restarts at
   `−L` past `len`; `v0` advances by `s` per step, `0.05·H` steps per frame.
   Inside the band (`v ≥ 0`) colours are multiplied by
   `clamp(1 − (v/len·Bar Darkness/100 + 0.3·sin(v/19 + f/40)), 0, 1)`.
7. **Vertical Hold** (when *Roll Speed* and *Roll Probability* are above 0) —
   frame by frame (seeded with the seed): when a hold period ends the picture
   snaps back; with *Roll Probability* % the next period rolls at `U·Roll Speed`
   rows per frame for `U·Roll Length` frames, otherwise it stays still for
   `(1.5U + 0.5)·Roll Length` frames. Output row `y` shows row `y + roll`
   (wrapping around).
8. **Blend With Original** — cross-fade back to the untouched source.

*Edges* (Transparent, Mirror, Tile, Black) decides what the distortion and the
ghosts show past the left and right sides of the layer. Every value is a pure
function of the controls and the frame: frames render in any order and in
parallel, and the same frame always renders the same.

## Differences from the plugin it replaces

Kept: the controls, their ranges and defaults, and the pipeline above.
Changed on purpose:

- *Blend With Original* is a real cross-fade (the original added the source on
  top of the effect); default 0% instead of 25%.
- After the warm-up the contrast walk used *Bright Variance*; it now uses
  *Contrast Variance*.
- The distortion strength now follows *Tempo Min./Max.* over time (the
  original fixed it per seed, so the tempo controls did nothing); swapped
  Min/Max values are handled.
- Rows, widths, offsets, tempos, the bar and the roll are measured on the
  full-resolution layer and from its top row (the original used the current
  preview resolution and the visible region, so previews differed).
- The line-noise segments are a fixed length (half of *Noise Width*, the
  original's average) with a random phase per row, so any pixel can be computed
  on its own; the original drew them one after another along the row.
- The distortion and ghosts read past the sides according to *Edges* (the
  original ran into the neighbouring row); the glow keeps transparent areas
  clear (the original grew a halo around every opaque pixel).
- 16/32 bpc, GPU (CUDA, OpenCL, Metal), multi-frame rendering (no global lock
  or global random state), macOS.
