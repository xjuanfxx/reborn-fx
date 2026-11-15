# Crosstalk — behaviour

Crosstalk splits the layer into horizontal bands of rows that alternate between
two "fields", **even (E)** and **odd (O)**, each with its own settings, like the
two fields of an interlaced signal that pick up different interference.

All rows, columns and distances are full-resolution layer pixels. At half or
quarter preview resolution every pixel uses the field, displacement and noise of
the full-resolution pixel it covers, so previews match the final render.
Colour values below are 0–1 (8 bpc: /255); 32 bpc keeps values above 1.

## Per pixel

1. **Field** — rows `0 … (E) Size−1` are even, the next *(O) Size* rows odd, and
   so on from the top of the layer.
2. **Displacement** — the row is shifted horizontally (whole pixels) by its
   field's *Distortion Method*, with `adj` = *Horiz. Adj.*, `I` = *Influence*,
   `W`/`H` = layer width/height, `t = row/H`, `r = int(t·4096)`:
   - Shift: `−adj·W/100 · I/1000`
   - Linear: `t · (−adj/100 · W) · I/100`
   - Noisy: `int(u · int(adj/100·W/4)) · I/1000`, `u` random in [0, 1) per row and frame
   - Sawtooth: `((r + int(9·I)) mod max(1, int(1 + I/1000)) − 10) · (−adj/(W/10))`
   - Wave: `sin((r + I)/(H/40)) · (−adj/400 · W)`
   - Wave Motion: `sin(r / ((H·I + 0.0001)/100)) · (−adj/200 · W)`

   A positive value moves the picture to the left. Where the shifted row reaches
   past the layer, *Edges* decides what shows: Transparent, Mirror, Tile (default)
   or Black.
3. **Tint** (when on) — `L = (R+G+B)/3`, tinted colour = `L × Tint Color`.
4. **Process Input** — `v = (1 − p)·source + p·tinted`, `p` = *Process Input*/100.
5. **Luminance and noise** — `v += Luminance/100 + n`, `n` random in
   ±*Noise Level*/100, one value per pixel, or one per channel with *Chroma Noise*.
   Above 50.2% noise the picture is replaced: `v = Luminance/100 + 2n`.
6. **Echo** (*Decay Amount* > 1) — for `k = 0 … Decay−1`, `e_k` the source pixel
   `k` pixels further along the displaced row:
   - Smear on: `v += (e_k − 0.5)/(k + 2)` (with `e_0` = the value from step 4)
   - Smear off: `v += (e_k − 0.5 + m_k)·(1 − k/Decay)`, `m_k` random in ±*Noise Level*/200

   The signal is centred on mid grey, so the echo also raises contrast.
7. Clip to [0, 1] (8/16 bpc) or [0, ∞) (32 bpc); alpha is the source alpha.
8. **Comb Filt. Level** — a column is blanked (transparent) when
   `column AND level` is not 0 (1: every other column, 2: two of every four, …).
   At reduced resolution a pixel keeps the share of its columns that survive.
9. **Blend With Original** — cross-fade back to the untouched source.

Noise and Noisy displacement change every frame and with *Random Seed*; the
same frame always renders the same.

## Differences from the plugin it replaces

Kept: the controls, their ranges and the formulas above. Changed on purpose:

- *Blend With Original* is a real cross-fade (the original added the source on
  top of the effect, doubling brightness at 0%); default 0% instead of 25%.
- *Luminance* is a plain percentage (the original applied the ×2.55 scale twice);
  the (E) default is −99 so the default look stays the same.
- *Process Input* mixes once (the original added the source a second time,
  brightening the picture).
- The noise is animated (the original used only the seed: frozen noise).
- Bands are exactly *Size* rows (the original made them one row taller).
- The echo reads the same channel of the next pixels (the original read the
  neighbouring bytes, i.e. other channels).
- Each row uses its own field's displacement method (the original computed the
  displacement from uninitialized values); *Wave Motion* works (an integer
  division made it a no-op).
- No unprocessed strip at the right edge, no stale pixels where a row is
  shifted out, pure white handled (a lookup table missed index 255).
- Sizes, distances and the comb no longer depend on the preview resolution.
- New *Edges* control; 16/32 bpc, GPU (CUDA, OpenCL, Metal), multi-frame
  rendering, macOS.
