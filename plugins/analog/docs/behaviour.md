# Analog — behaviour

Analog treats every row of the layer as one line of an NTSC composite signal
sampled at 4× the color subcarrier (14.318 MHz, four samples per subcarrier
cycle). One full-resolution pixel is one sample. At reduced preview resolution
each row is first resampled to its full-resolution width, processed, then
averaged back, so previews match the final render.

Signal values are kept in 8-bit-style units (white = 255) while a line is
processed; 32 bpc keeps over-range values, 8/16 bpc clip.

## Pipeline

For every frame, in this order:

1. **Roll** — the picture slides down by *Roll Speed* full-resolution pixels per
   second (wrapping around).
2. **Line jitter** (*TV Distortion*) — each line shifts horizontally, wrapping
   around, by `r1·T + sign·0.1·T + r2·F + sign·0.1·F` whole pixels, where
   `T = Thick/100 · width`, `F = Fine/10 · 2.5/100 · width` and `r1`, `r2` are
   random in [-1, 1). With *Distortion Speed* 0 the values change every frame;
   otherwise they drift smoothly, *Distortion Speed*/10 new values per second.
3. **RGB → YIQ**: `Y = .30R + .59G + .11B`, `I = .74(R−Y) − .27(B−Y)`,
   `Q = .48(R−Y) + .41(B−Y)`.
4. **Low-Pass Filter In** — I through three one-pole low-passes at 1.3 MHz
   (moved back 2 samples), Q at 600 kHz (moved back 4).
5. **Modulation** — `composite = Y + (I·cos + Q·sin) · Strength/50`, with the
   subcarrier at four samples per cycle. The cycle's starting phase per line
   follows *Phase Shift* (0: constant, 90: +line/2, 180: +(line & 2),
   270: −line/2) plus *Phase Shift Offset* quarter cycles.
6. **Signal Amplification** — pre-emphasis:
   `composite += highpass(composite) · Amplification`, corner
   `315 MHz / Cutoff Filter` (88 → 3.58 MHz).
7. **Luma Noise** — low-frequency noise added to the composite: per sample
   `walk ← (walk + u·n)/2`, `u` random in [-1, 1), `n = Luma Noise·1000/256`.
8. **Demodulation** — luma = composite through a `[1 2 2 2 1]/8` notch (zero at
   the subcarrier); chroma = the rest, holding I on even and Q on odd phases;
   the missing samples are the mean of their neighbours; both divided by
   Strength/50 (no chroma at all when Strength is 0).
9. **Luma Modulation** (FM, when enabled) — the tape records luma as frequency:
   `phase += Frequency + map(Y)` with `map` taking Y from [−195, 391] to
   [−Phase, +Phase]; the carrier `s = sin(phase)` is optionally quantized to
   *Quantize* levels; the detector output is `|s − s_prev|` through three
   one-pole low-passes, mapped back with
   `Y = ((2(d − Frequency·Lightness) + Phase)/(2·Phase))·586 − 195`.
   With the defaults this is close to identity plus the FM ringing and moiré.
   **Tape Errors**: on lines between *Start* and *End* (fractions of the
   height), each sample has a chance *Probability* to become a spike of height
   *Gamma*·random, weighted by `(sin(t·4.71 − 1.57) + 1)/2 · Fade` across the band.
10. **Chroma Noise** — like Luma Noise on I and Q, `n = Chroma Noise·3750/256`.
11. **Hue Noise** — each line's I/Q is rotated by `walk·3.14/100` radians, where
    `walk` halves and adds a random value in [−Hue Noise, Hue Noise] per line
    down the picture (24 lines of history).
12. **Emulating VHS** (SP / LP / EP): luma through three low-passes at
    2.4 / 1.9 / 1.4 MHz plus `1.6 × highpass` at a third of that; chroma through
    three low-passes at 320 / 300 / 280 kHz, delayed 9 / 12 / 14 samples.
13. **Chroma Vertical Blend** — I and Q averaged with the line above.
14. **Sharpen** (with *Sharpen On/Off*) — `Y += 2·(Y − lowpass₃(Y))·Sharpen`,
    corner 4× the VHS luma corner.
15. **S-Video Out** off — the signal is modulated and demodulated once more
    (dot crawl, rainbowing).
16. **Chroma Loss** — each line loses its color with that probability.
17. **Device-to-Device Connections** — that many extra modulate/demodulate passes.
18. **Low-Pass Filter Out** — Lite: I and Q through three low-passes at 2.6 MHz;
    Strong: the same filter as step 4.
19. **YIQ → RGB**: `R = Y + .956I + .621Q`, `G = Y − .272I − .647Q`,
    `B = Y − 1.106I + 1.703Q`. **Destruction** wraps values like an 8-bit
    overflow instead of clipping. Alpha is the source alpha at the same place;
    color never exceeds alpha (premultiplied).
20. **Scanlines Blur** (> 1) — the frame is area-averaged down to
    `width/Blur × height/Blur` (full-resolution terms) and scaled back up bilinearly.

One-pole low-pass coefficient: `a = (1/rate) / (1/(2π·fc) + 1/rate)`; filters
start each line at the line's first value.

## Differences from the plugin it replaces

Kept: the controls, their ranges and defaults, and the signal chain above.
Changed on purpose:

- Stable and thread-safe: no global state, no re-checkout of the input layer,
  balanced parameter checkouts, all buffer accesses bounded (the original
  crashed next to effects such as ReelSmart Motion Blur).
- Deterministic noise: the same frame always renders the same (the original
  advanced one global random generator per render).
- *Roll Speed* uses continuous time (the original moved once per second);
  *Distortion Speed* (unused in the original) makes the jitter drift smoothly.
- *Emulating VHS* gains **Off**; *Phase Shift* is a popup of the four values that
  have an effect; noise amounts are no longer rounded to whole numbers.
- The luma/chroma separation is a symmetric notch (the original was shifted by
  half a sample); line edges no longer start from black.
- 16/32 bpc, GPU (CUDA, OpenCL, Metal), multi-frame rendering, macOS,
  resolution-independent previews.
