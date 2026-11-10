# Tremor — behaviour

Tremor simulates an unsteady camera: the layer sways, zooms and turns around its
center, the color channels fringe apart as they turn, and on some frames the
picture jumps. Everything is a function of the parameters and the layer time, so
frames render in any order and in parallel.

## Time

`ω = 2π · t`, with `t` the layer time in seconds. Every motion is a cosine wave
`cos(Freq · ω + Phase)`, with Freq in cycles per second and Phase in degrees.

## Motion (per frame)

| Motion | Value |
| --- | --- |
| Horizontal shift, px | `wave(H) · Horiz. Amp. · Horizontal Max/100 · width` |
| Vertical shift, px | `wave(V) · Vert. Amp. · Vertical Max/100 · height` |
| Zoom `z` (magnification `1 + z`) | `Zoom Base + wave(Zoom) · Zoom Factor · Zoom Amp. · 0.1` |
| Rotation, degrees | `wave(Rot) · Rotation Amp. · Rotation Max` |
| Channel separation, degrees | `wave(Ch) · Ch.Separation Amp. · Channel Separate` |

Width and height are those of the buffer being rendered, so the motion keeps its
size relative to the frame at any preview resolution.

### Erratic jumps

Each frame draws from a pseudo-random sequence seeded with `Random Seed + time`
(the raw After Effects time value). The sequence is the Microsoft C runtime
`rand()`, so a seed gives the same jumps on every platform. In order:

1. With probability `(V) Erratic`%, the vertical shift is replaced by a random
   value in `±height · Erratic(V) Adj./100`.
2. With probability `(H) Erratic`%, the horizontal shift is replaced by a random
   value in `±width · Erratic(H) Adj./100`.
3. With probability `Zoom Erratic`%, the zoom becomes `z · r · Erratic(Zoom) Adj./100`,
   `r` random in `[0, 10)`.

## Rendering

Pixels are premultiplied; pixel centers sit at integer coordinates and the
center of the layer is `((w − 1)/2, (h − 1)/2)`.

1. **Deinterlace** (color channels only, always reading the original rows):
   - *Lower Only*: even rows are rebuilt as the average of the rows above and below.
   - *Upper Only*: odd rows are rebuilt the same way.
   - *Merge*: every row becomes `¼ above + ½ row + ¼ below`.
2. **Rotate**: around the center, red turns by the rotation, green by
   rotation + separation, blue by rotation + 2 · separation. Alpha is the mean of
   the three samples.
3. **Zoom and shift**: output pixel `p` shows the rotated picture at
   `c + (p − c) / (1 + z) − shift`.
4. **Edges**: what shows where the moved picture leaves the layer —
   *Transparent*, *Mirror*, *Tile* or *Black* (opaque).
5. **Color**: `R/G/B Adjust`% is added to the moved picture's channels, scaled by
   its alpha so transparent areas stay clear.
6. **Enable Alpha Composite**: the moved picture is composited over the source,
   so the source shows wherever the moved picture is transparent.
7. **Blend With Original**: cross-fade between the result (0%) and the source (100%).

Sampling is bilinear. At 8 and 16 bpc results clip at white; at 32 bpc color
values above 1 are kept.
