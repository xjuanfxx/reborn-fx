# Smear — behaviour

Smear adds motion blur to footage that has none (or not enough). It measures
how every pixel moves between frames (optical flow) and blurs each frame along
that motion, as a camera with an open shutter would have recorded it.

Its main design goal is **no ghosting**: background never smears over a
foreground object, objects do not leave copies of themselves at their edges,
and the blur of a moving object still spills over whatever is behind it.

Everything is a function of three frames (previous, current, next) and the
controls, so frames render in any order and in parallel.

## Controls

| Control | Range | What it does |
| --- | --- | --- |
| **Blur Amount** | 0–2, default 0.5 | Fraction of a frame the virtual shutter stays open, centred on the frame: 0.5 is a 180° shutter, 1 is 360°. A pixel moving `v` pixels per frame is smeared over `v × Blur Amount` pixels. 0 turns the blur off. |
| **Track** | Previous / Next / Both | Which neighbour frames measure the motion. *Both* (default) is the most reliable: it uses the two flows to find occlusions. *Previous* or *Next* use one neighbour only (useful next to a cut); occlusions are then found from colour mismatches alone. |
| **Max Samples** | 4–64, default 32 | Colour samples per pixel along the streaks. Fewer is faster; long streaks with few samples look stepped. Static areas take no samples at all. |
| **Motion Sensitivity** | 0–100, default 50 | Detail of the motion analysis. Higher follows smaller details and sharper motion boundaries (6-pixel matching patches at 100); lower gives smoother, steadier vectors (12-pixel patches at 0), better on noisy or soft footage. |
| **Quality** | Draft / Normal / Best | Resolution of the motion analysis: quarter, half or full. Draft is fastest; Best catches the finest motion. All three produce full-resolution vectors (see *Flow*). |
| **Show Vectors** | on / off | Debug view of the motion instead of the blur (see *Vector view*). |

The first and last frames of a layer have only one neighbour; Smear then
tracks with the one it has. Without any neighbour the frame passes through.

## How it works

### 1. Flow

For the current frame, Smear computes the optical flow to the previous and to
the next frame: for each pixel, where it went. The method is coarse-to-fine
*dense inverse search* (Kroeger et al., 2016), chosen because every step is a
small fixed-size job per patch or per pixel, which suits GPUs, and it is
accurate and much faster than global variational methods.

1. Both frames are turned into brightness and reduced into a pyramid of
   half-size levels.
2. From the smallest level up, the current frame is cut into overlapping square
   patches. Each patch looks for itself in the other frame:
   - it starts from the motion found on the coarser level, trying the value at
     its centre and one patch away in eight directions (so a patch on an
     object's edge can pick the side it belongs to);
   - it tries every whole-pixel shift within a few pixels of that (much wider
     on the two smallest levels), to catch motions too fast for the coarser
     levels;
   - it then refines the shift to a fraction of a pixel with a few
     Gauss-Newton steps, keeping the best match seen. Patches are compared
     after removing their average brightness, so small exposure changes do
     not matter.
   - On the last level searched, patches that still match poorly get a wide
     search: thin, fast objects (a pole, a racket) disappear from the small
     levels, so nothing coarser could have found their motion.
3. Each pixel then averages the motions of the patches that cover it, weighted
   by how well each motion explains *that pixel*. A patch straddling an
   object's edge only lends its motion to the side it matches.
4. Draft and Normal stop at quarter or half resolution and bring the motion to
   full resolution the same way: each pixel weighs the nearby coarse motions by
   how well they match it.
5. Finally, each pixel may adopt the motion of a neighbour a few pixels away if
   that explains a 3×3 window around it better. Motion boundaries snap to the
   edges in the image.

### 2. Motion vectors and occlusions

Each pixel gets one motion vector (pixels per frame) and an *occlusion* score.

- With both neighbours, a pixel visible in all three frames has consistent
  flows (to the next frame ≈ minus to the previous one) that both land on its
  own colour. Its motion is their average.
- Where they disagree, the flow whose destination does not show the pixel's
  colour is the one that lost the pixel: the pixel is **occluded** on that
  side (covered or not yet uncovered by something moving in front), and its
  motion comes from the other flow. A destination outside the frame is a pixel
  leaving the picture, not an occlusion.
- Each pixel then may adopt the vector of a neighbour if that single velocity
  explains it better in *both* neighbour frames (at `+v` in the next, `−v` in
  the previous). Three frames constrain much more than two: vectors that bled
  across an object's edge fail one side and get corrected.

The blur vector is the motion times half the Blur Amount (half of the streak,
on each side of the pixel), limited to 128 pixels.

### 3. Who is in front

Occluded pixels always belong to whatever is *behind*: you only lose sight of
something when something else passes in front of it. So the motion of the
occluded pixels around an area is the motion of the local background.

The frame is split into 16 × 16 tiles. For each tile Smear records the longest
blur vector and the sum of the occluded pixels' vectors; then, per tile, it
gathers the neighbouring tiles (more of them for faster motion) to get the
local background motion. This is repeated twice more, each time trusting less
the occluded pixels whose motion is far from the previous estimate, so a few
mistaken pixels cannot drag the estimate towards the foreground.

A pixel is **in front** if its motion differs from the local background motion
by more than about 1 pixel per frame (plus 10% of the background speed), and
**behind** otherwise. Where there are no occlusions nearby, all pixels count as
one layer. This works whether the object moves over a still background or the
camera follows the object while the background races by.

### 4. Blur

Each output pixel is built from the current frame only — neighbour frames are
never blended in, so they cannot leave double images. Following McGuire et al.
(2012) and Jimenez (2014):

- The pixel's own layer is blurred along its own vector: samples along the
  streak that are behind or level with it are averaged with it, samples in
  front are left out. A still pixel next to a fast background stays sharp.
- Samples along the dominant motion of the surrounding tiles that are *in
  front* and whose streak reaches the pixel cover it, for the fraction of the
  shutter they spend over it. The moving object's blur spills over the
  background, with the right transparency.
- Within one layer, a faster neighbour also smears over a slower pixel; this
  keeps blur edges soft where no occlusion can be measured (an object over a
  flat colour).

Samples are taken about once per pixel of streak, up to Max Samples. They all
go along the pixel's own streak, unless something nearby moves faster, in
which case they are split between the dominant motion and the pixel's own.
Beyond the frame edges the edge pixels are extended, so blur stays even up to
the border. Colours are premultiplied and all four channels are blurred
alike, so blurred edges of layers with transparency become partially
transparent.

## Vector view

Show Vectors displays the motion instead of the blur:

- **Hue** = direction (rightward cyan, leftward red, up and down in between);
- **Saturation** = speed (grey = still, fully saturated ≈ 50 px per frame and up);
- **Brightness** = layer: bright for pixels judged in front, dimmer for background;
- **Magenta tint** = occluded pixels.

## Limitations

- Very thin, very fast objects (a few pixels wide, moving many times their
  width per frame) can still be missed by the flow; they then stay sharp
  rather than ghost.
- Layering is local and two-level: where three or more layers with different
  motions overlap, the middle one is treated as front or back as a whole.
- Motion is linear across the shutter (no curved streaks for fast rotation).
- Areas hidden behind a foreground object at the current frame cannot be
  revealed by the blur; the background blur next to the object is rebuilt from
  the background that is visible.

## Benchmark

`smear_bench` renders synthetic scenes with known motion (an object turning
over a panning background, a still object over a fast pan, a thin pole, an
object leaving the frame, two crossing objects, grain, a still scene) and
compares the blur with ground truth averaged from 64 sub-frames, against a
naive "blur along the flow" baseline. Run `smear_bench <out_dir>`; it also
writes PPM images of each scene and times the CPU path at 1920 × 1080.
