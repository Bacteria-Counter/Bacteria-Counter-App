# AgarScopeKit — on-device colony counting

Swift port of the colony-counting pipelines, replacing the local Python
inference server. The app ships three of them — `sam_micro`, `mac1` and
`csrnet` — chosen by measurement (see "Choosing which models to ship"). The
engine and CLI still run all ten, because the harnesses have to be able to
measure the models that were cut in order to justify cutting them. The APHA
2002 CFU rules are ported too.

The app links this package (`bacteriaapp` → AgarScopeKit); the CLI below is the
same code, which is the only reason the numbers here say anything about the app.

## Building and running

```
swift build -c release
.build/release/agarscope <modelDir> <image...> [--model <key>] [--micro]
```

`modelDir` is a directory of `.mlpackage` or `.mlmodelc` files. `<key>` is one
of `sam_tuned` `sam_micro` `yolo_old` `yolo_new` `mac1` `mac2` `dog_blend`
`clahe` `lab_ab` `csrnet`, defaulting to `sam_tuned`. Output is one line per
image: `name|count|imgsz|escalated|seconds`.

Diagnostic modes, used to compare against the Python reference stage by stage:

| Flag | Purpose |
|---|---|
| `--trace <dir> <img> <size>` | per-stage numbers: dish, CLAHE mean, letterbox mean, NMS, mask areas, final count |
| `--lchan <img> <L.bin>` | LAB L channel against an OpenCV dump |
| `--clahe-bin <L.bin> <w> <h> [ref]` | CLAHE against an OpenCV dump |
| `--dump-lab2rgb <lab.bin> <w> <h> <out>` | the LAB inverse alone, fed OpenCV's own LAB bytes |
| `--dump-prep <key\|area> <img> <out>` | preprocessed pixels: dog_blend, clahe, lab_ab, or CSRNet's INTER_AREA resize |
| `--dump-boxes <dir> <key> <img> <out>` | post-NMS YOLO boxes in original-image coordinates |
| `--dump-density <dir> <img> <out>` | the CSRNet density map, not just its sum |
| `--dump-cand <dir> <img> <out>` | pre-NMS candidates, so torchvision can run NMS on identical boxes |
| `--lab-test`, `--inv-test`, `--contour-test` | fixed-value checks on the colour and contour maths |
| `--cfu-test` | the 23 worked CFU examples from the APHA document |

The harnesses that drive these live in `coreml_tools/` at the repository root.
They need benchmark data that is not in the repository; see that directory's
README for what is required and for a known defect in the PCA ground truth
that shifts the bias figures below.

## What is verified, and how

Each component was checked against the Python implementation rather than
assumed correct, and each check isolates one variable.

| Component | Result |
|---|---|
| PNG loading | identical byte for byte with `cv2.imread` |
| LAB forward | 16.9% of pixels differ by exactly 1, none by more |
| LAB inverse | 2.9% of bytes differ by 1, none by more |
| CLAHE, given identical L | 0.00% of pixels differ (see the fix below) |
| `make_dog_blend` | 0.00% of bytes differ, max 1 |
| `make_lab_ab` | 1.9% of bytes differ, mean 0.06, max 5 |
| CSRNet `INTER_AREA` resize | 0.04% of bytes differ, mean 0.0004, max 1 |
| NMS | `torchvision.ops.nms` on the same candidate boxes returns the same count |
| Contours / circularity | identical to 4 decimals on discs and squares |
| CFU calculator | 21/23 worked examples, matching the Python case for case |

End-to-end against the Python Core ML path, on 42 holdout plates (accuracy) and
34 empty plates (false positives). Both sides read the same PNGs and both run
Core ML, so the only variable is the port:

| Model | holdout MAE Py → Sw | within ±1 | empty Py → Sw | bias |
|---|---|---|---|---|
| `yolo_old` | 10.48 → 10.52 | 42/42 | 33 → 31 | +0.05 |
| `yolo_new` | 10.36 → 10.38 | 42/42 | 0 → 0 | −0.02 |
| `mac1` | 8.69 → 8.71 | 42/42 | 0 → 0 | −0.12 |
| `mac2` | 10.76 → 10.71 | 41/42 | 0 → 0 | −0.05 |
| `dog_blend` | 8.83 → 8.79 | 41/42 | 1 → 0 | −0.14 |
| `clahe` | 10.02 → 10.12 | 40/42 | 3 → 3 | +0.10 |
| `lab_ab` | 11.67 → 11.64 | 41/42 | 2 → 1 | +0.02 |
| `csrnet` | 8.67 → 8.67 | 42/42 | 0 → 0 | **0.00, every plate identical** |
| `sam_tuned` | 5.38 → 5.43 | 34/42 | 61 → 60 | 0.00 |

CSRNet agrees exactly on all 76 plates, which is what a model with no NMS and
no area filter should do — nothing in its chain can amplify a sub-unit pixel
difference into a different count. The FastSAM path is the loosest, for the
opposite reason: masks, contours and an area filter all sit downstream of it.

These MAE figures are not comparable to anything recorded before 15 August
2026. The original ground truth was destroyed when macOS cleared a scratchpad
and was rebuilt from the YOLO labels in `pca_light.zip`, which count about 3
colonies lower per plate.

## Before and after, measured end to end

Everything above compares one link at a time, which is right for finding
faults and wrong for answering "what changed for the person using the app" --
assuming three measured deltas compose is the sort of assumption this project
has been caught by. `eval_before_after.py` runs the two ENDS against each
other on the 42 holdout plates: `server.py`'s PyTorch path, verbatim, against
the shipped Swift binary on int8 weights.

| Model | MAE server → now | seconds/photo |
|---|---|---|
| `mac1` | 8.71 → **8.67** | 0.34 → 0.10 |
| `csrnet` | 8.62 → 8.74 | 0.61 → <0.1 |
| `yolo_new` | 9.95 → 10.33 | 0.36 → 0.10 |
| `sam_tuned` | 4.64 → **5.40** | 1.40 → 0.80 |

Attributing the accuracy change to each step, using the per-link measurements
already recorded above:

| Model | Core ML conversion | Swift port | int8 | total |
|---|---|---|---|---|
| `mac1` | −0.02 | +0.02 | −0.04 | −0.04 |
| `csrnet` | +0.05 | 0.00 | +0.07 | +0.12 |
| `yolo_new` | +0.41 | +0.02 | −0.05 | +0.38 |
| `sam_tuned` | **+0.74** | +0.05 | −0.03 | +0.76 |

The port and the quantisation are essentially free. Nearly all of the cost is
the Core ML conversion, and it is model-dependent — zero for `mac1`, 0.74 for
`sam_tuned`. The reason is structural rather than a conversion bug: ultralytics
feeds PyTorch a rectangular input padded to a stride multiple (1952×2560 for a
2560×1922 plate), while an exported Core ML model accepts only the square size
it was built at. FastSAM suffers most because its output is masks that then
pass an area and circularity filter, so a small input difference is amplified
by the filter chain rather than averaged away.

Size, for completeness: the server needed ~2.6 GB on the machine (134 MB of
`.pt` checkpoints, ~1.4 GB of Python/torch/ultralytics, ~1.1 GB for the
separate gsam2 checkout) and had to be started by hand. The app is 105 MB and
needs nothing. The model files themselves grew, 134 MB → 206 MB, because one
6.4 MB YOLO checkpoint becomes four Core ML files to preserve adaptive imgsz.

## Choosing which models to ship

Ten models went in; three ship. The cut was made by measuring all of them the
same way on three populations, not by reasoning about architectures.

**The rankings invert completely between benchmarks, and that is the finding.**

| Model | AGAR bright (99) | PCA bright (126) | lab: padat2 / sepi |
|---|---|---|---|
| `yolo_old` | **0.80** | 9.47 | 452 / 19 |
| `dog_blend` | **0.80** | 8.29 | 493 / 6 |
| `mac2` | 0.96 | 10.18 | 415 / 10 |
| `yolo_new` | 1.03 | 9.63 | 391 / 9 |
| `clahe` | 1.16 | 9.17 | 473 / 5 |
| `mac1` | 1.24 | 8.33 | 428 / 21 |
| `lab_ab` | 1.82 | 10.12 | 106 / 9 |
| `csrnet` | 3.13 | 7.97 | 386 / 63 |
| `sam_micro` | 25.07 | **5.37** | **274 / 120** |

The lab photos' true counts, ~280 and ~104, were confirmed by eye. AGAR is
in-domain for every YOLO variant (yolo_old trained on it entirely, the rest
with 600 AGAR images mixed into fine-tuning) and for CSRNet; FastSAM is
zero-shot on both benchmarks. The test split was never trained on, so the AGAR
figures are honest — they are just not comparable across models.

Reading AGAR as a model ranking would ship `yolo_old`, which is 76% wrong on a
real plate from this lab. PCA was built because it resembles this lab rather
than AGAR, so it and the lab photos are the relevant evidence; AGAR is a
robustness probe.

Stratified by colony size on PCA (median colony radius over dish radius),
`sam_micro` wins two of three buckets and no cut model was a hidden
small-colony specialist — the best YOLO on small colonies (`dog_blend`, 11.90)
is still twice as bad as `sam_micro` (5.79). Note PCA spans only 1.30%–7.62%
and this lab's pinpoint plate is 0.80%, so PCA cannot rank models on pinpoint
colonies at all; AGAR bright reaches 0.78% and can.

**What the escalation actually does.** `sam_micro` is `sam_tuned` plus a second
pass at 4480 when the median colony is under 1.5% of the dish radius. On AGAR
bright that doubles the error — and all of the damage is in the small-colony
bucket, where it fires:

| | MAE all | bias | small | medium | large |
|---|---|---|---|---|---|
| `sam_tuned` | 12.45 | −5.14 | 10.18 | 8.26 | 19.25 |
| `sam_micro` | 25.07 | +11.82 | **48.03** | 8.26 | 19.25 |

The mechanism matters more than the number. Escalation is guarded by a minimum
of 8 detections, on the assumption that fewer than that is too little to judge.
On AGAR, `sam_tuned` already reports 7–16 colonies on plates holding 1 — so the
guard is satisfied by FALSE detections and escalation then multiplies them.
Escalation is not dangerous on its own; it is dangerous when the base count is
already wrong. On this lab's plates the base count is sound (PCA 5.2, padat2
274 against ~280), and there escalation turns an error of 79 into 16.

That leaves one real risk to live with: FastSAM is fragile to backgrounds it
has not seen, and it fails silently. `mac1` is kept as a visual cross-check
(it draws per-colony circles, and produced 0 false positives on 34 empty plates
where SAM produced 64) and `csrnet` as a numeric one. A large disagreement
between them and SAM is a signal that the photography setup has changed, not
that the plate is unusual.

## A defect in the PCA ground truth, and what it invalidates

`bench_data/pca_benchmark.json` holds 126 plates totalling 4,690 colonies. The
handover records the original annotation as 127 plates and 5,059 colonies: the
counts in use were rebuilt from YOLO label files after the mask-derived
originals were lost with a cleared scratchpad, and they run about **2.9
colonies per plate low, roughly 7 per cent**.

Comparisons between pipelines survive this, because every pipeline was scored
against the same annotation — an under-reading meter still ranks the runners
correctly. What does not survive is any statement about a pipeline's absolute
bias, because a uniform shift in the reference moves every bias by the same
amount:

| Pipeline | Bias as measured | Bias if the reference is 2.9 low |
|---|---|---|
| `sam_micro` | +4.55 | ~+1.7 |
| `mac1` | +7.41 | ~+4.5 |
| `csrnet` | −0.02 | ~−2.9 |

So the claim made earlier in this file and in the handover — that `csrnet` is
the only pipeline without a systematic bias — is **withdrawn**. It is more
likely undercounting by about three colonies and appearing neutral only because
the reference is low by about three. `csrnet` is still a useful cross-check,
being a different architecture that fails differently, but not for that reason.

Correcting the reference would, if anything, strengthen the selection: the
overcounting pipelines all move toward zero, `sam_micro` most of all, while
`csrnet` moves away. Nothing measured on AGAR, on the lab photographs, or
between two implementations of the same pipeline is affected — those never use
this annotation.

The dataset is described in the handover as publicly available but is cited
nowhere, and it is not among the eleven Kaggle and Roboflow datasets listed in
Section 2.1.6. Recovering the original annotation starts with asking whoever
assembled it.

## A CLAHE bug this found

The previous README claimed CLAHE differed on 0.11% of pixels, never by more
than 1. That figure had been measured on an image whose height happened to
divide by 8, and it did not generalise: on a 2560×1922 plate the same
comparison gave 26.5% of pixels differing, by up to 4.

The cause is that OpenCV does **not** shrink tiles that hang off the edge of
the image. It grows the image to a multiple of the tile grid with
`BORDER_REFLECT_101` and normalises every LUT by the full tile area. This port
truncated instead, so an eighth of the image was equalised against a different
pixel population *and* a different divisor. Cropping the same plate to 1920
rows dropped the difference to 0.20% — which is what identified the cause.

There is an oddity worth knowing: OpenCV pads **both** axes as soon as
**either** is indivisible, so an exactly-divisible width still gains a full
extra tile of columns. That looks like a slip in its condition rather than a
decision, but it is what the thresholds in this project were calibrated
against, so it is reproduced.

Fixing it moved the already-shipped FastSAM path closer to the reference on
every population: holdout bias +0.07 → 0.00 with the totals now matching
exactly (1816 both sides) and within-±1 rising 29/42 → 34/42; lab photos
+6.33 → +5.67. Empty plates were unaffected, because at 3200×1800 they were
already divisible by 8 — which is itself confirmation of the mechanism.

## Known limitation: rescaled images

Images the pipeline **rescales** do not match Python exactly. Benchmark plates
letterbox at scale 1.0 — padding only, no resampling — and agree closely. The
lab photos are rescaled 1.4× on the way to 4480 for the `sam_micro`
escalation, and there the counts run about 6 colonies high out of 108.

The cause is `cv2.resize(INTER_LINEAR)`, which uses 5-bit fixed-point
arithmetic with integer weight tables. `ImageOps.resizeBilinear` reproduces the
sampling convention — pixel centres, `(dst + 0.5) * ratio - 0.5` — but in
floating point, leaving about 0.7 of a unit average difference per byte. That
is enough to move a handful of marginal detections across the threshold.

Accepted deliberately rather than pursued: the gap is smaller than the system's
own uncertainty (the best model here has MAE 3.86) and the true count for those
photos has never been measured. Closing it would mean reimplementing OpenCV's
fixed-point resampler.

A second, smaller residue has the same character. `make_clahe` as a whole still
differs from OpenCV by a mean of 1.06 per byte even though CLAHE and the LAB
inverse are now exact, because the LAB **forward** conversion differs by 1 unit
on 16.9% of pixels and CLAHE amplifies that: a shifted histogram shifts the
tile's whole LUT, so pixels that were themselves correct come out different.
Closing it would mean reproducing OpenCV's fixed-point `RGB2Lab_b` tables.

## Quantisation

627 MB of float32 models is awkward to bundle. Weight-only int8, measured
rather than assumed:

| Scheme | yolo_new_1920 | count (float32 = 34) |
|---|---|---|
| per-channel symmetric | 3.4 MB | **129** |
| per-channel asymmetric | 3.4 MB | **51** |
| per-channel, only weights > 65536 | 6.5 MB | **130** |
| per-channel, only weights > 262144 | 12.2 MB | 34 |
| **per-block, block size 32** | **4.2 MB** | **34** |

Per-channel int8 is unusable here: it nearly quadruples the count. A YOLO
detect head produces a score per anchor, and there are tens of thousands of
anchors, so a small upward shift in score calibration pushes a large absolute
number of boxes past `conf=0.4`. Restricting quantisation to the layers where
it is safe leaves almost no saving, because those mid-size layers *are* the
bulk of the weights.

Per-block quantisation is accurate enough to leave the counts alone, but
`coremltools` refuses to apply it below spec version 9 (iOS 18 / macOS 15), and
ultralytics' CoreML exporter does not expose `minimum_deployment_target` — so
all 34 files came out at version 6. Rather than redo step 1, `quantize.py`
re-stamps the spec version before compressing: the ops were emitted for an
older spec and every one is still valid in the newer one, which is additive.
It raises the minimum OS to macOS 15; the app targets 26.5, so it costs
nothing.

Result: **627 MB → 215 MB**, and the app bundle is 212 MB.

What that costs, measured through identical Swift code on the same plates —
float32 weights against int8, so the only variable is the weights:

| Model | holdout MAE f32 → int8 | empty-plate false positives | note |
|---|---|---|---|
| `yolo_new` | 10.38 → 10.33 | 0 → 0 | |
| `mac1` | 8.71 → 8.67 | 0 → 0 | |
| `mac2` | 10.71 → 10.74 | 0 → 0 | |
| `lab_ab` | 11.64 → 11.64 | 1 → 1 | |
| `clahe` | 10.12 → 10.26 | 3 → 3 | |
| `yolo_old` | 10.52 → 10.45 | 31 → 30 | |
| `dog_blend` | 8.79 → 8.95 | 0 → 0 | |
| `sam_tuned` | 5.43 → 5.40 | **60 → 64** | the one real cost |
| `csrnet` | 8.67 → 8.74 | 0 → 0 | systematic bias −1.79 |

Every YOLO variant that produced zero false detections on 34 empty plates still
produces zero — the property that matters most for sterility checks survives
quantisation. Two models pay something: `sam_tuned` gains 4 false positives
across 34 empty plates (1.76 → 1.88 per plate), and `csrnet` picks up a
systematic bias of −1.79 colonies.

**A bug this surfaced.** Quantised CSRNet reported **−1** colonies on empty
plates. Nothing constrains a predicted density map to be positive: on an empty
plate the sum lands either side of zero, and float32 happened to give −0.4
(rounding to 0) where int8 gives about −0.6 (rounding to −1). `run_csrnet()`
in Python has the same hole and simply never met an input that exposed it. The
count is now clamped at zero — a deliberate divergence from the Python, because
−1 colonies is not a wrong answer, it is not an answer.

## Two findings worth keeping

**Order of operations in mask assembly.** `process_mask` upsamples the
prototype logits to letterbox resolution and binarises *there*; `scale_masks`
then interpolates those 0/1 values again on the way down. Binarising once at
the end instead produced masks 16% smaller — enough to drop roughly ten
colonies per plate under the area filter. The two-pass order is not an
implementation detail.

**`align_corners=False`.** PyTorch's `F.interpolate` samples at pixel centres.
Omitting the half-pixel terms offset every sample by 0.375 of a prototype cell,
and a prototype cell covers 4 letterbox pixels.

Both were found by comparing stage by stage, not by reading the source — and
both had been masked earlier by cancelling against each other.

## Not ported

`sam` (the frozen original FastSAM settings) and `gsam2`. See
`AgarScope.Model` for why, and `../README.md` for what that means in the app.
