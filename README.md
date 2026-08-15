# Bacteria-Counter-App

A macOS app for counting bacterial colonies from a captured petri dish photo
(via Continuity Camera or the built-in Mac camera).

## Running colony analysis

Nothing to start. Counting runs on-device through `AgarScopeKit/`, a Swift
package in this repo that drives Core ML models bundled with the app. Open it,
connect the camera, pick a model in the sidebar, capture a plate.

This replaced a local Python server (`Server/server.py`) that had to be
launched by hand before the app was any use. The move was made model by model
rather than in one step, and each pipeline was measured against the Python path
on the same 76 plates before it was allowed to replace it — those numbers, and
the flags that reproduce them, are in `AgarScopeKit/README.md`.

`Server/` is still here. It is no longer needed to run the app, and it is where
every accuracy figure quoted below was originally measured.

## Which model to pick

All ten run on-device. Each one's caveat is shown next to the picker in the
app; the short version:

- **YOLO (Lama) / (Baru)** — the production YOLOv8n counters. Validated on the
  AGAR benchmark, but only for that visual domain (plain agar, no printed grid).
- **YOLO Mac1 / Mac2** — fine-tuned further on external colony datasets. Mac1
  is the most accurate YOLO here and, like every YOLO variant, produces zero
  false detections on 34 empty plates. Reach for it when a false positive costs
  more than a miss: sterility checks, negative controls, anything reported as
  "no growth".
- **SAM+ (`sam_tuned`)** — CLAHE + FastSAM + dish/shape filtering, tuned on a
  bright-background benchmark. Best general accuracy on bright plates
  (MAE 3.86). Its blind spot is pinpoint colonies.
- **SAM Mikro (`sam_micro`)** — SAM+ plus a second pass at imgsz 4480 when the
  plate's median colony is under 1.5% of the dish diameter. Identical to SAM+
  on ordinary plates; slower on the ones where it fires.
- **YOLO + DoG-blend / + CLAHE / + LAB a/b** — experimental variants with
  preprocessing baked into training. LAB a/b scores well on AGAR but is more
  likely to miss pale, low-colour-contrast colonies.
- **CSRNet (density map)** — a different architecture: density-map regression
  rather than box detection. No per-colony circles, just a heatmap and a total.
  Cleanest of all on empty plates; training was never finished, and it
  overcounts on dense plates.

Two options the server offered are deliberately gone:

- **`sam`**, the frozen ORIGINAL FastSAM settings, existed only to reproduce
  results from before August 2026 and was the least accurate option here
  (MAE 35.11 against SAM+'s 3.86). It needed a Core ML export at 3840 that was
  never made. The FastSAM actually relied on — SAM+ and SAM Mikro — is
  untouched.
- **Colony Grounded SAM2** was never converted: 1.1 GB, a separate dependency
  tree, and it failed the empty-plate safety check at 36/36 photos, inventing
  colonies out of paper texture.

Both still work through `Server/server.py` if an old figure ever needs
reproducing exactly.

## Models

`bacteriaapp/coreml_models/` holds the 34 Core ML files, quantised to int8
(627 MB to 215 MB). Xcode compiles each into a `.mlmodelc` in the app bundle.
Set `AGARSCOPE_MODELS` to a directory of `.mlpackage` or `.mlmodelc` files to
run against a different set — that is how the quantised models were measured
against the float32 originals.
