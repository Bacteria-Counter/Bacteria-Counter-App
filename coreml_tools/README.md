# coreml_tools — conversion, verification and evaluation harnesses

Every figure quoted in `AgarScopeKit/README.md` and in Section 7 of the
handover document was produced by a script in this directory. They are here so
those figures can be checked rather than taken on trust, and so that a change
to the pipeline can be measured the same way it was measured before.

**Read this first: the scripts are here, the data is not.** They need
benchmark images and annotations totalling roughly 700 MB, plus the 36,000-file
AGAR corpus. Neither belongs in a git repository. Without that data these
scripts will not run, and no amount of path configuration changes that. The
handover document's deliverable list records that the datasets are supplied via
Dropbox on request.

## Configuring paths

Three environment variables, each with a default pointing at the layout on the
machine where this work was done:

| Variable | Default | What it must contain |
|---|---|---|
| `AGARSCOPE_ROOT` | `~/bacteriaserius` | `bench_data/`, `coreml_models*/`, and the app repository |
| `AGAR_DATASET` | `~/Downloads/AGAR_dataset/dataset` | The AGAR images and their per-image JSON |
| `AGAR_MANIFEST` | `~/Downloads/bacteriacounter/kaggle_manifest.csv` | The split and background labels for AGAR |

`AGARSCOPE_MODELS` overrides the model directory for the verification scripts,
which is how the quantised models were measured against the float32 originals
through an unchanged harness.

## The Python environment

`requirements.txt` is a freeze from the environment these were run in
(coremltools 9.0, torch 2.7.1, ultralytics 8.4.115, opencv 5.0.0). The Swift
side needs no Python at all — only these harnesses do.

## What each script answers

| Script | Question |
|---|---|
| `convert_coreml.py` | Convert every checkpoint to Core ML, checking each against PyTorch on real plate photographs. |
| `quantize.py` | Compress the weights. `--mode block32` is what ships; `--mode int8` reproduces the per-channel failure that quadruples YOLO counts. |
| `coreml_fastsam.py`, `coreml_yolo.py` | The post-processing written out in Python, checked against Ultralytics before it was translated to Swift. |
| `verify_2a.py`, `verify_2b.py` | Core ML against PyTorch, then Swift against Core ML, for the FastSAM pipeline. |
| `verify_3.py` | The same for the seven YOLO variants and CSRNet. |
| `verify_3_components.py` | Each preprocessing stage against OpenCV, one variable at a time. Run this first when a count disagrees — a count is one number at the end of a long chain. |
| `verify_5.py` | float32 against int8 weights through identical Swift code. |
| `eval_before_after.py` | The Python server against the shipped app, end to end. |
| `eval_yolo.py` | All pipelines on PCA bright, empty plates, and the lab photos. |
| `eval_by_colony_size.py` | The same, stratified by colony size rather than by colonies per plate. |
| `eval_agar_bright.py` | The same on AGAR's 99 held-out bright images. |
| `eval_sam_escalation.py` | Where sam_micro's high-resolution escalation fires, and whether it helps there. |

## A known defect in the PCA ground truth

`bench_data/pca_benchmark.json` carries 126 plates totalling 4,690 colonies.
The handover records the original annotation as 127 plates and 5,059 colonies.
The counts in use were rebuilt from YOLO-format label files after the original
mask-derived annotations were lost with a cleared scratchpad, and they are
about 2.9 colonies per plate low — roughly 7 per cent.

This does not invalidate comparisons between pipelines, since every pipeline
was scored against the same annotation. It does mean the absolute MAE and
especially the **bias** figures on PCA are shifted, and any claim about a
pipeline being unbiased should be treated as unproven until the original
annotation is recovered. The dataset is described in the handover as publicly
available but is cited nowhere, so recovering it starts with asking whoever
assembled it.

Nothing measured on AGAR, on the lab photographs, or between two
implementations of the same pipeline is affected by this.
