# Bacteria-Counter-App inference server

This folder is the local Python server that **Bacteria-Counter-App** (the
macOS app one level up) talks to over HTTP to run colony counting. It started
as a fork of the university project documented below, but `server.py` and
everything it imports (`fastsam_colony_count.py`, `csrnet_model.py`, and the
fine-tuned `*.pt` checkpoints in this folder) are custom-built for this app
and are not part of that original project.

## Setup

```bash
cd Server
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
./start_server.sh
```

Leave that terminal running — it loads every model once at startup (takes a
few seconds) and listens on `http://127.0.0.1:8721`. Start this before
opening the app.

## The 8 model options

See the module docstring at the top of `server.py` for full validation
numbers and caveats for each. Briefly:

| Key | What it is |
| --- | --- |
| `yolo_old` | Original YOLOv8n counter, trained on AGAR only |
| `yolo_new` | Same, fine-tuned with SAM Copy-Paste synthetic data — **default/recommended** |
| `sam` | CLAHE + FastSAM zero-shot segmentation, no training needed |
| `dog_blend` / `clahe` / `lab_ab` | `yolo_new` further fine-tuned with a preprocessing technique baked into training |
| `gsam2` | Colony Grounded SAM2 — zero-shot, never trained on our data |
| `csrnet` | Density-map regression (no per-colony boxes, returns a heatmap instead) |

All 8 are kept in the app deliberately so real-world photos — not just
ground-truth benchmarks — can be the tiebreaker on which is most robust.

### Setting up `gsam2` (optional, extra steps)

7 of the 8 models load in-process and work out of the box once `Setup` above
is done. `gsam2` is the exception — it runs as a subprocess into two
*separate* repos/venvs, each with its own large pretrained checkpoint
(**not** included in this repo — they're large, public, and downloadable
directly, so there's no reason to bloat this repo with them). Every other
model works fine without doing any of this; only skip it if you specifically
want to test `gsam2`.

1. **Clone Grounded-SAM-2** (the official IDEA-Research framework this
   builds on) *outside* this repo, e.g. as a sibling folder:
   ```bash
   git clone https://github.com/IDEA-Research/Grounded-SAM-2.git
   cd Grounded-SAM-2
   python3.10 -m venv .venv && source .venv/bin/activate
   pip install torch torchvision torchaudio
   pip install -e .
   pip install --no-build-isolation -e grounding_dino
   cd checkpoints && bash download_ckpts.sh && cd ..
   ```
2. **Clone ColonyGroundedSam2** (Korporaal et al. 2026 — the paper this
   fine-tuned checkpoint comes from), also as a sibling folder:
   ```bash
   git clone https://github.com/WFSRDataScience/ColonyGroundedSam2.git
   cd ColonyGroundedSam2
   wget https://huggingface.co/DataScienceWFSR/ColonyGroundedSAM2/resolve/main/checkpoints/colony_gd.pth -P checkpoints
   ```
3. **Copy in our bridge script** — `infer_count_only.py` in
   [`gsam2_bridge/`](gsam2_bridge/) (next to this README) is custom code
   that calls Grounding DINO directly (skipping SAM2, since only the count
   + boxes are needed) and prints one JSON line for `server.py` to parse.
   It's not part of the official ColonyGroundedSam2 repo, so copy it in:
   ```bash
   cp gsam2_bridge/infer_count_only.py /path/to/ColonyGroundedSam2/
   ```
   (`gsam2_bridge/test_agar_groundtruth.py` and `test_empty_backgrounds.py`
   are the validation scripts used to get the numbers in `server.py`'s
   docstring — copy those in too if you want to reproduce that testing.)
4. **Point `server.py` at your checkouts** — set two environment variables
   before running `start_server.sh` (or edit the fallback defaults directly
   in `server.py`):
   ```bash
   export GSAM2_PYTHON=/path/to/Grounded-SAM-2/.venv/bin/python
   export GSAM2_SCRIPT=/path/to/ColonyGroundedSam2/infer_count_only.py
   ```
   `infer_count_only.py` itself also needs to know where you put
   Grounded-SAM-2 — set this too (same value as step 1's clone location):
   ```bash
   export GSAM2_REPO_DIR=/path/to/Grounded-SAM-2
   ```

With all three env vars set, `start_server.sh` will pick them up and
`gsam2` becomes usable. Without them, it falls back to this machine's
original paths, so nothing breaks for the existing setup.


## Attribution

This folder started as a fork of
[Husseinchr/bacterial-colony-detection](https://github.com/Husseinchr/bacterial-colony-detection),
a university computer-vision project (classical CV / YOLO / U-Net on the
AGAR-primary dataset). Only the pretrained YOLO counter checkpoint
(`models_trained/YOLO/counter/best.pt`, used as the `yolo_old` option) is
carried over from that project — everything else here (`server.py`,
`fastsam_colony_count.py`, `csrnet_model.py`, `gsam2_bridge/`, and the
fine-tuned checkpoints) was built for this app.
