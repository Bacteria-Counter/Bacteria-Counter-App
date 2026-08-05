# Bacteria-Counter-App

A macOS app for counting bacterial colonies from a captured petri dish photo
(via Continuity Camera or the built-in Mac camera).

## Running colony analysis

Colony counting runs through a local Python server, not bundled into the app.
Before capturing a plate:

1. Start the inference server (only needs to be done once per session):
   ```bash
   /path/to/bacterial-colony-detection/start_server.sh
   ```
   Leave that terminal running. It loads models once at startup, so the
   first run takes a few seconds; keep it open while using the app.
2. Open the app, connect the camera, capture a plate.
3. In the sidebar, pick which model to use before capturing. All 8 are kept
   in the app on purpose so results can be compared on real lab photos
   rather than ground-truth benchmarks alone — see each one's in-app caveat,
   and `server.py` in the bacterial-colony-detection repo for full
   validation notes:
   - **YOLO (Lama)** / **YOLO (Baru)** — the production YOLOv8n counters.
     Validated on the AGAR benchmark dataset, but only for that same visual
     domain (plain agar, no printed grid/background).
   - **SAM** — CLAHE contrast enhancement + FastSAM zero-shot segmentation.
     No training data needed, so it degrades more gracefully on new
     backgrounds, but isn't validated to the same accuracy standard as YOLO —
     treat its counts as an estimate, not ground truth.
   - **YOLO + DoG-blend / + CLAHE / + LAB a/b** — experimental variants with
     preprocessing baked into training. LAB a/b scores best on the AGAR
     benchmark but is more likely to miss pale/same-hue colonies.
   - **Colony Grounded SAM2** — zero-shot, never fine-tuned on our data.
     Known to hallucinate detections on empty backgrounds — use carefully.
   - **CSRNet (density map)** — a different architecture (density-map
     regression instead of box detection): no per-colony detection circles,
     just a heatmap + total count. Training isn't finished; tends to
     overcount on dense/complex plates.

If the app shows "Can't reach the local model server," the server in step 1
isn't running.
