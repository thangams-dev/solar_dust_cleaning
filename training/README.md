# Solar-panel image training and app helpers

The public-image classifier is a **two-class pretraining model** (`clean`, `dusty`).
It is not validated for the fixed, close-up webcam until own-panel photos are
available. Cracked-model training was explicitly skipped after visual review.

Run commands from the project root using the existing virtualenv:

```bash
.venv/bin/python -m pip install -r training/requirements.txt

# Download the public sources. Kaggle authentication is read by the CLI.
.venv/bin/kaggle datasets download -d alicjalena/pv-panel-defect-dataset \
  -p training/data/raw/pv_defect --unzip
.venv/bin/kaggle datasets download -d hemanthsai7/solar-panel-dust-detection \
  -p training/data/raw/dust_detection --unzip

# Build the source inventory/cache, inspect full-resolution crops, and split.
.venv/bin/python training/prepare_data.py
.venv/bin/python training/build_patches.py
.venv/bin/python training/split_data.py

# Run the mandatory 20-patch smoke test before the full training.
.venv/bin/python training/train.py --smoke-test

# Five-fold grouped OOF review of possible label issues (train+validation only).
.venv/bin/python training/cross_validate_suspects.py

# Compare regularized pretrained models, train a synthetic-dust variant, calibrate
# on validation, then perform one final test evaluation at the end.
.venv/bin/python training/compare_models.py

# Train MobileNetV2 with ImageNet initialization; epoch progress is logged.
.venv/bin/python training/train.py

# Evaluate the fixed test set once after training.
.venv/bin/python training/evaluate.py

# Predict (use --roi after camera calibration).
.venv/bin/python training/predict.py path/to/image.jpg
.venv/bin/python training/predict.py path/to/image.jpg --roi
# The default is the accepted, temperature-scaled ResNet18 v2 export; override with --model if needed.
.venv/bin/python training/predict.py path/to/image.jpg
```

Train and validation losses control early stopping and checkpoint selection. The
test split is never used for training or model selection. Reports include
patch-level and parent-image-level metrics; parent probabilities are the average
of all test patch probabilities for that parent. Bootstrap intervals resample
test parents, not patches. The six-patch PV dusty test subset is too small for a
stable source-specific conclusion.

## Own-photo fine-tuning

Follow [data/own/README.md](./data/own/README.md) to create folders by class and
capture session. The current model supports only clean and dusty; the script
skips unsupported own-photo classes such as cracked and shaded with a warning.
Whole sessions are held out for own-test. Fine-tuning mixes own photos with
public replay patches; `--own-weight` defaults to 3.

```bash
.venv/bin/python training/finetune_own.py --self-test
.venv/bin/python training/finetune_own.py --own-weight 3
```

Own-test metrics are reported separately and are the only metrics that count for
deployment. Fewer than three sessions for a class causes an unreliability
warning. A tuned checkpoint is written separately to
`model/own_finetuned.pt`; it does not overwrite the public-pretrained model.

## Fixed ROI calibration

Use a well-lit frame or a connected camera. Click the panel corners in order:
top-left, top-right, bottom-right, bottom-left. ROI JSON is written under
`app/config/roi.json`.

```bash
.venv/bin/python training/calibrate_roi.py --image path/to/panel_frame.jpg
.venv/bin/python training/calibrate_roi.py --camera-index 0
```

`app.panel_roi.warp_to_panel()` produces a 480×400 perspective-corrected crop
(matching the 168:140 panel ratio). It raises an actionable error if calibration
is missing.

## Reference-based dust index

Capture a genuinely clean panel after ROI calibration, then save it using
`app.dust_index.save_reference(warp_to_panel(frame), already_warped=True)`.
`dust_index(frame)` expects the fixed-panel crop by default; pass
`already_warped=False` for a raw camera frame.

The heuristic first matches candidate mean brightness to the clean reference,
then returns `100 * (0.70 * (1 - SSIM) / 2 + 0.30 * relative_contrast_loss)`,
clipped to 0–100. It is an **uncalibrated comparison score**, not a physical
dust measurement; shadows, reflections, exposure, movement, and panel changes
can affect it. Calibrate it with real dust on the actual panel before relying on
any threshold.

Synthetic checks:

```bash
.venv/bin/python -m unittest training.test_pipeline_helpers -v
.venv/bin/python training/finetune_own.py --self-test
```

The experimental large-crop cracked patches and their review grid are retained
for reference, but no cracked model is trained or included in the main labels.

## Capturing webcam photos

Calibrate the fixed panel ROI first. The default capture tool reads
`app/config/roi.json`, saves only perspective-corrected 480×400 crops, and asks
for a session, label, and (for dusty photos) dust level. Press `s` to save a
frame and `q` to finish. It reports blur/glare warnings and records each
capture in that session's `metadata.csv`. Image names are UTC timestamps; an
existing image is never overwritten. Exposure and automatic white balance are
locked with `v4l2-ctl` when available; install the system `v4l-utils` package if
the tool is missing, or use `--no-lock-controls` to opt out.
Blur warnings use Laplacian variance below 45; glare warnings mean at least 1%
of crop pixels have high value and low saturation. These are capture-quality
prompts, not calibrated image-quality guarantees.

The collector accepts `clean`, `dusty`, `cracked`, and `shaded` for future data
collection. The deployed v2 model only predicts clean/dusty.

```bash
.venv/bin/python training/capture_own.py
.venv/bin/python -m pytest training/tests/test_capture_own.py -q
```

The grouped OOF review writes `training/reports/suspect_labels.csv`,
`suspect_labels_top150.png`, and `suspect_labels_cv.md`. Its frozen ImageNet
feature probe is for prioritizing human review only; it does not change any
labels. Do not treat disagreement alone as label error, especially where source
style or weakly visible dust differs from the intended close-up deployment.

The comparison command trains MobileNetV2, EfficientNet-B0, and ResNet18 using
label smoothing 0.1, weight decay 1e-4, head/fine-tune stages, and validation-loss
early stopping. It compares a synthetic-dust variant on real validation images,
calibrates the selected model on validation, then reads the test split once as
the final step. It exports `model/model_v2.pt`, `model/labels_v2.json`, and
`model/model_card_v2.md`.

Generate synthetic dust from future clean webcam images using a session folder:

```bash
.venv/bin/python training/synth_dust.py \
  --input-folder training/data/own/clean/my_session \
  --session my_session --level all --seed 42
```

The generator writes `training/data/own_synth/my_session/` with light, medium,
and heavy variants and a manifest. Synthetic images are augmentations, not
replacements for real dusty validation images.
