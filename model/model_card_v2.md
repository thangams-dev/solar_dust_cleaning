# Solar-panel clean/dusty model v2

- Generated: 2026-11-10
- Selected architecture: `resnet18`.
- Task: clean versus dusty, binary classification only.
- Input: 224×224 RGB; ImageNet mean/std normalization.
- Selected run training time: 999.1 seconds.
- Validation-selected temperature: 0.740264.
- Validation ECE before temperature scaling: 0.0640.
- Validation ECE after temperature scaling: 0.0272.
- Selection used validation parent accuracy; ties were broken by dusty recall, then lower false-dusty rate. The held-out test split was not used for selection.

## Data sources and split sizes

- `solar-panel-dust-detection` ([Kaggle](https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection)): CC0-1.0.
- `pv-panel-defect-dataset` ([Kaggle](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)): CC-BY-NC-SA-4.0; non-commercial/share-alike terms apply.
- Only clean and dusty classes are included; the cracked experiment was skipped after visual review.

| Split | Patches | Parent images |
|---|---:|---:|
| Train | 7433 | 1634 |
| Validation | 1607 | 350 |
| Test | 1582 | 350 |

## Runtime versions

- Python: 3.12.3
- Pillow: 13.0.0
- matplotlib: 3.11.2
- numpy: 2.5.4
- scikit-learn: 1.9.1
- torch: 2.15.0+cpu
- torchvision: 0.29.2+cpu

## Public test (single final evaluation)

- Parent accuracy: 0.8343; 95% parent-bootstrap CI [0.7943, 0.8743].
- Dusty recall: 0.7422; 95% CI [0.6639, 0.8182].
- False-dusty rate on clean: 0.1126; 95% CI [0.0735, 0.1535].
- Confusion matrix (rows true clean/dusty, columns predicted clean/dusty): `[[197, 25], [33, 95]]`.

## Validation comparison

| Variant | Architecture | Parent accuracy | Dusty recall | False-dusty rate | Training seconds |
|---|---|---:|---:|---:|---:|
| mobilenet_v2_current | mobilenet_v2 | 0.8371 | 0.7578 | 0.1171 | existing |
| mobilenet_v2_regularized | mobilenet_v2 | 0.8457 | 0.8359 | 0.1486 | 1386.3 |
| efficientnet_b0_regularized | efficientnet_b0 | 0.8571 | 0.7812 | 0.0991 | 1557.2 |
| resnet18_regularized | resnet18 | 0.8600 | 0.8125 | 0.1126 | 999.1 |
| resnet18_synthetic | resnet18 | 0.8229 | 0.7031 | 0.1081 | 1204.0 |

Synthetic dusty was included only when its real-validation dusty recall improved over the selected architecture's real-only variant.
- Real dusty training patches: 2762.
- Synthetic dusty training patches: 2762.
- Synthetic examples were created only from clean training patches. No synthetic data was added to validation or test.

## Calibration

- Validation reliability plot: `training/reports/model_v2_reliability.png`.
- Parent-level dusty precision and recall at thresholds:

| Dusty threshold | Precision | Recall | Predicted dusty parents |
|---:|---:|---:|---:|
| 0.6 | 0.8500 | 0.7969 | 120 |
| 0.7 | 0.8738 | 0.7031 | 103 |
| 0.8 | 0.9167 | 0.6016 | 84 |
| 0.9 | 0.9355 | 0.4531 | 62 |

## Known limitations

- Public images are predominantly broad panel/array imagery and do not match the intended close-up fixed webcam view.
- Validation selection and temperature calibration use the same validation split, so the resulting validation metrics are selection/calibration estimates, not an independent final quality estimate.
- Synthetic dust is an approximation; real dust grain, adhesion, glare, camera optics, and lighting on the actual panel may differ.
- Test results describe public data only. Own-panel session-separated photos are still needed before deployment or setting a cleaning threshold.
- The classifier has no cracked or shaded output.
