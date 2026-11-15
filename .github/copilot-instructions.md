# Solar-panel application instructions

- The deployed classifier is the accepted `model/model_v2.pt` ResNet18 warm start. Do not start or resume model training unless explicitly requested.
- The deployed model has exactly two classes, in order: `clean`, `dusty`.
- Load label order, input size, normalisation, and temperature from `model/labels_v2.json`, which accompanies `model/model_v2.pt`. Apply the stored temperature to logits before softmax.
- The default dusty decision threshold is `0.70`. Keep it configurable through application configuration; do not hard-code it in multiple components.
- `cracked` and `shaded` are not deployed model classes. Capture tools may collect those labels for future review, but inference must not claim to detect them.
- Use the fixed, calibrated panel ROI and fail clearly if ROI calibration is missing. Do not add a panel detector; the camera and panel are fixed.
- Public-data metrics do not establish performance on the deployed panel. Treat session-separated own-panel test results as the deployment measure.
