# Deployed solar-panel clean/dusty model

The accepted deployment warm start is [model_v2.pt](./model_v2.pt), a
temperature-scaled ResNet18 with the two labels `clean` and `dusty`. Its label
order, preprocessing, and temperature (0.7402637038997142) are defined in
[labels_v2.json](./labels_v2.json). The default dusty decision threshold is
0.70 and is configurable through `app/config.py`.

The single final public-test evaluation reported parent accuracy of 83.43%
(95% bootstrap CI: 79.43–87.43%), dusty recall of 74.22%, and a false-dusty
rate on clean of 11.26%. Own-panel metrics are pending. Public test performance
does not establish performance on the intended close-up webcam view.

See the [v2 model card](./model_card_v2.md) for validation comparisons,
calibration, detailed metrics, runtime versions, data licenses, and limitations.
