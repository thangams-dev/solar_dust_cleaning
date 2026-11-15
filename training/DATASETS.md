# Dataset inventory

| Source | URL | License reported by Kaggle | Images after validation and exact de-duplication | Classes | Decision |
|---|---|---|---:|---|---|
| dust_detection | https://www.kaggle.com/datasets/hemanthsai7/solar-panel-dust-detection | CC0-1.0 | 2259 | ambiguous-near-duplicate, clean, dusty | Accepted selectively: clean and dusty. RGB panel images are present, but some samples are elevated/aerial views; inspected aerial examples were excluded. |
| pv_defect | https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset | CC-BY-NC-SA-4.0 | 543 | ambiguous-near-duplicate, bird-drop, clean, cracked, dusty, electrical-damage, snow-covered, unclear-physical-damage | Accepted selectively: clean, dusty, and physical-damage (mapped to cracked); bird-drop, snow, and electrical damage are held out. Visual inspection found watermarks and non-crack physical-damage examples; selected unclear examples were held out. |

## Counts by source and mapped class

| Source | Mapped class | Role | Images |
|---|---|---|---:|
| dust_detection | ambiguous-near-duplicate | heldout | 25 |
| dust_detection | clean | train_candidate | 1387 |
| dust_detection | dusty | train_candidate | 847 |
| pv_defect | ambiguous-near-duplicate | heldout | 23 |
| pv_defect | bird-drop | heldout | 92 |
| pv_defect | clean | train_candidate | 91 |
| pv_defect | cracked | train_candidate | 70 |
| pv_defect | dusty | train_candidate | 9 |
| pv_defect | electrical-damage | heldout | 92 |
| pv_defect | snow-covered | heldout | 165 |
| pv_defect | unclear-physical-damage | heldout | 1 |

Counts in this inventory include valid decoded images after exact-duplicate removal. Images under 64 px, unreadable files, known aerial examples, and exact duplicates are excluded. Near-duplicate images are retained and have group IDs in `data/manifest.csv` so a later split can keep each group together.

Exclusions recorded: exact duplicate: 1093; exact duplicate has conflicting labels: 232; visually identified aerial view: 9.

The public class names are mapped as follows: `Clean` → `clean`, `Dusty` → `dusty`, and visibly cracked `Physical-Damage` examples → `cracked`. `Bird-drop`, `Snow-Covered`, `Electrical-damage`, selected visually unclear physical-damage examples, and near-duplicate groups carrying conflicting labels are kept in `data/heldout/` and are not intended for training. A small visual sample of each retained/held-out label is in `reports/sample_*.png`.

Both Kaggle license identifiers are recorded as displayed by the Kaggle CLI. The CC-BY-NC-SA-4.0 dataset is non-commercial and share-alike; intended deployment/commercial use must be checked against its terms before use.

Rejected-file details are recorded in `data/rejected.csv`.
