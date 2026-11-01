# SHAPClean

Code for the paper *SHAPClean: Removing Backdoor Poisoning from Intrusion Detection Datasets without Clean Data*
(submitted to ICCIES 2027).

SHAPClean audits a labelled network-intrusion training set for dirty-label backdoor poisoning and removes the poisoned
records before the IDS is retrained. It needs no clean reference data:

1. train the model to be audited on the (possibly poisoned) training set and compute TreeSHAP attributions for the
   target class;
2. standardise, reduce with PCA (95% variance) and cluster every record with HDBSCAN;
3. **layer A** short-lists benign-labelled clusters whose nearest clusters are mostly attack-labelled (kNN label
   consistency at cluster level);
4. **layer B** labels each short-listed cluster with a clean XGBoost model and exemplar matching, both built from the
   records outside the short list;
5. the LABOR removal rule and a per-record rule for HDBSCAN noise decide which records are removed; the model is then
   retrained on what is left.

## Repository layout

```text
configs/config.yaml                  every parameter of the method and of the experiments
src/core/                            config loading, seeding, logging
src/feature_selection/               mRMR feature ranking
src/poison_audit/                    the method, the poisoning builder, the baselines and the reports
scripts/component1_poison_audit/     numbered entry points (see "Reproducing the paper")
run_component1.slurm                 SLURM launcher for any script
```

The config blocks `llm_cluster_labeling` and `llm_layer_b` keep their
names from an earlier LLM-based variant of layer B; they hold only the tie-break of record votes and the exemplar
matching settings, and SHAPClean itself does not use an LLM.

## Setup

```bash
python -m pip install -r requirements.txt
```

Tested locally with Python 3.13, numpy 2.3.4, pandas 2.3.3, scikit-learn 1.9.0, xgboost 3.2.0, shap 0.52.0,
hdbscan 0.8.44, mrmr-selection 0.2.8 and lightgbm 4.6.0. Every script reads
`configs/config.yaml` relative to the working directory, so run them from the repository root. Result paths under
`data/` reach about 140 characters, so on Windows keep the repository path short or enable long paths.

## Data

The datasets are not part of the repository. Place them under `data/` (ignored by Git):

| Path | Content | SHA256 of the file we used |
|---|---|---|
| `data/cicids2017_combined.csv` | CIC-IDS2017, built as below (2,827,876 rows, 79 columns) | `429c02f61c6011268bf3396963611ee870cff928e3ea64a8e911d0ea50f9dbcb` |
| `data/TON_IoT/TON_IoT.csv` | network part of TON_IoT (211,043 rows, 44 columns) | `26ddc513552de36de6428b2e578efaed2b57504c716dfba847cc0109a64e1974` |
| `data/ML_EdgeIIoT/ML_EdgeIIoT_dataset.csv` | Edge-IIoTset `ML-EdgeIIoT-dataset.csv`, zip-compressed under this name | `f9c586c0a7610bfcd95c5a802e6af62140b2f59bfe3d5073be88fb0dd4a7f972` |

`cicids2017_combined.csv` is the eight `MachineLearningCVE` files of CIC-IDS2017 concatenated in file-name order, with
column names stripped and the 2,867 rows holding NaN or infinite values dropped. A record's `sample_id` is its row
index in this file, so the order matters:

```python
import glob

import numpy as np
import pandas as pd

files = sorted(glob.glob("MachineLearningCVE/*.csv"))
frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
frame.columns = frame.columns.str.strip()
frame = frame[np.isfinite(frame.select_dtypes("number").to_numpy(dtype=float)).all(axis=1)]
frame.to_csv("data/cicids2017_combined.csv", index=False)
```

## Reproducing the paper

All outputs go to `data/component1_poison_audit/`. Seed 42 (`project.random_state`) is the calibration seed: its sets
were used only to select the configuration, by a rule fixed before their results were seen (`calibration_report` in
the config). The paper reports the five `selection_feature.evaluation_seeds` (8459, 1563, 9767, 4714, 3366). Most
scripts accept `--dataset`, `--trigger`, `--rate` and `--seed` to run part of the grid, and `--check` to list what
would run. The scripts are numbered in the order they run:

```bash
# 1. Clean train/test splits
python scripts/component1_poison_audit/01_split_cicids2017.py             # CIC-IDS2017
python scripts/component1_poison_audit/02_split_iot_datasets.py           # TON_IoT, Edge-IIoTset

# 2. Poisoned training sets and their mRMR features, 3 datasets x 3 triggers x 5 rates for every seed
#    (optional: script 04 prepares any set that is missing)
python scripts/component1_poison_audit/03_prepare_poisoned_datasets.py

# 3. Configuration selection on the calibration seed
python scripts/component1_poison_audit/04_poison_rate_sweep.py
python scripts/component1_poison_audit/05_calibration_report.py
python scripts/component1_poison_audit/06_select_configuration.py

# 4. SHAPClean, its ablations and the oracle references on every set; repeat for each evaluation seed
python scripts/component1_poison_audit/04_poison_rate_sweep.py --seed 8459

# 5. Baselines and other learners (repeat for each evaluation seed), and the 0% poison control
python scripts/component1_poison_audit/07_confident_learning_baseline.py --seed 8459
python scripts/component1_poison_audit/08_baseline_defenses.py --seed 8459
python scripts/component1_poison_audit/09_downstream_learners.py --seed 8459
python scripts/component1_poison_audit/10_clean_control.py

# 6. Clustering on raw features instead of attributions, and the layer-A diagnostic
python scripts/component1_poison_audit/11_raw_space_clustering.py
python scripts/component1_poison_audit/12_raw_space_audit.py
python scripts/component1_poison_audit/13_layer_a_sibling_diagnostic.py
```

`poison_rate_sweep.layer_a_settings` now holds only the selected layer-A setting. Step 3 compares the four candidates
listed in `calibration_report.selection_layer_a`, so to rerun the selection from scratch, list all four in
`layer_a_settings` before running script 04 on seed 42: `{k: 15, eta: 0.8}`, `{k: 25, eta: 0.5}`,
`{k: 15, eta: 0.8, reference: kept}` and `{k: 25, eta: 0.5, reference: kept}`.

A CIC-IDS2017 set takes a few hours. We ran the grid on a SLURM cluster (the launcher requests 8 CPUs and 64 GB), one
job per seed and trigger:

```bash
mkdir -p logs
sbatch --export=ALL,CONDA_BASE=<miniconda dir>,CONDA_ENV=<env name> run_component1.slurm \
    scripts/component1_poison_audit/04_poison_rate_sweep.py --seed 8459 --trigger representative
```
