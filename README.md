# MoE Subtyping

Training implementation for **Discovering Heterogeneous Neurodegenerative Disease Patterns From MRI Data for Improved Prediction**, Yuanwang Zhang, Hongming Li, and Yong Fan, MICCAI 2026.
[Paper DOI](https://doi.org/10.1007/978-3-032-38239-9_16).

The method jointly trains a router and specialized prediction experts. Hard expert assignments define subtypes. This release supports stable/progressive MCI classification, Cox progression-risk training, and reference/patient classification on prepared semi-simulated data.

## Installation

Use Python 3.10-3.12 (tested with Python 3.12.8 on Windows).

```bash
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell, or
`source .venv/bin/activate` on Linux/macOS, then install:

```bash
python -m pip install -r requirements.txt
```

## Dataset organization

Keep your private data outside version control. One possible layout is:

```text
MoESubtyping/
  data.py
  features.txt
  losses.py
  model.py
  train.py
  requirements.txt
  data/                         # not distributed; ignored by Git
    real_data_cls.csv
    real_data_surv.csv
    semi_simulated_data.csv
    covariates.csv              # optional
  runs/                         # generated checkpoints and training losses
```

Each CSV needs the 96 named feature columns listed in [`features.txt`](features.txt). That file defines the model's feature order; CSV column order can differ because the loader selects features by name. Extra metadata columns are ignored. Column names must be unique, and features must be numeric, finite, and complete. Perform any required quality control or imputation before training.

Features comprise 68 Desikan-Killiany cortical thickness measurements (34 per hemisphere) and 28 Aseg subcortical volumes. Following the original preprocessing notebook, divide each subcortical volume by that scan's estimated total intracranial volume (`EstimatedTotalIntraCranialVol`) before exporting the CSV.

| Task | Required metadata | Row represents |
| --- | --- | --- |
| `classification` | `subject_id`, `DX` | One subject's baseline MRI; `DX=0` stable MCI, `DX=1` progressive MCI |
| `survival` | `subject_id`, `E`, `T` | One subject's baseline MRI; `E=1` observed dementia conversion, `E=0` right-censored, `T` is the conversion time (for E=1) or observed time (for E=0) |

For the paper's classification cohort, subjects have baseline MCI; progressive MCI converts to dementia within three years, and stable MCI has sufficient follow-up beyond three years without conversion during that interval. Exclude subjects whose outcome cannot be determined. For survival, use baseline MCI subjects with follow-up; `T` is positive time from baseline to the event or last follow-up. Use one consistent unit.

### Final training CSV structure

Each row contains one subject's baseline MRI measurements and task labels. `subject_id` must be present and unique. The examples below use invented values and abbreviate the feature columns with `...`. In your actual CSV, replace `...` with all remaining named columns from [`features.txt`](features.txt) and their values; do not include a literal `...` column.

**Classification — `real_data_cls.csv`:** `subject_id`, `DX`, and 96 MRI features (98 required columns in total).

```csv
subject_id,DX,lh_bankssts_thickness,lh_caudalanteriorcingulate_thickness,...,3rd-Ventricle,4th-Ventricle
example_001,0,2.45,2.61,...,0.0012,0.0003
example_002,1,2.18,2.39,...,0.0018,0.0004
```

**Survival — `real_data_surv.csv`:** `subject_id`, `E`, `T`, and 96 MRI features (99 required columns in total). Here, `T` is expressed in days.

```csv
subject_id,E,T,lh_bankssts_thickness,lh_caudalanteriorcingulate_thickness,...,3rd-Ventricle,4th-Ventricle
example_001,0,1461,2.45,2.61,...,0.0012,0.0003
example_002,1,730,2.18,2.39,...,0.0018,0.0004
```


### Optional covariates

By default, only the 96 MRI features are used. `--add-covariates` appends `age` and `sex`, giving 98 inputs, and includes them in normalization and initial clustering. Supply them in the training table or use a separate CSV:

```csv
subject_id,age,sex
example_001,72,Female
example_002,68,Male
```

Sex can be `Female`/`Male` or numeric `0`/`1`. A separate covariate table must have one row per subject and is joined by `subject_id`. Alternatively, append `age` and `sex` columns directly to either training CSV above. Both values must be complete when `--add-covariates` is enabled; otherwise these columns are ignored.

## Train

Run from this repository's root. Every command trains on the entire supplied
training CSV and writes to a new or empty directory.

```bash
python train.py --task classification --data data/real_data_cls.csv --output runs/classification
python train.py --task survival --data data/real_data_surv.csv --output runs/survival
```

### Hyperparameters

Defaults apply to both real-data tasks unless a task-specific value is noted.

| Argument | Description and default |
| --- | --- |
| `--experts` | Number of experts/subtypes. Default: **4**. |
| `--epochs` | Number of training epochs. Default: **80** |
| `--lr` | SGD learning rate. Default: **0.02** |
| `--batch-size` | Samples per training batch; `0` uses the full training set. Default: **64**. |
| `--hidden-dim` | Hidden-layer width in the router and expert networks. Default: **128**. |
| `--expert-dropout` | Dropout probability in each expert. Default: **0.5** |
| `--router-dropout` | Dropout probability in the router. Default: **0.2** for classification; **0** for survival. |
| `--temperature` | Gumbel-softmax temperature used for hard routing during training. Default: **1.0**. |
| `--initializer` | Initial clustering method for guidance: `kmeans` or `gmm`. Default: **`kmeans`**. |
| `--guidance-epochs` | Number of epochs over which guidance decays to zero; `0` disables guidance. Default: **20**. |
| `--guidance-power` | Exponent controlling guidance decay; `1` is linear, larger values decay faster, and values between `0` and `1` decay slower. Default: **1.0**. |
| `--loss-balance` | Weight of the load-balancing loss. Default: **0.6**. |
| `--loss-sparsity` | Weight of the sparse-assignment loss. Default: **0.2**. |
| `--loss-guide` | Initial weight of clustering guidance, before decay. Default: **0.2**. |
| `--seed` | Random seed for model initialization, clustering, and batch shuffling. Default: **42**. |
| `--device` | Training device: `cpu` or `cuda`. Default: **`cpu`**. |

Choose initialization, guidance strength/duration, and optional covariates:

```bash
python train.py --task classification --data data/real_data_cls.csv --output runs/classification_gmm --initializer gmm --guidance-epochs 30 --loss-guide 0.3 --guidance-power 1 --add-covariates --covariates data/covariates.csv
```

For externally generated guidance (e.g. DecoNet simulation assignments), pass `--pseudo-labels path/to/labels.csv`. Use `subject_id,pseudo_subtype` for real-data tasks, or `id,pseudo_subtype` for simulation, with integer subtype labels from `0` to `K-1`. Every guidance sample must be present and all K labels must be represented. Labels are aligned by ID; this option overrides `--initializer`.
