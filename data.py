"""Validated MRI feature tables and training-only preprocessing."""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from torch.utils.data import TensorDataset


FEATURES = Path(__file__).with_name("features.txt").read_text().splitlines()


def read_table(path):
    # Check before pandas silently renames duplicate columns with .1 suffixes.
    with open(path, newline="", encoding="utf-8-sig") as stream:
        header = next(csv.reader(stream), [])
    if not header or len(header) != len(set(header)):
        raise ValueError("CSV must have a nonempty header with unique column names.")
    return pd.read_csv(path, dtype={"subject_id": str, "id": str})


def binary_column(frame, column):
    if column not in frame:
        raise ValueError(f"Missing required column: {column}")
    values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"{column} must contain only 0 and 1, without missing values.")
    return values


def prepare_training_data(path, task, experts, seed, initializer="kmeans",
                          pseudo_labels=None, add_covariates=False, covariates_path=None):
    """Fit scaling and pseudo-labels on this training table only.

    Returns a TensorDataset of (features, target, event, time, pseudo-label),
    plus JSON-serializable preprocessing metadata. Simulation reference rows
    have pseudo-label -1 and are excluded from router regularization.
    """
    if task not in {"classification", "survival", "simulation"}:
        raise ValueError(f"Unknown task: {task}")
    if initializer not in {"kmeans", "gmm"}:
        raise ValueError(f"Unknown initializer: {initializer}")
    frame = read_table(path)
    if frame.empty:
        raise ValueError("Training table is empty.")
    id_column = "id" if task == "simulation" else "subject_id"
    if id_column not in frame or frame[id_column].isna().any():
        raise ValueError(f"A complete {id_column} column is required.")
    if frame[id_column].duplicated().any():
        raise ValueError(f"{id_column} must be unique within the training table.")
    missing = [column for column in FEATURES if column not in frame]
    if missing:
        raise ValueError(f"Missing MRI feature columns: {', '.join(missing)}")
    features = list(FEATURES)
    if covariates_path is not None and not add_covariates:
        raise ValueError("--covariates requires --add-covariates.")
    if add_covariates:
        if covariates_path is not None:
            covariates = read_table(covariates_path)
            required = {"subject_id", "age", "sex"}
            if not required.issubset(covariates.columns) or "subject_id" not in frame:
                raise ValueError("Covariate merging requires subject_id, age, sex.")
            if covariates.subject_id.isna().any() or covariates.subject_id.duplicated().any():
                raise ValueError("Covariate subject_id values must be complete and unique.")
            frame = frame.drop(columns=["age", "sex"], errors="ignore").merge(
                covariates[["subject_id", "age", "sex"]], on="subject_id",
                how="left", validate="many_to_one", sort=False,
            )
        if not {"age", "sex"}.issubset(frame.columns):
            raise ValueError("Provide age and sex in the training CSV or --covariates CSV.")
        frame["sex"] = frame["sex"].map(
            lambda value: {"Female": 0, "Male": 1}.get(value, value)
        )
        frame["sex"] = binary_column(frame, "sex")
        features += ["age", "sex"]
    raw = frame[features].to_numpy(dtype=np.float64)
    if not np.isfinite(raw).all():
        raise ValueError("MRI features and enabled covariates must be numeric and finite.")

    n = len(frame)
    target, event, time = np.zeros(n), np.zeros(n), np.zeros(n)
    if task == "survival":
        event = binary_column(frame, "E")
        if "T" not in frame:
            raise ValueError("Survival training requires T (event/censoring time).")
        time = pd.to_numeric(frame["T"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(time).all() or (time <= 0).any():
            raise ValueError("T must contain finite, strictly positive follow-up times.")
        if not event.any():
            raise ValueError("Survival training requires at least one observed event.")
    else:
        target = binary_column(frame, "dx" if task == "simulation" else "DX")
        if len(np.unique(target)) != 2:
            raise ValueError("Classification training requires both binary classes.")

    normalization_rows = target == 0 if task == "simulation" else np.ones(n, dtype=bool)
    mean = raw[normalization_rows].mean(axis=0)
    std = raw[normalization_rows].std(axis=0)
    std[std == 0] = 1.0
    x = ((raw - mean) / std).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Normalization produced nonfinite features.")
    guidance_rows = target == 1 if task == "simulation" else np.ones(n, dtype=bool)
    if guidance_rows.sum() < experts:
        raise ValueError("There must be at least as many guidance samples as experts.")

    pseudo = np.full(n, -1, dtype=np.int64)
    if pseudo_labels is not None:
        labels = read_table(pseudo_labels)
        if id_column not in labels or "pseudo_subtype" not in labels:
            raise ValueError(f"Pseudo-label CSV requires {id_column},pseudo_subtype.")
        if labels[id_column].isna().any() or labels[id_column].duplicated().any():
            raise ValueError("Pseudo-label IDs must be complete and unique.")
        aligned = labels.set_index(id_column)["pseudo_subtype"].reindex(
            frame.loc[guidance_rows, id_column]
        )
        values = pd.to_numeric(aligned, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isin(values, np.arange(experts)).all():
            raise ValueError("Every guidance sample needs an integer pseudo-label in [0, K).")
        pseudo[guidance_rows] = values.astype(np.int64)
    else:
        if initializer == "kmeans":
            clustering = KMeans(n_clusters=experts, n_init=10, random_state=seed)
        else:
            clustering = GaussianMixture(n_components=experts, random_state=seed)
        # Retain original preprocessing: raw real-data inputs; normalized
        # simulated patient inputs. The neural networks always use z-scores.
        clustering_input = x if task == "simulation" else raw
        pseudo[guidance_rows] = clustering.fit_predict(clustering_input[guidance_rows])
    if len(np.unique(pseudo[guidance_rows])) != experts:
        raise ValueError("Initialization did not produce all K clusters; check data or K.")

    dataset = TensorDataset(
        torch.from_numpy(x), torch.tensor(target, dtype=torch.float32),
        torch.tensor(event, dtype=torch.bool), torch.tensor(time, dtype=torch.float32),
        torch.from_numpy(pseudo),
    )
    metadata = {
        "features": features, "mean": mean.tolist(), "std": std.tolist(),
        "normalization_population": "reference" if task == "simulation" else "training",
        "initializer": "external" if pseudo_labels is not None else initializer,
    }
    return dataset, metadata
