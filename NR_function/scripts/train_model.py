#!/usr/bin/env python3
"""Train final XGBoost functional classification models from prepared features.

Expected upload-folder layout:

repo_root/
  ago_ant/{random,butina}/{TARGET}.csv
  ago_ant_features/processed_features/{TARGET}_features.csv
  ago_ant_features/selected_features/{random,butina}/{TARGET}_selected_features.txt
  models/{random,butina}/params/{TARGET}_best_hyperparameters.json

The script trains one model using the selected feature list for every requested
target/split, evaluates the held-out test set at the default probability cutoff,
and writes a compact summary table to results/test_metrics.csv.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_curve,
    roc_auc_score,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DEFAULT_TARGETS = [
    "AR",
    "ESR1",
    "ESR2",
    "ESRRA",
    "FXR",
    "GR",
    "PPARG",
    "PR",
    "RXRA",
    "THRB",
]
DEFAULT_SPLITS = ["random", "butina"]
METRIC_NAMES = ["SE", "SP", "ACC", "BA", "MCC", "AUC", "PR_AUC"]


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    root = repo_root_from_script()
    parser = argparse.ArgumentParser(
        description="Train final XGBoost models using prepared functional features."
    )
    parser.add_argument("--root", type=Path, default=root, help="Upload repository root.")
    parser.add_argument(
        "--split-types",
        nargs="+",
        default=DEFAULT_SPLITS,
        choices=DEFAULT_SPLITS,
        help="Dataset splits to train. Default: random butina.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_TARGETS,
        help="Targets to train. Default: all 10 targets.",
    )
    parser.add_argument(
        "--xgb-n-jobs",
        type=int,
        default=1,
        help="Threads used inside each XGBoost fit. Default: 1.",
    )
    parser.add_argument(
        "--skip-existing-models",
        action="store_true",
        help="Do not overwrite existing model files. Metrics are still regenerated from the newly fitted models.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Also save per-molecule test predictions under results/predictions/.",
    )
    return parser.parse_args()


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing selected feature file: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_feature_table(root: Path, target: str) -> pd.DataFrame:
    candidates = [
        root / "ago_ant_features" / "processed_features" / f"{target}_features.csv",
        root / "ago_ant_features" / "processed_features" / f"{target}_processed_features.csv",
        root / "ago_ant_features" / "processed_features" / f"{target}.csv",
    ]
    for path in candidates:
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        "Processed feature file not found for "
        f"{target}. Run `python scripts/feature.py all` first. Tried: "
        + ", ".join(str(path) for path in candidates)
    )


def normalize_smiles_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "SMILES" not in df.columns and "smiles" in df.columns:
        df = df.rename(columns={"smiles": "SMILES"})
    if "SMILES" not in df.columns:
        raise ValueError("Input table must contain a SMILES column")
    df["SMILES"] = df["SMILES"].astype(str)
    return df


def load_split_table(root: Path, split_type: str, target: str) -> pd.DataFrame:
    path = root / "ago_ant" / split_type / f"{target}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    df = normalize_smiles_column(pd.read_csv(path))
    required = {"SMILES", "Label", "split"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    df["split"] = df["split"].astype(str).str.lower()
    if not set(df["split"]).issubset({"train", "test"}):
        bad = sorted(set(df["split"]) - {"train", "test"})
        raise ValueError(f"Unexpected split values in {path}: {bad}")
    return df


def load_selected_features(root: Path, split_type: str, target: str) -> List[str]:
    path = (
        root
        / "ago_ant_features"
        / "selected_features"
        / split_type
        / f"{target}_selected_features.txt"
    )
    return read_lines(path)


def load_param_template(root: Path, split_type: str, target: str) -> Dict:
    path = root / "models" / split_type / "params" / f"{target}_best_hyperparameters.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing hyperparameter file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if "best_params_template" in data:
        params = data["best_params_template"]
    elif "best_params" in data:
        params = data["best_params"]
    else:
        params = data
    if not isinstance(params, dict):
        raise ValueError(f"Could not parse hyperparameter template from {path}")
    return params


def compute_scale_pos_weight(y_train: np.ndarray) -> float:
    n_pos = int(np.sum(y_train == 1))
    n_neg = int(np.sum(y_train == 0))
    return n_neg / n_pos if n_pos else 1.0


def materialize_params(template: Dict, y_train: np.ndarray, xgb_n_jobs: int) -> Dict:
    params = dict(template)
    if params.get("scale_pos_weight") is None:
        params["scale_pos_weight"] = compute_scale_pos_weight(y_train)
    params["n_jobs"] = xgb_n_jobs
    params.setdefault("eval_metric", "logloss")
    params.setdefault("verbosity", 0)
    return params


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    se = tp / (tp + fn) if (tp + fn) else 0.0
    sp = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc_value = roc_auc_score(y_true, y_proba)
    except Exception:
        auc_value = np.nan
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
    except Exception:
        pr_auc = np.nan
    return {
        "SE": float(se),
        "SP": float(sp),
        "ACC": float(accuracy_score(y_true, y_pred)),
        "BA": float(balanced_accuracy_score(y_true, y_pred)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "AUC": float(auc_value) if not pd.isna(auc_value) else np.nan,
        "PR_AUC": float(pr_auc) if not pd.isna(pr_auc) else np.nan,
    }


def align_split_and_features(split_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = normalize_smiles_column(feature_df)
    if "Label" in feature_df.columns:
        feature_df = feature_df.drop(columns=["Label"])
    if "split" in feature_df.columns:
        feature_df = feature_df.drop(columns=["split"])

    if len(feature_df) != len(split_df):
        raise ValueError(
            "Split table and feature table have different row counts: "
            f"split={len(split_df)}, features={len(feature_df)}. "
            "Regenerate processed features from the same ago_ant files."
        )

    if not split_df["SMILES"].astype(str).reset_index(drop=True).equals(
        feature_df["SMILES"].astype(str).reset_index(drop=True)
    ):
        # Fall back to row order because processed features are generated from the same
        # target CSVs. This keeps duplicate SMILES safe without many-to-many merging.
        mismatch_count = int(
            (
                split_df["SMILES"].astype(str).reset_index(drop=True)
                != feature_df["SMILES"].astype(str).reset_index(drop=True)
            ).sum()
        )
        raise ValueError(
            f"SMILES order mismatch in {mismatch_count} rows. "
            "Please regenerate processed features from the same split/source CSVs."
        )

    merged = pd.concat(
        [split_df[["SMILES", "Label", "split"]].reset_index(drop=True), feature_df.drop(columns=["SMILES"]).reset_index(drop=True)],
        axis=1,
    )
    return merged


def feature_matrix(df: pd.DataFrame, features: Sequence[str]) -> np.ndarray:
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise ValueError(f"Missing selected features from processed feature table: {missing[:10]}")
    return df.loc[:, list(features)].astype(float).to_numpy()


def train_one(
    root: Path,
    split_type: str,
    target: str,
    selected_features: Sequence[str],
    xgb_n_jobs: int,
    skip_existing_models: bool,
    save_predictions: bool,
) -> Dict:
    split_df = load_split_table(root, split_type, target)
    feature_df = load_feature_table(root, target)
    data = align_split_and_features(split_df, feature_df)

    train_df = data[data["split"] == "train"].copy()
    test_df = data[data["split"] == "test"].copy()
    if train_df.empty or test_df.empty:
        raise ValueError(f"{split_type}/{target} must contain both train and test rows")

    X_train = feature_matrix(train_df, selected_features)
    y_train = train_df["Label"].astype(int).to_numpy()
    X_test = feature_matrix(test_df, selected_features)
    y_test = test_df["Label"].astype(int).to_numpy()

    template = load_param_template(root, split_type, target)
    params = materialize_params(template, y_train, xgb_n_jobs=xgb_n_jobs)
    model = XGBClassifier(**params)
    model.fit(X_train, y_train)

    model_dir = root / "models" / split_type / "models"
    model_path = model_dir / f"{target}_xgb_fusion_model.joblib"
    model_dir.mkdir(parents=True, exist_ok=True)
    if not skip_existing_models or not model_path.exists():
        joblib.dump(model, model_path)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = calculate_metrics(y_test, y_pred, y_proba)

    if save_predictions:
        pred_dir = root / "results" / "predictions" / split_type
        pred_dir.mkdir(parents=True, exist_ok=True)
        pred_df = test_df[["SMILES", "Label"]].copy()
        pred_df.insert(0, "target", target)
        pred_df.insert(1, "split_type", split_type)
        pred_df["proba_1"] = y_proba
        pred_df["pred"] = y_pred
        pred_df["correct"] = (pred_df["pred"].astype(int) == pred_df["Label"].astype(int)).astype(int)
        pred_df.to_csv(pred_dir / f"{target}_test_predictions.csv", index=False)

    row = {
        "target": target,
        "split_type": split_type,
    }
    row.update(metrics)
    return row


def macro_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split_type, group in metrics_df.groupby("split_type", sort=False):
        row = {
            "split_type": split_type,
            "n_targets": int(group["target"].nunique()),
        }
        for metric in METRIC_NAMES:
            row[metric] = float(group[metric].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    print(f"Using upload root: {root}", flush=True)

    rows = []
    for split_type in args.split_types:
        for target in args.targets:
            selected_features = load_selected_features(root, split_type, target)
            print(
                f"Training {split_type}/{target} with {len(selected_features)} selected features",
                flush=True,
            )
            row = train_one(
                root=root,
                split_type=split_type,
                target=target,
                selected_features=selected_features,
                xgb_n_jobs=args.xgb_n_jobs,
                skip_existing_models=args.skip_existing_models,
                save_predictions=args.save_predictions,
            )
            rows.append(row)

    results_dir = root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(results_dir / "test_metrics.csv", index=False)
    macro_summary(metrics_df).to_csv(results_dir / "test_metrics_macro_summary.csv", index=False)
    print(f"Saved: {results_dir / 'test_metrics.csv'}", flush=True)
    print(f"Saved: {results_dir / 'test_metrics_macro_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
