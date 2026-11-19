#!/usr/bin/env python
"""Task 4: high-imbalance clinical disease predictor.

Detects pathological fetal heart rate traces in the UCI Cardiotocography data
(NSP = 3, about 8% of records). Three classifiers (SVM, random forest and
XGBoost) are each trained with no imbalance handling, with class weights and
with SMOTE. Hyperparameters are tuned by stratified 5-fold cross-validation to
maximise recall while keeping mean precision at or above a floor (0.5 by
default), and the chosen configuration is evaluated once on a held-out test set.

Usage:
    python src/train.py
    python src/train.py --n-iter 30 --precision-floor 0.5 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
import urllib.request
from functools import partial
from pathlib import Path

import imblearn
import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             confusion_matrix, fbeta_score, make_scorer,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                     StratifiedKFold, train_test_split)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "fetal_health.csv"
DATA_URL = ("https://huggingface.co/datasets/sayanroy058/"
            "Fetal_Health_Classification/resolve/main/fetal_health.csv")

TARGET = "fetal_health"   # NSP label: 1 normal, 2 suspect, 3 pathologic
POSITIVE_NSP = 3
PRECISION_FLOOR = 0.75   # minimum mean CV precision for selection

# CSV column names mapped to the short codes used by the UCI repository.
FEATURES = {
    "baseline value": "LB",
    "accelerations": "AC",
    "fetal_movement": "FM",
    "uterine_contractions": "UC",
    "light_decelerations": "DL",
    "severe_decelerations": "DS",
    "prolongued_decelerations": "DP",
    "abnormal_short_term_variability": "ASTV",
    "mean_value_of_short_term_variability": "MSTV",
    "percentage_of_time_with_abnormal_long_term_variability": "ALTV",
    "mean_value_of_long_term_variability": "MLTV",
    "histogram_width": "Width",
    "histogram_min": "Min",
    "histogram_max": "Max",
    "histogram_number_of_peaks": "Nmax",
    "histogram_number_of_zeroes": "Nzeros",
    "histogram_mode": "Mode",
    "histogram_mean": "Mean",
    "histogram_median": "Median",
    "histogram_variance": "Variance",
    "histogram_tendency": "Tendency",
}

FEATURE_DESCRIPTIONS = {
    "LB": "FHR baseline (beats per minute)",
    "AC": "Accelerations per second",
    "FM": "Fetal movements per second",
    "UC": "Uterine contractions per second",
    "DL": "Light decelerations per second",
    "DS": "Severe decelerations per second",
    "DP": "Prolonged decelerations per second",
    "ASTV": "Percentage of time with abnormal short-term variability",
    "MSTV": "Mean value of short-term variability",
    "ALTV": "Percentage of time with abnormal long-term variability",
    "MLTV": "Mean value of long-term variability",
    "Width": "Width of the FHR histogram",
    "Min": "Minimum of the FHR histogram",
    "Max": "Maximum of the FHR histogram",
    "Nmax": "Number of histogram peaks",
    "Nzeros": "Number of histogram zeros",
    "Mode": "Histogram mode",
    "Mean": "Histogram mean",
    "Median": "Histogram median",
    "Variance": "Histogram variance",
    "Tendency": "Histogram tendency",
}

MODELS = ("SVM", "Random Forest", "XGBoost")
STRATEGIES = ("None", "Class weights", "SMOTE")

# SVM is searched over the full grid; random forest and XGBoost by random search.
PARAM_SPACES = {
    "SVM": {
        "clf__C": [0.1, 0.3, 1, 3, 10, 30, 100],
        "clf__gamma": ["scale", 0.003, 0.01, 0.03, 0.1, 0.3],
    },
    "Random Forest": {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [None, 4, 8, 12, 16],
        "clf__min_samples_leaf": [1, 2, 4, 8],
        "clf__max_features": ["sqrt", 0.3, 0.5],
    },
    "XGBoost": {
        "clf__n_estimators": [200, 400, 600],
        "clf__max_depth": [2, 3, 4, 6],
        "clf__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.6, 0.8, 1.0],
        "clf__min_child_weight": [1, 3, 5],
    },
}

def true_positives(y_true, y_pred) -> int:
    return int(((np.asarray(y_true) == 1) & (np.asarray(y_pred) == 1)).sum())


def false_positives(y_true, y_pred) -> int:
    return int(((np.asarray(y_true) == 0) & (np.asarray(y_pred) == 1)).sum())


SCORERS = {
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "specificity": make_scorer(recall_score, pos_label=0, zero_division=0),
    "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
    "tp": make_scorer(true_positives),
    "fp": make_scorer(false_positives),
}


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_data(path: Path = DATA_PATH, log: logging.Logger | None = None) -> pd.DataFrame:
    """Read the CTG table, downloading it first if the file is missing."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        if log:
            log.info("Data file not found; downloading %s", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def prepare_data(df: pd.DataFrame):
    """Drop exact duplicate rows, rename features and build the binary target."""
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    X = df[list(FEATURES)].rename(columns=FEATURES)
    nsp = df[TARGET].astype(int)
    y = (nsp == POSITIVE_NSP).astype(int)
    info = {
        "n_raw": int(n_raw),
        "n_duplicates_removed": int(n_raw - len(df)),
        "n_records": int(len(df)),
        "n_features": int(X.shape[1]),
        "n_missing_values": int(X.isna().sum().sum()),
        "nsp_counts": {int(k): int(v) for k, v in nsp.value_counts().sort_index().items()},
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "imbalance_ratio": float((y == 0).sum() / y.sum()),
    }
    return X, y, nsp, info


def split_data(X, y, nsp, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split on the three-level NSP label."""
    X_tr, X_te, y_tr, y_te, nsp_tr, nsp_te = train_test_split(
        X, y, nsp, test_size=test_size, stratify=nsp, random_state=seed)
    info = {
        "test_size": test_size,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "train_positive": int(y_tr.sum()), "test_positive": int(y_te.sum()),
        "train_nsp_counts": {int(k): int(v) for k, v in nsp_tr.value_counts().sort_index().items()},
        "test_nsp_counts": {int(k): int(v) for k, v in nsp_te.value_counts().sort_index().items()},
    }
    return X_tr, X_te, y_tr, y_te, info


# --------------------------------------------------------------------------
# Models and tuning
# --------------------------------------------------------------------------

def build_pipeline(model: str, strategy: str, pos_weight: float, seed: int = 42) -> Pipeline:
    """Imputation, scaling, optional SMOTE (training folds only) and the classifier."""
    weighted = strategy == "Class weights"
    steps = [("impute", SimpleImputer(strategy="median")),
             ("scale", StandardScaler())]
    if strategy == "SMOTE":
        steps.append(("smote", SMOTE(k_neighbors=5, random_state=seed)))
    if model == "SVM":
        clf = SVC(kernel="rbf", class_weight="balanced" if weighted else None)
    elif model == "Random Forest":
        clf = RandomForestClassifier(
            class_weight="balanced_subsample" if weighted else None,
            n_jobs=1, random_state=seed)
    elif model == "XGBoost":
        clf = XGBClassifier(
            objective="binary:logistic", eval_metric="logloss", tree_method="hist",
            scale_pos_weight=pos_weight if weighted else 1.0,
            n_jobs=1, random_state=seed, verbosity=0)
    else:
        raise ValueError(f"Unknown model: {model}")
    steps.append(("clf", clf))
    return Pipeline(steps)


def select_by_recall(cv_results, floor: float = PRECISION_FLOOR) -> int:
    """Index of the candidate with the highest mean CV recall among those whose
    mean CV precision is at least `floor`; ties are broken by mean F2. If no
    candidate reaches the floor, the candidate with the highest F2 is returned."""
    recall = np.round(np.asarray(cv_results["mean_test_recall"], dtype=float), 6)
    precision = np.asarray(cv_results["mean_test_precision"], dtype=float)
    f2 = np.asarray(cv_results["mean_test_f2"], dtype=float)
    eligible = np.flatnonzero(precision >= floor)
    if eligible.size == 0:
        return int(np.argmax(f2))
    order = np.lexsort((-f2[eligible], -recall[eligible]))
    return int(eligible[order[0]])


def tune(model, strategy, X, y, cv, *, n_iter=30, floor=PRECISION_FLOOR, seed=42, n_jobs=2):
    """Cross-validated hyperparameter search for one model/strategy pair."""
    pos_weight = float((y == 0).sum() / (y == 1).sum())
    pipe = build_pipeline(model, strategy, pos_weight, seed)
    kwargs = dict(scoring=SCORERS, refit=partial(select_by_recall, floor=floor),
                  cv=cv, n_jobs=n_jobs, error_score="raise")
    if model == "SVM":
        search = GridSearchCV(pipe, PARAM_SPACES[model], **kwargs)
    else:
        search = RandomizedSearchCV(pipe, PARAM_SPACES[model], n_iter=n_iter,
                                    random_state=seed, **kwargs)
    t0 = time.perf_counter()
    search.fit(X, y)
    return search, time.perf_counter() - t0


def summarise_search(search, n_splits: int) -> dict:
    """Per-fold scores, mean and SD of the selected candidate."""
    i, r = search.best_index_, search.cv_results_
    folds = {m: [float(r[f"split{k}_test_{m}"][i]) for k in range(n_splits)]
             for m in SCORERS}
    return {
        "best_params": {k.replace("clf__", ""): _py(v) for k, v in search.best_params_.items()},
        "n_candidates": int(len(r["params"])),
        "cv_folds": folds,
        "cv_mean": {m: float(np.mean(v)) for m, v in folds.items()},
        "cv_sd": {m: float(np.std(v, ddof=1)) for m, v in folds.items()},
        "cv_total": {m: int(round(sum(folds[m]))) for m in ("tp", "fp")},
    }


def tune_all(X, y, cv, *, n_iter=30, floor=PRECISION_FLOOR, seed=42, n_jobs=2, log=None):
    """Tune every model under every imbalance strategy."""
    n_splits = cv.get_n_splits()
    searches, summaries = {}, {}
    for model in MODELS:
        for strategy in STRATEGIES:
            search, secs = tune(model, strategy, X, y, cv, n_iter=n_iter,
                                floor=floor, seed=seed, n_jobs=n_jobs)
            s = summarise_search(search, n_splits)
            s["search_seconds"] = round(secs, 1)
            searches[(model, strategy)] = search
            summaries[(model, strategy)] = s
            if log:
                log.info("CV  %-13s | %-13s | %3d candidates | %6.1f s | recall %.3f ± %.3f"
                         " | precision %.3f | F2 %.3f | best %s",
                         model, strategy, s["n_candidates"], secs,
                         s["cv_mean"]["recall"], s["cv_sd"]["recall"],
                         s["cv_mean"]["precision"], s["cv_mean"]["f2"],
                         s["best_params"])
    return searches, summaries


def _choose(summaries: dict, keys: list, floor: float):
    table = {f"mean_test_{m}": [summaries[k]["cv_mean"][m] for k in keys]
             for m in ("recall", "precision", "f2")}
    return keys[select_by_recall(table, floor)]


def select_final(summaries: dict, floor: float = PRECISION_FLOOR):
    """Apply the selection rule across all nine tuned configurations."""
    return _choose(summaries, list(summaries), floor)


def best_strategy_per_model(summaries: dict, floor: float = PRECISION_FLOOR) -> dict:
    """Apply the selection rule within each model."""
    return {m: _choose(summaries, [k for k in summaries if k[0] == m], floor)[1]
            for m in MODELS}


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def continuous_score(model, X) -> np.ndarray:
    """Probability for tree models, decision value for the SVM."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(c - h), float(c + h)]


def bootstrap_ci(y, s, fn, n_boot: int = 2000, seed: int = 42) -> list:
    """Percentile interval from a bootstrap stratified by class."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])
        stats[b] = fn(y[idx], s[idx])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return [float(lo), float(hi)]


def evaluate(model, X, y, seed: int = 42) -> dict:
    """Test-set metrics at the model's default decision rule."""
    y = np.asarray(y)
    pred = model.predict(X)
    s = continuous_score(model, X)
    tn, fp, fn, tp = (int(v) for v in confusion_matrix(y, pred, labels=[0, 1]).ravel())
    return {
        "recall": tp / (tp + fn),
        "specificity": tn / (tn + fp),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "npv": tn / (tn + fn) if tn + fn else 0.0,
        "f1": float(fbeta_score(y, pred, beta=1, zero_division=0)),
        "f2": float(fbeta_score(y, pred, beta=2, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": (tp + tn) / len(y),
        "roc_auc": float(roc_auc_score(y, s)),
        "average_precision": float(average_precision_score(y, s)),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "recall_ci95": wilson_ci(tp, tp + fn),
        "specificity_ci95": wilson_ci(tn, tn + fp),
        "precision_ci95": wilson_ci(tp, tp + fp),
        "roc_auc_ci95": bootstrap_ci(y, s, roc_auc_score, seed=seed),
        "average_precision_ci95": bootstrap_ci(y, s, average_precision_score, seed=seed),
    }


def evaluate_all(searches: dict, X, y, seed: int = 42, log=None) -> dict:
    out = {}
    for key, search in searches.items():
        t = evaluate(search.best_estimator_, X, y, seed)
        out[key] = t
        if log:
            log.info("TEST %-13s | %-13s | recall %.3f (%d/%d) | precision %.3f | "
                     "specificity %.3f | F2 %.3f | ROC-AUC %.3f | AP %.3f",
                     key[0], key[1], t["recall"], t["tp"], t["tp"] + t["fn"],
                     t["precision"], t["specificity"], t["f2"], t["roc_auc"],
                     t["average_precision"])
    return out


def test_curves(model, X, y) -> dict:
    """ROC and precision-recall curves plus the default operating point."""
    y = np.asarray(y)
    s = continuous_score(model, X)
    pred = model.predict(X)
    fpr, tpr, _ = roc_curve(y, s)
    prec, rec, _ = precision_recall_curve(y, s)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"fpr": fpr, "tpr": tpr, "precision": prec, "recall": rec,
            "operating_point": {"fpr": fp / (fp + tn), "tpr": tp / (tp + fn),
                                "precision": tp / (tp + fp) if tp + fp else 0.0}}


def importance_table(model, X, y, seed: int = 42, n_repeats: int = 30) -> pd.DataFrame:
    """Permutation importance on the test set, scored by average precision."""
    r = permutation_importance(model, X, y, scoring="average_precision",
                               n_repeats=n_repeats, random_state=seed, n_jobs=1)
    df = pd.DataFrame({
        "feature": X.columns,
        "description": [FEATURE_DESCRIPTIONS[c] for c in X.columns],
        "importance_mean": r.importances_mean,
        "importance_sd": r.importances_std,
    })
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _py(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def comparison_table(summaries: dict, tests: dict, final_key) -> pd.DataFrame:
    rows = []
    for key, s in summaries.items():
        t = tests[key]
        rows.append({
            "model": key[0], "strategy": key[1],
            "cv_recall_mean": s["cv_mean"]["recall"], "cv_recall_sd": s["cv_sd"]["recall"],
            "cv_precision_mean": s["cv_mean"]["precision"],
            "cv_specificity_mean": s["cv_mean"]["specificity"],
            "cv_f2_mean": s["cv_mean"]["f2"],
            "cv_average_precision_mean": s["cv_mean"]["average_precision"],
            "cv_tp_total": s["cv_total"]["tp"], "cv_fp_total": s["cv_total"]["fp"],
            "test_recall": t["recall"], "test_precision": t["precision"],
            "test_specificity": t["specificity"], "test_f2": t["f2"],
            "test_roc_auc": t["roc_auc"], "test_average_precision": t["average_precision"],
            "tp": t["tp"], "fn": t["fn"], "fp": t["fp"], "tn": t["tn"],
            "final_model": key == final_key,
        })
    return pd.DataFrame(rows)


def folds_table(summaries: dict, metric: str = "recall") -> pd.DataFrame:
    rows = []
    for key, s in summaries.items():
        folds = s["cv_folds"][metric]
        row = {"model": key[0], "strategy": key[1]}
        row.update({f"fold_{k + 1}": v for k, v in enumerate(folds)})
        row.update({"mean": s["cv_mean"][metric], "sd": s["cv_sd"][metric]})
        rows.append(row)
    return pd.DataFrame(rows)


def search_table(searches: dict, floor: float) -> pd.DataFrame:
    rows = []
    for (model, strategy), search in searches.items():
        r = search.cv_results_
        n_splits = sum(1 for k in r if k.startswith("split") and k.endswith("_test_recall"))
        for i, params in enumerate(r["params"]):
            row = {"model": model, "strategy": strategy,
                   "params": json.dumps({k.replace("clf__", ""): _py(v) for k, v in params.items()})}
            for m in SCORERS:
                vals = [r[f"split{k}_test_{m}"][i] for k in range(n_splits)]
                row[f"mean_{m}"] = float(np.mean(vals))
                row[f"sd_{m}"] = float(np.std(vals, ddof=1))
            for m in ("tp", "fp"):
                row[f"total_{m}"] = int(round(sum(r[f"split{k}_test_{m}"][i] for k in range(n_splits))))
            row["meets_precision_floor"] = bool(r["mean_test_precision"][i] >= floor)
            row["selected"] = bool(i == search.best_index_)
            rows.append(row)
    return pd.DataFrame(rows)


def choose_from_table(table: pd.DataFrame, floor: float) -> pd.Series:
    """Apply the selection rule to the pooled candidates of all configurations."""
    cv = {f"mean_test_{m}": table[f"mean_{m}"].to_numpy() for m in ("recall", "precision", "f2")}
    return table.iloc[select_by_recall(cv, floor)]


def floor_sensitivity(table: pd.DataFrame, floors) -> list:
    """Which candidate the rule would pick under different precision floors."""
    out = []
    for f in floors:
        row = choose_from_table(table, f)
        out.append({"floor": f, "model": row["model"], "strategy": row["strategy"],
                    "params": json.loads(row["params"]),
                    "cv_recall": float(row["mean_recall"]), "cv_precision": float(row["mean_precision"]),
                    "cv_specificity": float(row["mean_specificity"]), "cv_f2": float(row["mean_f2"]),
                    "cv_tp_total": int(row["total_tp"]), "cv_fp_total": int(row["total_fp"])})
    return out


def versions() -> dict:
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "imbalanced-learn": imblearn.__version__, "xgboost": xgboost.__version__,
            "matplotlib": matplotlib.__version__}


def setup_logging(log_path: Path | None = None) -> logging.Logger:
    log = logging.getLogger("task4")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        handlers.append(logging.FileHandler(log_path, mode="w", encoding="utf-8"))
    for h in handlers:
        h.setFormatter(fmt)
        log.addHandler(h)
    return log


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Task 4: imbalanced CTG classifier (pathologic vs rest).")
    p.add_argument("--data", type=Path, default=DATA_PATH, help="Path to fetal_health.csv")
    p.add_argument("--out-dir", type=Path, default=ROOT, help="Folder that receives results/, figures/ and models/")
    p.add_argument("--n-iter", type=int, default=30, help="Random-search candidates for random forest and XGBoost")
    p.add_argument("--folds", type=int, default=5, help="Number of stratified CV folds")
    p.add_argument("--precision-floor", type=float, default=PRECISION_FLOOR, help="Minimum mean CV precision for model selection")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraction of records held out for testing")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--n-jobs", type=int, default=2, help="Parallel jobs for the CV search")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import plots  # local module in src/

    args = parse_args(argv)
    out = args.out_dir.resolve()
    for sub in ("results", "figures", "models"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    log = setup_logging(out / "results" / "train.log")
    t_start = time.perf_counter()
    log.info("Task 4 | pathologic vs rest | seed=%d | folds=%d | n_iter=%d | precision floor=%.2f",
             args.seed, args.folds, args.n_iter, args.precision_floor)

    df = load_data(args.data, log)
    X, y, nsp, data_info = prepare_data(df)
    log.info("Data: %d rows, %d duplicates removed -> %d records, %d features, %d missing values",
             data_info["n_raw"], data_info["n_duplicates_removed"], data_info["n_records"],
             data_info["n_features"], data_info["n_missing_values"])
    log.info("NSP counts %s | pathologic %d (%.1f%%) | imbalance %.1f:1",
             data_info["nsp_counts"], data_info["n_positive"],
             100 * data_info["prevalence"], data_info["imbalance_ratio"])

    X_tr, X_te, y_tr, y_te, split_info = split_data(X, y, nsp, args.test_size, args.seed)
    log.info("Split: train %d (%d pathologic) | test %d (%d pathologic)",
             split_info["n_train"], split_info["train_positive"],
             split_info["n_test"], split_info["test_positive"])

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    searches, summaries = tune_all(X_tr, y_tr, cv, n_iter=args.n_iter, floor=args.precision_floor,
                                   seed=args.seed, n_jobs=args.n_jobs, log=log)
    final_key = select_final(summaries, args.precision_floor)
    best = best_strategy_per_model(summaries, args.precision_floor)
    log.info("Best strategy per model (CV rule): %s", best)
    log.info("Final model (CV rule): %s + %s", *final_key)

    tests = evaluate_all(searches, X_te, y_te, args.seed, log)
    final_model = searches[final_key].best_estimator_
    imp = importance_table(final_model, X_te, y_te, args.seed)
    log.info("Top features (permutation, AP drop): %s",
             ", ".join(f"{f} {v:.3f}" for f, v in imp[["feature", "importance_mean"]].head(5).values))

    # Tables, metrics and model
    res = out / "results"
    comparison_table(summaries, tests, final_key).round(4).to_csv(res / "model_comparison.csv", index=False)
    folds_table(summaries, "recall").round(4).to_csv(res / "cv_recall_folds.csv", index=False)
    table = search_table(searches, args.precision_floor)
    table.round(4).to_csv(res / "cv_search_results.csv", index=False)
    sensitivity = floor_sensitivity(table, sorted({0.5, args.precision_floor}))
    for r in sensitivity:
        log.info("Floor %.2f -> %s + %s | CV recall %.3f (%d TP) | precision %.3f (%d FP)",
                 r["floor"], r["model"], r["strategy"], r["cv_recall"], r["cv_tp_total"],
                 r["cv_precision"], r["cv_fp_total"])
    imp.round(4).to_csv(res / "permutation_importance.csv", index=False)
    joblib.dump(final_model, out / "models" / "final_model.joblib", compress=3)

    # Test-set predictions of the final model, and where its missed cases rank
    y_true = np.asarray(y_te)
    scores = continuous_score(final_model, X_te)
    pred = final_model.predict(X_te)
    pd.DataFrame({"record": X_te.index, "nsp": nsp.loc[X_te.index].to_numpy(),
                  "y_true": y_true, "probability": scores, "predicted": pred}
                 ).round(4).to_csv(res / "test_predictions.csv", index=False)
    neg_scores = scores[y_true == 0]
    missed = [{"record": int(r), "probability": float(p),
               "negatives_scoring_higher": int((neg_scores > p).sum())}
              for r, p, t, yhat in zip(X_te.index, scores, y_true, pred) if t == 1 and yhat == 0]
    for c in missed:
        log.info("Missed test case: record %d | probability %.3f | %d of %d negatives score higher",
                 c["record"], c["probability"], c["negatives_scoring_higher"], neg_scores.size)

    metrics = {
        "task": "Pathological CTG (NSP = 3) vs normal or suspect",
        "dataset": data_info,
        "split": split_info,
        "cv": {"n_splits": args.folds, "shuffle": True, "seed": args.seed},
        "selection_rule": (f"highest mean CV recall among candidates with mean CV precision "
                           f">= {args.precision_floor}; ties broken by mean F2"),
        "precision_floor": args.precision_floor,
        "n_iter_random_search": args.n_iter,
        "configs": [{"model": k[0], "strategy": k[1], **summaries[k], "test": tests[k]}
                    for k in summaries],
        "best_strategy_per_model": best,
        "final_model": {"model": final_key[0], "strategy": final_key[1],
                        "best_params": summaries[final_key]["best_params"],
                        "file": "models/final_model.joblib"},
        "floor_sensitivity": sensitivity,
        "final_model_missed_test_cases": missed,
        "permutation_importance": imp.to_dict(orient="records"),
        "versions": versions(),
    }

    # Figures
    fig_dir = out / "figures"
    labels = {m: plots.config_label(m, best[m]) for m in MODELS}
    plots.plot_class_distribution(data_info["nsp_counts"], fig_dir / "fig1_class_distribution.png")
    plots.plot_cv_folds(summaries, final_key, args.precision_floor, fig_dir / "fig2_cv_folds.png")
    curves = {m: test_curves(searches[(m, best[m])].best_estimator_, X_te, y_te) for m in MODELS}
    plots.plot_test_curves(curves, {m: tests[(m, best[m])] for m in MODELS}, labels,
                           split_info["test_positive"] / split_info["n_test"],
                           fig_dir / "fig3_roc_pr_test.png")
    plots.plot_confusion_matrices({m: tests[(m, best[m])] for m in MODELS}, labels,
                                  fig_dir / "fig4_confusion_matrices.png")
    plots.plot_importance(imp, plots.MODEL_COLORS[final_key[0]], path=fig_dir / "fig5_permutation_importance.png")

    metrics["runtime_seconds"] = round(time.perf_counter() - t_start, 1)
    with open(res / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_py)
    log.info("Done in %.1f min. Outputs written to results/, figures/ and models/",
             metrics["runtime_seconds"] / 60)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
