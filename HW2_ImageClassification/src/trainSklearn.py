"""
Third approach: features extracted and saved + sklearn classifier

- Load features.npz
- Standardize with StandardScaler fit only on train to avoid leakage
- Tain more classifier and chose the best on validation macro-f1
- Save in results/approach3_selected.json: name of the best, fit time, val metrics
"""

import json
import time
import joblib

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from config import CHECKPOINTS_DIR, FEATURES_DIR, RESULTS_DIR, SEED


def fit_and_score(name, clf, X_tr, y_tr, X_val, y_val):
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    fit_time = time.perf_counter() - t0
    pred_val = clf.predict(X_val)
    metrics = {
        "classifier": name,
        "fit_time_sec": fit_time,
        "val_accuracy": float(accuracy_score(y_val, pred_val)),
        "val_macro_f1": float(f1_score(y_val, pred_val, average="macro", zero_division=0)),
    }

    if hasattr(clf, "coef_"):
        metrics["n_parameters"] = int(clf.coef_.size + clf.intercept_.size)
        metrics["tree_nodes"] = None
    else:
        metrics["n_parameters"] = None
        metrics["tree_nodes"] = int(sum(t.tree_.node_count for t in clf.estimators_))
    return clf, metrics


def main() -> None:
    data = np.load(FEATURES_DIR / "features.npz", allow_pickle=True)
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]

    # StandardScaler: fit only on training, transform on val (niente leakage)
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_val_s = scaler.transform(X_tr), scaler.transform(X_val)

    # Candidates: linear and non linear
    candidates = {
        "logreg": LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1, random_state=SEED),
        "rf": RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=SEED),
    }

    selected, all_metrics = {}, {}
    for name, clf in candidates.items():
        print(f"\nFit {name} on train features {X_tr_s.shape}")
        fitted, metrics = fit_and_score(name, clf, X_tr_s, y_tr, X_val_s, y_val)
        print(f"  val acc={metrics['val_accuracy']:.4f}  val macro-F1={metrics['val_macro_f1']:.4f}  fit={metrics['fit_time_sec']:.1f}s")
        all_metrics[name] = metrics
        selected[name] = (fitted, scaler)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (fitted, scaler) in selected.items():
        pipe = Pipeline([("scaler", scaler), ("clf", fitted)])
        out_path = CHECKPOINTS_DIR / f"approach3_{name}.joblib"
        joblib.dump(pipe, out_path)
        print(f"Salvato {out_path}")

    # Best on validation
    best = max(all_metrics.values(), key=lambda m: m["val_macro_f1"])["classifier"]
    print(f"\n-> Classifier selected on validation: {best}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {
        "best_classifier": best,
        "selected": {name: {k: v for k, v in m.items() if k != "classifier"}
                     for name, m in all_metrics.items()},
        "feature_extraction_time_train_val_sec": None,  # aggiornato da evaluateTest.py
    }
    # read the time of extraction
    try:
        with open(RESULTS_DIR / "approach3_extraction_time.json") as f:
            times = json.load(f)
        out["feature_extraction_time_train_val_sec"] = times["train"] + times["val"]
    except FileNotFoundError:
        pass

    with open(RESULTS_DIR / "approach3_selected.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"Salvato {RESULTS_DIR / 'approach3_selected.json'}")


if __name__ == "__main__":
    main()