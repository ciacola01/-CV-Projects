"""
This is the final evaluation of all the selected models on the test set. 

So this script is meant to be run ONE time, when all the training and all the
choices (epochs, learning rates, classifiers, hyperparameters) are finished.

"""

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, f1_score
from torch import nn

from config import CHECKPOINTS_DIR, FEATURES_DIR, RESULTS_DIR
from data import get_dataloaders, get_train_eval_loader
from engine import evaluate
from model import build_model
from utils import get_device, set_seed


def metrics(y_true, y_pred) -> dict:
    return {"acc": float(np.mean(y_true == y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))}


def save_predictions_and_confusion(key, y_true, y_pred, class_names) -> None:
    """Saves one CSV with every test prediction and one with the confusion matrix."""
    names = list(class_names)
    pd.DataFrame({
        "test_index": np.arange(len(y_true)),
        "y_true": y_true, "y_pred": y_pred,
        "true_breed": [names[i] for i in y_true],
        "pred_breed": [names[i] for i in y_pred],
        "correct": y_true == y_pred,
    }).to_csv(RESULTS_DIR / f"predictions_{key}.csv", index=False)

    # Row = true breed, column = predicted breed. Off-diagonal cells are mistakes.
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(names))))
    pd.DataFrame(cm, index=names, columns=names).to_csv(RESULTS_DIR / f"confusion_{key}.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full_run", default="approach1_full",
                        help="run name of the selected approach 1 model")
    parser.add_argument("--head_run", default="approach2_head",
                        help="run name of the selected approach 2 model")
    parser.add_argument("--force", action="store_true",
                        help="run again even if test_results.json exists")
    args = parser.parse_args()

    out_path = RESULTS_DIR / "test_results.json"
    if out_path.exists() and not args.force:
        print(f"{out_path} already exists: the test set has already been evaluated.\n"
              "The homework asks to evaluate each selected model ONCE. "
              "Use --force only if you are sure.")
        return

    set_seed()
    device = get_device()
    criterion = nn.CrossEntropyLoss()

    # Clean train (no augmentation), validation and test loaders, all in fixed order
    train_loader = get_train_eval_loader()
    _, val_loader, test_loader, class_names = get_dataloaders()

    results = {}
    reference_labels = None   # test labels as seen by the networks: used for sanity checks

    for key, run, freeze in [("approach1_full", args.full_run, False), ("approach2_head", args.head_run, True)]:
        print(f"\n=== {key} (run: {run}) ===")
        with open(RESULTS_DIR / f"{run}_summary.json") as f:
            summary = json.load(f)

        model = build_model(freeze_backbone=freeze)
        state = torch.load(CHECKPOINTS_DIR / f"{run}.pt", map_location=device,
                           weights_only=True)
        model.load_state_dict(state)
        model.to(device)

        r_train = evaluate(model, train_loader, criterion, device)
        r_val = evaluate(model, val_loader, criterion, device)
        r_test = evaluate(model, test_loader, criterion, device)

        if reference_labels is None:
            reference_labels = r_test["labels"]
        
        save_predictions_and_confusion(key, r_test["labels"], r_test["predictions"], class_names)

        results[key] = {
            "run_name": run, "best_epoch": summary["best_epoch"],
            "train_acc": r_train["accuracy"], "train_macro_f1": r_train["macro_f1"],
            "val_acc": r_val["accuracy"], "val_macro_f1": r_val["macro_f1"],
            "test_acc": r_test["accuracy"], "test_macro_f1": r_test["macro_f1"],
        }
    
        print(f"train acc {r_train['accuracy']:.4f} | val acc {r_val['accuracy']:.4f} | TEST acc {r_test['accuracy']:.4f}, macro-F1 {r_test['macro_f1']:.4f}")

    # Approach 3 (scikit-learn pipelines on the saved features)
    feats = np.load(FEATURES_DIR / "features.npz")
    X_train, y_train = feats["X_train"], feats["y_train"]
    X_val, y_val = feats["X_val"], feats["y_val"]
    X_test, y_test = feats["X_test"], feats["y_test"]
    assert np.array_equal(y_test, reference_labels), "Test order differs between approaches!"

    with open(RESULTS_DIR / "approach3_selected.json") as f:
        sel3 = json.load(f)

    for name, chosen in sel3["selected"].items():
        key = f"approach3_{name}"
        print(f"\n=== {key} ===")
        pipe = joblib.load(CHECKPOINTS_DIR / f"{key}.joblib")
        m_train = metrics(y_train, pipe.predict(X_train))
        m_val = metrics(y_val, pipe.predict(X_val))
        test_pred = pipe.predict(X_test)
        m_test = metrics(y_test, test_pred)
        save_predictions_and_confusion(key, y_test, test_pred, class_names)

        results[key] = {
            "run_name": key, "selected_param": f"classifier={name}",
            "train_acc": m_train["acc"], "train_macro_f1": m_train["macro_f1"],
            "val_acc": m_val["acc"], "val_macro_f1": m_val["macro_f1"],
            "test_acc": m_test["acc"], "test_macro_f1": m_test["macro_f1"],
        }
        print(f"train acc {m_train['acc']:.4f} | val acc {m_val['acc']:.4f} | "
              f"TEST acc {m_test['acc']:.4f}, macro-F1 {m_test['macro_f1']:.4f}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()