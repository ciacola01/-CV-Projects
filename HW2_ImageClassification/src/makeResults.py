"""
Builds ALL the tables and figures of the report from saved files.

It does not train or evaluate anything: it only reads what the other scripts
saved, so it can be run as many times as you like.
"""

import json

import matplotlib

matplotlib.use("Agg")  # no windows: figures are only saved to files
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import RESULTS_DIR, SEED

LABELS = {"approach1_full": "1. Full fine-tuning",
          "approach2_head": "2. Classifier-head training"}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def pct(x) -> str:
    return f"{100 * x:.2f}"


# Results table
def build_table(test, sel3) -> pd.DataFrame:
    """One row per evaluated model: parameters, cost and metrics (in %)."""
    rows = []
    for key, r in test.items():
        if key.startswith("approach3_"):
            name = key[len("approach3_"):]
            s = sel3["selected"][name]
            label = f"3. Features + {name} ({r['selected_param']})"
            # n_parameters is None for a random forest (it has trees, not weights)
            params = f"{s['n_parameters']:,}" if s["n_parameters"] is not None \
                else f"n/a ({s['tree_nodes']:,} tree nodes)"
            # cost = extraction of train+val features + fitting the selected classifier
            seconds = sel3["feature_extraction_time_train_val_sec"] + s["fit_time_sec"]
        else:
            summary = load_json(RESULTS_DIR / f"{r['run_name']}_summary.json")
            label = LABELS[key]
            params = f"{summary['trainable_parameters']:,}"
            seconds = summary["train_time_sec"]
        rows.append({"Approach": label, "Trainable params": params,
                     "Training time (s)": f"{seconds:.0f}",
                     "Train acc (%)": pct(r["train_acc"]), "Val acc (%)": pct(r["val_acc"]),
                     "Test acc (%)": pct(r["test_acc"]),
                     "Test macro-F1 (%)": pct(r["test_macro_f1"])})
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    """Tiny markdown table writer (avoids the extra 'tabulate' dependency)."""
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join([head, sep] + body)


# Figures
def plot_training_curves(test) -> None:
    """Loss and accuracy per epoch for approaches 1 and 2 (train vs validation)."""
    keys = [k for k in ("approach1_full", "approach2_head") if k in test]
    fig, axes = plt.subplots(2, len(keys), figsize=(6 * len(keys), 7), squeeze=False)
    for col, key in enumerate(keys):
        h = pd.read_csv(RESULTS_DIR / f"{test[key]['run_name']}_history.csv")
        for row, (metric, title) in enumerate([("loss", "Loss"), ("acc", "Accuracy")]):
            ax = axes[row, col]
            ax.plot(h["epoch"], h[f"train_{metric}"], marker="o", label="train (augmented)")
            ax.plot(h["epoch"], h[f"val_{metric}"], marker="o", linestyle="--", label="validation")
            ax.axvline(test[key]["best_epoch"], color="gray", linestyle=":", label="selected epoch")
            ax.set_title(f"{LABELS[key]}: {title}")
            ax.set_xlabel("epoch")
            ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def plot_confusion(key, title) -> None:
    """37x37 confusion matrix with breed names, normalized by row.

    Each row is a TRUE breed and sums to 1: the diagonal shows the fraction of
    correct predictions, the off-diagonal cells show where the mistakes go.
    """
    cm = pd.read_csv(RESULTS_DIR / f"confusion_{key}.csv", index_col=0)
    names = list(cm.index)
    norm = cm.values / np.maximum(cm.values.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=6)
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04, label="fraction of the true breed")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"confusion_{key}.png", dpi=150)
    plt.close(fig)


def top_confusions(key, k=10) -> list:
    """The k most frequent (true breed -> predicted breed) mistakes."""
    cm = pd.read_csv(RESULTS_DIR / f"confusion_{key}.csv", index_col=0)
    names = list(cm.index)
    arr = cm.values.copy()
    np.fill_diagonal(arr, 0)  # ignore the correct predictions
    rows = []
    for flat in np.argsort(arr, axis=None)[::-1][:k]:
        i, j = np.unravel_index(flat, arr.shape)
        if arr[i, j] == 0:
            break
        rows.append({"approach": key, "true_breed": names[i],
                     "predicted_breed": names[j], "count": int(arr[i, j])})
    return rows


def plot_examples(key, test_set, n=4) -> None:
    """n correct (top row) and n incorrect (bottom row) test predictions, picked
    at random with a fixed seed (so the figure is reproducible)."""
    rng = np.random.RandomState(SEED)
    pred = pd.read_csv(RESULTS_DIR / f"predictions_{key}.csv")
    groups = [(pred[pred["correct"]], "green"), (pred[~pred["correct"]], "red")]

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.8))
    for row, (subset, color) in enumerate(groups):
        picks = rng.choice(len(subset), size=min(n, len(subset)), replace=False)
        for col in range(n):
            ax = axes[row, col]
            ax.axis("off")
            if col >= len(picks):
                continue
            r = subset.iloc[picks[col]]
            img, _ = test_set[int(r["test_index"])]   # raw PIL image (no transform)
            ax.imshow(img)
            ax.set_title(f"True: {r['true_breed']}\nPred: {r['pred_breed']}",
                         fontsize=8, color=color)
    fig.suptitle(f"{key}: correct (top) and incorrect (bottom) test predictions")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"examples_{key}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    test = load_json(RESULTS_DIR / "test_results.json")
    sel3 = load_json(RESULTS_DIR / "approach3_selected.json")
    best3 = f"approach3_{sel3['best_classifier']}"   # chosen on VALIDATION
    main_keys = ["approach1_full", "approach2_head", best3]

    # 1) Table
    table = build_table(test, sel3)
    table.to_csv(RESULTS_DIR / "results_table.csv", index=False)
    (RESULTS_DIR / "results_table.md").write_text(to_markdown(table) + "\n")
    print(to_markdown(table))

    # 2) Training curves
    plot_training_curves(test)

    # 3) Confusion matrices + the most confused breeds
    all_confusions = []
    for key in main_keys:
        plot_confusion(key, f"Confusion matrix (test): {key}")
        all_confusions += top_confusions(key)
    pd.DataFrame(all_confusions).to_csv(RESULTS_DIR / "top_confusions.csv", index=False)
    print("\nMost frequent confusions (test):")
    print(pd.DataFrame(all_confusions).to_string(index=False))

    # 4) Prediction examples (needs the raw test images)
    from data import load_dataset   # imported here: needs torchvision
    test_set = load_dataset("test", transform=None, download=False)
    for key in main_keys:
        plot_examples(key, test_set)

    print(f"\nTables and figures saved in {RESULTS_DIR}")


if __name__ == "__main__":
    main()