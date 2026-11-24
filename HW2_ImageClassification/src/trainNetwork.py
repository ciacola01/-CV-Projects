"""
Trains MobileNetV3-Small for approach 1 and 2

Process
- Fixes the seeds and builds the model
- Prints which parameters are trainable
- Trains on the train set and evaluates on the validation set
- Saves the checkpoints (weights) with the highest validation macro-F1
- Saves CSV for every epoch history and a JSON a summary with hyperparameters, trainable parameters, training time and selected checkpoint
"""

import argparse
import json
import time

import pandas as pd
import torch
from torch import nn

from config import(BATCH_SIZE, CHECKPOINTS_DIR, RESULTS_DIR, SEED, TRAIN_DEFAULTS, WEIGHT_DECAY)
from data import get_dataloaders
from engine import evaluate, train_one_epoch
from model import build_model, count_trainable_params, trainable_layer_names
from utils import get_device, set_seed

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["head", "full"], required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--tag", type=str, default=None, help="run name to compare lerning rates")
    args = parser.parse_args()

    mode = args.mode
    freeze = mode == "head"
    lr = args.lr if args.lr is not None else TRAIN_DEFAULTS[mode]["lr"]
    epochs = args.epochs if args.epochs is not None else TRAIN_DEFAULTS[mode]["epochs"]
    approach = "approach2_head" if freeze else "approach1_full"
    run_name = args.tag or approach

    # Set seed before building the model
    set_seed(SEED)
    device = get_device()
    model = build_model(freeze_backbone=freeze).to(device)
    train_loader, val_loader, _, _ = get_dataloaders()

    # Trainable parameters
    trainable, total = count_trainable_params(model)
    layers = trainable_layer_names(model)
    print(f"Run: {run_name} | mode={mode} | lr={lr} | epochs={epochs} | batch={BATCH_SIZE}")
    print(f"Trainable parameters: {trainable:,} of {total:,}")
    if freeze:
        print("Trainable parameters tensors:", layers)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss() # standard loss for multi-classification
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=WEIGHT_DECAY)

    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CHECKPOINTS_DIR / f"{run_name}.pt"

    history, best_f1, best_epoch, best_acc = [], -1.0, -1, -1.0
    train_time, total_start = 0.0, time.perf_counter()

    for epoch in range(1, epochs+1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, freeze)
        epoch_time = time.perf_counter() - t0
        train_time += epoch_time

        val = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {val['loss']:.4f} acc {val['accuracy']:.4f} F1 {val['macro_f1']:.4f} | {epoch_time:.0f}s")
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc, "val_loss": val["loss"], "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"], "epoch_time_sec": epoch_time})

        # Selection rule
        if val["macro_f1"] > best_f1:
            best_f1, best_acc, best_epoch = val["macro_f1"], val["accuracy"], epoch
            torch.save(model.state_dict(), ckpt_path)
            print(f"    -> new best checkpoint (val macro-f1 { best_f1:.4f})")

    # Final files
    pd.DataFrame(history).to_csv(RESULTS_DIR / f"{run_name}_history.csv", index=False)
    summary = {
        "run_name": run_name, "mode":mode, "loss": "CrossEntropyLoss", "optimizer": "AdamW", "lr": lr, "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
        "epochs": epochs, "seed": SEED, "device": str(device), "selection_rule": "epoch with the highest validation macro-F1",
        "best_epoch": best_epoch, "best_val_macro_f1": best_f1, "best_val_acc": best_acc, "trainable_parameters": trainable, "total_parameters": total,
        "trainable_layers": layers if freeze else "all parameters",
        "train_time_sec": train_time,
        "total_time_sec": time.perf_counter() - total_start,
        "checkpoint": ckpt_path.name,
    }
    with open(RESULTS_DIR / f"{run_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. Best epoch: {best_epoch} (val macro-f1 {best_f1:.4f}, val acc {best_acc:.4f}). Training time {train_time /60:.1f} min")


if __name__ == "__main__":
    main()