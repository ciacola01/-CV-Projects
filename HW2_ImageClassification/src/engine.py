"""
Approach 1 (full) and approach 2 (head) share the same engine: the difference is which parameters are trainable

Process:
1. forward: network produces 37 scores for every image
2. loss: measures how wrong the scroes are
3. backward: PyTorch computes in which direction it should move to reduce the loss
4. step: the optimizer moves every trainable parameter by a small stes
"""

import torch
from sklearn.metrics import f1_score

from model import set_train_mode

"""You'll train one time and the accuracy is measured while training"""
def train_one_epoch(model, loader, criterion, optimizer, device, freeze_backbone):
    set_train_mode(model, freeze_backbone)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad() # Clear the gradients of the previous step
        out = model(x) # forward
        loss = criterion(out, y) # loss
        loss.backward() #backward
        optimizer.step() # update the weights

        total_loss += loss.item() * x.size(0) #weight by batch size
        correct +=(out.argmax(dim=1) == y).sum().item()
        n += x.size(0)

    return total_loss / n, correct / n


"""Evaluate a model on a loader (val, test or clean train) and returns a dict with: loss, accuracy, macro_f1, preds, labels"""
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n = 0.0, 0
    all_preds, all_labels = [] , []

    for x, y in loader:
        x,y = x.to(device), y.to(device)
        out = model(x)
        total_loss += criterion(out, y).item() * x.size(0)
        all_preds.append(out.argmax(dim=1).cpu())
        all_labels.append(y.cpu())
        n += x.size(0)

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    return{
        "loss": total_loss / n,
        "accuracy": float((preds == labels).mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "predictions": preds,
        "labels": labels,
    }