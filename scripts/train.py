import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader

from dataset import SPLITS_CSV, PcosDataset, eval_tf, get_fold_loaders
from model import PcosModel

WEIGHTS_JSON = SPLITS_CSV.parent / "class_weights.json"
CKPT_DIR = Path("../checkpoints")
CKPT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_class_weights():
    # Order matches label_id: 0 = noninfected, 1 = infected
    with open(WEIGHTS_JSON) as f:
        w = json.load(f)["weights"]
    return torch.tensor(w, dtype=torch.float32, device=DEVICE)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    losses, probs, labels = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            logits = model(x)
        logits = logits.float()
        losses.append(criterion(logits, y).item() * len(y))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        labels.append(y.cpu().numpy())

    probs = np.concatenate(probs)
    y_true = np.concatenate(labels)
    y_pred = probs.argmax(axis=1)
    return {
        "loss": sum(losses) / len(y_true),
        "acc": float((y_pred == y_true).mean()),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_noninf": f1_score(y_true, y_pred, pos_label=0),
        "recall_noninf": recall_score(y_true, y_pred, pos_label=0),
        "prec_noninf": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "recall_inf": recall_score(y_true, y_pred, pos_label=1),
        "auc": roc_auc_score(y_true, probs[:, 1]),
        "pr_auc_noninf": average_precision_score(y_true == 0, probs[:, 0]),
    }


def fit(model, train_loader, val_loader, epochs, lr, tag):
    criterion = nn.CrossEntropyLoss(weight=load_class_weights())
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )
    best_f1, history = -1.0, []
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
                logits = model(x)
            loss = criterion(logits.float(), y)  # loss in float32 under bf16 autocast
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * len(y)
            n += len(y)

        scheduler.step()
        m = evaluate(model, val_loader, criterion)
        m["epoch"], m["train_loss"] = epoch, run_loss / n
        history.append(m)
        print(
            f"[{tag}] ep {epoch:02d} | train {m['train_loss']:.4f} | val {m['loss']:.4f} "
            f"| acc {m['acc']:.3f} | f1m {m['f1_macro']:.3f} "
            f"| rec_non {m['recall_noninf']:.3f} | prec_non {m['prec_noninf']:.3f} | auc {m['auc']:.3f}"
        )

        # Select checkpoints by macro-F1, not accuracy (classes are imbalanced)
        if m["f1_macro"] > best_f1:
            best_f1 = m["f1_macro"]
            torch.save(model.state_dict(), CKPT_DIR / f"{tag}_best.pt")

    with open(CKPT_DIR / f"{tag}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    return history


def run_sanity(lr, epochs):
    # Overfit 100 training images to check that the pipeline can learn
    df = pd.read_csv(SPLITS_CSV)
    small = df[df["split"] == "train"].sample(100, random_state=0)
    print("sanity labels:", small["label_id"].value_counts().to_dict())
    loader = DataLoader(PcosDataset(small, eval_tf), batch_size=16, shuffle=True)
    eval_loader = DataLoader(PcosDataset(small, eval_tf), batch_size=16)
    fit(PcosModel().to(DEVICE), loader, eval_loader, epochs, lr, tag="sanity")


def run_fold(fold, epochs, lr, rank):
    train_loader, val_loader = get_fold_loaders(fold)
    model = PcosModel(rank=rank, alpha=2 * rank).to(DEVICE)
    return fit(model, train_loader, val_loader, epochs, lr, tag=f"fold{fold}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--rank", type=int, default=8)
    a = ap.parse_args()

    if a.sanity:
        run_sanity(lr=3e-4, epochs=a.epochs if a.epochs != 8 else 30)
    elif a.fold >= 0:
        run_fold(a.fold, a.epochs, a.lr, a.rank)
    else:
        # --fold -1 runs all five folds and reports last-epoch mean x std
        last = [run_fold(k, a.epochs, a.lr, a.rank)[-1] for k in range(5)]
        for key in ["acc", "f1_macro", "recall_noninf", "prec_noninf", "auc", "pr_auc_noninf"]:
            v = [m[key] for m in last]
            print(f"{key}: {np.mean(v):.3f} ± {np.std(v):.3f}")