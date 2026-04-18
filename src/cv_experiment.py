"""
Stratified 5-Fold Cross-Validation deneyi (kucuk veri savunmasi).

Ders Notu Bolum 2.1 (Small DL): "If validation set is too small, it may be
more reliable to use stratified/group-aware K-fold on the training side
while keeping the test set fixed."

Bu modul:
  1. Test setini sabit tutar (30 ornek).
  2. Kalan 170 ornek uzerinde stratified 5-fold CV ile her iki modeli kisa
     sureli (10 epoch) egitir.
  3. Fold bazli mean +- std validation accuracy/F1 raporlar.
  4. Sonuclari results/cv_results.json ve results/cv_summary.png olarak yazar.

Cikis: tek bir hold-out sayisi yerine guven araligi → "%96.67 sansli mi yoksa
gercekten boyle mi?" sorusuna kanit.
"""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import DataLoader

from config import (
    DEVICE, MODELS_DIR, RESULTS_DIR, RANDOM_SEED, TEST_RATIO
)
from data_preprocessing import (
    HeadCTDataset, compute_train_statistics, get_transforms, load_labels
)
from custom_cnn import get_custom_cnn
from pretrained_model import get_convnext_model
from train import set_seed, train_one_epoch, validate

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


N_FOLDS = 5
QUICK_EPOCHS = 10
BATCH_SIZE = 8


def _split_test_first(seed: int = RANDOM_SEED):
    """Test setini sabit ayir (orijinal split ile ayni mantik)."""
    df = load_labels()
    rng = np.random.RandomState(seed)
    idx = np.arange(len(df))
    labels = df["label"].values

    # Stratified test holdout
    skf = StratifiedKFold(n_splits=int(round(1 / TEST_RATIO)), shuffle=True, random_state=seed)
    train_idx, test_idx = next(skf.split(idx, labels))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def _train_quick(model_fn, train_paths, train_labels, val_paths, val_labels,
                 mean, std, epochs: int = QUICK_EPOCHS) -> Dict:
    set_seed(RANDOM_SEED)

    tr_tf = get_transforms(mean, std, is_train=True, augment=True)
    va_tf = get_transforms(mean, std, is_train=False, augment=False)

    tr_loader = DataLoader(
        HeadCTDataset(train_paths, train_labels, tr_tf),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
    )
    va_loader = DataLoader(
        HeadCTDataset(val_paths, val_labels, va_tf),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    model = model_fn().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_loss = float("inf")
    best_acc = 0.0
    best_f1 = 0.0
    best_recall = 0.0

    for ep in range(1, epochs + 1):
        train_one_epoch(model, tr_loader, criterion, optimizer, DEVICE,
                        use_mixup=False, max_grad_norm=1.0)
        val_loss, val_acc = validate(model, va_loader, criterion, DEVICE)

        # F1 ve recall
        model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for x, y in va_loader:
                logits = model(x.to(DEVICE))
                all_p.extend(logits.argmax(1).cpu().numpy())
                all_t.extend(y.numpy())
        f1 = f1_score(all_t, all_p, average="weighted")
        rec = recall_score(all_t, all_p, average="weighted")

        if val_loss < best_loss:
            best_loss = val_loss
            best_acc = val_acc
            best_f1 = f1
            best_recall = rec

    return {"acc": best_acc, "f1": best_f1, "recall": best_recall, "loss": best_loss}


def run_cv(n_folds: int = N_FOLDS, epochs: int = QUICK_EPOCHS) -> Dict:
    """5-fold CV her iki model icin."""
    train_pool_df, _test_df = _split_test_first()
    paths = train_pool_df["image_path"].values
    labels = train_pool_df["label"].values

    print(f"\n[CV] Stratified {n_folds}-Fold CV  ·  pool={len(paths)}  ·  epochs/fold={epochs}")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)

    results = {"convnext": [], "custom_cnn": []}

    for fold, (tr_idx, va_idx) in enumerate(skf.split(paths, labels), start=1):
        tr_paths, va_paths = paths[tr_idx].tolist(), paths[va_idx].tolist()
        tr_labels, va_labels = labels[tr_idx].tolist(), labels[va_idx].tolist()

        print(f"\n[CV] Fold {fold}/{n_folds}  train={len(tr_paths)}  val={len(va_paths)}")

        # Stats sadece bu fold'un train'inden
        mean, std = compute_train_statistics(tr_paths)

        for name, model_fn in [
            ("convnext", lambda: get_convnext_model(pretrained=True)),
            ("custom_cnn", get_custom_cnn),
        ]:
            print(f"  [CV] {name}")
            res = _train_quick(model_fn, tr_paths, tr_labels, va_paths, va_labels,
                               mean, std, epochs=epochs)
            res["fold"] = fold
            results[name].append(res)
            print(f"    acc={res['acc']:.4f}  f1={res['f1']:.4f}  recall={res['recall']:.4f}")

    # Ozet
    summary = {}
    for name, runs in results.items():
        accs = [r["acc"] for r in runs]
        f1s = [r["f1"] for r in runs]
        recs = [r["recall"] for r in runs]
        summary[name] = {
            "folds": runs,
            "acc_mean": float(np.mean(accs)),
            "acc_std": float(np.std(accs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s)),
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs)),
        }

    out_path = RESULTS_DIR / "cv_results.json"
    with open(str(out_path), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[CV] Sonuclar: {out_path}")

    _plot_cv(summary, RESULTS_DIR / "cv_summary.png")

    print("\n[CV] OZET")
    print(f"  ConvNeXt-Tiny: acc = {summary['convnext']['acc_mean']:.3f} +- {summary['convnext']['acc_std']:.3f}")
    print(f"  Custom CNN  : acc = {summary['custom_cnn']['acc_mean']:.3f} +- {summary['custom_cnn']['acc_std']:.3f}")

    return summary


def _plot_cv(summary: Dict, save_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    models = list(summary.keys())
    display = {"convnext": "ConvNeXt-Tiny", "custom_cnn": "Custom CNN"}
    color = {"convnext": "#1e3a5f", "custom_cnn": "#0d9488"}

    # Sol: Per-fold accuracy
    ax = axes[0]
    width = 0.35
    folds = list(range(1, len(summary[models[0]]["folds"]) + 1))
    x = np.arange(len(folds))

    for i, m in enumerate(models):
        accs = [r["acc"] for r in summary[m]["folds"]]
        ax.bar(x + (i - 0.5) * width, accs, width,
               color=color[m], label=display[m], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}" for f in folds])
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Stratified 5-Fold CV  ·  Fold-bazli accuracy",
                 fontsize=11, fontweight="600")
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9)

    # Sag: mean +- std
    ax = axes[1]
    means = [summary[m]["acc_mean"] for m in models]
    stds = [summary[m]["acc_std"] for m in models]
    bars = ax.bar(
        [display[m] for m in models], means,
        yerr=stds, capsize=8,
        color=[color[m] for m in models], alpha=0.85,
        edgecolor="white", linewidth=1.5,
    )
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.015,
                f"{m:.3f} ± {s:.3f}",
                ha="center", fontsize=10, fontweight="600")

    ax.set_ylabel("Mean Validation Accuracy")
    ax.set_title("Model robustlugu  ·  CV mean ± std",
                 fontsize=11, fontweight="600")
    ax.set_ylim(0.5, 1.05)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Cross-Validation Deneyimi  —  Tek hold-out sansi mi, gercek performans mi?",
        fontsize=12.5, fontweight="700", y=1.02,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"[CV] Grafik: {save_path}")


if __name__ == "__main__":
    run_cv()
