import sys, os
sys.path.append(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import csv
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, f1_score,
                              precision_score, recall_score)

from config import (RESULT_DIR, TEST_SPLIT, SEED, BATCH_SIZE,
                    LR, KELAS_LIST, IMG_SIZE, NUM_CLASSES,
                    GROUP_AWARE_SPLIT)
from src.seeding import set_seed
from src.data.data_loader import load_dataset
from src.data.grouping import compute_group_labels, group_aware_holdout
from src.data.preprocessing import get_transforms
from src.data.augmentation import KidneyDataset
from src.models.efficientnet_model import build_model
from src.models.gradcam import generate_gradcam

import torch.nn as nn
import torch.optim as optim

EPOCHS   = 10
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = os.path.join(RESULT_DIR, "baseline")
os.makedirs(SAVE_DIR, exist_ok=True)

set_seed()
print(f"Device: {DEVICE}")

# ── Load & split ──────────────────────────────────────────────────────────────
all_paths, all_labels = load_dataset()
idx_all = np.arange(len(all_paths))

# GROUP_AWARE_SPLIT: cegah irisan near-identical dari scan yang sama tersebar
# ke train dan test sekaligus (lihat src/data/grouping.py).
if GROUP_AWARE_SPLIT:
    print("Group-aware splitting AKTIF — mengelompokkan citra serupa...")
    groups = compute_group_labels(all_paths)

    # StratifiedGroupKFold: group-aware DAN stratified, sehingga komposisi
    # kelas pada test/val tetap mendekati populasi (lihat src/data/grouping.py).
    idx_tv,    idx_test = group_aware_holdout(
        idx_all, all_labels, groups, TEST_SPLIT, SEED)
    idx_train, idx_val  = group_aware_holdout(
        idx_tv,  all_labels, groups, 0.20, SEED)
else:
    idx_tv, idx_test = train_test_split(
        idx_all, test_size=TEST_SPLIT,
        stratify=all_labels, random_state=SEED)

    idx_train, idx_val = train_test_split(
        idx_tv, test_size=0.20,
        stratify=all_labels[idx_tv], random_state=SEED)

print(f"Train : {len(idx_train):,} ({len(idx_train)/len(all_paths)*100:.1f}%)")
print(f"Val   : {len(idx_val):,} ({len(idx_val)/len(all_paths)*100:.1f}%)")
print(f"Test  : {len(idx_test):,} ({len(idx_test)/len(all_paths)*100:.1f}%)")

transform_train, transform_val = get_transforms()

loader_train = DataLoader(
    KidneyDataset(all_paths[idx_train], all_labels[idx_train], transform_train),
    batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
loader_val = DataLoader(
    KidneyDataset(all_paths[idx_val], all_labels[idx_val], transform_val),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
loader_test = DataLoader(
    KidneyDataset(all_paths[idx_test], all_labels[idx_test], transform_val),
    batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
model     = build_model().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Kriteria pemilihan model terbaik = macro F1 validasi, MENGIKUTI Bab 3.9
# proposal dan konsisten dengan sisi FL (KidneyFedProxStrategy.aggregate_fit).
# Sebelumnya memakai val_acc, sehingga baseline dan FL memilih checkpoint
# dengan kriteria berbeda dan tidak sebanding.
best_val_f1  = 0.0
best_val_acc = 0.0
best_weights = None

# ── Siapkan CSV ───────────────────────────────────────────────────────────────
csv_path = os.path.join(SAVE_DIR, "metrics.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "epoch", "train_loss", "train_acc",
        "val_loss", "val_acc", "val_f1", "is_best"
    ])
    writer.writeheader()

# ── Training loop ─────────────────────────────────────────────────────────────
print("\nMemulai training baseline...")
print("-" * 75)

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train()
    run_loss, run_correct = 0.0, 0
    for imgs, labels in loader_train:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        run_loss    += loss.item() * imgs.size(0)
        run_correct += (out.argmax(1) == labels).sum().item()

    train_loss = run_loss    / len(loader_train.dataset)
    train_acc  = run_correct / len(loader_train.dataset)

    # Val
    model.eval()
    v_loss, v_correct = 0.0, 0
    v_preds, v_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader_val:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out   = model(imgs)
            loss  = criterion(out, labels)
            v_loss    += loss.item() * imgs.size(0)
            v_correct += (out.argmax(1) == labels).sum().item()
            v_preds.extend(out.argmax(1).cpu().numpy())
            v_labels.extend(labels.cpu().numpy())

    val_loss = v_loss    / len(loader_val.dataset)
    val_acc  = v_correct / len(loader_val.dataset)
    val_f1   = f1_score(v_labels, v_preds, average='macro', zero_division=0)

    scheduler.step()

    # Cek best
    is_best = val_f1 > best_val_f1
    if is_best:
        best_val_f1  = val_f1
        best_val_acc = val_acc
        best_weights = {k: v.cpu().clone()
                        for k, v in model.state_dict().items()}
        torch.save(best_weights,
                   os.path.join(SAVE_DIR, "best_model.pth"))

    # Simpan ke CSV
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "epoch", "train_loss", "train_acc",
            "val_loss", "val_acc", "val_f1", "is_best"
        ])
        writer.writerow({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "val_f1":     round(val_f1,     6),
            "is_best":    1 if is_best else 0
        })

    print(f"Epoch {epoch:02d}/{EPOCHS} | "
          f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}%  "
          f"F1: {val_f1:.4f}"
          + (" ★ BEST" if is_best else ""))

print("-" * 75)
print(f"Best Val Macro F1: {best_val_f1:.4f}  (Val Accuracy: {best_val_acc*100:.2f}%)")

# ── Evaluasi test set ─────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("HASIL EVALUASI TEST SET")
print("=" * 50)

model.load_state_dict(best_weights)
model.eval()

preds_all, labels_all = [], []
with torch.no_grad():
    for imgs, labels in loader_test:
        imgs = imgs.to(DEVICE)
        preds_all.extend(model(imgs).argmax(1).cpu().numpy())
        labels_all.extend(labels.numpy())

preds_all  = np.array(preds_all)
labels_all = np.array(labels_all)

acc       = accuracy_score(labels_all, preds_all)
precision = precision_score(labels_all, preds_all,
                            average='macro', zero_division=0)
recall    = recall_score(labels_all, preds_all,
                         average='macro', zero_division=0)
f1        = f1_score(labels_all, preds_all,
                     average='macro', zero_division=0)

print(f"Accuracy       : {acc*100:.4f}%")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall   : {recall:.4f}")
print(f"Macro F1-Score : {f1:.4f}")
print()
print(classification_report(labels_all, preds_all,
                             target_names=KELAS_LIST, digits=4))

# Simpan hasil test ke CSV terpisah
test_csv_path = os.path.join(SAVE_DIR, "test_results.csv")
with open(test_csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "accuracy", "precision", "recall", "f1"
    ])
    writer.writeheader()
    writer.writerow({
        "accuracy":  round(acc,       6),
        "precision": round(precision, 6),
        "recall":    round(recall,    6),
        "f1":        round(f1,        6)
    })

# ── Confusion matrix ──────────────────────────────────────────────────────────
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(labels_all, preds_all)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=KELAS_LIST, yticklabels=KELAS_LIST,
            linewidths=0.5, linecolor='white',
            annot_kws={"size": 12, "weight": "bold"})
ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual Label',    fontsize=12, fontweight='bold')
ax.set_title('Confusion Matrix — Baseline', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "confusion_matrix.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Confusion matrix tersimpan.")

# ── Kurva training ────────────────────────────────────────────────────────────
import pandas as pd
df = pd.read_csv(csv_path)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.patch.set_facecolor('#F8F9FA')

axes[0].plot(df["epoch"], df["train_loss"], 'o-', color='#2E86AB',
             label='Train Loss', linewidth=2)
axes[0].plot(df["epoch"], df["val_loss"],   's-', color='#C73E1D',
             label='Val Loss',   linewidth=2)
best_epoch = df[df["is_best"] == 1]["epoch"].values
axes[0].axvline(x=best_epoch[0], color='green',
                linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch[0]})')
axes[0].set_title('Loss', fontweight='bold')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True, linestyle='--', alpha=0.5)

axes[1].plot(df["epoch"], df["train_acc"] * 100, 'o-', color='#2E86AB',
             label='Train Acc', linewidth=2)
axes[1].plot(df["epoch"], df["val_acc"]   * 100, 's-', color='#C73E1D',
             label='Val Acc',   linewidth=2)
axes[1].set_title('Accuracy', fontweight='bold')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
axes[1].grid(True, linestyle='--', alpha=0.5)

axes[2].plot(df["epoch"], df["val_f1"], 'o-', color='#A23B72',
             label='Val F1', linewidth=2)
axes[2].set_title('Macro F1-Score', fontweight='bold')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('F1-Score')
axes[2].legend()
axes[2].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, "kurva_training.png"),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"Kurva training tersimpan.")

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
generate_gradcam(model, all_paths[idx_test],
                 all_labels[idx_test], DEVICE,
                 save_dir=os.path.join(SAVE_DIR, "gradcam"))

print("\nBaseline selesai.")
print(f"Semua hasil tersimpan di: {SAVE_DIR}")