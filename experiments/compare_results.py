"""
experiments/compare_results.py
Membandingkan hasil semua 25 eksperimen (5 alpha × 5 fold).
Membuat tabel ringkasan dan menyimpan ke results/final/comparison_table.csv.

Cara pakai:
    python experiments/compare_results.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import csv
import numpy as np
import torch
from config import ALPHAS, RESULT_DIR, K_FOLD, dir_alpha, dir_final
from src.evaluation.evaluate import evaluate_all_alphas

FINAL_DIR = dir_final()
os.makedirs(FINAL_DIR, exist_ok=True)

rows = []

print("=" * 65)
print("TABEL PERBANDINGAN HASIL EKSPERIMEN")
print(f"{'Alpha':>6}  {'Fold':>4}  {'Best Val F1':>11}  {'Status':>10}")
print("=" * 65)

for alpha in ALPHAS:
    alpha_dir   = dir_alpha(alpha)
    status_path = os.path.join(alpha_dir, "fold_status.json")
    status      = {}
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = json.load(f)

    for fold in range(1, K_FOLD + 1):
        info = status.get(f"fold_{fold}", {})
        if info.get("done"):
            f1     = info["metrics"]["best_val_f1"]
            label  = "Selesai"
        else:
            f1     = None
            # Cek apakah ada checkpoint parsial
            fold_d = os.path.join(alpha_dir, f"fold_{fold}")
            if os.path.exists(fold_d):
                ckpts = [f for f in os.listdir(fold_d)
                         if f.startswith("checkpoint_round_")]
                label = f"Terjeda (ronde {len(ckpts)*10})" if ckpts else "Belum mulai"
            else:
                label = "Belum mulai"

        f1_str = f"{f1:.4f}" if f1 is not None else "—"
        print(f"  {alpha:>4}  Fold {fold}  {f1_str:>11}  {label}")
        rows.append({
            "alpha": alpha,
            "fold":  fold,
            "best_val_f1": f1 if f1 is not None else "",
            "status": label,
        })

print("=" * 65)

# ── Ringkasan per alpha ────────────────────────────────────────────────────────
print("\nRINGKASAN PER ALPHA (rata-rata Best Val F1 dari fold yang selesai):")
print(f"{'Alpha':>6}  {'Folds Done':>10}  {'Avg F1':>8}  {'Max F1':>8}")
print("-" * 45)
for alpha in ALPHAS:
    alpha_rows = [r for r in rows
                  if r["alpha"] == alpha and r["best_val_f1"] != ""]
    if alpha_rows:
        vals    = [r["best_val_f1"] for r in alpha_rows]
        avg_f1  = np.mean(vals)
        max_f1  = np.max(vals)
        done    = len(alpha_rows)
    else:
        avg_f1 = max_f1 = 0.0
        done   = 0
    print(f"  {alpha:>4}  {done:>10}  {avg_f1:.4f}  {max_f1:.4f}")

# ── Simpan CSV ─────────────────────────────────────────────────────────────────
out_csv = os.path.join(FINAL_DIR, "comparison_table.csv")
with open(out_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["alpha", "fold",
                                           "best_val_f1", "status"])
    writer.writeheader()
    writer.writerows(rows)

print(f"\nTabel disimpan: {out_csv}")

# ── Tabel perbandingan metrik TEST SET per alpha (Bab 3.9 proposal) ────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
evaluate_all_alphas(DEVICE)
