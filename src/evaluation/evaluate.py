import os
import json
import csv
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (confusion_matrix, classification_report,
                              accuracy_score, f1_score,
                              precision_score, recall_score)
from torchinfo import summary
from src.models.efficientnet_model import build_model
from src.data.augmentation import KidneyDataset
from src.data.preprocessing import get_transforms
from config import (BATCH_SIZE, NUM_CLASSES, KELAS_LIST,
                    RESULT_DIR, IMG_SIZE, ALPHAS, K_FOLD)


def select_best_model(device):
    """
    Memilih model terbaik berdasarkan rata-rata macro F1 validasi
    dari seluruh fold per alpha (sesuai spesifikasi Bagian 11.1).

    Langkah:
    1. Untuk setiap alpha, hitung rata-rata F1 dari semua fold yang selesai
    2. Pilih alpha dengan rata-rata F1 tertinggi
    3. Dari alpha terpilih, ambil bobot model dari fold dengan F1 tertinggi

    Return: state_dict model terbaik, info alpha dan fold terbaiknya.
    """
    alpha_stats = {}  # alpha -> {avg_f1, best_fold, best_fold_f1, num_done}

    for alpha in ALPHAS:
        alpha_dir   = os.path.join(RESULT_DIR, f"alpha_{alpha}")
        status_path = os.path.join(alpha_dir, "fold_status.json")
        if not os.path.exists(status_path):
            continue
        with open(status_path) as f:
            status = json.load(f)

        fold_f1s     = []
        best_fold    = None
        best_fold_f1 = 0.0

        for fold in range(1, K_FOLD + 1):
            info = status.get(f"fold_{fold}", {})
            if not info.get("done"):
                continue
            f1 = info["metrics"]["best_val_f1"]
            fold_f1s.append(f1)
            if f1 > best_fold_f1:
                best_fold_f1 = f1
                best_fold    = fold

        if fold_f1s:
            alpha_stats[alpha] = {
                "avg_f1":       np.mean(fold_f1s),
                "best_fold":    best_fold,
                "best_fold_f1": best_fold_f1,
                "num_done":     len(fold_f1s),
            }

    if not alpha_stats:
        raise RuntimeError("Tidak ada eksperimen yang selesai.")

    # Pilih alpha dengan rata-rata F1 tertinggi
    best_alpha = max(alpha_stats, key=lambda a: alpha_stats[a]["avg_f1"])
    info       = alpha_stats[best_alpha]

    # Muat bobot dari fold terbaik pada alpha terpilih
    model_path = os.path.join(RESULT_DIR, f"alpha_{best_alpha}",
                              f"fold_{info['best_fold']}", "best_model.pth")
    best_state = torch.load(model_path, map_location=device)

    best_info = {
        "alpha":   best_alpha,
        "fold":    info["best_fold"],
        "avg_f1":  info["avg_f1"],
        "fold_f1": info["best_fold_f1"],
    }

    # Tampilkan ringkasan
    print("Rata-rata Macro F1 per \u03b1:")
    for alpha in ALPHAS:
        if alpha in alpha_stats:
            s = alpha_stats[alpha]
            marker = " \u2190 terbaik" if alpha == best_alpha else ""
            print(f"  \u03b1={alpha}: Avg F1={s['avg_f1']:.4f} "
                  f"({s['num_done']} fold){marker}")
        else:
            print(f"  \u03b1={alpha}: belum selesai")

    print(f"\nModel terpilih: \u03b1={best_alpha} | "
          f"Avg F1={info['avg_f1']:.4f} | "
          f"Best Fold {info['best_fold']} (F1={info['best_fold_f1']:.4f})")
    return best_state, best_info


def full_evaluation(model_state, paths_test, labels_test, device,
                    save_dir=None):
    """
    Evaluasi lengkap pada test set:
    accuracy, precision, recall, F1, confusion matrix, efisiensi.
    """
    if save_dir is None:
        save_dir = os.path.join(RESULT_DIR, "final")
    os.makedirs(save_dir, exist_ok=True)

    _, transform_val = get_transforms()
    loader_test = DataLoader(
        KidneyDataset(paths_test, labels_test, transform_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = build_model().to(device)
    model.load_state_dict(model_state)
    model.eval()

    # ── Metrik klasifikasi ──
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in loader_test:
            imgs = imgs.to(device)
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

    print("\n" + "="*50)
    print("HASIL EVALUASI TEST SET")
    print("="*50)
    print(f"Accuracy       : {acc*100:.4f}%")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall   : {recall:.4f}")
    print(f"Macro F1-Score : {f1:.4f}")
    print()
    print(classification_report(labels_all, preds_all,
                                 target_names=KELAS_LIST, digits=4))

    # ── Confusion matrix ──
    cm  = confusion_matrix(labels_all, preds_all)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=KELAS_LIST, yticklabels=KELAS_LIST,
                linewidths=0.5, linecolor='white',
                annot_kws={"size": 12, "weight": "bold"})
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label',    fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix — Test Set',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix_final.png"),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ── Efisiensi ──
    dummy  = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)
    info   = summary(model, input_size=(1, 3, IMG_SIZE, IMG_SIZE),
                     verbose=0)
    flops  = info.total_mult_adds * 2
    params = sum(p.numel() for p in model.parameters())

    times = []
    is_cuda = device.type == "cuda"
    with torch.no_grad():
        for _ in range(100):
            if is_cuda:
                torch.cuda.synchronize()
            t0 = time.time()
            model(dummy)
            if is_cuda:
                torch.cuda.synchronize()
            times.append(time.time() - t0)
    avg_time = np.mean(times) * 1000

    laporan = (f"Total Parameter : {params:,}\n"
               f"FLOPs (approx)  : {flops/1e9:.3f} GFLOPs\n"
               f"Waktu Inferensi : {avg_time:.3f} ms/citra\n")
    print("\n" + "="*50)
    print("EVALUASI EFISIENSI MODEL")
    print("="*50)
    print(laporan)

    with open(os.path.join(save_dir, "laporan_efisiensi.txt"), "w") as f:
        f.write(laporan)

    return {"acc": acc, "precision": precision,
            "recall": recall, "f1": f1}


def evaluate_all_alphas(device, save_dir=None):
    """
    Evaluasi model TERBAIK DARI SETIAP ALPHA (bukan hanya pemenang keseluruhan)
    pada test set yang sama, lalu susun tabel perbandingan accuracy/macro-P/
    macro-R/macro-F1 antar alpha sesuai spesifikasi Bagian 11.4 / Bab 3.9
    proposal ("tabel perbandingan metrik untuk masing-masing nilai alpha").
    """
    from src.data.data_loader import load_dataset

    if save_dir is None:
        save_dir = os.path.join(RESULT_DIR, "final")
    os.makedirs(save_dir, exist_ok=True)

    all_paths, all_labels = load_dataset()
    idx_test = np.load(os.path.join(RESULT_DIR, "test_indices.npy"))
    paths_test, labels_test = all_paths[idx_test], all_labels[idx_test]

    _, transform_val = get_transforms()
    loader_test = DataLoader(
        KidneyDataset(paths_test, labels_test, transform_val),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    rows = []
    for alpha in ALPHAS:
        alpha_dir   = os.path.join(RESULT_DIR, f"alpha_{alpha}")
        status_path = os.path.join(alpha_dir, "fold_status.json")
        if not os.path.exists(status_path):
            continue
        with open(status_path) as f:
            status = json.load(f)

        best_fold, best_fold_f1 = None, -1.0
        for fold in range(1, K_FOLD + 1):
            info = status.get(f"fold_{fold}", {})
            if info.get("done") and info["metrics"]["best_val_f1"] > best_fold_f1:
                best_fold_f1 = info["metrics"]["best_val_f1"]
                best_fold    = fold
        if best_fold is None:
            continue

        model_path = os.path.join(alpha_dir, f"fold_{best_fold}", "best_model.pth")
        if not os.path.exists(model_path):
            continue

        model = build_model().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels in loader_test:
                imgs = imgs.to(device)
                preds_all.extend(model(imgs).argmax(1).cpu().numpy())
                labels_all.extend(labels.numpy())

        rows.append({
            "alpha":           alpha,
            "best_fold":       best_fold,
            "accuracy":        accuracy_score(labels_all, preds_all),
            "macro_precision": precision_score(labels_all, preds_all,
                                               average='macro', zero_division=0),
            "macro_recall":    recall_score(labels_all, preds_all,
                                            average='macro', zero_division=0),
            "macro_f1":        f1_score(labels_all, preds_all,
                                        average='macro', zero_division=0),
        })

    print("\n" + "="*70)
    print("TABEL PERBANDINGAN METRIK TEST SET ANTAR ALPHA")
    print("="*70)
    print(f"{'Alpha':>6}  {'Fold':>4}  {'Accuracy':>9}  {'Macro-P':>8}  "
          f"{'Macro-R':>8}  {'Macro-F1':>9}")
    for r in rows:
        print(f"  {r['alpha']:>4}  {r['best_fold']:>4}  {r['accuracy']*100:8.2f}%  "
              f"{r['macro_precision']:.4f}    {r['macro_recall']:.4f}    "
              f"{r['macro_f1']:.4f}")

    out_csv = os.path.join(save_dir, "alpha_comparison_test_metrics.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "alpha", "best_fold", "accuracy",
            "macro_precision", "macro_recall", "macro_f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nTabel disimpan: {out_csv}")

    return rows


def plot_training_curves(save_dir=None):
    """
    Plot kurva training loss, validation loss, dan macro F1 per ronde
    untuk setiap alpha (semua fold dalam satu plot).
    Sesuai spesifikasi Bagian 11.2.
    """
    import pandas as pd

    if save_dir is None:
        save_dir = os.path.join(RESULT_DIR, "final")
    os.makedirs(save_dir, exist_ok=True)

    for alpha in ALPHAS:
        alpha_dir = os.path.join(RESULT_DIR, f"alpha_{alpha}")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f'Kurva Training \u2014 \u03b1 = {alpha}',
                     fontsize=14, fontweight='bold')

        has_data = False
        for fold in range(1, K_FOLD + 1):
            csv_path = os.path.join(alpha_dir, f"fold_{fold}", "metrics.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            if df.empty:
                continue
            has_data = True

            # Plot Loss (train + val)
            axes[0].plot(df["round"], df["val_loss"],
                         label=f"Fold {fold} (val)", alpha=0.8)
            if "train_loss" in df.columns:
                train_col = pd.to_numeric(df["train_loss"], errors='coerce')
                if train_col.notna().any():
                    axes[0].plot(df["round"], train_col,
                                 label=f"Fold {fold} (train)",
                                 linestyle='--', alpha=0.5)

            # Plot Macro F1
            axes[1].plot(df["round"], df["val_f1"],
                         label=f"Fold {fold}", alpha=0.8)

        if not has_data:
            plt.close(fig)
            continue

        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training & Validation Loss per Round')
        axes[0].legend(fontsize=7)
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Macro F1')
        axes[1].set_title('Validation Macro F1 per Round')
        axes[1].legend(fontsize=7)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = os.path.join(save_dir, f"training_curves_alpha_{alpha}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Kurva training tersimpan: {out_path}")