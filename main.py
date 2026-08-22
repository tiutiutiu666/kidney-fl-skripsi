"""
main.py — Orkestrasi semua 25 eksperimen FL (5 alpha × 5 fold).

Alur:
1. Muat dataset → pisahkan test set (dikunci, tidak digunakan saat training)
2. Untuk setiap alpha: partisi Dirichlet → jalankan simulasi FL (Flower)
3. Pilih model terbaik dari semua 25 eksperimen berdasarkan macro F1
4. Evaluasi akhir pada test set: akurasi, presisi, recall, F1, confusion matrix
5. Generate visualisasi Grad-CAM pada model terbaik
"""
import os
import sys
import torch
import numpy as np
from sklearn.model_selection import train_test_split

from config import (ALPHAS, RESULT_DIR, TEST_SPLIT, SEED, GROUP_AWARE_SPLIT,
                    dir_alpha, dir_final)
from src.seeding import set_seed
from src.data.data_loader import load_dataset
from src.data.grouping import compute_group_labels, group_aware_holdout
from src.data.partitioner import dirichlet_partition
from src.federated.server import run_simulation
from src.evaluation.evaluate import (
    select_best_model, full_evaluation, plot_training_curves, evaluate_all_alphas,
)
from src.models.efficientnet_model import build_model
from src.models.gradcam import generate_gradcam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("SISTEM FEDERATED LEARNING — KLASIFIKASI CT SCAN GINJAL")
    set_seed()
    print(f"Device : {DEVICE}")
    print(f"{'='*60}\n")

    # ── 1. Muat dataset ───────────────────────────────────────────────────────
    all_paths, all_labels = load_dataset()

    # ── 2. Pisahkan test set (dikunci, hanya digunakan saat evaluasi akhir) ───
    # CATATAN METODOLOGI: test set 20% dipisah secara GLOBAL (stratified atas
    # seluruh dataset) SEBELUM partisi Dirichlet dijalankan, alih-alih setiap
    # klien memisahkan 20% dari data lokalnya sendiri setelah dipartisi.
    # Ini disengaja: partisi Dirichlet digambar ulang secara acak untuk setiap
    # nilai alpha (lihat pemanggilan dirichlet_partition di bawah), sehingga
    # jika test set dibentuk SETELAH partisi, isi test set akan berbeda-beda
    # antar skenario alpha dan antar klien — membuat tabel perbandingan
    # antar-alpha (Bab 3.9 proposal) tidak apple-to-apple karena tiap alpha
    # dievaluasi pada citra test yang berbeda. Dengan test set global yang
    # tetap (disimpan di test_indices.npy dan dipakai ulang di semua alpha),
    # seluruh 25 eksperimen dievaluasi pada test set yang identik.
    idx_all = np.arange(len(all_paths))

    # GROUP_AWARE_SPLIT: cegah irisan near-identical dari scan yang sama
    # tersebar ke train dan test sekaligus (lihat src/data/grouping.py).
    groups = None
    if GROUP_AWARE_SPLIT:
        print("Group-aware splitting AKTIF — mengelompokkan citra serupa...")
        groups = compute_group_labels(all_paths)
        idx_tv, idx_test = group_aware_holdout(
            idx_all, all_labels, groups, TEST_SPLIT, SEED)
    else:
        idx_tv, idx_test = train_test_split(
            idx_all, test_size=TEST_SPLIT,
            stratify=all_labels, random_state=SEED,
        )

    os.makedirs(RESULT_DIR, exist_ok=True)
    # Nama file dibedakan per mode split. Tanpa ini, test_indices.npy dari
    # skenario split acak akan dimuat ulang saat GROUP_AWARE_SPLIT diaktifkan
    # dan menimpa idx_test yang baru — sementara idx_tv tetap versi group-aware,
    # sehingga train dan test justru saling tumpang tindih (leakage kembali).
    suffix        = "_group" if GROUP_AWARE_SPLIT else ""
    test_idx_path = os.path.join(RESULT_DIR, f"test_indices{suffix}.npy")
    if not os.path.exists(test_idx_path):
        np.save(test_idx_path, idx_test)
        print(f"Test indices disimpan: {test_idx_path}")
    else:
        idx_test = np.load(test_idx_path)
        print(f"Test indices dimuat: {test_idx_path}")

    # Pengaman: test set yang dimuat harus tetap terpisah dari train+val.
    overlap = np.intersect1d(idx_tv, idx_test)
    if len(overlap) > 0:
        raise RuntimeError(
            f"Test set tumpang tindih dengan train+val sebanyak {len(overlap):,} "
            f"indeks!\nFile '{test_idx_path}' kemungkinan berasal dari skenario "
            f"split yang berbeda. Hapus/arsipkan file tersebut lalu jalankan ulang."
        )

    print(f"Total data     : {len(all_paths):,}")
    print(f"Train+Val (80%): {len(idx_tv):,}")
    print(f"Test set  (20%): {len(idx_test):,}\n")

    # ── 3. Loop semua alpha ───────────────────────────────────────────────────
    for alpha in ALPHAS:
        print(f"\n{'='*60}")
        print(f"EKSPERIMEN α = {alpha}")
        print(f"{'='*60}")

        result_dir_alpha = dir_alpha(alpha)
        os.makedirs(result_dir_alpha, exist_ok=True)

        partitions = dirichlet_partition(idx_tv, all_labels, alpha)

        from collections import Counter
        print(f"Distribusi klien (α={alpha}):")
        for k, part in enumerate(partitions):
            dist = Counter(all_labels[part])
            print(f"  Klien {k+1}: {len(part):,} sampel | {dict(dist)}")

        run_simulation(
            all_paths=all_paths,
            all_labels=all_labels,
            partitions=partitions,
            idx_tv=idx_tv,
            result_dir=result_dir_alpha,
            device=DEVICE,
            groups=groups,
        )

    # ── 4. Pilih model terbaik dari semua 25 eksperimen ───────────────────────
    print(f"\n{'='*60}")
    print("SELEKSI MODEL TERBAIK")
    print(f"{'='*60}")
    best_state, best_info = select_best_model(DEVICE)

    # ── 5. Evaluasi akhir pada test set ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUASI AKHIR — TEST SET")
    print(f"{'='*60}")
    final_dir = dir_final()
    full_evaluation(
        model_state=best_state,
        paths_test=all_paths[idx_test],
        labels_test=all_labels[idx_test],
        device=DEVICE,
        save_dir=final_dir,
    )

    # ── 6. Grad-CAM pada model terbaik ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("VISUALISASI GRAD-CAM")
    print(f"{'='*60}")
    model_best = build_model().to(DEVICE)
    model_best.load_state_dict(best_state)
    generate_gradcam(
        model=model_best,
        paths_test=all_paths[idx_test],
        labels_test=all_labels[idx_test],
        device=DEVICE,
    )

    # ── 7. Plot kurva training per alpha ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("PLOT KURVA TRAINING")
    print(f"{'='*60}")
    plot_training_curves()

    # ── 8. Tabel perbandingan metrik test set antar alpha ─────────────────────
    print(f"\n{'='*60}")
    print("TABEL PERBANDINGAN ANTAR ALPHA (TEST SET)")
    print(f"{'='*60}")
    evaluate_all_alphas(DEVICE)

    print(f"\n{'='*60}")
    print("Semua eksperimen selesai.")
    print(f"Hasil tersimpan di: {RESULT_DIR}")
    print(f"{'='*60}")