"""
experiments/finalisasi.py

Menjalankan seluruh tahap PASCA-PELATIHAN, tanpa melatih apa pun.

Dipakai apabila pelatihan dilakukan bertahap melalui experiments/run_alpha.py
(satu nilai alpha per eksekusi), sehingga tahap 4-8 pada main.py belum pernah
dijalankan. Skrip ini aman dieksekusi kapan saja: ia hanya membaca hasil
pelatihan yang sudah ada dan tidak akan memulai pelatihan baru.

Tahapan yang dijalankan:
  1. Pemilihan model global terbaik (rata-rata macro F1 validasi per alpha)
  2. Evaluasi test set: accuracy, macro precision/recall/F1, confusion matrix,
     jumlah parameter, FLOPs, dan waktu inferensi
  3. Visualisasi Grad-CAM untuk keempat kelas
  4. Kurva training loss / validation loss / macro F1 per ronde
  5. Tabel perbandingan metrik test set antar alpha

Alpha yang belum selesai dilatih akan dilewati secara otomatis, sehingga
skrip ini tetap dapat dipakai untuk hasil parsial.

Cara pakai:
    python experiments/finalisasi.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch

from config import (RESULT_DIR, ALPHAS, K_FOLD, TEST_SPLIT, SEED,
                    GROUP_AWARE_SPLIT)
from src.seeding import set_seed
from src.data.data_loader import load_dataset
from src.evaluation.evaluate import (select_best_model, full_evaluation,
                                     plot_training_curves, evaluate_all_alphas)
from src.models.efficientnet_model import build_model
from src.models.gradcam import generate_gradcam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ringkas_progres():
    """Tampilkan alpha dan fold mana saja yang sudah selesai dilatih."""
    print(f"\n{'='*60}")
    print("PROGRES PELATIHAN")
    print(f"{'='*60}")

    ada = False
    for alpha in ALPHAS:
        path = os.path.join(RESULT_DIR, f"alpha_{alpha}", "fold_status.json")
        if not os.path.exists(path):
            print(f"  alpha {alpha}: belum dimulai")
            continue
        with open(path) as f:
            status = json.load(f)
        selesai = [fo for fo in range(1, K_FOLD + 1)
                   if status.get(f"fold_{fo}", {}).get("done")]
        if selesai:
            ada = True
            f1 = [status[f"fold_{fo}"]["metrics"]["best_val_f1"]
                  for fo in selesai]
            print(f"  alpha {alpha}: {len(selesai)}/{K_FOLD} fold selesai | "
                  f"rata-rata macro F1 validasi = {np.mean(f1):.4f}")
        else:
            print(f"  alpha {alpha}: belum ada fold yang selesai")

    if not ada:
        raise RuntimeError(
            "Belum ada fold yang selesai dilatih.\n"
            "Jalankan pelatihan terlebih dahulu: python experiments/run_alpha.py"
        )


if __name__ == "__main__":
    set_seed()
    print(f"\n{'='*60}")
    print("FINALISASI — EVALUASI, GRAD-CAM, DAN TABEL PERBANDINGAN")
    print(f"Device            : {DEVICE}")
    print(f"Group-aware split : {GROUP_AWARE_SPLIT}")
    print(f"{'='*60}")

    ringkas_progres()

    # ── Muat dataset dan indeks uji yang dikunci saat pelatihan ──────────────
    all_paths, all_labels = load_dataset()

    suffix   = "_group" if GROUP_AWARE_SPLIT else ""
    idx_path = os.path.join(RESULT_DIR, f"test_indices{suffix}.npy")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"Berkas indeks uji tidak ditemukan: {idx_path}\n"
            f"Berkas ini dibuat saat pelatihan pertama kali dijalankan. "
            f"Pastikan nilai GROUP_AWARE_SPLIT sama dengan yang dipakai "
            f"ketika melatih."
        )
    idx_test = np.load(idx_path)
    print(f"\nIndeks uji dimuat : {idx_path} ({len(idx_test):,} citra)")

    # ── 1. Pilih model global terbaik ────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SELEKSI MODEL TERBAIK")
    print(f"{'='*60}")
    best_state, best_info = select_best_model(DEVICE)

    # ── 2. Evaluasi akhir pada test set ──────────────────────────────────────
    print(f"\n{'='*60}")
    print("EVALUASI AKHIR — TEST SET")
    print(f"{'='*60}")
    final_dir = os.path.join(RESULT_DIR, "final")
    full_evaluation(
        model_state=best_state,
        paths_test=all_paths[idx_test],
        labels_test=all_labels[idx_test],
        device=DEVICE,
        save_dir=final_dir,
    )

    # ── 3. Grad-CAM pada model terbaik ───────────────────────────────────────
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

    # ── 4. Kurva pelatihan per alpha ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print("KURVA PELATIHAN")
    print(f"{'='*60}")
    plot_training_curves()

    # ── 5. Tabel perbandingan antar alpha ────────────────────────────────────
    print(f"\n{'='*60}")
    print("TABEL PERBANDINGAN ANTAR ALPHA (TEST SET)")
    print(f"{'='*60}")
    evaluate_all_alphas(DEVICE)

    print(f"\n{'='*60}")
    print("Finalisasi selesai.")
    print(f"Seluruh keluaran tersimpan di: {final_dir}")
    print(f"{'='*60}")
