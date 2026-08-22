"""
experiments/gradcam_saja.py

Menghasilkan visualisasi Grad-CAM SAJA, tanpa menjalankan evaluasi test set,
kurva pelatihan, maupun tabel perbandingan antar alpha. Berguna ketika hanya
tampilan Grad-CAM yang perlu diperbarui, karena finalisasi.py mengevaluasi
seluruh model pada test set dan memakan waktu jauh lebih lama.

Cara pakai:
    python experiments/gradcam_saja.py            # model global terbaik
    python experiments/gradcam_saja.py 0.1        # fold terbaik pada alpha 0.1
    python experiments/gradcam_saja.py 0.1 3      # alpha 0.1 fold 3

Opsi tambahan, dapat digabung:
    --mask    hitamkan area di luar tubuh sebelum citra masuk ke model,
              sehingga meja pemindai tidak dapat muncul pada heatmap
    --halus   pakai lapisan yang lebih dangkal (peta 14x14, bukan 7x7)
              sehingga heatmap jauh lebih rinci

    python experiments/gradcam_saja.py 1.0 --mask
    python experiments/gradcam_saja.py 1.0 --mask --halus

Memilih alpha tertentu berguna untuk membandingkan fokus perhatian model
antar tingkat non-IID, misalnya alpha 0.1 dibanding alpha 1.0.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import torch

from config import RESULT_DIR, K_FOLD, GROUP_AWARE_SPLIT, dir_alpha, dir_final
from src.seeding import set_seed
from src.data.data_loader import load_dataset
from src.evaluation.evaluate import select_best_model
from src.models.efficientnet_model import build_model
from src.models.gradcam import generate_gradcam

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def muat_model_alpha(alpha, fold=None):
    """
    Kembalikan (state_dict, keterangan, subfolder) untuk alpha tertentu.
    Bila fold tidak diberikan, dipilih fold dengan macro F1 validasi tertinggi.
    """
    alpha_dir   = dir_alpha(alpha)
    status_path = os.path.join(alpha_dir, "fold_status.json")
    if not os.path.exists(status_path):
        raise FileNotFoundError(
            f"Hasil untuk alpha {alpha} tidak ditemukan: {status_path}")

    with open(status_path) as f:
        status = json.load(f)

    if fold is None:
        fold, f1_terbaik = None, -1.0
        for k in range(1, K_FOLD + 1):
            info = status.get(f"fold_{k}", {})
            if info.get("done") and info["metrics"]["best_val_f1"] > f1_terbaik:
                f1_terbaik = info["metrics"]["best_val_f1"]
                fold       = k
        if fold is None:
            raise RuntimeError(f"Belum ada fold yang selesai pada alpha {alpha}.")
    else:
        info = status.get(f"fold_{fold}", {})
        if not info.get("done"):
            raise RuntimeError(f"Fold {fold} pada alpha {alpha} belum selesai.")
        f1_terbaik = info["metrics"]["best_val_f1"]

    model_path = os.path.join(alpha_dir, f"fold_{fold}", "best_model.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Bobot model tidak ditemukan: {model_path}")

    state = torch.load(model_path, map_location=DEVICE)
    ket   = f"alpha {alpha} fold {fold} (macro F1 validasi {f1_terbaik:.4f})"
    return state, ket, f"alpha_{alpha}_fold_{fold}"


if __name__ == "__main__":
    set_seed()
    argv    = [a for a in sys.argv[1:] if not a.startswith("--")]
    opsi    = {a for a in sys.argv[1:] if a.startswith("--")}
    pakai_mask = "--mask" in opsi
    lapisan    = "halus" if "--halus" in opsi else "conv_head"

    alpha = float(argv[0]) if len(argv) > 0 else None
    fold  = int(argv[1])   if len(argv) > 1 else None

    print(f"\n{'='*60}")
    print("VISUALISASI GRAD-CAM")
    print(f"Device            : {DEVICE}")
    print(f"Group-aware split : {GROUP_AWARE_SPLIT}")
    print(f"Mask tubuh        : {pakai_mask}")
    print(f"Lapisan target    : {lapisan}")
    print(f"{'='*60}")

    # ── Muat bobot model ─────────────────────────────────────────────────────
    if alpha is None:
        state, info = select_best_model(DEVICE)
        ket      = (f"model global terbaik — alpha {info['alpha']} "
                    f"fold {info['fold']}")
        subfolder = None
    else:
        state, ket, subfolder = muat_model_alpha(alpha, fold)
    print(f"\nModel dipakai: {ket}")

    # ── Muat citra uji yang dikunci saat pelatihan ──────────────────────────
    all_paths, all_labels = load_dataset()
    suffix   = "_group" if GROUP_AWARE_SPLIT else ""
    idx_path = os.path.join(RESULT_DIR, f"test_indices{suffix}.npy")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"Berkas indeks uji tidak ditemukan: {idx_path}\n"
            f"Pastikan nilai GROUP_AWARE_SPLIT sama dengan saat pelatihan.")
    idx_test = np.load(idx_path)
    print(f"Citra uji     : {len(idx_test):,}")

    # ── Bangun model dan hasilkan Grad-CAM ──────────────────────────────────
    model = build_model().to(DEVICE)
    model.load_state_dict(state)

    save_dir = os.path.join(dir_final(), "gradcam")
    if subfolder:
        save_dir = os.path.join(save_dir, subfolder)
    # Varian disimpan terpisah agar tidak menimpa versi sebelumnya
    varian = ("mask" if pakai_mask else "utuh") + ("_halus" if lapisan == "halus" else "")
    save_dir = os.path.join(save_dir, varian)

    generate_gradcam(
        model=model,
        paths_test=all_paths[idx_test],
        labels_test=all_labels[idx_test],
        device=DEVICE,
        save_dir=save_dir,
        pakai_mask_tubuh=pakai_mask,
        lapisan=lapisan,
    )

    print(f"\n{'='*60}")
    print("Selesai.")
    print(f"{'='*60}")
