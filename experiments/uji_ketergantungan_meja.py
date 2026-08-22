"""
experiments/uji_ketergantungan_meja.py

Menguji apakah model bergantung pada meja pemindai CT — struktur non-anatomis
di tepi bawah citra — untuk membuat prediksi.

LATAR BELAKANG
Visualisasi Grad-CAM pada model dengan regularisasi menunjukkan dua area
aktivasi: gumpalan terfokus di regio ginjal, dan pita kuat di tepi bawah citra
yang bertepatan dengan garis meja pemindai. Bila prediksi ikut ditopang oleh
struktur tersebut, kenaikan metrik tidak sepenuhnya mencerminkan pemahaman
anatomi.

CARA KERJA
Model yang sudah dilatih dievaluasi berulang pada test set yang sama, dengan
empat perlakuan berbeda:

  asli    : citra apa adanya, sebagai acuan
  bawah   : bagian bawah citra ditutup (area meja)
  atas    : bagian atas ditutup dengan luas yang SAMA — KONTROL
  tubuh   : hanya tubuh yang dipertahankan, seluruh area di luarnya ditutup

Perlakuan "atas" adalah kunci penafsiran. Menutup sebagian citra selalu
menciptakan ketidaksesuaian dengan kondisi pelatihan, sehingga penurunan
akurasi bisa muncul semata karena citra menjadi asing bagi model. Dengan
membandingkan "bawah" terhadap "atas" pada luas tutupan yang identik, efek
ketidaksesuaian tersebut dapat dikurangkan:

  - Penurunan pada "bawah" jauh lebih besar daripada "atas"
      -> model memang bergantung pada meja pemindai
  - Keduanya turun serupa
      -> yang terjadi hanya efek ketidaksesuaian, meja bukan penentu

Perlakuan "tubuh" memakai segmentasi pada src/data/masking.py, yang
mempertahankan setiap komponen terhubung yang menyentuh kotak tengah citra
sehingga ginjal tidak mungkin terpotong. Skrip memverifikasi keutuhan wilayah
tengah dan letak vertikal piksel yang terbuang, serta menghasilkan montase
visual untuk diperiksa secara manual.

Perlu diperhatikan: pada citra KORONAL, bagian bawah gambar berisi panggul dan
tungkai, bukan meja. Perlakuan "bawah" dan "atas" karenanya ikut memotong
anatomi pada citra tersebut, sehingga perlakuan "tubuh" merupakan ukuran yang
paling sahih lintas orientasi.

CATATAN
Skrip ini TIDAK melatih apa pun dan tidak mengubah berkas hasil eksperimen.
Pelatihan ulang hanya diperlukan apabila hasil uji ini menunjukkan
ketergantungan nyata dan Anda memutuskan untuk menghilangkannya.

Cara pakai:
    python experiments/uji_ketergantungan_meja.py            # alpha 1.0
    python experiments/uji_ketergantungan_meja.py 0.1        # alpha lain
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from scipy import ndimage

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (RESULT_DIR, BATCH_SIZE, K_FOLD, KELAS_LIST, IMG_SIZE,
                    GROUP_AWARE_SPLIT, dir_alpha, dir_final)
from src.seeding import set_seed
from src.data.data_loader import load_dataset
from src.data.preprocessing import get_transforms
from src.data.masking import mask_tubuh, AMBANG_TUBUH
from src.models.efficientnet_model import build_model

DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA_TARGET = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

# Tinggi area yang ditutup pada perlakuan "bawah" dan "atas", sebagai proporsi
# tinggi citra. Grad-CAM memakai peta fitur 7x7, sehingga satu baris peta
# mencakup sekitar 1/7 (14%) tinggi citra; nilai 0,18 menutupi baris terbawah
# tersebut dengan sedikit margin.
PROPORSI_TUTUP = 0.18

# Ambang, margin, dan fungsi mask_tubuh dipusatkan di src/data/masking.py
# agar identik dengan yang dipakai pada visualisasi Grad-CAM.


def terapkan(img_pil, perlakuan):
    """Terapkan satu perlakuan pada citra PIL. Return (citra, proporsi_ditutup)."""
    if perlakuan == "asli":
        return img_pil, 0.0

    arr = np.array(img_pil)                      # RGB
    abu = np.array(img_pil.convert("L"))
    h   = arr.shape[0]

    if perlakuan == "bawah":
        batas = int(h * (1 - PROPORSI_TUTUP))
        arr[batas:, :, :] = 0
        return Image.fromarray(arr), PROPORSI_TUTUP

    if perlakuan == "atas":
        batas = int(h * PROPORSI_TUTUP)
        arr[:batas, :, :] = 0
        return Image.fromarray(arr), PROPORSI_TUTUP

    if perlakuan == "tubuh":
        m = mask_tubuh(abu)
        arr[~m] = 0
        return Image.fromarray(arr), float((~m).mean())

    raise ValueError(f"Perlakuan tidak dikenal: {perlakuan}")


class DatasetPerlakuan(Dataset):
    """Dataset yang menerapkan perlakuan penutupan sebelum transform."""

    def __init__(self, paths, labels, transform, perlakuan):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform
        self.perlakuan = perlakuan

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img, _ = terapkan(img, self.perlakuan)
        return self.transform(img), int(self.labels[i])


def fold_tersedia(alpha):
    """Daftar fold yang sudah selesai dilatih beserta path bobotnya."""
    status_path = os.path.join(dir_alpha(alpha), "fold_status.json")
    if not os.path.exists(status_path):
        raise FileNotFoundError(f"Tidak ditemukan: {status_path}")
    with open(status_path) as f:
        status = json.load(f)

    hasil = []
    for k in range(1, K_FOLD + 1):
        if not status.get(f"fold_{k}", {}).get("done"):
            continue
        p = os.path.join(dir_alpha(alpha), f"fold_{k}", "best_model.pth")
        if os.path.exists(p):
            hasil.append((k, p))
    if not hasil:
        raise RuntimeError(f"Tidak ada bobot model pada alpha {alpha}.")
    return hasil


def evaluasi(model, loader):
    model.eval()
    pr, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            pr.extend(model(x.to(DEVICE)).argmax(1).cpu().numpy())
            ys.extend(y.numpy())
    ys, pr = np.array(ys), np.array(pr)
    return {
        "accuracy":        accuracy_score(ys, pr),
        "macro_precision": precision_score(ys, pr, average="macro", zero_division=0),
        "macro_recall":    recall_score(ys, pr, average="macro", zero_division=0),
        "macro_f1":        f1_score(ys, pr, average="macro", zero_division=0),
        "f1_per_kelas":    f1_score(ys, pr, average=None, zero_division=0).tolist(),
    }


def montase(paths, out_path):
    """Simpan montase verifikasi: citra asli dan hasil keempat perlakuan."""
    perlakuan = ["asli", "bawah", "atas", "tubuh"]
    n = len(paths)
    fig, axes = plt.subplots(n, len(perlakuan),
                             figsize=(len(perlakuan) * 3.1, n * 3.2),
                             squeeze=False)
    fig.suptitle("Verifikasi Perlakuan Penutupan Area Citra",
                 fontsize=14, fontweight="bold", y=0.995)

    for r, p in enumerate(paths):
        base = Image.open(p).convert("RGB")
        for c, pl in enumerate(perlakuan):
            img, prop = terapkan(base.copy(), pl)
            ax = axes[r][c]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(pl, fontsize=11, fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(os.path.basename(p)[:18], fontsize=8)
            if pl != "asli":
                ax.set_xlabel(f"ditutup {prop*100:.1f}%", fontsize=8.5,
                              color="#c73e1d")

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Montase verifikasi disimpan: {out_path}")


if __name__ == "__main__":
    set_seed()
    print(f"\n{'='*74}")
    print("UJI KETERGANTUNGAN MODEL PADA MEJA PEMINDAI")
    print(f"  Device            : {DEVICE}")
    print(f"  Group-aware split : {GROUP_AWARE_SPLIT}")
    print(f"  Alpha diuji       : {ALPHA_TARGET}")
    print(f"{'='*74}")

    # ── Data uji ─────────────────────────────────────────────────────────────
    all_paths, all_labels = load_dataset()
    suffix   = "_group" if GROUP_AWARE_SPLIT else ""
    idx_path = os.path.join(RESULT_DIR, f"test_indices{suffix}.npy")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Tidak ditemukan: {idx_path}")
    idx_test = np.load(idx_path)
    paths_test, labels_test = all_paths[idx_test], all_labels[idx_test]
    print(f"\nCitra uji: {len(idx_test):,}")

    _, transform_val = get_transforms()
    folds = fold_tersedia(ALPHA_TARGET)
    print(f"Fold tersedia: {[k for k, _ in folds]}")

    out_dir = os.path.join(dir_final(), "uji_meja")
    os.makedirs(out_dir, exist_ok=True)

    # ── Verifikasi visual sebelum pengukuran ────────────────────────────────
    print("\nMembuat montase verifikasi...")
    contoh = [paths_test[i] for i in
              np.linspace(0, len(paths_test) - 1, 4).astype(int)]
    montase(contoh, os.path.join(out_dir, f"verifikasi_alpha_{ALPHA_TARGET}.png"))

    # ── Verifikasi mask tubuh tidak memotong anatomi ────────────────────────
    #
    # Piksel terang yang terbuang sebagian besar justru MEJA pemindai, yang
    # memang menjadi sasaran. Karena itu proporsi piksel terang yang hilang
    # bukan ukuran yang tepat. Dua pemeriksaan berikut lebih relevan:
    #
    #   1. Letak vertikal piksel terang yang terbuang. Bila terpusat di bagian
    #      bawah citra, yang terbuang adalah meja, bukan jaringan tubuh.
    #   2. Keutuhan wilayah tengah citra, tempat kedua ginjal berada. Wilayah
    #      ini harus dipertahankan sepenuhnya.
    print("\nMemverifikasi mask tubuh tidak memotong anatomi...")
    posisi_buang, utuh_tengah, terbuang = [], [], []
    for p in paths_test[np.linspace(0, len(paths_test) - 1, 200).astype(int)]:
        abu = np.array(Image.open(p).convert("L"))
        h, w = abu.shape
        jar  = abu > AMBANG_TUBUH
        if jar.sum() == 0:
            continue
        m     = mask_tubuh(abu)
        hilang = jar & ~m
        terbuang.append(hilang.sum() / jar.sum())
        if hilang.any():
            # posisi vertikal rata-rata piksel yang terbuang, 0 = atas, 1 = bawah
            posisi_buang.append(np.nonzero(hilang)[0].mean() / h)
        # wilayah tengah citra (30%-70% tinggi dan lebar) tempat ginjal berada
        tengah = jar[int(h*.30):int(h*.70), int(w*.30):int(w*.70)]
        m_tgh  = m[int(h*.30):int(h*.70), int(w*.30):int(w*.70)]
        if tengah.sum() > 0:
            utuh_tengah.append((tengah & m_tgh).sum() / tengah.sum())

    terbuang     = np.array(terbuang)
    posisi_buang = np.array(posisi_buang)
    utuh_tengah  = np.array(utuh_tengah)

    print(f"  Piksel terang yang terbuang        : rata-rata {terbuang.mean()*100:.2f}%")
    print(f"  Letak vertikalnya (0=atas, 1=bawah): rata-rata {posisi_buang.mean():.3f} "
          f"| median {np.median(posisi_buang):.3f}")
    print(f"  Keutuhan wilayah tengah (ginjal)   : rata-rata {utuh_tengah.mean()*100:.2f}% "
          f"| minimum {utuh_tengah.min()*100:.2f}%")
    print()
    if posisi_buang.mean() > 0.75 and utuh_tengah.min() > 0.99:
        print("  Aman — yang terbuang terpusat di bagian bawah citra (meja),")
        print("  sementara wilayah ginjal dipertahankan seutuhnya.")
    elif utuh_tengah.min() > 0.99:
        print("  Wilayah ginjal utuh, namun yang terbuang tidak seluruhnya di")
        print("  bagian bawah. Periksa montase sebelum menafsirkan hasil.")
    else:
        print("  PERHATIAN: ada citra yang wilayah tengahnya ikut terpotong.")
        print("  Naikkan MARGIN_TUBUH atau turunkan AMBANG_TUBUH, lalu ulangi.")

    # ── Pengukuran ───────────────────────────────────────────────────────────
    PERLAKUAN = ["asli", "bawah", "atas", "tubuh"]
    kumpulan  = {pl: [] for pl in PERLAKUAN}
    baris_fold = []

    for k, path_bobot in folds:
        print(f"\n{'─'*74}")
        print(f"Fold {k}")
        model = build_model().to(DEVICE)
        model.load_state_dict(torch.load(path_bobot, map_location=DEVICE))

        for pl in PERLAKUAN:
            loader = DataLoader(
                DatasetPerlakuan(paths_test, labels_test, transform_val, pl),
                batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
            m = evaluasi(model, loader)
            kumpulan[pl].append(m)
            baris_fold.append({"alpha": ALPHA_TARGET, "fold": k, "perlakuan": pl,
                               **{a: round(m[a], 6) for a in
                                  ("accuracy","macro_precision","macro_recall","macro_f1")}})
            print(f"  {pl:<7} acc {m['accuracy']*100:6.2f}% | "
                  f"macro-F1 {m['macro_f1']:.4f}")

    # ── Ringkasan ────────────────────────────────────────────────────────────
    def rerata(pl, kunci):
        v = np.array([m[kunci] for m in kumpulan[pl]])
        return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)

    acc_asli, _ = rerata("asli", "accuracy")
    f1_asli,  _ = rerata("asli", "macro_f1")

    print("\n" + "=" * 74)
    print(f"RINGKASAN — alpha {ALPHA_TARGET}, rata-rata {len(folds)} fold")
    print("=" * 74)
    print(f"{'Perlakuan':<9}{'Accuracy':>18}{'Macro-F1':>18}"
          f"{'ΔAcc':>10}{'ΔF1':>10}")
    ringkas = []
    for pl in PERLAKUAN:
        a, sa = rerata(pl, "accuracy")
        f, sf = rerata(pl, "macro_f1")
        print(f"{pl:<9}{a*100:>11.2f}% ±{sa*100:<5.2f}{f:>12.4f} ±{sf:<5.4f}"
              f"{(a-acc_asli)*100:>+9.2f}p{(f-f1_asli)*100:>+9.2f}p")
        ringkas.append({"alpha": ALPHA_TARGET, "perlakuan": pl,
                        "n_fold": len(folds),
                        "accuracy_mean": round(a, 6), "accuracy_std": round(sa, 6),
                        "macro_f1_mean": round(f, 6), "macro_f1_std": round(sf, 6),
                        "delta_accuracy": round(a - acc_asli, 6),
                        "delta_macro_f1": round(f - f1_asli, 6)})

    # ── Penafsiran ───────────────────────────────────────────────────────────
    d_bawah = acc_asli - rerata("bawah", "accuracy")[0]
    d_atas  = acc_asli - rerata("atas",  "accuracy")[0]

    print("\n" + "=" * 74)
    print("PENAFSIRAN")
    print("=" * 74)
    print(f"  Penurunan akurasi karena menutup BAWAH (area meja) : {d_bawah*100:.2f} poin")
    print(f"  Penurunan akurasi karena menutup ATAS  (kontrol)   : {d_atas*100:.2f} poin")
    print(f"  Selisih                                            : "
          f"{(d_bawah - d_atas)*100:.2f} poin")
    print()
    if d_bawah > d_atas + 0.03:
        print("  -> Menutup area meja jauh lebih merugikan daripada kontrol.")
        print("     KETERGANTUNGAN PADA MEJA PEMINDAI TERBUKTI.")
        print("     Pertimbangkan melatih ulang dengan area di luar tubuh ditutup.")
    elif d_bawah > d_atas + 0.01:
        print("  -> Ada indikasi ketergantungan, namun selisihnya kecil.")
        print("     Sajikan sebagai indikasi, bukan kesimpulan tegas.")
    else:
        print("  -> Penurunan pada kedua sisi setara.")
        print("     Yang terjadi hanya efek ketidaksesuaian dengan kondisi")
        print("     pelatihan; meja pemindai BUKAN penentu prediksi.")

    # ── Simpan ───────────────────────────────────────────────────────────────
    p1 = os.path.join(out_dir, f"ringkasan_alpha_{ALPHA_TARGET}.csv")
    with open(p1, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ringkas[0].keys()))
        w.writeheader(); w.writerows(ringkas)

    p2 = os.path.join(out_dir, f"per_fold_alpha_{ALPHA_TARGET}.csv")
    with open(p2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(baris_fold[0].keys()))
        w.writeheader(); w.writerows(baris_fold)

    print(f"\nRingkasan : {p1}")
    print(f"Per fold  : {p2}")
