"""
experiments/alur_pembagian_data.py

Membuat diagram alur pembagian data, dari 12.446 citra hingga menjadi data
latih tiap klien pada setiap fold. Angka yang ditampilkan dihitung langsung
dari pipeline sesungguhnya, bukan ilustrasi perkiraan.

Keluaran: results/final/alur_pembagian_data.png

Cara pakai:
    python experiments/alur_pembagian_data.py
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.model_selection import StratifiedGroupKFold

from config import (RESULT_DIR, TEST_SPLIT, SEED, K_FOLD, NUM_CLIENTS,
                    KELAS_LIST)
from src.data.data_loader import load_dataset
from src.data.grouping import compute_group_labels, group_aware_holdout
from src.data.partitioner import dirichlet_partition

# Nilai alpha yang diilustrasikan. Dapat ditimpa lewat argumen baris perintah:
#     python experiments/alur_pembagian_data.py 0.1
ALPHA_ILUSTRASI = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0

BIRU   = "#4C72B0"
ORANYE = "#DD8452"
MERAH  = "#C44E52"
HIJAU  = "#55A868"
ABU    = "#8C8C8C"


def kotak(ax, x, y, w, h, teks, warna, fontsize=9.5, tebal=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.012",
                                facecolor=warna, edgecolor="white",
                                linewidth=1.6, alpha=0.92))
    ax.text(x + w / 2, y + h / 2, teks, ha="center", va="center",
            fontsize=fontsize, color="white",
            fontweight="bold" if tebal else "normal", linespacing=1.45)


def panah(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                 arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.5, color=ABU))


def main():
    all_paths, all_labels = load_dataset()
    idx_all = np.arange(len(all_paths))
    groups  = compute_group_labels(all_paths)
    n_grup  = int(groups.max() + 1)

    idx_tv, idx_test = group_aware_holdout(idx_all, all_labels, groups,
                                           TEST_SPLIT, SEED)
    partisi = dirichlet_partition(idx_tv, all_labels, ALPHA_ILUSTRASI)
    concat  = np.concatenate(partisi)

    sgkf  = StratifiedGroupKFold(n_splits=K_FOLD, shuffle=True,
                                 random_state=42)
    folds = list(sgkf.split(concat, all_labels[concat],
                            groups=groups[concat]))
    blok  = [concat[b] for _, b in folds]     # blok = himpunan validasi tiap fold

    # Matriks klien x blok
    M = np.zeros((NUM_CLIENTS, K_FOLD), dtype=int)
    for k in range(NUM_CLIENTS):
        for b in range(K_FOLD):
            M[k, b] = len(np.intersect1d(partisi[k], blok[b]))

    fig = plt.figure(figsize=(17, 9.5))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.14)
    fig.suptitle("Alur Pembagian Data — dari Dataset Utuh hingga Data Latih Tiap Klien",
                 fontsize=15.5, fontweight="bold", y=0.975)

    # ══ KIRI: alur bertahap ═════════════════════════════════════════════════
    ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(0.5, 0.965, "TAHAP 1 — Pemisahan Data Uji", ha="center",
            fontsize=11.5, fontweight="bold", color="#333")

    kotak(ax, 0.13, 0.845, 0.74, 0.085,
          f"DATASET UTUH — {len(all_paths):,} citra", BIRU, 11.5)
    panah(ax, 0.5, 0.843, 0.5, 0.795)

    kotak(ax, 0.06, 0.695, 0.88, 0.095,
          f"Pengelompokan citra serupa\n{len(all_paths):,} citra  →  {n_grup} grup "
          f"(±{len(all_paths)/n_grup:.0f} citra/grup)\nTIDAK ADA DATA YANG DIBUANG",
          ABU, 9.2)
    panah(ax, 0.5, 0.693, 0.5, 0.645)

    kotak(ax, 0.05, 0.535, 0.45, 0.10,
          f"TRAIN + VAL\n{len(idx_tv):,} citra\n({len(idx_tv)/len(all_paths)*100:.1f}%)",
          BIRU, 10.5)
    kotak(ax, 0.54, 0.535, 0.41, 0.10,
          f"TEST\n{len(idx_test):,} citra\n({len(idx_test)/len(all_paths)*100:.1f}%)",
          MERAH, 10.5)
    ax.text(0.745, 0.522, "dikunci — hanya dipakai\npada evaluasi akhir",
            ha="center", va="top", fontsize=8.2, style="italic", color=MERAH)

    panah(ax, 0.275, 0.533, 0.275, 0.465)
    ax.text(0.5, 0.435, "TAHAP 2 — Dua Pembagian yang Saling Tegak Lurus",
            ha="center", fontsize=11.5, fontweight="bold", color="#333")

    kotak(ax, 0.03, 0.285, 0.44, 0.115,
          f"POTONGAN MENDATAR\nPartisi Dirichlet → {NUM_CLIENTS} klien\n"
          f"(α = {ALPHA_ILUSTRASI})\nSAMA untuk kelima fold", ORANYE, 9.2)
    kotak(ax, 0.52, 0.285, 0.45, 0.115,
          f"POTONGAN TEGAK\nStratifiedGroupKFold → {K_FOLD} blok\n"
          f"(±{len(idx_tv)//K_FOLD:,} citra/blok)\nSAMA untuk semua α", HIJAU, 9.2)

    ax.text(0.5, 0.245, "Keduanya membagi {:,} citra yang sama,\n"
                        "seperti memotong kue mendatar dan tegak".format(len(idx_tv)),
            ha="center", fontsize=9, style="italic", color="#555")

    ax.text(0.5, 0.175, "TAHAP 3 — Rotasi Fold", ha="center",
            fontsize=11.5, fontweight="bold", color="#333")
    ax.text(0.5, 0.055,
            "Setiap fold memakai 1 blok sebagai VALIDASI\n"
            "dan 4 blok sisanya sebagai LATIH.\n"
            "Diulang 5 kali, tiap blok menjadi validasi tepat sekali.\n"
            f"Seluruh simulasi FL ({NUM_CLIENTS} klien × 100 ronde) diulang tiap fold.",
            ha="center", fontsize=9.3, color="#333", linespacing=1.7)

    # ══ KANAN: matriks klien × blok ═════════════════════════════════════════
    ax2 = fig.add_subplot(gs[0, 1]); ax2.axis("off")
    ax2.set_xlim(-0.16, 1.02); ax2.set_ylim(-0.30, 1.02)
    ax2.text(0.43, 0.975, f"Pembagian {len(idx_tv):,} Citra Train+Val "
                          f"(α = {ALPHA_ILUSTRASI}) — contoh Fold 1",
             ha="center", fontsize=11.5, fontweight="bold", color="#333")

    x0, y0, cw, ch = 0.0, 0.30, 0.172, 0.155
    for b in range(K_FOLD):
        val = (b == 0)
        ax2.text(x0 + b * cw + cw / 2, y0 + NUM_CLIENTS * ch + 0.055,
                 f"Blok {b+1}", ha="center", fontsize=10,
                 fontweight="bold", color=MERAH if val else "#333")
        ax2.text(x0 + b * cw + cw / 2, y0 + NUM_CLIENTS * ch + 0.018,
                 "VALIDASI" if val else "latih", ha="center", fontsize=8.5,
                 fontweight="bold" if val else "normal",
                 color=MERAH if val else ABU)

    for k in range(NUM_CLIENTS):
        baris = NUM_CLIENTS - 1 - k
        # Kelas dominan klien ini — memperlihatkan label distribution skew
        cnt  = np.array([(all_labels[partisi[k]] == c).sum()
                         for c in range(len(KELAS_LIST))])
        top  = int(cnt.argmax())
        pers = cnt[top] / max(cnt.sum(), 1) * 100
        ax2.text(x0 - 0.025, y0 + baris * ch + ch / 2,
                 f"Klien {k+1}\n{len(partisi[k]):,}\n"
                 f"{KELAS_LIST[top]} {pers:.0f}%",
                 ha="right", va="center", fontsize=9.2,
                 fontweight="bold", color=ORANYE, linespacing=1.5)
        for b in range(K_FOLD):
            val = (b == 0)
            ax2.add_patch(FancyBboxPatch(
                (x0 + b * cw + 0.007, y0 + baris * ch + 0.007),
                cw - 0.014, ch - 0.014, boxstyle="round,pad=0.004",
                facecolor=MERAH if val else BIRU,
                alpha=0.80 if val else 0.42,
                edgecolor="white", linewidth=1.5))
            ax2.text(x0 + b * cw + cw / 2, y0 + baris * ch + ch / 2,
                     f"{M[k, b]:,}", ha="center", va="center", fontsize=11,
                     fontweight="bold", color="white" if val else "#123")

    for b in range(K_FOLD):
        ax2.text(x0 + b * cw + cw / 2, y0 - 0.045, f"{M[:, b].sum():,}",
                 ha="center", fontsize=9.5, fontweight="bold", color="#333")
    ax2.text(x0 - 0.025, y0 - 0.045, "total blok", ha="right",
             fontsize=9, color="#555")

    ax2.text(0.43, 0.185,
             f"Fold 1  →  latih {M[:, 1:].sum():,} citra  ·  "
             f"validasi {M[:, 0].sum():,} citra",
             ha="center", va="center", fontsize=11, fontweight="bold",
             color="#333")
    ax2.text(0.43, 0.135,
             "Tiap sel = irisan antara data milik satu klien dengan satu blok.\n"
             "Pada Fold 1, ketiga klien melatih model lokalnya memakai\n"
             "sel-sel biru miliknya; server menilai model global pada sel merah.",
             ha="center", va="top", fontsize=9.2, color="#444",
             linespacing=1.7)
    ax2.text(0.43, -0.10,
             "Fold 2 memindahkan kotak merah ke Blok 2, dan seterusnya —\n"
             "sehingga setiap citra menjadi validasi tepat satu kali.",
             ha="center", va="top", fontsize=9.2, style="italic", color="#555",
             linespacing=1.7)

    out = os.path.join(RESULT_DIR, "final",
                       f"alur_pembagian_data_alpha_{ALPHA_ILUSTRASI}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ── Ringkasan angka di terminal ─────────────────────────────────────────
    print("\n" + "=" * 64)
    print("RINGKASAN ANGKA")
    print("=" * 64)
    print(f"  Dataset utuh          : {len(all_paths):,} citra ({n_grup} grup)")
    print(f"  Train+Val             : {len(idx_tv):,}")
    print(f"  Test (dikunci)        : {len(idx_test):,}")
    print(f"  Jumlah                : {len(idx_tv) + len(idx_test):,}")
    print(f"\n  Partisi Dirichlet α={ALPHA_ILUSTRASI}:")
    for k in range(NUM_CLIENTS):
        print(f"    Klien {k+1}: {len(partisi[k]):,}")
    print(f"\n  Ukuran tiap blok      : "
          f"{', '.join(f'{M[:, b].sum():,}' for b in range(K_FOLD))}")
    print(f"\nGambar disimpan: {out}")


if __name__ == "__main__":
    main()
