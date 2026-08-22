"""
experiments/ablasi_regularisasi.py

Ablation study untuk menguji pengaruh regularisasi terhadap overfitting pada
skenario group-aware. Dijalankan pada model terpusat (baseline) karena jauh
lebih cepat daripada mengulang 25 simulasi Federated Learning: satu konfigurasi
selesai dalam hitungan puluhan menit, bukan berjam-jam.

LATAR BELAKANG
Pada skenario group-aware, model mencapai akurasi latih 99,9% namun hanya
sekitar 80% pada data uji. Dataset hanya memuat sekitar 250 kasus independen,
sehingga model menghafal kasus latih alih-alih mempelajari ciri yang dapat
digeneralisasi. Skrip ini menguji tiga bentuk regularisasi untuk menekan
kesenjangan tersebut.

Skrip ini TIDAK mengubah berkas konfigurasi maupun modul mana pun, dan menulis
keluarannya ke folder terpisah, sehingga seluruh hasil eksperimen yang sudah
ada tetap utuh dan dapat direproduksi.

Cara pakai:
    python experiments/ablasi_regularisasi.py dasar        # replikasi baseline
    python experiments/ablasi_regularisasi.py augmentasi
    python experiments/ablasi_regularisasi.py wd
    python experiments/ablasi_regularisasi.py ls
    python experiments/ablasi_regularisasi.py gabungan
    python experiments/ablasi_regularisasi.py mask_tubuh
    python experiments/ablasi_regularisasi.py gabungan_mask
    python experiments/ablasi_regularisasi.py --ringkas    # tabel perbandingan

Jalankan konfigurasi "dasar" terlebih dahulu sebagai titik acuan, sebab hasil
baseline yang lama memakai pemilihan model berdasarkan akurasi, bukan macro F1.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import transforms
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, classification_report)

from config import (RESULT_DIR, TEST_SPLIT, SEED, BATCH_SIZE, LR, IMG_SIZE,
                    KELAS_LIST, DENOISE_SIGMA, GROUP_AWARE_SPLIT)
from src.seeding import set_seed
from src.data.data_loader import load_dataset
from src.data.grouping import compute_group_labels, group_aware_holdout
from src.data.preprocessing import GaussianDenoising
from src.data.augmentation import KidneyDataset
from src.data.masking import terapkan_mask_tubuh
from src.models.efficientnet_model import build_model

DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR  = os.path.join(RESULT_DIR, "ablasi")
RINGKAS  = os.path.join(OUT_DIR, "ringkasan.csv")
EPOCHS   = 15          # sedikit lebih panjang dari baseline agar regularisasi
                       # sempat menunjukkan pengaruhnya

# ── Definisi konfigurasi ──────────────────────────────────────────────────────
# augmentasi : 'standar' sesuai Tabel 3.1 proposal, 'kuat' diperluas
# weight_decay : penalti L2 pada optimizer (0 = tanpa)
# label_smoothing : melunakkan target one-hot (0 = tanpa)
KONFIG = {
    "dasar": {
        "keterangan":      "Sesuai Tabel 3.1 dan 3.2 proposal",
        "augmentasi":      "standar",
        "weight_decay":    0.0,
        "label_smoothing": 0.0,
        "mask_tubuh":      False,
    },
    "augmentasi": {
        "keterangan":      "Augmentasi diperkuat",
        "augmentasi":      "kuat",
        "weight_decay":    0.0,
        "label_smoothing": 0.0,
        "mask_tubuh":      False,
    },
    # CATATAN HASIL: konfigurasi ini terbukti TIDAK BEREFEK sama sekali —
    # metrics.csv-nya bit-identik dengan "dasar". Penyebabnya, AdamW meluruhkan
    # bobot sebesar p *= (1 - lr*wd) = 1 - 1e-8 per langkah, sedangkan epsilon
    # mesin float32 adalah 1,192e-7. Faktor peluruhan membulat kembali menjadi
    # tepat 1,0 sehingga bobot tidak pernah berubah. Dipertahankan sebagai bukti
    # temuan tersebut; gunakan "wd_kuat" untuk weight decay yang benar-benar
    # aktif pada learning rate 1e-4.
    "wd": {
        "keterangan":      "Weight decay 1e-4 (tidak berefek pada float32)",
        "augmentasi":      "standar",
        "weight_decay":    1e-4,
        "label_smoothing": 0.0,
        "mask_tubuh":      False,
    },
    "wd_kuat": {
        "keterangan":      "Weight decay 0,05 (AdamW)",
        "augmentasi":      "standar",
        "weight_decay":    0.05,
        "label_smoothing": 0.0,
        "mask_tubuh":      False,
    },
    "ls": {
        "keterangan":      "Label smoothing 0,1",
        "augmentasi":      "standar",
        "weight_decay":    0.0,
        "label_smoothing": 0.1,
        "mask_tubuh":      False,
    },
    "gabungan": {
        "keterangan":      "Augmentasi kuat + weight decay + label smoothing",
        "augmentasi":      "kuat",
        "weight_decay":    1e-4,
        "label_smoothing": 0.1,
        "mask_tubuh":      False,
    },
    # Dua konfigurasi berikut melatih model pada citra yang area di luar
    # tubuhnya sudah dihitamkan, sehingga meja pemindai tidak pernah dilihat
    # model sejak awal. Uji perturbasi sebelumnya hanya mencabut meja saat
    # inferensi dari model yang terlanjur belajar dengannya; melatih ulang
    # dengan citra bersih dapat menghasilkan representasi yang berbeda.
    #
    # Manfaat yang paling diharapkan bukan pada angka, melainkan pada kualitas
    # Grad-CAM: pada model sekarang, aktivasi liar muncul tepat di batas buatan
    # hasil masking karena model tidak pernah melihat citra bermask saat
    # pelatihan. Bila batas itu menjadi hal biasa baginya, artefak tersebut
    # berpeluang hilang.
    "mask_tubuh": {
        "keterangan":      "Mask tubuh, augmentasi standar",
        "augmentasi":      "standar",
        "weight_decay":    0.0,
        "label_smoothing": 0.0,
        "mask_tubuh":      True,
    },
    "gabungan_mask": {
        "keterangan":      "Gabungan + mask tubuh",
        "augmentasi":      "kuat",
        "weight_decay":    1e-4,
        "label_smoothing": 0.1,
        "mask_tubuh":      True,
    },
}

_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]


def bangun_transform(mode):
    """
    Transform latih sesuai mode augmentasi, dan transform validasi/uji yang
    selalu tanpa augmentasi.

    'standar' mengikuti Tabel 3.1 proposal. 'kuat' memperluas rentang setiap
    teknik yang sama tanpa menambah teknik baru, sehingga tetap sejalan dengan
    pertimbangan citra medis pada proposal: rotasi dan pergeseran diperbesar,
    rentang skala dan kecerahan diperlebar, namun tidak ada pembalikan vertikal
    maupun distorsi yang dapat mengubah makna anatomi.
    """
    if mode == "kuat":
        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(25),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15),
                                    scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
        ]
    else:
        augment = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1),
                                    scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.1),
        ]

    tf_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        GaussianDenoising(sigma=DENOISE_SIGMA),
        *augment,
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        GaussianDenoising(sigma=DENOISE_SIGMA),
        transforms.ToTensor(),
        transforms.Normalize(mean=_MEAN, std=_STD),
    ])
    return tf_train, tf_eval


class DatasetMask(Dataset):
    """
    Sama seperti KidneyDataset, namun area di luar tubuh dihitamkan lebih dulu.
    Mask diterapkan pada resolusi ASLI sebelum resize, identik dengan yang
    dipakai pada uji perturbasi dan visualisasi Grad-CAM, agar ketiganya
    sebanding.
    """

    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img, _ = terapkan_mask_tubuh(img)
        return self.transform(img), int(self.labels[i])


def siapkan_data(tf_train, tf_eval, pakai_mask=False):
    """Pembagian data identik dengan eksperimen utama agar hasilnya sebanding."""
    all_paths, all_labels = load_dataset()
    idx_all = np.arange(len(all_paths))

    if not GROUP_AWARE_SPLIT:
        raise RuntimeError(
            "GROUP_AWARE_SPLIT bernilai False. Ablation ini dimaksudkan untuk "
            "skenario group-aware; setel True pada config.py terlebih dahulu."
        )

    groups = compute_group_labels(all_paths)
    idx_tv,    idx_test = group_aware_holdout(idx_all, all_labels, groups,
                                              TEST_SPLIT, SEED)
    idx_train, idx_val  = group_aware_holdout(idx_tv, all_labels, groups,
                                              0.20, SEED)

    print(f"  Train {len(idx_train):,} | Val {len(idx_val):,} | "
          f"Test {len(idx_test):,}")

    Kelas = DatasetMask if pakai_mask else KidneyDataset
    buat = lambda idx, tf, acak: DataLoader(
        Kelas(all_paths[idx], all_labels[idx], tf),
        batch_size=BATCH_SIZE, shuffle=acak, num_workers=0)

    return (buat(idx_train, tf_train, True),
            buat(idx_val,   tf_eval,  False),
            buat(idx_test,  tf_eval,  False))


def jalankan(nama):
    cfg = KONFIG[nama]
    set_seed()

    print(f"\n{'='*70}")
    print(f"ABLATION: {nama}  —  {cfg['keterangan']}")
    print(f"  augmentasi {cfg['augmentasi']} | weight_decay {cfg['weight_decay']} "
          f"| label_smoothing {cfg['label_smoothing']} "
          f"| mask_tubuh {cfg['mask_tubuh']}")
    print(f"{'='*70}")

    tf_train, tf_eval = bangun_transform(cfg["augmentasi"])
    loader_train, loader_val, loader_test = siapkan_data(
        tf_train, tf_eval, pakai_mask=cfg["mask_tubuh"])

    model     = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    if cfg["weight_decay"] > 0:
        optimizer = optim.AdamW(model.parameters(), lr=LR,
                                weight_decay=cfg["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=LR)

    dir_cfg = os.path.join(OUT_DIR, nama)
    os.makedirs(dir_cfg, exist_ok=True)
    path_csv = os.path.join(dir_cfg, "metrics.csv")
    with open(path_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1"])

    best_f1, best_weights = -1.0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        rl, rc = 0.0, 0
        for imgs, labels in loader_train:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            rl += loss.item() * imgs.size(0)
            rc += (out.argmax(1) == labels).sum().item()
        train_loss = rl / len(loader_train.dataset)
        train_acc  = rc / len(loader_train.dataset)

        model.eval()
        vl, vc, vp, vy = 0.0, 0, [], []
        with torch.no_grad():
            for imgs, labels in loader_val:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out  = model(imgs)
                vl  += criterion(out, labels).item() * imgs.size(0)
                vc  += (out.argmax(1) == labels).sum().item()
                vp.extend(out.argmax(1).cpu().numpy())
                vy.extend(labels.cpu().numpy())
        val_loss = vl / len(loader_val.dataset)
        val_acc  = vc / len(loader_val.dataset)
        val_f1   = f1_score(vy, vp, average="macro", zero_division=0)

        # Pemilihan model terbaik berdasarkan macro F1, sesuai Bab 3.9 proposal
        tandai = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_weights = {k: v.cpu().clone()
                            for k, v in model.state_dict().items()}
            torch.save(best_weights, os.path.join(dir_cfg, "best_model.pth"))
            tandai = "  <- terbaik"

        with open(path_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, round(train_loss, 6),
                                    round(train_acc, 6), round(val_loss, 6),
                                    round(val_acc, 6), round(val_f1, 6)])

        print(f"  Epoch {epoch:02d}/{EPOCHS} | latih loss {train_loss:.4f} "
              f"acc {train_acc*100:.2f}% | val loss {val_loss:.4f} "
              f"acc {val_acc*100:.2f}% F1 {val_f1:.4f}{tandai}")

    # ── Evaluasi test set ────────────────────────────────────────────────────
    model.load_state_dict(best_weights)
    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for imgs, labels in loader_test:
            preds.extend(model(imgs.to(DEVICE)).argmax(1).cpu().numpy())
            ys.extend(labels.numpy())

    hasil = {
        "konfigurasi":     nama,
        "keterangan":      cfg["keterangan"],
        "train_acc_akhir": round(train_acc, 6),
        "val_f1_terbaik":  round(best_f1, 6),
        "accuracy":        round(accuracy_score(ys, preds), 6),
        "macro_precision": round(precision_score(ys, preds, average="macro",
                                                 zero_division=0), 6),
        "macro_recall":    round(recall_score(ys, preds, average="macro",
                                              zero_division=0), 6),
        "macro_f1":        round(f1_score(ys, preds, average="macro",
                                          zero_division=0), 6),
    }
    hasil["selisih_train_test"] = round(hasil["train_acc_akhir"] -
                                        hasil["accuracy"], 6)

    print(f"\n  HASIL TEST SET")
    print(f"    Accuracy        : {hasil['accuracy']*100:.2f}%")
    print(f"    Macro F1        : {hasil['macro_f1']:.4f}")
    print(f"    Selisih latih-uji: {hasil['selisih_train_test']*100:.2f} poin "
          f"(makin kecil makin sedikit overfitting)")
    print()
    print(classification_report(ys, preds, target_names=KELAS_LIST, digits=4))

    with open(os.path.join(dir_cfg, "test_results.json"), "w") as f:
        json.dump(hasil, f, indent=2)

    # Perbarui ringkasan; baris konfigurasi yang sama ditimpa
    baris = []
    if os.path.exists(RINGKAS):
        with open(RINGKAS) as f:
            baris = [r for r in csv.DictReader(f)
                     if r["konfigurasi"] != nama]
    baris.append({k: str(v) for k, v in hasil.items()})
    with open(RINGKAS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(hasil.keys()))
        w.writeheader()
        w.writerows(baris)
    print(f"  Ringkasan diperbarui: {RINGKAS}")


def ringkas():
    if not os.path.exists(RINGKAS):
        print("Belum ada hasil ablation. Jalankan minimal satu konfigurasi.")
        return
    with open(RINGKAS) as f:
        baris = list(csv.DictReader(f))
    urut = {k: i for i, k in enumerate(KONFIG)}
    baris.sort(key=lambda r: urut.get(r["konfigurasi"], 99))

    print("\n" + "=" * 92)
    print("PERBANDINGAN HASIL ABLATION REGULARISASI (test set)")
    print("=" * 92)
    print(f"{'Konfigurasi':<13}{'Accuracy':>10}{'Macro-F1':>11}"
          f"{'Val F1':>10}{'Latih':>9}{'Selisih':>10}   Keterangan")
    for r in baris:
        print(f"{r['konfigurasi']:<13}"
              f"{float(r['accuracy'])*100:>9.2f}%"
              f"{float(r['macro_f1']):>11.4f}"
              f"{float(r['val_f1_terbaik']):>10.4f}"
              f"{float(r['train_acc_akhir'])*100:>8.1f}%"
              f"{float(r['selisih_train_test'])*100:>9.1f}p   {r['keterangan']}")

    if len(baris) > 1:
        dasar = next((r for r in baris if r["konfigurasi"] == "dasar"), None)
        if dasar:
            print("\nSelisih macro-F1 terhadap konfigurasi 'dasar':")
            for r in baris:
                if r["konfigurasi"] == "dasar":
                    continue
                d = (float(r["macro_f1"]) - float(dasar["macro_f1"])) * 100
                tanda = "naik" if d > 0 else "turun"
                print(f"  {r['konfigurasi']:<13}{d:+6.2f} poin  ({tanda})")
            print("\nAmbang keputusan: jalankan ulang 25 eksperimen FL hanya "
                  "bila kenaikan melebihi 3 poin.")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    if arg in ("--ringkas", "-r"):
        ringkas()
    elif arg in KONFIG:
        jalankan(arg)
        ringkas()
    else:
        print(__doc__)
        print("Konfigurasi tersedia:")
        for k, v in KONFIG.items():
            print(f"  {k:<13} {v['keterangan']}")
