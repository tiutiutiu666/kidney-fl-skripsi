import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import IMG_SIZE, NUM_CLASSES, KELAS_LIST, RESULT_DIR


def get_transform_val():
    """
    Transform untuk Grad-CAM — harus IDENTIK dengan transform validasi/uji
    di src/data/preprocessing.py (termasuk denoising), agar heatmap dihitung
    pada distribusi input yang sama dengan saat model dievaluasi.
    """
    from src.data.preprocessing import get_transforms
    _, transform_val = get_transforms()
    return transform_val


# Pilihan lapisan target Grad-CAM beserta resolusi peta fiturnya pada masukan
# 224x224. Lapisan yang lebih dangkal menghasilkan peta lebih rinci, namun
# menjauh dari "features[-1]" yang ditetapkan Bab 3.8 proposal.
LAPISAN_TARGET = {
    # Padanan persis features[-1] torchvision: conv head 1x1 (320 -> 1280)
    # setelah seluruh blok MBConv. Sesuai proposal — dipakai sebagai default.
    "conv_head":    (lambda m: m.conv_head,               "7x7"),
    # Conv proyeksi di DALAM blok MBConv terakhir.
    "blok_akhir":   (lambda m: m.blocks[-1][-1].conv_pwl, "7x7"),
    # Dua tahap lebih dangkal — peta 14x14, empat kali lebih rinci.
    "halus":        (lambda m: m.blocks[-3][-1].conv_pwl, "14x14"),
}


def generate_gradcam(model, paths_test, labels_test, device,
                     save_dir=None, num_samples=2,
                     pakai_mask_tubuh=False, lapisan="conv_head"):
    """
    Generate dan simpan visualisasi Grad-CAM.

    num_samples      : jumlah sampel per kelas
    save_dir         : folder penyimpanan hasil
    pakai_mask_tubuh : bila True, area di luar tubuh dihitamkan sebelum citra
                       masuk ke model. Meja pemindai karenanya tidak dapat
                       muncul pada heatmap. Uji perturbasi menunjukkan
                       perlakuan ini hanya menurunkan akurasi 0,92 poin
                       (90,12% -> 89,19%), sehingga perilaku model yang
                       divisualisasikan tetap mewakili model sesungguhnya.
    lapisan          : kunci pada LAPISAN_TARGET. Default "conv_head" sesuai
                       Bab 3.8 proposal; "halus" memberi peta 14x14 yang jauh
                       lebih rinci namun menyimpang dari proposal.
    """
    if save_dir is None:
        save_dir = os.path.join(RESULT_DIR, "final", "gradcam")
    os.makedirs(save_dir, exist_ok=True)

    if lapisan not in LAPISAN_TARGET:
        raise ValueError(f"Lapisan '{lapisan}' tidak dikenal. "
                         f"Pilihan: {list(LAPISAN_TARGET)}")
    ambil, resolusi = LAPISAN_TARGET[lapisan]
    target_layer    = [ambil(model)]

    transform = get_transform_val()
    if pakai_mask_tubuh:
        from src.data.masking import terapkan_mask_tubuh

    # Kumpulkan sampel per kelas
    sampel = {i: [] for i in range(NUM_CLASSES)}
    # Sampel diambil menyebar merata pada tiap kelas, bukan beberapa citra
    # pertama. Citra uji tersimpan berurutan, sehingga mengambil yang pertama
    # akan menghasilkan irisan-irisan dari pemindaian yang sama — dua kolom
    # sampel menjadi nyaris identik dan visualisasi kehilangan keragamannya.
    for kelas in range(NUM_CLASSES):
        idx_kelas = np.flatnonzero(np.asarray(labels_test) == kelas)
        if len(idx_kelas) == 0:
            continue
        pilih = idx_kelas[np.linspace(0, len(idx_kelas) - 1,
                                      num_samples).astype(int)]
        sampel[kelas] = [paths_test[i] for i in pilih]

    model.eval()
    n_col = num_samples * 3
    n_row = NUM_CLASSES
    fig, axes = plt.subplots(n_row, n_col,
                              figsize=(n_col * 3.2, n_row * 3.4))
    fig.patch.set_facecolor('#F0F2F5')
    ket_mask = 'area luar tubuh dihitamkan' if pakai_mask_tubuh else 'citra utuh'
    fig.suptitle('Visualisasi Grad-CAM — EfficientNet-B0\n'
                 f'lapisan {lapisan} ({resolusi}) · {ket_mask}',
                 fontsize=14, fontweight='bold', y=1.015)

    WARNA_BENAR = '#2d6a4f'
    WARNA_SALAH = '#c73e1d'

    # CATATAN: jangan memakai ax.axis('off') di sini. Perintah tersebut
    # mematikan seluruh elemen sumbu, termasuk set_xlabel/set_ylabel dan
    # spine — sehingga nama kelas aktual, kelas prediksi, probabilitas, serta
    # pewarnaan tepi benar/salah tidak pernah tampil pada gambar. Yang benar
    # adalah menyembunyikan tick-nya saja agar label tetap terlihat.
    def _bersihkan(ax):
        ax.set_xticks([])
        ax.set_yticks([])

    JUDUL_KOLOM = ['Citra Asli', 'Heatmap Grad-CAM', 'Overlay']

    for row_idx, kelas_nama in enumerate(KELAS_LIST):
        axes[row_idx, 0].set_ylabel(f'Aktual:\n{kelas_nama}',
                                     fontsize=10, fontweight='bold',
                                     rotation=90, labelpad=10)
        for s_idx, img_path in enumerate(sampel[row_idx]):
            # Preprocess. Mask tubuh diterapkan pada resolusi ASLI sebelum
            # resize, sama seperti pada uji perturbasi, agar keduanya sebanding.
            img_full = Image.open(img_path).convert("RGB")
            if pakai_mask_tubuh:
                img_full, _ = terapkan_mask_tubuh(img_full)
            img_pil = img_full.resize((IMG_SIZE, IMG_SIZE))
            img_np  = np.array(img_pil).astype(np.float32) / 255.0
            tensor  = transform(img_pil).unsqueeze(0).to(device)

            # Grad-CAM
            with GradCAM(model=model, target_layers=target_layer) as cam:
                grayscale = cam(input_tensor=tensor, targets=None)[0]
            overlay = show_cam_on_image(img_np, grayscale,
                                        use_rgb=True, image_weight=0.5)

            # Prediksi
            with torch.no_grad():
                probs      = torch.softmax(model(tensor), dim=1).cpu().numpy()[0]
            pred_label = int(probs.argmax())
            pred_prob  = float(probs.max())

            col_base = s_idx * 3
            warna    = WARNA_BENAR if pred_label == row_idx else WARNA_SALAH
            status   = '✓' if pred_label == row_idx else '✗'

            # Plot
            axes[row_idx, col_base].imshow(np.array(img_pil))
            _bersihkan(axes[row_idx, col_base])

            axes[row_idx, col_base + 1].imshow(grayscale, cmap='jet',
                                                vmin=0, vmax=1)
            _bersihkan(axes[row_idx, col_base + 1])

            axes[row_idx, col_base + 2].imshow(overlay)
            _bersihkan(axes[row_idx, col_base + 2])
            axes[row_idx, col_base + 2].set_xlabel(
                f"{status} Prediksi: {KELAS_LIST[pred_label]}  "
                f"({pred_prob*100:.1f}%)",
                fontsize=9, color=warna, fontweight='bold', labelpad=6)

            # Judul kolom hanya pada baris teratas
            if row_idx == 0:
                for j, judul in enumerate(JUDUL_KOLOM):
                    axes[0, col_base + j].set_title(
                        judul, fontsize=10, fontweight='bold',
                        color='#333', pad=8)

            for ax in axes[row_idx, col_base:col_base + 3]:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(warna)
                    spine.set_linewidth(2.5)

    # Keterangan warna tepi
    fig.text(0.5, -0.012,
             "Tepi hijau = prediksi benar    ·    Tepi merah = prediksi salah",
             ha='center', fontsize=10, color='#444', fontweight='bold')

    plt.tight_layout()
    out_path = os.path.join(save_dir, "gradcam_hasil.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM tersimpan: {out_path}")