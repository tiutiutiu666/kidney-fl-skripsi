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


def generate_gradcam(model, paths_test, labels_test, device,
                     save_dir=None, num_samples=2):
    """
    Generate dan simpan visualisasi Grad-CAM.
    - num_samples : jumlah sampel per kelas
    - save_dir    : folder penyimpanan hasil
    """
    if save_dir is None:
        save_dir = os.path.join(RESULT_DIR, "final", "gradcam")
    os.makedirs(save_dir, exist_ok=True)

    transform   = get_transform_val()
    # Target layer: conv 1x1 terakhir (conv head) SETELAH seluruh blok MBConv,
    # yang memetakan 320 -> 1280 channel sebelum global average pooling.
    # Di timm EfficientNet-B0 ini adalah `model.conv_head`, padanan persis dari
    # `features[-1]` pada torchvision.models.efficientnet_b0 (bukan conv di
    # DALAM blok MBConv terakhir, yang hanya menghasilkan 320 channel).
    target_layer = [model.conv_head]

    # Kumpulkan sampel per kelas
    sampel = {i: [] for i in range(NUM_CLASSES)}
    for path, label in zip(paths_test, labels_test):
        if len(sampel[label]) < num_samples:
            sampel[label].append(path)
        if all(len(v) == num_samples for v in sampel.values()):
            break

    model.eval()
    n_col = num_samples * 3
    n_row = NUM_CLASSES
    fig, axes = plt.subplots(n_row, n_col,
                              figsize=(n_col * 3.2, n_row * 3.4))
    fig.patch.set_facecolor('#F0F2F5')
    fig.suptitle('Visualisasi Grad-CAM — EfficientNet-B0',
                 fontsize=14, fontweight='bold', y=1.01)

    WARNA_BENAR = '#2d6a4f'
    WARNA_SALAH = '#c73e1d'

    for row_idx, kelas_nama in enumerate(KELAS_LIST):
        axes[row_idx, 0].set_ylabel(f'Aktual:\n{kelas_nama}',
                                     fontsize=9, fontweight='bold',
                                     rotation=90, labelpad=8)
        for s_idx, img_path in enumerate(sampel[row_idx]):
            # Preprocess
            img_pil = Image.open(img_path).convert("RGB").resize(
                (IMG_SIZE, IMG_SIZE))
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
            axes[row_idx, col_base].axis('off')

            axes[row_idx, col_base + 1].imshow(grayscale, cmap='jet',
                                                vmin=0, vmax=1)
            axes[row_idx, col_base + 1].axis('off')

            axes[row_idx, col_base + 2].imshow(overlay)
            axes[row_idx, col_base + 2].axis('off')
            axes[row_idx, col_base + 2].set_xlabel(
                f"{status} Pred: {KELAS_LIST[pred_label]}\n"
                f"Prob: {pred_prob*100:.1f}%",
                fontsize=7.5, color=warna, fontweight='bold')

            for ax in axes[row_idx, col_base:col_base + 3]:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_edgecolor(warna)
                    spine.set_linewidth(2)

    plt.tight_layout()
    out_path = os.path.join(save_dir, "gradcam_hasil.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Grad-CAM tersimpan: {out_path}")