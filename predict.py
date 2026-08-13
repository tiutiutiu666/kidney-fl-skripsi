"""
predict.py — Prediksi kelas CT Scan Ginjal menggunakan model terbaik.

Cara pakai:
1. Ganti variabel IMAGE_PATH di bawah dengan path gambar yang ingin di-cek.
2. Jalankan: python predict.py
3. Script akan otomatis mencari model terbaik (FL atau baseline),
   lalu menampilkan hasil klasifikasi, probabilitas, dan Grad-CAM.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from config import (
    RESULT_DIR, KELAS_LIST, NUM_CLASSES, IMG_SIZE,
    ALPHAS, K_FOLD, BASELINE_DIR
)
from src.models.efficientnet_model import build_model
from src.data.preprocessing import get_transforms, GaussianDenoising

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  GANTI PATH GAMBAR DI SINI                                                ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
IMAGE_PATH = r"/root/kidneyfl/P040_FA_M_S_I05_DO.jpg"
# Contoh:
# IMAGE_PATH = r"D:\data\test_cyst_001.jpg"
# IMAGE_PATH = r"D:\KULI AH\888\code\kidney fl\data\raw\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone\Cyst\Cyst- (1).jpg"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_best_model():
    """
    Mencari model terbaik yang tersedia.
    Prioritas:
    1. Model FL terbaik (berdasarkan rata-rata F1 per alpha dari fold_status.json)
    2. Model baseline (baseline_model.pth)

    Return: (state_dict, info_dict) atau raise RuntimeError jika tidak ada.
    """
    # ── Coba cari model FL terbaik ────────────────────────────────────────────
    alpha_stats = {}
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

    if alpha_stats:
        best_alpha = max(alpha_stats, key=lambda a: alpha_stats[a]["avg_f1"])
        info       = alpha_stats[best_alpha]
        model_path = os.path.join(
            RESULT_DIR, f"alpha_{best_alpha}",
            f"fold_{info['best_fold']}", "best_model.pth"
        )
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=DEVICE)
            desc  = {
                "sumber":     "Federated Learning (FedProx)",
                "alpha":      best_alpha,
                "fold":       info["best_fold"],
                "avg_f1":     info["avg_f1"],
                "fold_f1":    info["best_fold_f1"],
                "model_path": model_path,
            }
            return state, desc

    # ── Fallback: model baseline ──────────────────────────────────────────────
    baseline_path = os.path.join(BASELINE_DIR, "baseline_model.pth")
    if os.path.exists(baseline_path):
        state = torch.load(baseline_path, map_location=DEVICE)
        desc  = {
            "sumber":     "Baseline (Centralized Training)",
            "model_path": baseline_path,
        }
        return state, desc

    raise RuntimeError(
        "Tidak ditemukan model yang sudah di-train!\n"
        "Jalankan training terlebih dahulu:\n"
        "  - FL:       python main.py\n"
        "  - Baseline: python experiments/baseline/run_baseline.py"
    )


def load_and_preprocess(image_path):
    """
    Memuat gambar dan menerapkan preprocessing yang sama dengan pipeline validasi.
    Return: (tensor [1,3,224,224], img_pil asli yang sudah di-resize, img_np [0-1])
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Gambar tidak ditemukan: {image_path}")

    img_pil = Image.open(image_path).convert("RGB")
    print(f"  Ukuran asli: {img_pil.size[0]}×{img_pil.size[1]} piksel")

    # Gunakan transform validasi (denoising + resize + normalize)
    _, transform_val = get_transforms()
    tensor = transform_val(img_pil).unsqueeze(0)   # [1, 3, 224, 224]

    # Versi untuk visualisasi (resize saja, tanpa normalize)
    img_resized = img_pil.resize((IMG_SIZE, IMG_SIZE))
    img_np      = np.array(img_resized).astype(np.float32) / 255.0

    return tensor, img_resized, img_np


def predict_single(model, tensor, device):
    """
    Jalankan inferensi pada 1 gambar.
    Return: (predicted_class_idx, probabilities_array)
    """
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
    pred_idx = int(probs.argmax())
    return pred_idx, probs


def generate_gradcam_single(model, tensor, img_np, pred_idx, device):
    """
    Generate Grad-CAM overlay untuk 1 gambar.
    Return: overlay image (numpy array RGB)
    """
    # Padanan persis `features[-1]` torchvision: conv head 1x1 (320->1280)
    # setelah seluruh blok MBConv, bukan conv di dalam blok MBConv terakhir.
    target_layer = [model.conv_head]

    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale = cam(input_tensor=tensor.to(device), targets=None)[0]

    overlay = show_cam_on_image(img_np, grayscale, use_rgb=True, image_weight=0.5)
    return overlay, grayscale


def visualize_result(img_resized, img_np, overlay, grayscale,
                     pred_idx, probs, save_path):
    """
    Buat visualisasi lengkap: gambar asli, heatmap, overlay, dan bar chart probabilitas.
    """
    fig = plt.figure(figsize=(16, 5))
    fig.patch.set_facecolor('#1a1a2e')

    # ── Judul utama ───────────────────────────────────────────────────────────
    pred_class = KELAS_LIST[pred_idx]
    pred_prob  = probs[pred_idx] * 100
    fig.suptitle(
        f"Prediksi: {pred_class}  ({pred_prob:.1f}%)",
        fontsize=18, fontweight='bold', color='white', y=0.98
    )

    # ── Panel 1: Gambar asli ──────────────────────────────────────────────────
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(np.array(img_resized))
    ax1.set_title("Gambar Input", fontsize=11, color='white', fontweight='bold')
    ax1.axis('off')

    # ── Panel 2: Heatmap Grad-CAM ─────────────────────────────────────────────
    ax2 = fig.add_subplot(1, 4, 2)
    ax2.imshow(grayscale, cmap='jet', vmin=0, vmax=1)
    ax2.set_title("Grad-CAM Heatmap", fontsize=11, color='white', fontweight='bold')
    ax2.axis('off')

    # ── Panel 3: Overlay ──────────────────────────────────────────────────────
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(overlay)
    ax3.set_title("Overlay", fontsize=11, color='white', fontweight='bold')
    ax3.axis('off')

    # ── Panel 4: Bar chart probabilitas ───────────────────────────────────────
    ax4 = fig.add_subplot(1, 4, 4)
    colors = ['#e74c3c' if i == pred_idx else '#3498db' for i in range(NUM_CLASSES)]
    bars = ax4.barh(KELAS_LIST, probs * 100, color=colors, edgecolor='white',
                    linewidth=0.5, height=0.6)
    ax4.set_xlim(0, 105)
    ax4.set_xlabel("Probabilitas (%)", fontsize=10, color='white')
    ax4.set_title("Confidence per Kelas", fontsize=11, color='white', fontweight='bold')
    ax4.tick_params(colors='white')
    ax4.set_facecolor('#16213e')
    for spine in ax4.spines.values():
        spine.set_color('#444')

    # Tambahkan label persentase di bar
    for bar, prob in zip(bars, probs):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{prob*100:.1f}%", va='center', fontsize=9, color='white',
                 fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Visualisasi tersimpan: {save_path}")


def sanity_check_analysis(probs, pred_idx):
    """
    Analisis sederhana untuk membantu menilai apakah prediksi masuk akal.
    Membantu mengatasi keraguan soal akurasi tinggi.
    """
    pred_prob   = probs[pred_idx]
    sorted_idx  = np.argsort(probs)[::-1]
    second_idx  = sorted_idx[1]
    second_prob = probs[second_idx]
    margin      = pred_prob - second_prob

    print("\n" + "─" * 60)
    print("  SANITY CHECK — ANALISIS KEPERCAYAAN PREDIKSI")
    print("─" * 60)

    # 1. Tingkat keyakinan model
    if pred_prob > 0.95:
        level = "SANGAT TINGGI"
        emoji = "🟢"
        note  = ("Model sangat yakin. Ini umum pada dataset CT Kidney\n"
                 "   karena perbedaan visual antar kelas cukup jelas\n"
                 "   (mis. batu ginjal vs jaringan normal).")
    elif pred_prob > 0.80:
        level = "TINGGI"
        emoji = "🟡"
        note  = "Model cukup yakin, tapi ada sedikit ambiguitas."
    elif pred_prob > 0.50:
        level = "SEDANG"
        emoji = "🟠"
        note  = ("Model kurang yakin — gambar mungkin ambigu atau\n"
                 "   berada di batas antar kelas.")
    else:
        level = "RENDAH"
        emoji = "🔴"
        note  = ("Model tidak yakin. Prediksi mungkin tidak akurat.\n"
                 "   Perlu review manual oleh ahli.")

    print(f"\n  {emoji} Tingkat keyakinan: {level} ({pred_prob*100:.1f}%)")
    print(f"     {note}")

    # 2. Margin terhadap kelas runner-up
    print(f"\n  Margin vs runner-up ({KELAS_LIST[second_idx]}): "
          f"{margin*100:.1f}%")

    if margin < 0.10:
        print("     ⚠ Margin sangat tipis — model ragu antara dua kelas.")
    elif margin < 0.30:
        print("     ℹ Margin moderat — ada sedikit kemiripan antar kelas.")
    else:
        print("     ✓ Margin besar — prediksi cukup tegas.")

# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MAIN                                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
if __name__ == "__main__":
    print(f"\n{'═'*60}")
    print("  PREDIKSI CT SCAN GINJAL — KIDNEY CLASSIFICATION")
    print(f"  Device: {DEVICE}")
    print(f"{'═'*60}\n")

    # ── 1. Validasi input ─────────────────────────────────────────────────────
    if IMAGE_PATH == r"GANTI_DENGAN_PATH_GAMBAR_ANDA.jpg":
        print("❌ ERROR: Anda belum mengganti IMAGE_PATH!")
        print("   Buka file predict.py dan ganti variabel IMAGE_PATH")
        print("   dengan path gambar CT scan ginjal yang ingin diprediksi.")
        sys.exit(1)

    print(f"📷 Gambar input: {IMAGE_PATH}")

    # ── 2. Cari dan muat model terbaik ────────────────────────────────────────
    print(f"\n🔍 Mencari model terbaik...")
    try:
        model_state, model_info = find_best_model()
    except RuntimeError as e:
        print(f"\n❌ {e}")
        sys.exit(1)

    print(f"  ✓ Model ditemukan!")
    print(f"  Sumber : {model_info['sumber']}")
    if 'alpha' in model_info:
        print(f"  Alpha  : {model_info['alpha']}")
        print(f"  Fold   : {model_info['fold']}")
        print(f"  Avg F1 : {model_info['avg_f1']:.4f}")
    print(f"  File   : {model_info['model_path']}")

    # ── 3. Build model dan load weights ───────────────────────────────────────
    print(f"\n🏗️  Membangun model EfficientNet-B0...")
    model = build_model().to(DEVICE)
    model.load_state_dict(model_state)
    model.eval()
    print(f"  ✓ Model siap (parameter: {sum(p.numel() for p in model.parameters()):,})")

    # ── 4. Preprocess gambar ──────────────────────────────────────────────────
    print(f"\n🖼️  Memproses gambar...")
    try:
        tensor, img_resized, img_np = load_and_preprocess(IMAGE_PATH)
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    print(f"  ✓ Preprocessing selesai (tensor shape: {tensor.shape})")

    # ── 5. Prediksi ───────────────────────────────────────────────────────────
    print(f"\n🧠 Menjalankan inferensi...")
    pred_idx, probs = predict_single(model, tensor, DEVICE)

    print(f"\n{'═'*60}")
    print(f"  HASIL PREDIKSI")
    print(f"{'═'*60}")
    print(f"\n  Kelas prediksi : {KELAS_LIST[pred_idx]}")
    print(f"  Confidence     : {probs[pred_idx]*100:.2f}%")
    print(f"\n  Probabilitas per kelas:")
    for i, kelas in enumerate(KELAS_LIST):
        bar_len = int(probs[i] * 40)
        bar     = "█" * bar_len + "░" * (40 - bar_len)
        marker  = " ← PREDIKSI" if i == pred_idx else ""
        print(f"    {kelas:8s} │ {bar} │ {probs[i]*100:6.2f}%{marker}")

    # ── 6. Grad-CAM ──────────────────────────────────────────────────────────
    print(f"\n🔬 Generating Grad-CAM...")
    overlay, grayscale = generate_gradcam_single(
        model, tensor, img_np, pred_idx, DEVICE
    )

    # ── 7. Visualisasi & Simpan ───────────────────────────────────────────────
    save_dir = os.path.join(RESULT_DIR, "predictions")
    os.makedirs(save_dir, exist_ok=True)

    img_basename = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
    save_path    = os.path.join(save_dir, f"prediksi_{img_basename}.png")

    visualize_result(img_resized, img_np, overlay, grayscale,
                     pred_idx, probs, save_path)

    # ── 8. Sanity check ──────────────────────────────────────────────────────
    sanity_check_analysis(probs, pred_idx)

    print(f"{'═'*60}")
    print(f"  Selesai! Cek visualisasi di: {save_path}")
    print(f"{'═'*60}\n")
