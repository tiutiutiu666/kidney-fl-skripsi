import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from config import ALPHAS, RESULT_DIR, K_FOLD, dir_alpha

print("=" * 55)
print("STATUS PROGRESS EKSPERIMEN")
print("=" * 55)

for alpha in ALPHAS:
    alpha_dir   = dir_alpha(alpha)
    status_path = os.path.join(alpha_dir, "fold_status.json")

    print(f"\nα = {alpha}")

    if not os.path.exists(alpha_dir):
        print("  Belum dimulai")
        continue

    status = {}
    if os.path.exists(status_path):
        with open(status_path) as f:
            status = json.load(f)

    for fold in range(1, K_FOLD + 1):
        info = status.get(f"fold_{fold}", {})
        if info.get("done"):
            f1 = info["metrics"]["best_val_f1"]
            print(f"  Fold {fold}: ✓ Selesai  | Best Val F1 = {f1:.4f}")
        else:
            fold_dir = os.path.join(alpha_dir, f"fold_{fold}")
            if os.path.exists(fold_dir):
                ckpts = sorted([f for f in os.listdir(fold_dir)
                                if f.startswith("checkpoint_round_")])
                if ckpts:
                    last = int(ckpts[-1].split("_")[-1].replace(".pth",""))
                    print(f"  Fold {fold}: ⟳ Terjeda  | Checkpoint: ronde {last}")
                else:
                    print(f"  Fold {fold}: ○ Belum dimulai")
            else:
                print(f"  Fold {fold}: ○ Belum dimulai")