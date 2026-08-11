"""
src/federated/server.py
Simulasi Federated Learning menggunakan Flower (flwr).

Arsitektur:
- KidneyFedProxStrategy : strategy kustom (extend FedAvg) dengan:
    - Agregasi berbobot sampel (FedProx / FedAvg server)
    - Evaluasi global setiap ronde
    - Early stopping berbasis macro F1
    - Checkpoint otomatis setiap CHECKPOINT_INTERVAL ronde
- run_simulation()      : orkestrasi K-Fold + flwr.simulation.start_simulation()
"""
import os
import json
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    Metrics, NDArrays, Parameters, Scalar, FitRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

from src.models.efficientnet_model import build_model
from src.federated.client import (
    KidneyFlowerClient, get_params, set_params, evaluate_global,
)
from src.federated.aggregation import fedprox_aggregate
from src.data.augmentation import KidneyDataset
from src.data.preprocessing import get_transforms
from config import (
    NUM_CLIENTS, NUM_ROUNDS, BATCH_SIZE,
    CHECKPOINT_INTERVAL, K_FOLD, EARLY_STOPPING_PATIENCE,
)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fold_dir(result_dir: str, fold: int) -> str:
    d = os.path.join(result_dir, f"fold_{fold}")
    os.makedirs(d, exist_ok=True)
    return d


def _save_checkpoint(result_dir, fold, round_num, payload: dict):
    path = os.path.join(_fold_dir(result_dir, fold),
                        f"checkpoint_round_{round_num:03d}.pth")
    torch.save(payload, path)


def _load_latest_checkpoint(result_dir, fold, device):
    """Kembalikan (state_dict_payload, last_round) atau (None, 0)."""
    d    = _fold_dir(result_dir, fold)
    ckpts = sorted([f for f in os.listdir(d)
                    if f.startswith("checkpoint_round_")])
    if not ckpts:
        return None, 0
    latest   = ckpts[-1]
    last_rnd = int(latest.split("_")[-1].replace(".pth", ""))
    payload  = torch.load(os.path.join(d, latest), map_location=device)
    print(f"  [Resume] Fold {fold} — lanjut dari ronde {last_rnd}")
    return payload, last_rnd


def _is_fold_done(result_dir, fold) -> bool:
    path = os.path.join(result_dir, "fold_status.json")
    if not os.path.exists(path):
        return False
    with open(path) as f:
        status = json.load(f)
    return status.get(f"fold_{fold}", {}).get("done", False)


def _mark_fold_done(result_dir, fold, metrics: dict):
    path   = os.path.join(result_dir, "fold_status.json")
    status = {}
    if os.path.exists(path):
        with open(path) as f:
            status = json.load(f)
    status[f"fold_{fold}"] = {"done": True, "metrics": metrics}
    with open(path, "w") as f:
        json.dump(status, f, indent=2)
    print(f"  [Done] Fold {fold} — F1: {metrics['best_val_f1']:.4f}")


def _append_metrics_csv(result_dir, fold, round_num, metrics: dict):
    path        = os.path.join(_fold_dir(result_dir, fold), "metrics.csv")
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["round", "train_loss", "val_loss", "val_acc", "val_f1"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({"round": round_num, **metrics})


def _truncate_metrics_csv(result_dir, fold, start_round):
    """
    Buang baris metrik dengan round > start_round.

    Checkpoint hanya disimpan tiap CHECKPOINT_INTERVAL ronde. Jika proses
    terhenti di antara dua checkpoint (mis. mati di ronde 22 sedangkan
    checkpoint terakhir ronde 20), training akan diulang dari ronde 21 —
    sementara baris ronde 21-22 dari percobaan sebelumnya masih tertinggal di
    metrics.csv. Akibatnya muncul baris duplikat dengan nomor ronde sama, yang
    merusak kurva training (sumbu-x tidak monoton) dan membuat analisis
    konvergensi keliru. Baris basi tersebut dibuang di sini saat resume.
    """
    path = os.path.join(_fold_dir(result_dir, fold), "metrics.csv")
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    kept = [r for r in rows if int(r["round"]) <= start_round]
    if len(kept) == len(rows):
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["round", "train_loss", "val_loss", "val_acc", "val_f1"])
        writer.writeheader()
        writer.writerows(kept)
    print(f"  [Resume] {len(rows) - len(kept)} baris metrik basi "
          f"(ronde > {start_round}) dibuang dari metrics.csv")


# ══════════════════════════════════════════════════════════════════════════════
# Custom Flower Strategy — FedProx dengan Early Stopping & Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

class KidneyFedProxStrategy(FedAvg):
    """
    Strategy Flower kustom untuk federated learning penyakit ginjal.

    Fitur tambahan di atas FedAvg standar:
    1. Agregasi berbobot sampel (identik dengan FedProx server-side)
    2. Evaluasi global (centralized) setiap akhir ronde
    3. Early stopping berbasis macro F1 (patience = EARLY_STOPPING_PATIENCE)
    4. Simpan checkpoint setiap CHECKPOINT_INTERVAL ronde
    5. Simpan best_model.pth saat F1 meningkat
    6. Mendukung resume dari checkpoint (start_round offset)
    """

    def __init__(
        self,
        *,
        initial_parameters: Parameters,
        val_loader: DataLoader,
        device: torch.device,
        result_dir: str,
        fold: int,
        start_round: int = 0,
        best_val_f1: float = 0.0,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 0.0,   # Evaluasi terpusat saja
        min_fit_clients: int = NUM_CLIENTS,
        min_available_clients: int = NUM_CLIENTS,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
        )
        self.val_loader   = val_loader
        self.device       = device
        self.result_dir   = result_dir
        self.fold         = fold
        self.start_round  = start_round   # offset untuk resume

        # State early stopping & best model
        self.best_val_f1   = best_val_f1
        self.patience_ctr  = 0
        self.stopped       = False

        # Cache bobot terbaik (CPU)
        self._best_weights: Optional[dict] = None

    # ── Override: hentikan training jika early stopping aktif ─────────────────

    def configure_fit(self, server_round, parameters, client_manager):
        if self.stopped:
            return []   # Tidak ada klien → Flower akan skip ronde ini
        return super().configure_fit(server_round, parameters, client_manager)

    # ── Override: agregasi + evaluasi + checkpoint + early stopping ────────────

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        if not results:
            return None, {}

        # ── 1. Kumpulkan bobot dan jumlah sampel tiap klien ──────────────────
        weights_list = []
        samples_list = []
        for _, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)
            # Rekonstruksi state_dict
            model_tmp = build_model()
            keys      = list(model_tmp.state_dict().keys())
            state     = {k: torch.tensor(v) for k, v in zip(keys, ndarrays)}
            weights_list.append(state)
            samples_list.append(fit_res.num_examples)

        # ── 1b. Hitung rata-rata training loss berbobot dari semua klien ─────
        train_losses  = []
        train_samples = []
        for _, fit_res in results:
            tl = fit_res.metrics.get("train_loss")
            if tl is not None:
                train_losses.append(tl)
                train_samples.append(fit_res.num_examples)
        if train_losses:
            total_s = sum(train_samples)
            avg_train_loss = sum(tl * ts / total_s
                                for tl, ts in zip(train_losses, train_samples))
        else:
            avg_train_loss = None

        # ── 2. Agregasi FedProx (weighted average, safe dtype) ───────────────
        aggregated_weights = fedprox_aggregate(weights_list, samples_list)

        # ── 3. Konversi ke Parameters Flower ─────────────────────────────────
        ndarrays_agg = [v.cpu().numpy()
                        for v in aggregated_weights.values()]
        aggregated_parameters = ndarrays_to_parameters(ndarrays_agg)

        # ── 4. Evaluasi global terpusat ───────────────────────────────────────
        global_model = build_model().to(self.device)
        global_model.load_state_dict(aggregated_weights)
        val_loss, val_acc, val_f1 = evaluate_global(
            global_model, self.val_loader, self.device)

        # Ronde aktual (dengan offset resume)
        actual_round = self.start_round + server_round

        _append_metrics_csv(self.result_dir, self.fold, actual_round, {
            "train_loss": round(avg_train_loss, 6) if avg_train_loss is not None else "",
            "val_loss": round(val_loss, 6),
            "val_acc":  round(val_acc,  6),
            "val_f1":   round(val_f1,   6),
        })

        is_best = val_f1 > self.best_val_f1
        print(f"  Ronde {actual_round:03d}/{self.start_round + NUM_ROUNDS} | "
              f"Loss: {val_loss:.4f} | "
              f"Acc: {val_acc*100:.2f}% | "
              f"F1: {val_f1:.4f}"
              + (" ← best" if is_best else ""))

        # ── 5. Perbarui best model ────────────────────────────────────────────
        if is_best:
            self.best_val_f1  = val_f1
            self.patience_ctr = 0
            self._best_weights = {k: v.cpu().clone()
                                  for k, v in aggregated_weights.items()}
            best_path = os.path.join(
                _fold_dir(self.result_dir, self.fold), "best_model.pth")
            torch.save(self._best_weights, best_path)
        else:
            self.patience_ctr += 1

        # ── 6. Checkpoint periodik ────────────────────────────────────────────
        if actual_round % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(self.result_dir, self.fold, actual_round, {
                "global_weights": aggregated_weights,
                "best_val_f1":    self.best_val_f1,
                "best_weights":   self._best_weights,
                "round":          actual_round,
                "fold":           self.fold,
            })
            print(f"  [Checkpoint] Ronde {actual_round} tersimpan")

        # ── 7. Early stopping ─────────────────────────────────────────────────
        if self.patience_ctr >= EARLY_STOPPING_PATIENCE:
            print(f"  [Early Stop] Fold {self.fold} berhenti di ronde "
                  f"{actual_round} (patience={EARLY_STOPPING_PATIENCE})")
            self.stopped = True

        return aggregated_parameters, {
            "val_loss": val_loss,
            "val_acc":  val_acc,
            "val_f1":   val_f1,
        }


# ══════════════════════════════════════════════════════════════════════════════
# Simulasi FL — K-Fold × 1 alpha
# ══════════════════════════════════════════════════════════════════════════════

def run_simulation(all_paths, all_labels, partitions,
                   idx_tv, result_dir, device, groups=None):
    """
    Menjalankan simulasi Flower FL untuk satu nilai alpha.
    Loop: K_FOLD fold × NUM_ROUNDS ronde maksimum.
    Mendukung resume otomatis dari checkpoint.

    groups : array label grup per citra (opsional). Jika diberikan, pembagian
             fold memakai StratifiedGroupKFold sehingga citra near-identical
             dari scan yang sama tidak terbelah antara train dan validasi.
    """
    transform_train, transform_val = get_transforms()

    # Gabungkan semua indeks klien (untuk split global via K-Fold)
    all_client_indices = np.concatenate([p for p in partitions])
    all_client_labels  = all_labels[all_client_indices]

    if groups is not None:
        skf        = StratifiedGroupKFold(n_splits=K_FOLD, shuffle=True,
                                          random_state=42)
        fold_split = skf.split(all_client_indices, all_client_labels,
                               groups=groups[all_client_indices])
    else:
        skf        = StratifiedKFold(n_splits=K_FOLD, shuffle=True,
                                     random_state=42)
        fold_split = skf.split(all_client_indices, all_client_labels)

    for fold_idx, (train_local_idx, val_local_idx) in enumerate(
            fold_split, start=1):

        print(f"\n{'─'*55}")
        print(f"FOLD {fold_idx}/{K_FOLD}")
        print(f"{'─'*55}")

        if _is_fold_done(result_dir, fold_idx):
            print(f"  [Skip] Fold {fold_idx} sudah selesai.")
            continue

        train_global = all_client_indices[train_local_idx]
        val_global   = all_client_indices[val_local_idx]

        # Validasi loader terpusat (digunakan oleh Strategy)
        loader_val = DataLoader(
            KidneyDataset(all_paths[val_global],
                          all_labels[val_global], transform_val),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )

        # ── Bagi train indices ke tiap klien sesuai partisi ───────────────────
        client_train_paths  = []
        client_train_labels = []
        client_val_paths    = []
        client_val_labels   = []

        for part in partitions:
            # Train: irisan antara train_global dan partisi klien ini
            mask_tr  = np.isin(train_global, part)
            idx_tr_k = train_global[mask_tr]
            # Val: irisan antara val_global dan partisi klien ini
            mask_va  = np.isin(val_global, part)
            idx_va_k = val_global[mask_va]

            client_train_paths.append(all_paths[idx_tr_k])
            client_train_labels.append(all_labels[idx_tr_k])
            client_val_paths.append(all_paths[idx_va_k])
            client_val_labels.append(all_labels[idx_va_k])

        # ── Resume: cek checkpoint ────────────────────────────────────────────
        ckpt, start_round = _load_latest_checkpoint(result_dir, fold_idx, device)
        # Buang baris metrik dari percobaan sebelumnya yang melewati checkpoint
        # terakhir, agar tidak ada nomor ronde duplikat di metrics.csv.
        _truncate_metrics_csv(result_dir, fold_idx, start_round)

        # Inisialisasi model global
        global_model = build_model().to(device)
        best_val_f1  = 0.0
        if ckpt is not None:
            global_model.load_state_dict(ckpt["global_weights"])
            best_val_f1 = ckpt.get("best_val_f1", 0.0)

        # ── Buat initial parameters untuk Flower ─────────────────────────────
        initial_params = ndarrays_to_parameters(get_params(global_model))

        # ── Buat Strategy ─────────────────────────────────────────────────────
        strategy = KidneyFedProxStrategy(
            initial_parameters=initial_params,
            val_loader=loader_val,
            device=device,
            result_dir=result_dir,
            fold=fold_idx,
            start_round=start_round,
            best_val_f1=best_val_f1,
        )

        # ── Buat client_fn (closure menangkap data fold ini) ──────────────────
        # Salin ke local var agar closure aman
        _cpaths  = client_train_paths
        _clabels = client_train_labels
        _vpaths  = client_val_paths
        _vlabels = client_val_labels
        _tftr    = transform_train
        _tftv    = transform_val
        _dev     = device

        def make_client_fn():
            def client_fn(cid: str) -> fl.client.Client:
                k = int(cid)
                if len(_cpaths[k]) == 0:
                    # Klien tanpa data: kembalikan dummy (skip training)
                    return KidneyFlowerClient(
                        k, _cpaths[k], _clabels[k],
                        _vpaths[k], _vlabels[k],
                        _tftr, _tftv, _dev,
                    ).to_client()
                return KidneyFlowerClient(
                    k, _cpaths[k], _clabels[k],
                    _vpaths[k], _vlabels[k],
                    _tftr, _tftv, _dev,
                ).to_client()
            return client_fn

        remaining_rounds = NUM_ROUNDS - start_round
        if remaining_rounds <= 0:
            print(f"  [Skip] Fold {fold_idx} sudah mencapai ronde maksimum.")
            _mark_fold_done(result_dir, fold_idx,
                            {"best_val_f1": round(best_val_f1, 6)})
            continue

        print(f"  Memulai simulasi Flower: {remaining_rounds} ronde "
              f"(ronde {start_round+1} s/d {NUM_ROUNDS})")

        # ── Jalankan simulasi Flower ──────────────────────────────────────────
        fl.simulation.start_simulation(
            client_fn=make_client_fn(),
            num_clients=NUM_CLIENTS,
            config=fl.server.ServerConfig(num_rounds=remaining_rounds),
            strategy=strategy,
            # Satu klien GPU penuh → eksekusi sekuensial (tidak paralel)
            client_resources={"num_cpus": 1, "num_gpus": 1.0
                              if str(device) != "cpu" else 0.0},
            ray_init_args={
                "ignore_reinit_error": True,
                "include_dashboard": False,
                "num_cpus": NUM_CLIENTS,
                "num_gpus": 1 if str(device) != "cpu" else 0,
            },
        )

        # ── Tandai fold selesai ───────────────────────────────────────────────
        _mark_fold_done(result_dir, fold_idx,
                        {"best_val_f1": round(strategy.best_val_f1, 6)})