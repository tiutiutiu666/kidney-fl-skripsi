"""
src/federated/client.py
Implementasi klien Flower (NumPyClient) untuk FL klasifikasi CT ginjal.
Training lokal menggunakan FedProx loss.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl

from src.data.augmentation import KidneyDataset
from src.models.efficientnet_model import build_model
from src.seeding import set_seed
from config import LOCAL_EPOCHS, LR, BATCH_SIZE, MU_FEDPROX, SEED


# ── Utilitas konversi parameter ────────────────────────────────────────────────

def get_params(model):
    """Ambil semua parameter model sebagai list numpy array (format Flower)."""
    return [val.cpu().numpy() for val in model.state_dict().values()]


def set_params(model, params):
    """Muat parameter dari list numpy array ke dalam model."""
    keys = list(model.state_dict().keys())
    new_state = {k: torch.tensor(v) for k, v in zip(keys, params)}
    model.load_state_dict(new_state, strict=True)


# ── Flower NumPyClient ─────────────────────────────────────────────────────────

class KidneyFlowerClient(fl.client.NumPyClient):
    """
    Klien FL untuk klasifikasi penyakit ginjal dari CT scan.

    Setiap klien merepresentasikan satu rumah sakit virtual yang:
    - Menyimpan data lokal (paths + labels)
    - Melakukan training lokal dengan FedProx
    - Mengirim bobot yang diperbarui ke server untuk agregasi
    """

    def __init__(self, client_id, paths_train, labels_train,
                 paths_val, labels_val,
                 transform_train, transform_val, device):
        self.client_id      = client_id
        self.paths_train    = paths_train
        self.labels_train   = labels_train
        self.paths_val      = paths_val
        self.labels_val     = labels_val
        self.transform_train = transform_train
        self.transform_val   = transform_val
        self.device         = device
        # Klien dijalankan sebagai proses terpisah oleh Ray, sehingga seed yang
        # ditetapkan di proses utama tidak diwariskan. Seed di-offset dengan
        # client_id agar tiap klien tetap deterministik namun tidak identik
        # (mis. urutan pengacakan batch dan augmentasinya berbeda antar klien).
        set_seed(SEED + client_id)
        self.model          = build_model().to(device)

    # ── Flower interface ───────────────────────────────────────────────────────

    def get_parameters(self, config):
        """Kembalikan parameter model saat ini ke server."""
        return get_params(self.model)

    def fit(self, parameters, config):
        """
        Terima bobot global, latih secara lokal dengan FedProx, kirim kembali.
        FedProx loss = CrossEntropy + (mu/2) * ||w_k - w*||^2
        """
        # Muat bobot global dari server
        set_params(self.model, parameters)
        # Simpan salinan bobot global sebagai anchor untuk proximal term
        global_params = [p.detach().clone() for p in self.model.parameters()]

        loader = DataLoader(
            KidneyDataset(self.paths_train, self.labels_train,
                          self.transform_train),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)
        self.model.train()
        total_train_loss = 0.0
        total_samples    = 0

        for epoch in range(LOCAL_EPOCHS):
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                outputs  = self.model(imgs)
                loss_ce  = criterion(outputs, labels)

                # Proximal term FedProx: mencegah drift dari bobot global
                prox = (MU_FEDPROX / 2) * sum(
                    (lp - gp.to(self.device)).norm() ** 2
                    for lp, gp in zip(self.model.parameters(), global_params)
                )
                loss_total = loss_ce + prox
                loss_total.backward()
                optimizer.step()

                total_train_loss += loss_total.item() * imgs.size(0)
                total_samples    += imgs.size(0)

        avg_train_loss = total_train_loss / total_samples if total_samples > 0 else 0.0
        print(f"    [Klien {self.client_id + 1}] Selesai latih "
              f"({len(self.paths_train):,} sampel, {LOCAL_EPOCHS} epoch)")
        return get_params(self.model), len(self.paths_train), {
            "train_loss": float(avg_train_loss),
        }

    def evaluate(self, parameters, config):
        """
        Evaluasi bobot global pada data validasi lokal.
        Return: loss, num_samples, {"f1": float}
        """
        from sklearn.metrics import f1_score

        set_params(self.model, parameters)
        loader = DataLoader(
            KidneyDataset(self.paths_val, self.labels_val, self.transform_val),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0
        )

        criterion  = nn.CrossEntropyLoss()
        self.model.eval()
        total_loss, preds_all, labels_all = 0.0, [], []

        with torch.no_grad():
            for imgs, labels in loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                outputs     = self.model(imgs)
                total_loss += criterion(outputs, labels).item() * imgs.size(0)
                preds_all.extend(outputs.argmax(1).cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(loader.dataset)
        f1       = f1_score(labels_all, preds_all,
                            average='macro', zero_division=0)
        return float(avg_loss), len(self.paths_val), {"f1": float(f1)}


# ── Fungsi utilitas untuk server ───────────────────────────────────────────────

def evaluate_global(model, loader, device):
    """
    Evaluasi model global pada loader validasi terpusat.
    Return: avg_loss, accuracy, macro_f1
    """
    from sklearn.metrics import f1_score, accuracy_score
    import torch.nn as nn

    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs     = model(imgs)
            total_loss += criterion(outputs, labels).item() * imgs.size(0)
            preds_all.extend(outputs.argmax(1).cpu().numpy())
            labels_all.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc      = accuracy_score(labels_all, preds_all)
    f1       = f1_score(labels_all, preds_all,
                        average='macro', zero_division=0)
    return avg_loss, acc, f1