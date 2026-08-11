import torch


def fedprox_aggregate(local_weights_list, num_samples_list):
    """
    Weighted average berbobot jumlah sampel tiap klien (FedProx / FedAvg aggregation).

    Menangani tensor integer (mis. BatchNorm num_batches_tracked) dengan aman:
    - Tensor float  → weighted average biasa
    - Tensor integer → diambil dari klien dengan sampel terbanyak (majority owner)

    Return: dict global weights (OrderedDict-compatible)
    """
    total = sum(num_samples_list)
    # Indeks klien dengan data terbanyak (fallback untuk tensor integer)
    majority_k = num_samples_list.index(max(num_samples_list))

    global_weights = {}
    for key in local_weights_list[0].keys():
        ref_tensor = local_weights_list[0][key]

        if not ref_tensor.is_floating_point():
            # Tensor integer: ambil dari klien terbesar, jangan di-average
            global_weights[key] = local_weights_list[majority_k][key].clone()
        else:
            # Tensor float: weighted average berbobot jumlah sampel
            weighted = sum(
                local_weights_list[k][key].float() * (num_samples_list[k] / total)
                for k in range(len(local_weights_list))
            )
            # Kembalikan ke dtype asli (float32 / float16 / bfloat16)
            global_weights[key] = weighted.to(ref_tensor.dtype)

    return global_weights