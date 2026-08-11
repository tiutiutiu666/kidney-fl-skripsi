import torch.nn as nn
import timm
from config import NUM_CLASSES


def build_model():
    """
    Membangun EfficientNet-B0 dengan classifier head yang diganti
    sesuai jumlah kelas (Linear 1280 → NUM_CLASSES).
    """
    model = timm.create_model('efficientnet_b0', pretrained=True)
    in_features       = model.classifier.in_features   # 1280
    model.classifier  = nn.Linear(in_features, NUM_CLASSES)
    return model