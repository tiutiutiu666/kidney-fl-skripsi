from torchvision import transforms
from PIL import ImageFilter
from config import IMG_SIZE, DENOISE_SIGMA


class GaussianDenoising:
    """
    Transform kustom: mengurangi noise pada gambar menggunakan Gaussian Blur (PIL).
    Diterapkan sebelum resize pada train maupun val/test.
    """
    def __init__(self, sigma=DENOISE_SIGMA):
        self.sigma = sigma

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(radius=self.sigma))

    def __repr__(self):
        return f"GaussianDenoising(sigma={self.sigma})"


def get_transforms():
    """
    Return dua transform:
    - transform_train : denoising + augmentasi + normalisasi
    - transform_val   : denoising + resize + normalisasi (tanpa augmentasi)
    """
    _mean = [0.485, 0.456, 0.406]
    _std  = [0.229, 0.224, 0.225]

    transform_train = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        GaussianDenoising(sigma=DENOISE_SIGMA),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.RandomAffine(degrees=0,
                                translate=(0.1, 0.1),
                                scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    transform_val = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        GaussianDenoising(sigma=DENOISE_SIGMA),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std),
    ])

    return transform_train, transform_val