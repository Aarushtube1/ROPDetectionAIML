import cv2
import numpy as np
import torch
from torchvision import transforms


class CLAHETransform:
    """
    Apply CLAHE on the green channel of a retinal image.
    """
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(
            clipLimit=clip_limit,
            tileGridSize=tile_grid_size
        )

    def __call__(self, img):
        # img: PIL Image → numpy array
        img = np.array(img)

        # Convert to RGB if needed
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Extract green channel
        green = img[:, :, 1]

        # Apply CLAHE
        enhanced = self.clahe.apply(green)

        # Stack back to 3 channels
        img = np.stack([enhanced, enhanced, enhanced], axis=2)

        return img


def get_transforms(train=True):
    base_transforms = [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        CLAHETransform(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]

    if train:
        base_transforms.insert(
            2,
            transforms.RandomHorizontalFlip(p=0.5)
        )

    return transforms.Compose(base_transforms)
