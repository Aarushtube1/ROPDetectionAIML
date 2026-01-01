import os
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image


class ROPDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        self.df = pd.read_csv(csv_file)
        self.transforms = transforms

        required_cols = {"Source", "ROP Label"}
        if not required_cols.issubset(self.df.columns):
            raise ValueError(f"CSV must contain columns: {required_cols}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = row["Source"]
        label = int(row["ROP Label"])

        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load image (PIL)
        image = Image.open(img_path).convert("RGB")

        if self.transforms:
            image = self.transforms(image)

        return image, torch.tensor(label, dtype=torch.float32)
