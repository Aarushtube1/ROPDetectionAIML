import torch
from torch.utils.data import DataLoader
from torch.optim import Adam

from src.data.dataset import ROPDataset
from src.data.transforms import get_transforms
from src.models.resnet import ResNet18ROP
from src.training.loss import get_loss
from src.training.train import train_one_epoch, validate

from pathlib import Path
import pandas as pd


# ---------------- CONFIG ----------------
BATCH_SIZE = 16
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_CSV = PROJECT_ROOT / "data/splits/train.csv"
VAL_CSV = PROJECT_ROOT / "data/splits/val.csv"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)
# ---------------------------------------


def compute_pos_weight(csv_path):
    df = pd.read_csv(csv_path)
    n_pos = (df["ROP Label"] == 1).sum()
    n_neg = (df["ROP Label"] == 0).sum()
    return torch.tensor([n_neg / n_pos], dtype=torch.float32)


def main():
    train_ds = ROPDataset(TRAIN_CSV, get_transforms(train=True))
    val_ds = ROPDataset(VAL_CSV, get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet18ROP(pretrained=True).to(DEVICE)

    pos_weight = compute_pos_weight(TRAIN_CSV).to(DEVICE)
    criterion = get_loss(pos_weight)
    optimizer = Adam(model.parameters(), lr=LR)

    best_auc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, DEVICE
        )
        val_loss, val_auc = validate(
            model, val_loader, criterion, DEVICE
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(
                model.state_dict(),
                CHECKPOINT_DIR / "best_model.pth"
            )
            print("✅ Best model saved")

    print(f"\nTraining complete. Best Val AUC = {best_auc:.4f}")


if __name__ == "__main__":
    main()
