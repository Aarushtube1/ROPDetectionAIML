import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from src.data.dataset import ROPDataset
from src.data.transforms import get_transforms
from src.models.resnet import ResNet18ROP
from pathlib import Path

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_CSV = PROJECT_ROOT / "data" / "splits" / "test.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"
# --------------------------------------


@torch.no_grad()  # VERY IMPORTANT: disables training
def main():
    # 1. Load test dataset
    test_dataset = ROPDataset(
        csv_file=TEST_CSV,
        transforms=get_transforms(train=False)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # 2. Load trained model
    model = ResNet18ROP(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # 3. Run inference
    all_probs = []
    all_preds = []
    all_labels = []

    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.numpy()

        logits = model(images)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels)

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 4. Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    sensitivity = recall_score(all_labels, all_preds, pos_label=1)
    specificity = recall_score(all_labels, all_preds, pos_label=0)
    auc = roc_auc_score(all_labels, all_probs)

    cm = confusion_matrix(all_labels, all_preds)

    print("\n📊 FINAL TEST SET RESULTS")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Sensitivity : {sensitivity:.4f}")
    print(f"Specificity : {specificity:.4f}")
    print(f"ROC-AUC     : {auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    # 5. Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Test Set)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
