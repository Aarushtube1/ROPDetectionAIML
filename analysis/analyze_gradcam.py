import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import pandas as pd

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet import ResNet18ROP
from src.data.transforms import get_transforms


# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
print("PROJECT_ROOT:", PROJECT_ROOT)
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
CSV_PATH = PROJECT_ROOT / "data" / "splits" / "val.csv"

OUTPUT_DIR = PROJECT_ROOT / "gradcam_results"
OUTPUT_DIR.mkdir(exist_ok=True)
# --------------------------------------


# ----------- Grad-CAM class ------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.eval()

        logits = self.model(input_tensor)
        score = logits[:, 0]

        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam
# --------------------------------------


def overlay_heatmap(image, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)


def main():
    # Load model
    model = ResNet18ROP(pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    gradcam = GradCAM(model, model.backbone.layer4)

    # Same transforms as validation
    transform = get_transforms(train=False)

    df = pd.read_csv(CSV_PATH)

    # Run on first N images only
    N = 10
    df = df.sample(N, random_state=42)

    for idx, row in df.iterrows():
        img_path = row["Source"]
        label = row["ROP Label"]

        # Load original image for visualization
        orig_img = cv2.imread(img_path)
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        # Prepare input tensor
        pil_img = Image.open(img_path).convert("RGB")
        input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

        # Generate Grad-CAM
        cam = gradcam.generate(input_tensor)
        heatmap = cam[0].cpu().numpy()

        overlay = overlay_heatmap(orig_img, heatmap)

        out_name = f"img_{idx}_label_{label}.png"
        out_path = OUTPUT_DIR / out_name

        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
