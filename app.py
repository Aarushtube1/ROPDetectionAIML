"""
ROP Detection Flask Application
Dark Theme Frontend for Eye Photo Analysis
"""

import os
import io
import uuid
import base64
import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from flask import Flask, render_template, request, jsonify
from torchvision import transforms

# Import project modules
import sys
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet import ResNet18ROP
from src.data.transforms import CLAHETransform

# ============== CONFIG ==============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
UPLOAD_FOLDER = PROJECT_ROOT / "frontend" / "static" / "uploads"
RESULTS_FOLDER = PROJECT_ROOT / "frontend" / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)

# ============== FLASK APP ==============
app = Flask(__name__, 
            template_folder=str(PROJECT_ROOT / "frontend" / "templates"),
            static_folder=str(PROJECT_ROOT / "frontend" / "static"))
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ============== GRAD-CAM ==============
class GradCAM:
    def __init__(self, model, target_layer, layer_name="layer"):
        self.model = model
        self.target_layer = target_layer
        self.layer_name = layer_name
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(_, __, output):
            self.activations = output.detach()

        def backward_hook(_, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor):
        self.model.eval()
        
        # Enable gradients for this inference
        input_tensor.requires_grad_(True)
        
        logits = self.model(input_tensor)
        score = logits[:, 0]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = torch.relu(cam)

        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        return cam


class MultiLayerGradCAM:
    """GradCAM for multiple layers simultaneously"""
    def __init__(self, model, target_layers_dict):
        self.model = model
        self.gradcams = {}
        for layer_name, layer in target_layers_dict.items():
            self.gradcams[layer_name] = GradCAM(model, layer, layer_name)
    
    def generate_all(self, input_tensor):
        """Generate GradCAM heatmaps for all layers"""
        results = {}
        self.model.eval()
        
        # Enable gradients
        input_tensor = input_tensor.clone().requires_grad_(True)
        
        # Forward pass
        logits = self.model(input_tensor)
        score = logits[:, 0]
        
        # Backward pass
        self.model.zero_grad()
        score.backward()
        
        # Generate CAMs for each layer
        for layer_name, gradcam in self.gradcams.items():
            if gradcam.gradients is not None and gradcam.activations is not None:
                weights = gradcam.gradients.mean(dim=(2, 3), keepdim=True)
                cam = (weights * gradcam.activations).sum(dim=1)
                cam = torch.relu(cam)
                cam -= cam.min()
                cam /= (cam.max() + 1e-8)
                results[layer_name] = cam
        
        return results


# ============== MODEL LOADING ==============
model = None
multi_gradcam = None

def load_model():
    global model, multi_gradcam
    if model is None:
        print(f"Loading model from {CHECKPOINT_PATH}...")
        model = ResNet18ROP(pretrained=False).to(DEVICE)
        if CHECKPOINT_PATH.exists():
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            print("Model loaded successfully!")
        else:
            print(f"Warning: Checkpoint not found at {CHECKPOINT_PATH}. Using random weights.")
        model.eval()
        
        # Setup GradCAM for all ResNet layers
        target_layers = {
            'layer1': model.backbone.layer1,
            'layer2': model.backbone.layer2,
            'layer3': model.backbone.layer3,
            'layer4': model.backbone.layer4
        }
        multi_gradcam = MultiLayerGradCAM(model, target_layers)
    return model, multi_gradcam


# ============== IMAGE PROCESSING HELPERS ==============
def numpy_to_base64(img_array, format='PNG'):
    """Convert numpy array to base64 string for display in HTML"""
    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_pil = Image.fromarray(img_array.astype(np.uint8))
    else:
        img_pil = Image.fromarray(img_array.astype(np.uint8), mode='L')
    
    buffer = io.BytesIO()
    img_pil.save(buffer, format=format)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def process_image_steps(image_path):
    """
    Process image through each step and return intermediate results
    """
    results = {}
    
    # Step 1: Load original image
    original_img = Image.open(image_path).convert("RGB")
    original_array = np.array(original_img)
    results['original'] = {
        'image': numpy_to_base64(original_array),
        'title': 'Original Image',
        'description': f'Loaded RGB image ({original_array.shape[1]}x{original_array.shape[0]})'
    }
    
    # Step 2: Resize to 224x224
    resize_transform = transforms.Resize((224, 224))
    resized_img = resize_transform(original_img)
    resized_array = np.array(resized_img)
    results['resized'] = {
        'image': numpy_to_base64(resized_array),
        'title': 'Resized (224x224)',
        'description': 'Resized to standard input size for ResNet18'
    }
    
    # Step 3: Extract Green Channel
    green_channel = resized_array[:, :, 1]
    green_rgb = np.stack([green_channel, green_channel, green_channel], axis=2)
    results['green_channel'] = {
        'image': numpy_to_base64(green_rgb),
        'title': 'Green Channel Extraction',
        'description': 'Green channel contains most retinal vessel information'
    }
    
    # Step 4: Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(green_channel)
    clahe_rgb = np.stack([clahe_enhanced, clahe_enhanced, clahe_enhanced], axis=2)
    results['clahe'] = {
        'image': numpy_to_base64(clahe_rgb),
        'title': 'CLAHE Enhanced',
        'description': 'Contrast Limited Adaptive Histogram Equalization for better feature visibility'
    }
    
    # Step 5: Normalize (show as visualization - shift to visible range)
    # Create the final tensor
    tensor_img = transforms.ToTensor()(clahe_rgb)
    normalized = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )(tensor_img)
    
    # For visualization, denormalize to show
    denorm_img = normalized.numpy().transpose(1, 2, 0)
    denorm_img = (denorm_img - denorm_img.min()) / (denorm_img.max() - denorm_img.min()) * 255
    results['normalized'] = {
        'image': numpy_to_base64(denorm_img.astype(np.uint8)),
        'title': 'Normalized (Model Input)',
        'description': 'ImageNet normalization applied - ready for model inference'
    }
    
    return results, normalized.unsqueeze(0)


def generate_gradcam_overlay(original_path, input_tensor, model, multi_gradcam):
    """Generate GradCAM heatmap overlays for all layers"""
    input_tensor = input_tensor.to(DEVICE)
    
    # Load original image for overlay
    orig_img = cv2.imread(str(original_path))
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    orig_img = cv2.resize(orig_img, (224, 224))
    
    # Generate CAMs for all layers
    with torch.enable_grad():
        cams = multi_gradcam.generate_all(input_tensor)
    
    layer_results = {}
    layer_info = {
        'layer1': {'name': 'Layer 1 (Early Features)', 'description': 'Detects basic edges and textures'},
        'layer2': {'name': 'Layer 2 (Low-Level)', 'description': 'Captures simple patterns and shapes'},
        'layer3': {'name': 'Layer 3 (Mid-Level)', 'description': 'Recognizes intermediate structures'},
        'layer4': {'name': 'Layer 4 (High-Level)', 'description': 'Identifies complex features for classification'}
    }
    
    for layer_name, cam in cams.items():
        heatmap = cam[0].cpu().numpy()
        
        # Create heatmap overlay
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlay = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
        
        layer_results[layer_name] = {
            'overlay': numpy_to_base64(overlay),
            'heatmap': numpy_to_base64(heatmap_colored),
            'name': layer_info[layer_name]['name'],
            'description': layer_info[layer_name]['description']
        }
    
    return layer_results


def predict(input_tensor, model):
    """Run model prediction"""
    input_tensor = input_tensor.to(DEVICE)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probability = torch.sigmoid(logits).item()
        prediction = 1 if probability >= 0.5 else 0
    
    return {
        'prediction': prediction,
        'probability': probability,
        'label': 'ROP Positive' if prediction == 1 else 'ROP Negative',
        'confidence': probability if prediction == 1 else (1 - probability)
    }


# ============== ROUTES ==============
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check file extension
    allowed_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({'error': f'Invalid file type. Allowed: {allowed_extensions}'}), 400
    
    try:
        # Save uploaded file
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{unique_id}_{file.filename}"
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Load model
        model, multi_gradcam_obj = load_model()
        
        # Process image through each step
        processing_steps, input_tensor = process_image_steps(filepath)
        
        # Run prediction
        prediction_result = predict(input_tensor, model)
        
        # Generate GradCAM for all layers
        gradcam_layers = generate_gradcam_overlay(
            filepath, input_tensor, model, multi_gradcam_obj
        )
        
        return jsonify({
            'success': True,
            'processing_steps': processing_steps,
            'prediction': prediction_result,
            'gradcam_layers': gradcam_layers
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'device': DEVICE,
        'model_loaded': model is not None,
        'checkpoint_exists': CHECKPOINT_PATH.exists()
    })


if __name__ == '__main__':
    print(f"Starting ROP Detection Server...")
    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    
    # Pre-load model
    load_model()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
