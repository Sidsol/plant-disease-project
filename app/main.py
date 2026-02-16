"""
Plant Disease Classification API.

FastAPI backend that serves predictions from trained models.
"""

import sys
from pathlib import Path

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io

# Add project root to path so we can import ml modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.models.efficientnet import EfficientNetB0
from ml.src.models.cnn import CustomCNN
from ml.src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

from torchvision import transforms

app = FastAPI(title="Plant Disease Classifier", version="1.0.0")

# ---------------------------------------------------------------------------
# Class names (38 PlantVillage classes, alphabetically sorted as ImageFolder)
# ---------------------------------------------------------------------------
CLASS_NAMES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___healthy",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
]

# Friendly display names
def friendly_name(raw: str) -> dict:
    """Convert folder name to readable plant + disease."""
    parts = raw.split("___")
    plant = parts[0].replace("_", " ")
    condition = parts[1].replace("_", " ") if len(parts) > 1 else ""
    is_healthy = condition.strip().lower() == "healthy"
    return {"plant": plant, "condition": condition, "healthy": is_healthy}

# ---------------------------------------------------------------------------
# Image preprocessing (must match training pipeline)
# ---------------------------------------------------------------------------
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

models_cache: dict = {}

def load_model(model_name: str):
    """Load a model from the exported weights."""
    if model_name in models_cache:
        return models_cache[model_name]

    models_dir = PROJECT_ROOT / "models_exported"

    if model_name == "efficientnet":
        model = EfficientNetB0(num_classes=38, pretrained=False)
        weights_path = models_dir / "efficientnet_best.pth"
    elif model_name == "custom_cnn":
        model = CustomCNN(num_classes=38)
        weights_path = models_dir / "custom_cnn_best.pth"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=DEVICE, weights_only=False)
    # Checkpoints may contain full training state or just a state_dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    models_cache[model_name] = model
    print(f"Loaded {model_name} on {DEVICE}")
    return model


# Pre-load the best model at startup
@app.on_event("startup")
async def startup():
    try:
        load_model("efficientnet")
    except Exception as e:
        print(f"Warning: Could not pre-load efficientnet model: {e}")
    try:
        load_model("custom_cnn")
    except Exception as e:
        print(f"Warning: Could not pre-load custom_cnn model: {e}")


# ---------------------------------------------------------------------------
# API routes
# ---------------------------------------------------------------------------
@app.get("/api/models")
async def list_models():
    """List available models and their metadata."""
    return {
        "models": [
            {
                "id": "efficientnet",
                "name": "EfficientNet-B0",
                "accuracy": 0.997,
                "description": "Transfer learning with EfficientNet-B0 (best model)",
            },
            {
                "id": "custom_cnn",
                "name": "Custom CNN",
                "accuracy": 0.956,
                "description": "Custom CNN trained from scratch (baseline)",
            },
        ]
    }


@app.get("/api/classes")
async def list_classes():
    """List all 38 plant disease classes."""
    return {
        "classes": [
            {"index": i, "raw": name, **friendly_name(name)}
            for i, name in enumerate(CLASS_NAMES)
        ]
    }


@app.post("/api/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = "efficientnet",
):
    """
    Classify an uploaded plant leaf image.

    Returns top-5 predictions with confidence scores.
    """
    # Validate file type
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=400, detail="File must be a JPEG, PNG, or WebP image.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")

    # Preprocess
    tensor = inference_transform(image).unsqueeze(0).to(DEVICE)

    # Inference
    try:
        model = load_model(model_name)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]

    # Top-5 results
    top5_probs, top5_indices = torch.topk(probs, k=5)
    predictions = []
    for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        raw_name = CLASS_NAMES[idx]
        info = friendly_name(raw_name)
        predictions.append({
            "class_index": idx,
            "class_name": raw_name,
            "plant": info["plant"],
            "condition": info["condition"],
            "healthy": info["healthy"],
            "confidence": round(prob, 5),
        })

    top = predictions[0]
    return {
        "model": model_name,
        "prediction": top,
        "top5": predictions,
    }


# ---------------------------------------------------------------------------
# Serve static frontend
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
