"""
Plant Disease Classification API.

FastAPI backend that serves predictions from trained models.
"""

import sys
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image
import io

# Add project root to path so we can import ml modules
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ml.src.models.efficientnet import EfficientNetB0
from ml.src.models.cnn import CustomCNN
from ml.src.data.dataset import IMAGENET_MEAN, IMAGENET_STD

from torchvision import transforms

app = FastAPI(title="Plant Disease Classifier", version="2.0.0")


# ---------------------------------------------------------------------------
# Pydantic response schemas
# ---------------------------------------------------------------------------
class ModelMetadata(BaseModel):
    model_name: str = Field(..., description="Model identifier used for inference")
    model_version: str = Field("1.0.0", description="Semantic version of the model weights")
    architecture: str = Field(..., description="Network architecture name")
    num_classes: int = Field(38, description="Number of output classes")
    device: str = Field(..., description="Compute device used for inference")


class PredictionItem(BaseModel):
    class_index: int
    class_name: str
    plant: str
    condition: str
    healthy: bool
    confidence_percentage: float = Field(
        ...,
        description="Confidence as a percentage rounded to 2 decimal places. "
                    "Examples: 99.72, 87.34, 0.15",
    )


class DiagnosisResponse(BaseModel):
    """
    Structured diagnosis returned by /api/predict.

    Few-shot examples for confidence_percentage rounding:
      - raw 0.997234  -> 99.72
      - raw 0.873421  -> 87.34
      - raw 0.001499  ->  0.15
    """
    class_name: str = Field(..., description="Raw PlantVillage class label")
    confidence_percentage: float = Field(
        ..., description="Top-1 confidence as a percentage (0-100), rounded to 2 dp"
    )
    model_metadata: ModelMetadata
    prediction: PredictionItem
    top5: List[PredictionItem]


class TreatmentTip(BaseModel):
    tip: str
    category: str  # e.g. "organic", "chemical", "cultural"


class TreatmentResponse(BaseModel):
    class_name: str
    plant: str
    condition: str
    healthy: bool
    tips: List[TreatmentTip]

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
# Treatment tips knowledge base
# ---------------------------------------------------------------------------
TREATMENT_TIPS: dict[str, list[dict]] = {
    # ---- Apple ----
    "Apple___Apple_scab": [
        {"tip": "Apply fungicide (captan or myclobutanil) at green-tip stage.", "category": "chemical"},
        {"tip": "Rake and destroy fallen leaves to reduce overwintering spores.", "category": "cultural"},
        {"tip": "Plant scab-resistant cultivars such as Liberty or Enterprise.", "category": "cultural"},
    ],
    "Apple___Black_rot": [
        {"tip": "Prune out dead or cankered wood and mummified fruit.", "category": "cultural"},
        {"tip": "Apply captan or thiophanate-methyl from pink through 2nd cover.", "category": "chemical"},
        {"tip": "Maintain balanced fertility to avoid tree stress.", "category": "cultural"},
    ],
    "Apple___Cedar_apple_rust": [
        {"tip": "Remove nearby juniper / red cedar hosts within 1-2 miles if possible.", "category": "cultural"},
        {"tip": "Apply myclobutanil or mancozeb at early bloom.", "category": "chemical"},
        {"tip": "Choose rust-resistant cultivars (Freedom, Redfree).", "category": "cultural"},
    ],
    # ---- Cherry ----
    "Cherry_(including_sour)___Powdery_mildew": [
        {"tip": "Apply sulfur-based sprays early in the season.", "category": "organic"},
        {"tip": "Ensure good air circulation through pruning.", "category": "cultural"},
        {"tip": "Neem oil can be used as an organic alternative.", "category": "organic"},
    ],
    # ---- Corn ----
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": [
        {"tip": "Rotate crops and avoid continuous corn planting.", "category": "cultural"},
        {"tip": "Apply foliar fungicides (strobilurins) at VT-R1 stage.", "category": "chemical"},
        {"tip": "Use resistant hybrids when available.", "category": "cultural"},
    ],
    "Corn_(maize)___Common_rust_": [
        {"tip": "Plant rust-tolerant hybrids.", "category": "cultural"},
        {"tip": "Apply foliar fungicide if rust appears before tasselling.", "category": "chemical"},
        {"tip": "Scout fields weekly during warm, humid conditions.", "category": "cultural"},
    ],
    "Corn_(maize)___Northern_Leaf_Blight": [
        {"tip": "Rotate with non-host crops to reduce inoculum.", "category": "cultural"},
        {"tip": "Apply strobilurin or triazole fungicides at early tassel.", "category": "chemical"},
        {"tip": "Tillage to bury infected crop residue.", "category": "cultural"},
    ],
    # ---- Grape ----
    "Grape___Black_rot": [
        {"tip": "Remove mummified berries and infected tendrils.", "category": "cultural"},
        {"tip": "Apply mancozeb or myclobutanil from shoot growth to veraison.", "category": "chemical"},
        {"tip": "Ensure canopy management for good air flow.", "category": "cultural"},
    ],
    "Grape___Esca_(Black_Measles)": [
        {"tip": "No curative treatment exists; remove severely affected vines.", "category": "cultural"},
        {"tip": "Protect pruning wounds with wound sealant or Trichoderma-based products.", "category": "organic"},
        {"tip": "Avoid heavy pruning during wet weather.", "category": "cultural"},
    ],
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": [
        {"tip": "Apply copper-based fungicides preventively.", "category": "chemical"},
        {"tip": "Remove and destroy infected leaves.", "category": "cultural"},
        {"tip": "Improve air circulation through vine training.", "category": "cultural"},
    ],
    # ---- Orange ----
    "Orange___Haunglongbing_(Citrus_greening)": [
        {"tip": "Control Asian citrus psyllid vector with systemic insecticides.", "category": "chemical"},
        {"tip": "Remove and destroy infected trees promptly.", "category": "cultural"},
        {"tip": "Use certified disease-free nursery stock.", "category": "cultural"},
    ],
    # ---- Peach ----
    "Peach___Bacterial_spot": [
        {"tip": "Apply copper sprays at leaf fall and again at bud swell.", "category": "chemical"},
        {"tip": "Plant resistant cultivars (e.g., Contender).", "category": "cultural"},
        {"tip": "Avoid overhead irrigation that keeps foliage wet.", "category": "cultural"},
    ],
    # ---- Pepper ----
    "Pepper,_bell___Bacterial_spot": [
        {"tip": "Use certified disease-free seed and transplants.", "category": "cultural"},
        {"tip": "Apply copper-based bactericide on a 7-day schedule.", "category": "chemical"},
        {"tip": "Rotate with non-solanaceous crops for 2-3 years.", "category": "cultural"},
    ],
    # ---- Potato ----
    "Potato___Early_blight": [
        {"tip": "Apply chlorothalonil or mancozeb at first sign of symptoms.", "category": "chemical"},
        {"tip": "Rotate with non-solanaceous crops.", "category": "cultural"},
        {"tip": "Maintain adequate fertility (especially nitrogen).", "category": "cultural"},
    ],
    "Potato___Late_blight": [
        {"tip": "Apply preventive fungicide (chlorothalonil, mefenoxam) before wet weather.", "category": "chemical"},
        {"tip": "Destroy volunteer potatoes and cull piles.", "category": "cultural"},
        {"tip": "Plant late-blight resistant varieties when available.", "category": "cultural"},
    ],
    # ---- Squash ----
    "Squash___Powdery_mildew": [
        {"tip": "Apply potassium bicarbonate or neem oil at first sign of white patches.", "category": "organic"},
        {"tip": "Ensure good air circulation and avoid overcrowding.", "category": "cultural"},
        {"tip": "Plant resistant varieties (e.g., PM-resistant squash lines).", "category": "cultural"},
    ],
    # ---- Strawberry ----
    "Strawberry___Leaf_scorch": [
        {"tip": "Apply captan or thiram at bloom and post-harvest.", "category": "chemical"},
        {"tip": "Renovate beds after harvest to remove infected foliage.", "category": "cultural"},
        {"tip": "Avoid overhead irrigation in evening.", "category": "cultural"},
    ],
    # ---- Tomato ----
    "Tomato___Bacterial_spot": [
        {"tip": "Use copper hydroxide sprays on a 5-7 day schedule.", "category": "chemical"},
        {"tip": "Rotate with non-solanaceous crops for 3 years.", "category": "cultural"},
        {"tip": "Use pathogen-free seed and transplants.", "category": "cultural"},
    ],
    "Tomato___Early_blight": [
        {"tip": "Apply chlorothalonil or mancozeb at first symptoms.", "category": "chemical"},
        {"tip": "Mulch around plants to prevent soil splashing.", "category": "cultural"},
        {"tip": "Remove lower infected leaves promptly.", "category": "cultural"},
    ],
    "Tomato___Late_blight": [
        {"tip": "Apply metalaxyl or chlorothalonil preventively in wet seasons.", "category": "chemical"},
        {"tip": "Destroy infected plant material immediately.", "category": "cultural"},
        {"tip": "Use resistant varieties (e.g., Mountain Magic).", "category": "cultural"},
    ],
    "Tomato___Leaf_Mold": [
        {"tip": "Improve greenhouse ventilation and reduce humidity.", "category": "cultural"},
        {"tip": "Apply chlorothalonil or mancozeb preventively.", "category": "chemical"},
        {"tip": "Remove infected leaves and stake plants for air flow.", "category": "cultural"},
    ],
    "Tomato___Septoria_leaf_spot": [
        {"tip": "Apply chlorothalonil at first sign of spots.", "category": "chemical"},
        {"tip": "Remove lower leaves showing symptoms.", "category": "cultural"},
        {"tip": "Avoid working in fields when foliage is wet.", "category": "cultural"},
    ],
    "Tomato___Spider_mites Two-spotted_spider_mite": [
        {"tip": "Release predatory mites (Phytoseiulus persimilis).", "category": "organic"},
        {"tip": "Apply insecticidal soap or neem oil to undersides of leaves.", "category": "organic"},
        {"tip": "Increase humidity around plants to discourage mites.", "category": "cultural"},
    ],
    "Tomato___Target_Spot": [
        {"tip": "Apply chlorothalonil or azoxystrobin at disease onset.", "category": "chemical"},
        {"tip": "Promote good air flow through staking and pruning.", "category": "cultural"},
        {"tip": "Rotate crops to avoid pathogen build-up.", "category": "cultural"},
    ],
    "Tomato___Tomato_mosaic_virus": [
        {"tip": "No chemical cure; remove and destroy infected plants.", "category": "cultural"},
        {"tip": "Disinfect tools with 10% bleach between plants.", "category": "cultural"},
        {"tip": "Plant TMV-resistant varieties.", "category": "cultural"},
    ],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": [
        {"tip": "Control whitefly vector with reflective mulches or insecticidal soap.", "category": "organic"},
        {"tip": "Use TYLCV-resistant tomato cultivars.", "category": "cultural"},
        {"tip": "Remove and destroy infected plants early.", "category": "cultural"},
    ],
}

# Default tip for healthy plants or classes without specific tips
HEALTHY_TIPS: list[dict] = [
    {"tip": "Your plant looks healthy! Continue regular watering and fertilization.", "category": "cultural"},
    {"tip": "Monitor leaves periodically for early signs of disease.", "category": "cultural"},
    {"tip": "Maintain proper spacing between plants for air circulation.", "category": "cultural"},
]

GENERIC_DISEASE_TIPS: list[dict] = [
    {"tip": "Remove visibly infected leaves or fruit to slow spread.", "category": "cultural"},
    {"tip": "Avoid overhead watering to keep foliage dry.", "category": "cultural"},
    {"tip": "Consult your local agricultural extension office for region-specific guidance.", "category": "cultural"},
]

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

    # Top-5 results  —  confidence as percentage rounded to 2 dp
    # Few-shot rounding examples:
    #   raw 0.997234  -> round(0.997234 * 100, 2) = 99.72
    #   raw 0.873421  -> round(0.873421 * 100, 2) = 87.34
    #   raw 0.001499  -> round(0.001499 * 100, 2) =  0.15
    top5_probs, top5_indices = torch.topk(probs, k=5)
    predictions: list[PredictionItem] = []
    for prob, idx in zip(top5_probs.tolist(), top5_indices.tolist()):
        raw_name = CLASS_NAMES[idx]
        info = friendly_name(raw_name)
        predictions.append(PredictionItem(
            class_index=idx,
            class_name=raw_name,
            plant=info["plant"],
            condition=info["condition"],
            healthy=info["healthy"],
            confidence_percentage=round(prob * 100, 2),
        ))

    top = predictions[0]

    # Build model metadata
    arch_map = {"efficientnet": "EfficientNet-B0", "custom_cnn": "CustomCNN"}
    metadata = ModelMetadata(
        model_name=model_name,
        model_version="1.0.0",
        architecture=arch_map.get(model_name, model_name),
        num_classes=38,
        device=str(DEVICE),
    )

    return DiagnosisResponse(
        class_name=top.class_name,
        confidence_percentage=top.confidence_percentage,
        model_metadata=metadata,
        prediction=top,
        top5=predictions,
    )


# ---------------------------------------------------------------------------
# Treatment tips endpoint
# ---------------------------------------------------------------------------
@app.get("/api/treatment/{class_name}", response_model=TreatmentResponse)
async def get_treatment(class_name: str):
    """
    Return treatment tips for a diagnosed plant disease class.

    If the class represents a healthy plant, general care tips are returned.
    If disease-specific tips are unavailable, generic advice is provided.
    """
    if class_name not in CLASS_NAMES:
        raise HTTPException(status_code=404, detail=f"Unknown class: {class_name}")

    info = friendly_name(class_name)

    if info["healthy"]:
        raw_tips = HEALTHY_TIPS
    else:
        raw_tips = TREATMENT_TIPS.get(class_name, GENERIC_DISEASE_TIPS)

    tips = [TreatmentTip(**t) for t in raw_tips]
    return TreatmentResponse(
        class_name=class_name,
        plant=info["plant"],
        condition=info["condition"],
        healthy=info["healthy"],
        tips=tips,
    )


# ---------------------------------------------------------------------------
# Serve static frontend
# ---------------------------------------------------------------------------
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
