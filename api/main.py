"""FastAPI backend for plant disease diagnosis and treatment tips."""

from __future__ import annotations

import io
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = PROJECT_ROOT / "ml"
FRONTEND_ROOT = PROJECT_ROOT / "frontend"
if str(ML_ROOT) not in sys.path:
    sys.path.append(str(ML_ROOT))

from src.models.cnn import CustomCNN
from src.models.efficientnet import EfficientNetB0


class ModelMetadata(BaseModel):
    model_name: str
    checkpoint: str
    input_size: str = "224x224"
    classes_supported: int


class DiagnosisResponse(BaseModel):
    class_name: str
    confidence_percentage: float = Field(description="Confidence as percentage rounded to 2 decimals")
    model_metadata: ModelMetadata
    top_predictions: List[dict]


class TreatmentTipsResponse(BaseModel):
    diagnosis: str
    treatment_tips: List[str]


CLASS_NAMES: List[str] = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy",
]

CHECKPOINTS = {
    "efficientnet": PROJECT_ROOT / "models_exported/efficientnet_best.pth",
    "custom_cnn": PROJECT_ROOT / "models_exported/custom_cnn_best.pth",
}

TREATMENT_TIPS: Dict[str, List[str]] = {
    "blight": ["Remove infected leaves promptly.", "Use copper-based fungicide weekly.", "Avoid overhead watering."],
    "rust": ["Increase plant spacing for airflow.", "Apply sulfur-based spray.", "Clean plant debris after harvest."],
    "mildew": ["Prune dense canopy areas.", "Apply potassium bicarbonate spray.", "Water early to reduce humidity."],
    "spot": ["Remove severely affected foliage.", "Disinfect tools between plants.", "Rotate crops next season."],
    "virus": ["Isolate infected plants.", "Control aphids/whiteflies.", "Use certified disease-free seedlings."],
    "healthy": ["Maintain balanced fertilization.", "Continue weekly leaf inspection.", "Keep irrigation consistent."]
}

app = FastAPI(title="Plant Disease API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@lru_cache(maxsize=2)
def load_model(model_key: str):
    if model_key not in CHECKPOINTS:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{model_key}'.")

    if model_key == "efficientnet":
        model = EfficientNetB0(num_classes=len(CLASS_NAMES), pretrained=False)
    else:
        model = CustomCNN(num_classes=len(CLASS_NAMES), dropout=0.5)

    checkpoint_path = CHECKPOINTS[model_key]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)


def to_percentage(probability: float) -> float:
    """Round confidence to 2 decimals (few-shot style examples: 0.93456->93.46, 0.5->50.0)."""
    return round(probability * 100.0, 2)


def friendly_label(raw_label: str) -> str:
    return raw_label.replace("___", " → ").replace("_", " ")


@app.post("/predict", response_model=DiagnosisResponse)
def predict(image: UploadFile = File(...), model: str = Form("efficientnet")):
    image_bytes = image.file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")

    model_instance = load_model(model)
    tensor = preprocess(image_bytes)
    with torch.inference_mode():
        logits = model_instance(tensor)
        probs = torch.softmax(logits, dim=1)

    top_probs, top_indices = probs.topk(3, dim=1)
    top_predictions = [
        {
            "class_name": friendly_label(CLASS_NAMES[idx]),
            "confidence_percentage": to_percentage(float(prob)),
        }
        for idx, prob in zip(top_indices[0].tolist(), top_probs[0].tolist())
    ]

    best = top_predictions[0]
    return DiagnosisResponse(
        class_name=best["class_name"],
        confidence_percentage=best["confidence_percentage"],
        model_metadata=ModelMetadata(
            model_name="EfficientNet-B0" if model == "efficientnet" else "Custom CNN",
            checkpoint=str(CHECKPOINTS[model].name),
            classes_supported=len(CLASS_NAMES),
        ),
        top_predictions=top_predictions,
    )


@app.get("/treatment-tips", response_model=TreatmentTipsResponse)
def get_treatment_tips(diagnosis: str = Query(..., min_length=3)):
    diagnosis_lower = diagnosis.lower()
    for key, tips in TREATMENT_TIPS.items():
        if key in diagnosis_lower:
            return TreatmentTipsResponse(diagnosis=diagnosis, treatment_tips=tips)

    return TreatmentTipsResponse(
        diagnosis=diagnosis,
        treatment_tips=[
            "Consult a local agronomist for targeted diagnosis.",
            "Capture a clearer close-up image and rerun prediction.",
            "Track plant symptoms for 3-5 days before treatment.",
        ],
    )


if FRONTEND_ROOT.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_ROOT, html=True), name="frontend")
