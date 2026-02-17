import { useState, useCallback } from "react";
import { classify, fetchTreatment } from "./api/client";
import type { DiagnosisResponse, TreatmentResponse } from "./types";
import UploadArea from "./components/UploadArea";
import ConfidenceBar from "./components/ConfidenceBar";
import AttentionMap from "./components/AttentionMap";
import TreatmentTips from "./components/TreatmentTips";
import TopPredictions from "./components/TopPredictions";
import ReportButton from "./components/ReportButton";
import HistoryPanel from "./components/HistoryPanel";
import ModelMetadataCard from "./components/ModelMetadataCard";
import "./App.css";

// 38 PlantVillage classes for the report autocomplete
const CLASS_NAMES = [
  "Apple___Apple_scab","Apple___Black_rot","Apple___Cedar_apple_rust",
  "Apple___healthy","Blueberry___healthy","Cherry_(including_sour)___healthy",
  "Cherry_(including_sour)___Powdery_mildew",
  "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
  "Corn_(maize)___Common_rust_","Corn_(maize)___healthy",
  "Corn_(maize)___Northern_Leaf_Blight","Grape___Black_rot",
  "Grape___Esca_(Black_Measles)","Grape___healthy",
  "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
  "Orange___Haunglongbing_(Citrus_greening)","Peach___Bacterial_spot",
  "Peach___healthy","Pepper,_bell___Bacterial_spot","Pepper,_bell___healthy",
  "Potato___Early_blight","Potato___healthy","Potato___Late_blight",
  "Raspberry___healthy","Soybean___healthy","Squash___Powdery_mildew",
  "Strawberry___healthy","Strawberry___Leaf_scorch","Tomato___Bacterial_spot",
  "Tomato___Early_blight","Tomato___healthy","Tomato___Late_blight",
  "Tomato___Leaf_Mold","Tomato___Septoria_leaf_spot",
  "Tomato___Spider_mites Two-spotted_spider_mite","Tomato___Target_Spot",
  "Tomato___Tomato_mosaic_virus","Tomato___Tomato_Yellow_Leaf_Curl_Virus",
];

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [model, setModel] = useState("efficientnet");
  const [loading, setLoading] = useState(false);
  const [diagnosis, setDiagnosis] = useState<DiagnosisResponse | null>(null);
  const [treatment, setTreatment] = useState<TreatmentResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [historyTrigger, setHistoryTrigger] = useState(0);

  const onFileSelect = useCallback((f: File) => {
    setFile(f);
    setDiagnosis(null);
    setTreatment(null);
    setError(null);
    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target?.result as string);
    reader.readAsDataURL(f);
  }, []);

  const onClear = useCallback(() => {
    setFile(null);
    setPreview(null);
    setDiagnosis(null);
    setTreatment(null);
    setError(null);
  }, []);

  const handleClassify = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setDiagnosis(null);
    setTreatment(null);
    try {
      const result = await classify(file, model);
      setDiagnosis(result);
      setHistoryTrigger((n) => n + 1);

      // Fetch treatment tips in parallel
      fetchTreatment(result.class_name)
        .then(setTreatment)
        .catch(() => {});
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Classification failed");
    } finally {
      setLoading(false);
    }
  };

  const top = diagnosis?.prediction;

  return (
    <div className="container">
      {/* Header */}
      <header>
        <div className="logo">{"\u{1F331}"}</div>
        <h1>Plant Disease Classifier</h1>
        <p className="subtitle">
          Upload a photo of a plant leaf and our AI will identify the disease
        </p>
      </header>

      {/* Model Selector */}
      <div className="model-selector">
        <label htmlFor="model-select">Model:</label>
        <select
          id="model-select"
          value={model}
          onChange={(e) => setModel(e.target.value)}
        >
          <option value="efficientnet">EfficientNet-B0 (99.7% acc)</option>
          <option value="custom_cnn">Custom CNN (95.6% acc)</option>
        </select>
      </div>

      {/* Upload */}
      <UploadArea
        onFileSelect={onFileSelect}
        preview={preview}
        onClear={onClear}
      />

      {/* Action Buttons */}
      <div className="actions">
        <button
          className="btn btn-primary"
          disabled={!file || loading}
          onClick={handleClassify}
        >
          {loading ? "Analyzing\u2026" : "Classify"}
        </button>
        {file && (
          <button className="btn btn-secondary" onClick={onClear}>
            Clear
          </button>
        )}
      </div>

      {/* Loading */}
      {loading && (
        <div className="loading">
          <div className="spinner" />
          <p>Analyzing image&hellip;</p>
        </div>
      )}

      {/* Error */}
      {error && <div className="error-banner" role="alert">{error}</div>}

      {/* Results */}
      {diagnosis && top && (
        <div className="results">
          {/* Diagnosis Card */}
          <div className="result-card">
            <div className="plant-name">{top.plant}</div>
            {top.healthy ? (
              <span className="condition healthy">{"\u2714"} Healthy</span>
            ) : (
              <span className="condition disease">{"\u26A0"} {top.condition}</span>
            )}
          </div>

          {/* Confidence Progress Bar */}
          <ConfidenceBar value={diagnosis.confidence_percentage} />

          {/* Model Metadata */}
          <ModelMetadataCard metadata={diagnosis.model_metadata} />

          {/* XAI Attention Map */}
          <AttentionMap
            originalPreview={preview}
            attentionMapBase64={diagnosis.attention_map}
          />

          {/* Treatment Tips */}
          {treatment && (
            <TreatmentTips tips={treatment.tips} healthy={treatment.healthy} />
          )}

          {/* Top 5 */}
          <TopPredictions top5={diagnosis.top5} />

          {/* Human-in-the-Loop: Report */}
          <ReportButton diagnosis={diagnosis} classNames={CLASS_NAMES} />
        </div>
      )}

      {/* History */}
      <HistoryPanel refreshTrigger={historyTrigger} />

      {/* Footer */}
      <footer>
        <p>
          Powered by PyTorch &middot; Trained on PlantVillage Dataset &middot;
          38 Classes &middot; Explainable AI with Grad-CAM
        </p>
      </footer>
    </div>
  );
}
