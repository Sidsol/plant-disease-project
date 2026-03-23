import { useState, useCallback } from "react";
import { classify, fetchTreatment } from "./api/client";
import type { DiagnosisResponse, TreatmentResponse } from "./types";
import UploadArea from "./components/UploadArea";
import ConfidenceBar from "./components/ConfidenceBar";
import AttentionMap from "./components/AttentionMap";
import TopPredictions from "./components/TopPredictions";
import ReportButton from "./components/ReportButton";
import HistoryPanel from "./components/HistoryPanel";
import ModelMetadataCard from "./components/ModelMetadataCard";
import ChatPanel from "./components/ChatPanel";
import "./App.css";

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

type Tab = "home" | "history" | "assistant";

export default function App() {
  const [tab, setTab] = useState<Tab>("home");
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

  // Navigate to assistant tab with diagnosis context
  const goToAssistant = () => setTab("assistant");

  return (
    <div className="app">
      {/* ---- Top App Bar ---- */}
      <header className="top-bar">
        <div className="top-bar-brand">
          <span className="material-symbols-outlined">eco</span>
          <span className="top-bar-title">Digital Herbarium</span>
        </div>
      </header>

      {/* ---- Page Content ---- */}
      {tab === "home" && (
        <main className={`page-content ${!diagnosis ? "hero-bg" : ""}`}>
          {!diagnosis ? (
            /* ======== HOME: Upload State ======== */
            <>
              <section className="hero-section">
                <h2 className="hero-title">
                  Identify plant diseases{" "}
                  <span className="accent">in seconds.</span>
                </h2>
                <p className="hero-subtitle">
                  Protect your garden with professional-grade AI analysis. Just
                  snap a photo and let our Digital Herbarium diagnose your
                  botanical concerns.
                </p>
              </section>

              <UploadArea onFileSelect={onFileSelect} preview={preview} onClear={onClear} />

              {/* Model selector */}
              <div className="model-selector">
                <label htmlFor="model-select">Model</label>
                <select
                  id="model-select"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                >
                  <option value="efficientnet">EfficientNet-B0 (99.7%)</option>
                  <option value="custom_cnn">Custom CNN (95.6%)</option>
                </select>
              </div>

              {/* Action buttons */}
              {file && (
                <div style={{ maxWidth: "18rem", margin: "0 auto 2rem" }}>
                  <button
                    className="btn-primary"
                    disabled={loading}
                    onClick={handleClassify}
                  >
                    <span className="material-symbols-outlined">scan</span>
                    {loading ? "Analyzing\u2026" : "Scan Leaf"}
                  </button>
                </div>
              )}

              {loading && (
                <div className="loading-state">
                  <div className="spinner" />
                  <p>Analyzing image&hellip;</p>
                </div>
              )}

              {error && <div className="error-banner" role="alert">{error}</div>}

              {/* Capture tips — shown only when no image */}
              {!file && (
                <section>
                  <div className="tips-header">
                    <h3 className="tips-title">Capture Tips</h3>
                    <span className="tips-badge">Best Results</span>
                  </div>
                  <div className="tips-grid">
                    <div className="tip-card">
                      <div className="tip-icon light">
                        <span className="material-symbols-outlined">light_mode</span>
                      </div>
                      <div>
                        <h4>Good lighting</h4>
                        <p>Use natural daylight, avoiding harsh direct sun or deep shadows.</p>
                      </div>
                    </div>
                    <div className="tip-card">
                      <div className="tip-icon light">
                        <span className="material-symbols-outlined">center_focus_strong</span>
                      </div>
                      <div>
                        <h4>Focus on leaf</h4>
                        <p>Keep the infected area in the center of the frame and tap to focus.</p>
                      </div>
                    </div>
                    <div className="tip-card accent">
                      <div className="tip-icon dark">
                        <span className="material-symbols-outlined">check_circle</span>
                      </div>
                      <div>
                        <h4>Single leaf</h4>
                        <p>Try to isolate one leaf against a neutral background for 99% accuracy.</p>
                      </div>
                    </div>
                  </div>
                </section>
              )}
            </>
          ) : (
            /* ======== HOME: Diagnosis Result ======== */
            <>
              {/* Image + XAI */}
              <AttentionMap originalPreview={preview} attentionMapBase64={diagnosis.attention_map} />

              {/* Diagnosis header */}
              <div className="diagnosis-header">
                <p className="diagnosis-label">Analysis Result</p>
                <h2 className="diagnosis-title">
                  {top?.plant} &mdash; {top?.healthy ? "Healthy" : top?.condition}
                </h2>
              </div>

              {/* Confidence */}
              <ConfidenceBar value={diagnosis.confidence_percentage} />

              {/* Model metadata */}
              <ModelMetadataCard metadata={diagnosis.model_metadata} />

              {/* Insights */}
              <div className="insights-grid">
                <div className="insight-card evidence">
                  <div className="insight-header">
                    <span className="material-symbols-outlined">neurology</span>
                    <h3>Visual Evidence</h3>
                  </div>
                  <p>
                    The attention map highlights the key features the model used
                    for the &ldquo;{top?.condition}&rdquo; diagnosis. Toggle the
                    XAI overlay above to see exactly where the model focused.
                  </p>
                </div>
                {!top?.healthy && (
                  <div className="insight-card priority">
                    <div className="insight-header">
                      <span className="material-symbols-outlined">info</span>
                      <h3>Next Priority</h3>
                    </div>
                    <p>
                      Review the treatment protocols and consult our botanical AI
                      assistant for personalized management advice.
                    </p>
                  </div>
                )}
              </div>

              {/* View Treatment CTA */}
              {!top?.healthy && treatment && (
                <button className="treatment-cta" onClick={goToAssistant}>
                  <div className="treatment-cta-text">
                    <span className="main">View Treatment &amp; Chat</span>
                    <span className="sub">Connect with our Botanical AI</span>
                  </div>
                  <span className="material-symbols-outlined">arrow_forward</span>
                </button>
              )}

              {/* Top 5 predictions */}
              <TopPredictions top5={diagnosis.top5} />

              {/* Report */}
              <ReportButton diagnosis={diagnosis} classNames={CLASS_NAMES} />

              {/* Clear / scan again */}
              <div style={{ textAlign: "center", marginTop: "1rem" }}>
                <button className="btn-secondary" onClick={onClear} style={{ maxWidth: "14rem" }}>
                  <span className="material-symbols-outlined">refresh</span>
                  Scan Another Leaf
                </button>
              </div>
            </>
          )}
        </main>
      )}

      {tab === "history" && (
        <main className="page-content">
          <HistoryPanel refreshTrigger={historyTrigger} />
        </main>
      )}

      {tab === "assistant" && (
        <main className="page-content">
          <ChatPanel
            diagnosis={diagnosis}
            treatment={treatment}
          />
        </main>
      )}

      {/* ---- Bottom Navigation ---- */}
      <nav className="bottom-nav">
        <button
          className={`nav-item ${tab === "home" ? "active" : ""}`}
          onClick={() => setTab("home")}
        >
          <span
            className="material-symbols-outlined"
            style={tab === "home" ? { fontVariationSettings: "'FILL' 1" } : undefined}
          >
            home_max
          </span>
          <span className="nav-label">Home</span>
        </button>
        <button
          className={`nav-item ${tab === "history" ? "active" : ""}`}
          onClick={() => setTab("history")}
        >
          <span
            className="material-symbols-outlined"
            style={tab === "history" ? { fontVariationSettings: "'FILL' 1" } : undefined}
          >
            potted_plant
          </span>
          <span className="nav-label">History</span>
        </button>
        <button
          className={`nav-item ${tab === "assistant" ? "active" : ""}`}
          onClick={() => setTab("assistant")}
        >
          <span
            className="material-symbols-outlined"
            style={tab === "assistant" ? { fontVariationSettings: "'FILL' 1" } : undefined}
          >
            forum
          </span>
          <span className="nav-label">Assistant</span>
        </button>
      </nav>
    </div>
  );
}
