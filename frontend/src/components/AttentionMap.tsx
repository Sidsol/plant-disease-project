import { useState } from "react";

interface Props {
  originalPreview: string | null;
  attentionMapBase64: string | null;
}

export default function AttentionMap({ originalPreview, attentionMapBase64 }: Props) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  if (!attentionMapBase64) return null;

  const heatmapSrc = `data:image/jpeg;base64,${attentionMapBase64}`;

  return (
    <div className="result-image-section">
      <div className="result-image-wrapper">
        <img
          src={showHeatmap ? heatmapSrc : originalPreview ?? heatmapSrc}
          alt={showHeatmap ? "Grad-CAM heatmap overlay" : "Original leaf image"}
          className="result-image"
        />
      </div>
      <button
        className="xai-toggle"
        onClick={() => setShowHeatmap((v) => !v)}
        aria-label="Toggle attention map"
      >
        <span className="material-symbols-outlined">visibility</span>
        <span>{showHeatmap ? "Show Original" : "XAI Attention Map"}</span>
      </button>
    </div>
  );
}
