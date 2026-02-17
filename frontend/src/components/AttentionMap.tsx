import { useState } from "react";

interface Props {
  originalPreview: string | null;
  attentionMapBase64: string | null;
}

/**
 * XAI Attention Map: shows the Grad-CAM heatmap overlay so users can see
 * WHERE the AI looked on the leaf. Toggle between original and heatmap.
 */
export default function AttentionMap({
  originalPreview,
  attentionMapBase64,
}: Props) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  if (!attentionMapBase64) return null;

  const heatmapSrc = `data:image/jpeg;base64,${attentionMapBase64}`;

  return (
    <div className="attention-map-card">
      <div className="attention-header">
        <h3>&#128269; AI Attention Map (XAI)</h3>
        <button
          className="toggle-btn"
          onClick={() => setShowHeatmap((v) => !v)}
          aria-label="Toggle heatmap overlay"
        >
          {showHeatmap ? "Show Original" : "Show Heatmap"}
        </button>
      </div>
      <p className="attention-explainer">
        The heatmap highlights the leaf regions the model focused on.
        <strong> Warm colors (red/yellow)</strong> indicate high attention areas
        that most influenced the prediction.
      </p>
      <div className="attention-image-wrapper">
        <img
          src={showHeatmap ? heatmapSrc : originalPreview ?? heatmapSrc}
          alt={showHeatmap ? "Grad-CAM heatmap overlay" : "Original leaf image"}
          className="attention-image"
        />
        <span className="attention-badge">
          {showHeatmap ? "HEATMAP" : "ORIGINAL"}
        </span>
      </div>
    </div>
  );
}
