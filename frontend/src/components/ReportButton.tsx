import { useState } from "react";
import { reportIncorrect } from "../api/client";
import type { DiagnosisResponse } from "../types";

interface Props {
  diagnosis: DiagnosisResponse;
  classNames: string[];
}

export default function ReportButton({ diagnosis, classNames }: Props) {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState("");
  const [correction, setCorrection] = useState("");
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setLoading(true);
    setError(null);
    try {
      await reportIncorrect({
        scan_id: diagnosis.scan_id,
        reason: reason || undefined,
        user_correction: correction || undefined,
      });
      setSubmitted(true);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Failed to submit report");
    } finally {
      setLoading(false);
    }
  };

  if (submitted) {
    return (
      <div className="report-section success">
        <p>Thank you! Your feedback will help improve our model.</p>
      </div>
    );
  }

  return (
    <div className="report-section">
      {!open ? (
        <div style={{ textAlign: "center" }}>
          <button className="btn-report" onClick={() => setOpen(true)}>
            <span className="material-symbols-outlined" style={{ fontSize: "1rem", verticalAlign: "middle", marginRight: "0.35rem" }}>flag</span>
            Report Incorrect
          </button>
        </div>
      ) : (
        <div className="report-form">
          <h4>Report Incorrect Prediction</h4>
          <p className="report-help">
            Your feedback helps us identify mistakes and improve the model through
            human-in-the-loop retraining.
          </p>
          <label>
            Reason (optional)
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="e.g. The leaf shows bacterial spot, not early blight"
              rows={2}
            />
          </label>
          <label>
            Correct diagnosis (optional)
            <input
              type="text"
              value={correction}
              onChange={(e) => setCorrection(e.target.value)}
              placeholder="e.g. Tomato___Bacterial_spot"
              list="class-suggestions"
            />
            <datalist id="class-suggestions">
              {classNames.map((c) => <option key={c} value={c} />)}
            </datalist>
          </label>
          {error && <p className="report-error">{error}</p>}
          <div className="report-actions">
            <button className="btn-primary" onClick={handleSubmit} disabled={loading}>
              {loading ? "Submitting\u2026" : "Submit Report"}
            </button>
            <button className="btn-secondary" onClick={() => setOpen(false)}>Cancel</button>
          </div>
        </div>
      )}
    </div>
  );
}
