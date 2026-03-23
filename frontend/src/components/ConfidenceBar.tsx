interface Props {
  value: number;
}

export default function ConfidenceBar({ value }: Props) {
  const pct = value.toFixed(2);

  return (
    <div className="confidence-card">
      <div className="confidence-row">
        <span className="confidence-label">Model Confidence</span>
        <span className="confidence-value">{pct}%</span>
      </div>
      <div className="confidence-track">
        <div className="confidence-fill" style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
