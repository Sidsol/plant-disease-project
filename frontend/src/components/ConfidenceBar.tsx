interface Props {
  value: number; // 0-100
}

export default function ConfidenceBar({ value }: Props) {
  const pct = value.toFixed(2);
  const level = value >= 80 ? "high" : value >= 50 ? "medium" : "low";

  return (
    <div className="confidence-bar-wrapper">
      <div className="confidence-label">
        <span>Confidence</span>
        <span>{pct}%</span>
      </div>
      <div className="confidence-track">
        <div
          className={`confidence-fill ${level}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
