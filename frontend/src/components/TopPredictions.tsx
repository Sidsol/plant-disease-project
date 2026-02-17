import type { PredictionItem } from "../types";

interface Props {
  top5: PredictionItem[];
}

export default function TopPredictions({ top5 }: Props) {
  const maxConf = top5[0]?.confidence_percentage ?? 1;

  return (
    <div className="top-predictions">
      <h3>Top 5 Predictions</h3>
      <div className="bar-chart">
        {top5.map((p, i) => {
          const label = `${p.plant} \u2013 ${p.condition}`;
          const widthPct = (p.confidence_percentage / maxConf) * 100;
          return (
            <div className="bar-row" key={p.class_index}>
              <div className="bar-label" title={label}>{label}</div>
              <div className="bar-track">
                <div
                  className={`bar-fill ${i === 0 ? "top" : ""}`}
                  style={{ width: `${widthPct}%` }}
                />
              </div>
              <div className="bar-value">
                {p.confidence_percentage.toFixed(2)}%
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
