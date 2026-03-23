import type { TreatmentTip } from "../types";

interface Props {
  tips: TreatmentTip[];
  healthy: boolean;
}

const ICON_MAP: Record<string, string> = {
  chemical: "biotech",
  organic: "compost",
  cultural: "grid_view",
};

export default function TreatmentTips({ tips, healthy }: Props) {
  if (healthy) {
    return (
      <div className="treatment-section">
        <div className="treatment-header">
          <h3 className="treatment-heading">Plant Care Tips</h3>
        </div>
        <div className="treatment-list">
          {tips.map((t, i) => (
            <div key={i} className="treatment-item cultural">
              <div className="treatment-icon cultural">
                <span className="material-symbols-outlined">spa</span>
              </div>
              <div>
                <h4>Care Tip</h4>
                <p>{t.tip}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // Group tips by category
  const grouped: Record<string, TreatmentTip[]> = {};
  for (const t of tips) {
    (grouped[t.category] ??= []).push(t);
  }

  return (
    <div className="treatment-section">
      <div className="treatment-header">
        <h3 className="treatment-heading">Treatment Protocols</h3>
        <span className="expert-badge">Expert Verified</span>
      </div>
      <div className="treatment-list">
        {Object.entries(grouped).map(([category, catTips]) => (
          <div key={category} className={`treatment-item ${category}`}>
            <div className={`treatment-icon ${category}`}>
              <span className="material-symbols-outlined">
                {ICON_MAP[category] ?? "spa"}
              </span>
            </div>
            <div>
              <h4>{category.charAt(0).toUpperCase() + category.slice(1)}</h4>
              {catTips.map((t, i) => (
                <p key={i}>{t.tip}</p>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
