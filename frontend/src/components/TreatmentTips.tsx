import type { TreatmentTip } from "../types";

interface Props {
  tips: TreatmentTip[];
  healthy: boolean;
}

const BADGE_CLASS: Record<string, string> = {
  chemical: "tip-badge chemical",
  cultural: "tip-badge cultural",
  organic: "tip-badge organic",
};

export default function TreatmentTips({ tips, healthy }: Props) {
  return (
    <div className="treatment-card">
      <h3>{healthy ? "\u{1F33F} Plant Care Tips" : "\u{1F48A} Treatment Tips"}</h3>
      <ul>
        {tips.map((t, i) => (
          <li key={i}>
            <span className={BADGE_CLASS[t.category] ?? "tip-badge cultural"}>
              {t.category}
            </span>{" "}
            {t.tip}
          </li>
        ))}
      </ul>
    </div>
  );
}
