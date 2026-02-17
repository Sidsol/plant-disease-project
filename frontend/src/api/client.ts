import type {
  DiagnosisResponse,
  TreatmentResponse,
  HistoryResponse,
  ReportRequest,
  ReportResponse,
} from "../types";

const BASE = "";

export async function classify(
  file: File,
  modelName: string
): Promise<DiagnosisResponse> {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(
    `${BASE}/api/predict?model_name=${encodeURIComponent(modelName)}`,
    { method: "POST", body: form }
  );
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Prediction failed");
  }
  return res.json();
}

export async function fetchTreatment(
  className: string
): Promise<TreatmentResponse> {
  const res = await fetch(
    `${BASE}/api/treatment/${encodeURIComponent(className)}`
  );
  if (!res.ok) throw new Error("Failed to fetch treatment tips");
  return res.json();
}

export async function fetchHistory(
  page = 1,
  limit = 10
): Promise<HistoryResponse> {
  const res = await fetch(`${BASE}/api/history?page=${page}&limit=${limit}`);
  if (!res.ok) throw new Error("Failed to fetch history");
  return res.json();
}

export async function reportIncorrect(
  req: ReportRequest
): Promise<ReportResponse> {
  const res = await fetch(`${BASE}/api/report`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Report failed");
  }
  return res.json();
}
