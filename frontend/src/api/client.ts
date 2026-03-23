import type {
  DiagnosisResponse,
  TreatmentResponse,
  HistoryResponse,
  ReportRequest,
  ReportResponse,
  ChatRequest,
  ChatMessage,
  OllamaStatus,
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

// ---------------------------------------------------------------------------
// Chat API (RAG + Ollama streaming)
// ---------------------------------------------------------------------------

export async function sendChatMessage(
  req: ChatRequest,
  onToken: (token: string) => void,
  onDone?: (sessionId?: string) => void,
  onError?: (error: string) => void
): Promise<void> {
  const res = await fetch(`${BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Chat request failed");
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error("No response body");

  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      const jsonStr = trimmed.slice(6);
      try {
        const data = JSON.parse(jsonStr);
        if (data.token) {
          onToken(data.token);
        }
        if (data.error) {
          onError?.(data.error);
        }
        if (data.done) {
          onDone?.(data.session_id);
        }
      } catch {
        // skip malformed JSON
      }
    }
  }
}

export async function fetchChatStatus(): Promise<OllamaStatus> {
  const res = await fetch(`${BASE}/api/chat/status`);
  if (!res.ok) {
    return { available: false, models: [], default_model: "llama3.1:8b" };
  }
  return res.json();
}

export async function fetchChatHistory(
  scanId?: string,
  sessionId?: string
): Promise<ChatMessage[]> {
  const params = new URLSearchParams();
  if (scanId) params.set("scan_id", scanId);
  if (sessionId) params.set("session_id", sessionId);
  const res = await fetch(`${BASE}/api/chat/history?${params}`);
  if (!res.ok) return [];
  const data = await res.json();
  return data.messages ?? [];
}
