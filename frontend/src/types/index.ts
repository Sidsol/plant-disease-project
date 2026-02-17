// ---------------------------------------------------------------------------
// Shared TypeScript types matching the FastAPI Pydantic models
// ---------------------------------------------------------------------------

export interface ModelMetadata {
  model_name: string;
  model_version: string;
  architecture: string;
  num_classes: number;
  device: string;
}

export interface PredictionItem {
  class_index: number;
  class_name: string;
  plant: string;
  condition: string;
  healthy: boolean;
  confidence_percentage: number;
}

export interface DiagnosisResponse {
  scan_id: string;
  class_name: string;
  confidence_percentage: number;
  model_metadata: ModelMetadata;
  prediction: PredictionItem;
  top5: PredictionItem[];
  attention_map: string | null;
}

export interface TreatmentTip {
  tip: string;
  category: "organic" | "chemical" | "cultural";
}

export interface TreatmentResponse {
  class_name: string;
  plant: string;
  condition: string;
  healthy: boolean;
  tips: TreatmentTip[];
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  model_name: string;
  class_name: string;
  plant: string;
  condition: string;
  healthy: boolean;
  confidence: number;
  thumbnail: string | null;
  attention_map: string | null;
}

export interface HistoryResponse {
  items: HistoryItem[];
  total: number;
  page: number;
  limit: number;
  pages: number;
}

export interface ReportRequest {
  scan_id: string;
  reason?: string;
  user_correction?: string;
}

export interface ReportResponse {
  report_id: string;
  message: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  accuracy: number;
  description: string;
}
