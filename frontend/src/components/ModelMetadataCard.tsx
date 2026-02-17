import type { ModelMetadata } from "../types";

interface Props {
  metadata: ModelMetadata;
}

export default function ModelMetadataCard({ metadata }: Props) {
  return (
    <div className="metadata-card">
      <div className="meta-row">
        <span className="meta-label">Model</span>
        <span>{metadata.architecture}</span>
      </div>
      <div className="meta-row">
        <span className="meta-label">Version</span>
        <span>v{metadata.model_version}</span>
      </div>
      <div className="meta-row">
        <span className="meta-label">Device</span>
        <span>{metadata.device}</span>
      </div>
      <div className="meta-row">
        <span className="meta-label">Classes</span>
        <span>{metadata.num_classes}</span>
      </div>
    </div>
  );
}
