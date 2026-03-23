import type { ModelMetadata } from "../types";

interface Props {
  metadata: ModelMetadata;
}

export default function ModelMetadataCard({ metadata }: Props) {
  return (
    <div className="metadata-row">
      <div className="meta-chip">
        <span className="label">Model</span>
        <span>{metadata.architecture}</span>
      </div>
      <div className="meta-chip">
        <span className="label">Version</span>
        <span>v{metadata.model_version}</span>
      </div>
      <div className="meta-chip">
        <span className="label">Device</span>
        <span>{metadata.device}</span>
      </div>
      <div className="meta-chip">
        <span className="label">Classes</span>
        <span>{metadata.num_classes}</span>
      </div>
    </div>
  );
}
