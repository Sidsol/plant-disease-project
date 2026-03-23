import { useRef, useState, type DragEvent } from "react";

interface Props {
  onFileSelect: (file: File) => void;
  preview: string | null;
  onClear: () => void;
}

export default function UploadArea({ onFileSelect, preview, onClear }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = (file: File) => {
    if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
      alert("Please upload a JPEG, PNG, or WebP image.");
      return;
    }
    onFileSelect(file);
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
  };

  return (
    <div
      className={`action-card ${preview ? "has-preview" : ""} ${dragOver ? "dragover" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      {/* Decorative background element */}
      {!preview && (
        <div className="action-card-decor">
          <span className="material-symbols-outlined">potted_plant</span>
        </div>
      )}

      {preview ? (
        <>
          <img src={preview} alt="Leaf preview" className="preview-image" />
          <div className="action-buttons">
            <button className="btn-secondary" onClick={(e) => { e.stopPropagation(); onClear(); }}>
              <span className="material-symbols-outlined">close</span>
              Remove Photo
            </button>
          </div>
        </>
      ) : (
        <>
          <div
            className="upload-circle"
            onClick={() => inputRef.current?.click()}
            style={{ cursor: "pointer" }}
          >
            <span className="material-symbols-outlined">photo_camera</span>
          </div>
          <div className="action-buttons">
            <button
              className="btn-primary"
              onClick={(e) => { e.stopPropagation(); inputRef.current?.click(); }}
            >
              <span className="material-symbols-outlined">upload_file</span>
              Upload Photo
            </button>
          </div>
        </>
      )}

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/webp"
        hidden
        onChange={(e) => {
          if (e.target.files?.length) handleFile(e.target.files[0]);
        }}
      />
    </div>
  );
}
