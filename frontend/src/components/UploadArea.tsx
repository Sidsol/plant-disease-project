import { useRef, useState, type DragEvent } from "react";

interface Props {
  onFileSelect: (file: File) => void;
  preview: string | null;
  onClear: () => void;
}

export default function UploadArea({ onFileSelect, preview }: Props) {
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
      className={`upload-area ${dragOver ? "dragover" : ""}`}
      onClick={() => inputRef.current?.click()}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)}
      onDrop={onDrop}
    >
      {preview ? (
        <img src={preview} alt="Preview" className="preview" />
      ) : (
        <div className="upload-content">
          <div className="upload-icon">&#128247;</div>
          <p className="upload-text">Drag &amp; drop a leaf image here</p>
          <p className="upload-hint">or click to browse &mdash; JPEG, PNG, WebP</p>
        </div>
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
