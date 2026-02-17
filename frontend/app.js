(function () {
  const h = React.createElement;
  const API_BASE = window.location.origin;

  function App() {
    const [model, setModel] = React.useState("efficientnet");
    const [file, setFile] = React.useState(null);
    const [previewUrl, setPreviewUrl] = React.useState("");
    const [diagnosis, setDiagnosis] = React.useState(null);
    const [tips, setTips] = React.useState([]);
    const [history, setHistory] = React.useState([]);
    const [page, setPage] = React.useState(1);
    const [totalPages, setTotalPages] = React.useState(1);
    const [loading, setLoading] = React.useState(false);
    const [notice, setNotice] = React.useState("");

    React.useEffect(() => {
      loadHistory(page);
    }, [page]);

    function onFileChange(event) {
      const picked = event.target.files && event.target.files[0];
      setFile(picked || null);
      setDiagnosis(null);
      setTips([]);
      setNotice("");
      if (picked) {
        const url = URL.createObjectURL(picked);
        setPreviewUrl(url);
      } else {
        setPreviewUrl("");
      }
    }

    async function runDiagnosis() {
      if (!file) return;
      setLoading(true);
      setNotice("");
      try {
        const fd = new FormData();
        fd.append("image", file);
        fd.append("model", model);
        const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: fd });
        if (!res.ok) throw new Error("Diagnosis failed.");
        const payload = await res.json();
        setDiagnosis(payload);

        const tipsRes = await fetch(`${API_BASE}/treatment-tips?diagnosis=${encodeURIComponent(payload.class_name)}`);
        if (!tipsRes.ok) throw new Error("Could not load treatment tips.");
        const tipsPayload = await tipsRes.json();
        setTips(tipsPayload.treatment_tips || []);
        await loadHistory(1);
        setPage(1);
      } catch (error) {
        setNotice(error.message || "Unexpected error");
      } finally {
        setLoading(false);
      }
    }

    async function loadHistory(targetPage) {
      try {
        const res = await fetch(`${API_BASE}/history?page=${targetPage}&page_size=5`);
        if (!res.ok) return;
        const payload = await res.json();
        setHistory(payload.items || []);
        setTotalPages(payload.total_pages || 1);
      } catch (_error) {
        // ignore history failures in UI
      }
    }

    async function reportIncorrect() {
      if (!diagnosis || !file) return;
      setNotice("");
      try {
        const fd = new FormData();
        fd.append("image", file);
        fd.append("predicted_class", diagnosis.class_name);
        fd.append("confidence_percentage", String(diagnosis.confidence_percentage));
        fd.append("model_name", diagnosis.model_metadata.model_name);
        fd.append("notes", "User reported this prediction as incorrect from frontend.");

        const res = await fetch(`${API_BASE}/report-incorrect`, { method: "POST", body: fd });
        if (!res.ok) throw new Error("Could not submit report.");
        const payload = await res.json();
        setNotice(payload.message || "Report submitted.");
      } catch (error) {
        setNotice(error.message || "Report failed.");
      }
    }

    const confidence = diagnosis ? Number(diagnosis.confidence_percentage).toFixed(2) : "0.00";

    return h("main", { className: "container" }, [
      h("header", { key: "header" }, [
        h("h1", { key: "h1" }, "🌿 Plant Disease Detector (HCAI + XAI)"),
        h("p", { key: "p" }, "Diagnose with confidence, review attention heatmaps, and keep humans in the loop via correction reporting."),
      ]),

      h("section", { className: "card controls", key: "controls" }, [
        h("label", { htmlFor: "model", key: "label-model" }, "Model"),
        h("select", { id: "model", value: model, onChange: (e) => setModel(e.target.value), key: "model" }, [
          h("option", { value: "efficientnet", key: "m1" }, "EfficientNet-B0"),
          h("option", { value: "custom_cnn", key: "m2" }, "Custom CNN"),
        ]),
        h("label", { htmlFor: "image", key: "label-image" }, "Leaf image"),
        h("input", { id: "image", type: "file", accept: "image/*", onChange: onFileChange, key: "image" }),
        h("button", { onClick: runDiagnosis, disabled: !file || loading, key: "btn" }, loading ? "Diagnosing..." : "Run diagnosis"),
      ]),

      notice ? h("p", { className: "notice", key: "notice" }, notice) : null,

      h("section", { className: "grid", key: "grid" }, [
        h("article", { className: "card", key: "preview" }, [
          h("h2", { key: "title" }, "Explainability View"),
          previewUrl
            ? h("div", { className: "overlay-wrap", key: "overlay" }, [
                h("img", { src: previewUrl, className: "base-image", alt: "Leaf" }),
                diagnosis && diagnosis.attention_map_data_url
                  ? h("img", { src: diagnosis.attention_map_data_url, className: "heatmap-image", alt: "Attention heatmap" })
                  : null,
              ])
            : h("p", { key: "empty" }, "Upload an image to see the leaf and attention map."),
          diagnosis ? h("small", { className: "explain-note", key: "note" }, diagnosis.explainability_note) : null,
        ]),

        h("article", { className: "card", key: "dx" }, [
          h("h2", { key: "h2" }, "Diagnosis"),
          h("p", { className: "diagnosis" }, diagnosis ? diagnosis.class_name : "Diagnosis will appear here."),
          h("div", { className: "progress-wrap" }, [
            h("div", { className: "progress-label", key: "pl" }, [
              h("span", { key: "t" }, "Confidence"),
              h("strong", { key: "v" }, `${confidence}%`),
            ]),
            h("div", { className: "progress-track", key: "pt" }, [
              h("div", { className: "progress-bar", style: { width: `${confidence}%` }, key: "pb" }),
            ]),
          ]),
          diagnosis
            ? h("small", { key: "meta", className: "meta" }, `${diagnosis.model_metadata.model_name} • ${diagnosis.model_metadata.checkpoint} • Scan #${diagnosis.scan_id}`)
            : null,
          h("h3", null, "Top predictions"),
          h(
            "ol",
            { className: "list" },
            diagnosis
              ? diagnosis.top_predictions.map((p, i) =>
                  h("li", { key: `p-${i}` }, `${p.class_name} (${Number(p.confidence_percentage).toFixed(2)}%)`),
                )
              : [h("li", { key: "none" }, "No predictions yet.")],
          ),
          h("h3", null, "Treatment Tips"),
          h(
            "ul",
            { className: "list" },
            tips.length ? tips.map((tip, i) => h("li", { key: `t-${i}` }, tip)) : [h("li", { key: "empty-tip" }, "Tips appear after diagnosis.")],
          ),
          h("button", { className: "warn", onClick: reportIncorrect, disabled: !diagnosis || !file }, "Report Incorrect"),
        ]),

        h("article", { className: "card", key: "history" }, [
          h("h2", null, "Scan History"),
          h(
            "ul",
            { className: "list" },
            history.length
              ? history.map((item) =>
                  h(
                    "li",
                    { key: `h-${item.scan_id}` },
                    `#${item.scan_id} • ${item.class_name} • ${Number(item.confidence_percentage).toFixed(2)}% • ${new Date(item.created_at).toLocaleString()}`,
                  ),
                )
              : [h("li", { key: "h-none" }, "No scans yet.")],
          ),
          h("div", { className: "pager" }, [
            h("button", { onClick: () => setPage(Math.max(1, page - 1)), disabled: page <= 1, key: "prev" }, "Previous"),
            h("span", { key: "p" }, `Page ${page} / ${totalPages}`),
            h("button", { onClick: () => setPage(Math.min(totalPages, page + 1)), disabled: page >= totalPages, key: "next" }, "Next"),
          ]),
        ]),
      ]),
    ]);
  }

  ReactDOM.createRoot(document.getElementById("root")).render(h(App));
})();
