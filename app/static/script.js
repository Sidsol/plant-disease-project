// ============================================================
// Plant Disease Classifier – Front-end Logic (v2)
// ============================================================

const uploadArea      = document.getElementById("upload-area");
const uploadContent   = document.getElementById("upload-content");
const preview         = document.getElementById("preview");
const fileInput       = document.getElementById("file-input");
const classifyBtn     = document.getElementById("classify-btn");
const clearBtn        = document.getElementById("clear-btn");
const loading         = document.getElementById("loading");
const resultsDiv      = document.getElementById("results");
const topResult       = document.getElementById("top-result");
const barChart        = document.getElementById("bar-chart");
const modelSelect     = document.getElementById("model-select");
const confidenceValue = document.getElementById("confidence-value");
const confidenceFill  = document.getElementById("confidence-fill");
const modelMetadata   = document.getElementById("model-metadata");
const treatmentCard   = document.getElementById("treatment-card");
const treatmentList   = document.getElementById("treatment-list");

let selectedFile = null;

// ---- Upload interactions ----

uploadArea.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
  if (e.target.files.length) handleFile(e.target.files[0]);
});

uploadArea.addEventListener("dragover", (e) => {
  e.preventDefault();
  uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
  uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
  e.preventDefault();
  uploadArea.classList.remove("dragover");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
  if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
    alert("Please upload a JPEG, PNG, or WebP image.");
    return;
  }
  selectedFile = file;
  const reader = new FileReader();
  reader.onload = (ev) => {
    preview.src = ev.target.result;
    preview.classList.remove("hidden");
    uploadContent.classList.add("hidden");
  };
  reader.readAsDataURL(file);
  classifyBtn.disabled = false;
  clearBtn.classList.remove("hidden");
  resultsDiv.classList.add("hidden");
}

clearBtn.addEventListener("click", resetUI);

function resetUI() {
  selectedFile = null;
  fileInput.value = "";
  preview.src = "";
  preview.classList.add("hidden");
  uploadContent.classList.remove("hidden");
  classifyBtn.disabled = true;
  clearBtn.classList.add("hidden");
  resultsDiv.classList.add("hidden");
  loading.classList.add("hidden");
  treatmentCard.classList.add("hidden");
}

// ---- Classification ----

classifyBtn.addEventListener("click", classify);

async function classify() {
  if (!selectedFile) return;

  classifyBtn.disabled = true;
  loading.classList.remove("hidden");
  resultsDiv.classList.add("hidden");
  treatmentCard.classList.add("hidden");

  const formData = new FormData();
  formData.append("file", selectedFile);

  const model = modelSelect.value;

  try {
    const res = await fetch(`/api/predict?model_name=${model}`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      let message = "Prediction failed";
      try {
        const err = await res.json();
        message = err.detail || message;
      } catch {
        message = await res.text() || `Server error (${res.status})`;
      }
      throw new Error(message);
    }
    const data = await res.json();
    showResults(data);

    // Fetch treatment tips for the diagnosed class
    fetchTreatment(data.class_name);
  } catch (err) {
    alert("Error: " + err.message);
  } finally {
    loading.classList.add("hidden");
    classifyBtn.disabled = false;
  }
}

// ---- Fetch treatment tips ----

async function fetchTreatment(className) {
  try {
    const res = await fetch(`/api/treatment/${encodeURIComponent(className)}`);
    if (!res.ok) return;
    const data = await res.json();
    showTreatment(data);
  } catch {
    // Silently ignore – treatment tips are supplementary
  }
}

function showTreatment(data) {
  treatmentList.innerHTML = "";
  data.tips.forEach((t) => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="tip-badge ${t.category}">${t.category}</span> ${t.tip}`;
    treatmentList.appendChild(li);
  });
  treatmentCard.classList.remove("hidden");
}

// ---- Display results ----

function showResults(data) {
  const top = data.prediction;
  const pct = top.confidence_percentage.toFixed(2);

  // Badge
  const badge = top.healthy
    ? '<span class="condition healthy">&#10004; Healthy</span>'
    : `<span class="condition disease">&#9888; ${top.condition}</span>`;

  topResult.innerHTML = `
    <div class="plant-name">${top.plant}</div>
    ${badge}
  `;

  // Confidence progress bar
  confidenceValue.textContent = `${pct}%`;
  confidenceFill.style.width = `${pct}%`;

  // Color the bar based on confidence
  if (top.confidence_percentage >= 80) {
    confidenceFill.className = "confidence-fill high";
  } else if (top.confidence_percentage >= 50) {
    confidenceFill.className = "confidence-fill medium";
  } else {
    confidenceFill.className = "confidence-fill low";
  }

  // Model metadata
  const meta = data.model_metadata;
  modelMetadata.innerHTML = `
    <div class="meta-row"><span class="meta-label">Model</span><span>${meta.architecture}</span></div>
    <div class="meta-row"><span class="meta-label">Version</span><span>v${meta.model_version}</span></div>
    <div class="meta-row"><span class="meta-label">Device</span><span>${meta.device}</span></div>
    <div class="meta-row"><span class="meta-label">Classes</span><span>${meta.num_classes}</span></div>
  `;

  // Bar chart – top 5
  barChart.innerHTML = "";
  const maxConf = data.top5[0].confidence_percentage;

  data.top5.forEach((p, i) => {
    const label = `${p.plant} – ${p.condition}`;
    const pctVal = p.confidence_percentage.toFixed(2);
    const widthPct = (p.confidence_percentage / maxConf) * 100;

    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <div class="bar-label" title="${label}">${label}</div>
      <div class="bar-track">
        <div class="bar-fill ${i === 0 ? "top" : ""}" style="width: ${widthPct}%"></div>
      </div>
      <div class="bar-value">${pctVal}%</div>
    `;
    barChart.appendChild(row);
  });

  resultsDiv.classList.remove("hidden");
}
