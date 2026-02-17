const API_BASE = window.location.origin;

const imageInput = document.querySelector("#imageInput");
const modelSelect = document.querySelector("#modelSelect");
const predictButton = document.querySelector("#predictButton");
const previewImage = document.querySelector("#previewImage");
const emptyPreview = document.querySelector("#emptyPreview");
const diagnosisText = document.querySelector("#diagnosisText");
const confidenceWrap = document.querySelector("#confidenceWrap");
const confidenceBar = document.querySelector("#confidenceBar");
const confidenceValue = document.querySelector("#confidenceValue");
const metadataText = document.querySelector("#metadataText");
const predictionList = document.querySelector("#predictionList");
const tipsList = document.querySelector("#tipsList");
const emptyTips = document.querySelector("#emptyTips");

let selectedFile = null;

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files?.[0] ?? null;
  predictButton.disabled = !selectedFile;

  if (!selectedFile) {
    previewImage.hidden = true;
    emptyPreview.hidden = false;
    return;
  }

  previewImage.src = URL.createObjectURL(selectedFile);
  previewImage.hidden = false;
  emptyPreview.hidden = true;
});

predictButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  try {
    predictButton.disabled = true;
    predictButton.textContent = "Diagnosing...";

    const diagnosis = await fetchDiagnosis(selectedFile, modelSelect.value);
    renderDiagnosis(diagnosis);

    const tipsPayload = await fetchTreatmentTips(diagnosis.class_name);
    renderTips(tipsPayload.treatment_tips);
  } catch (error) {
    diagnosisText.textContent = `Error: ${error.message}`;
  } finally {
    predictButton.disabled = false;
    predictButton.textContent = "Run diagnosis";
  }
});

async function fetchDiagnosis(file, model) {
  const formData = new FormData();
  formData.append("image", file);
  formData.append("model", model);

  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Diagnosis request failed.");
  }

  return response.json();
}

async function fetchTreatmentTips(diagnosisName) {
  const response = await fetch(
    `${API_BASE}/treatment-tips?diagnosis=${encodeURIComponent(diagnosisName)}`,
  );
  if (!response.ok) {
    throw new Error("Treatment tips request failed.");
  }
  return response.json();
}

function renderDiagnosis(payload) {
  diagnosisText.textContent = payload.class_name;
  confidenceWrap.hidden = false;
  confidenceBar.style.width = `${payload.confidence_percentage}%`;
  confidenceValue.textContent = `${payload.confidence_percentage.toFixed(2)}%`;

  const modelData = payload.model_metadata;
  metadataText.textContent = `${modelData.model_name} • ${modelData.checkpoint} • ${modelData.classes_supported} classes`;

  predictionList.innerHTML = "";
  payload.top_predictions.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = `${item.class_name} (${item.confidence_percentage.toFixed(2)}%)`;
    predictionList.appendChild(li);
  });
}

function renderTips(tips) {
  tipsList.innerHTML = "";
  emptyTips.hidden = tips.length > 0;

  tips.forEach((tip) => {
    const li = document.createElement("li");
    li.textContent = tip;
    tipsList.appendChild(li);
  });
}
