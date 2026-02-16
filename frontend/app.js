const CLASS_NAMES = [
  "Apple → Apple scab",
  "Apple → Black rot",
  "Cherry (including sour) → Powdery mildew",
  "Corn (maize) → Common rust",
  "Grape → Black rot",
  "Potato → Early blight",
  "Potato → Late blight",
  "Tomato → Bacterial spot",
  "Tomato → Early blight",
  "Tomato → Late blight",
  "Tomato → healthy",
];

const imageInput = document.querySelector("#imageInput");
const modelSelect = document.querySelector("#modelSelect");
const predictButton = document.querySelector("#predictButton");
const previewImage = document.querySelector("#previewImage");
const emptyPreview = document.querySelector("#emptyPreview");
const predictionList = document.querySelector("#predictionList");
const emptyPrediction = document.querySelector("#emptyPrediction");

let selectedFile = null;

imageInput.addEventListener("change", () => {
  selectedFile = imageInput.files?.[0] ?? null;
  predictButton.disabled = !selectedFile;

  if (!selectedFile) {
    previewImage.hidden = true;
    emptyPreview.hidden = false;
    return;
  }

  const previewUrl = URL.createObjectURL(selectedFile);
  previewImage.src = previewUrl;
  previewImage.hidden = false;
  emptyPreview.hidden = true;
});

predictButton.addEventListener("click", async () => {
  if (!selectedFile) return;

  predictButton.disabled = true;
  predictButton.textContent = "Predicting...";

  const predictions = await runPrediction(selectedFile, modelSelect.value);
  renderPredictions(predictions);

  predictButton.disabled = false;
  predictButton.textContent = "Run prediction";
});

async function runPrediction(file, model) {
  // If a backend exists, use it. Otherwise, use fallback mock predictions.
  try {
    const formData = new FormData();
    formData.append("image", file);
    formData.append("model", model);

    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const json = await response.json();
      if (Array.isArray(json.predictions)) return json.predictions;
    }
  } catch (_error) {
    // Fallback below.
  }

  return buildMockPredictions();
}

function buildMockPredictions() {
  const shuffled = [...CLASS_NAMES].sort(() => Math.random() - 0.5).slice(0, 3);
  const raw = [0.64, 0.23, 0.13];
  return shuffled.map((label, idx) => ({
    label,
    confidence: raw[idx],
  }));
}

function renderPredictions(predictions) {
  predictionList.innerHTML = "";
  emptyPrediction.hidden = true;

  predictions.forEach((item) => {
    const li = document.createElement("li");
    const confidencePct = (Number(item.confidence) * 100).toFixed(1);
    li.textContent = `${item.label} (${confidencePct}%)`;
    predictionList.appendChild(li);
  });
}
