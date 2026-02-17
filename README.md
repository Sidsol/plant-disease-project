# plant-disease-project
Plant disease classification project

## Quick Start

### 1. Install Dependencies

**For GPU (CUDA 12.1) - Recommended:**
```bash
pip install -r requirements-gpu.txt
```

**For CPU only:**
```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

1. Download the PlantVillage dataset from Kaggle:
   - https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset

2. Unzip it into:
   `data/raw/plantvillage/`

   You should have, for example:
   `data/raw/plantvillage/color/Apple___Black_rot/...`

3. Create train/val/test splits:
   ```bash
   cd ml
   python -m src.data.prepare_data \
       --source ../data/raw/plantvillage/color \
       --dest   ../data/processed/plantvillage_color_80_10_10 \
       --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
   ```

## API + React Frontend (HCAI + XAI)

Run the integrated API and UI:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Then open: `http://localhost:8000/`

> The frontend is implemented with ReactJS (loaded from CDN in `frontend/index.html`).

### API Endpoints

- `POST /predict`
  - Input: `multipart/form-data` with `image` and `model` (`efficientnet` or `custom_cnn`)
  - Output: structured `DiagnosisResponse` with diagnosis, confidence, metadata, and `attention_map_data_url`

- `GET /treatment-tips?diagnosis=<class_name>`
  - Output: treatment guidance list for the diagnosis

- `GET /history?page=1&page_size=10`
  - Output: paginated scan history for past diagnoses

- `POST /report-incorrect`
  - Input: image + predicted metadata
  - Output: acknowledgment that sample was flagged for human review/retraining
