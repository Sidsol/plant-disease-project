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

## Frontend (basic)

A basic frontend UI is included in `frontend/`.

### Run locally

```bash
python -m http.server 8000
```

Then open: `http://localhost:8000/frontend/`

Notes:
- The UI is intentionally lightweight and focuses on image upload + prediction display.
- It will call `POST /predict` if a backend is available.
- If no backend exists yet, it falls back to mock predictions so the UI can still be tested.
