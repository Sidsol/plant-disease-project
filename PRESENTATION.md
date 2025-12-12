# Plant Disease Classification
## Deep Learning for Agricultural Health Monitoring

---

# Agenda

1. **Project Overview** - Problem statement & dataset
2. **Exploratory Data Analysis** - Understanding the data
3. **Model Architectures** - EfficientNet-B0 vs Custom CNN
4. **Training & Results** - Performance comparison
5. **Model Interpretability** - Grad-CAM attention maps
6. **Conclusions & Future Work**

---

# 1. Project Overview

## Problem Statement

**Challenge:** Manual identification of plant diseases is:
- Time-consuming and labor-intensive
- Requires expert knowledge
- Prone to human error
- Not scalable for large agricultural operations

**Solution:** Automated plant disease classification using deep learning

---

## Dataset: PlantVillage

| Metric | Value |
|--------|-------|
| **Total Images** | ~54,000+ |
| **Plant Species** | 14 |
| **Disease Classes** | 38 |
| **Image Size** | 256×256 pixels |
| **Format** | RGB Color |

### Data Split
- **Training:** 80%
- **Validation:** 10%
- **Test:** 10%

---

# 2. Exploratory Data Analysis
## Notebook: `01_eda.ipynb`

---

## Class Distribution

### Key Findings:
- **38 unique classes** (diseases + healthy states)
- **Class imbalance exists** - ratio up to ~5.7x between largest/smallest classes
- Plants covered: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato

### Plants by Number of Classes:
| Plant | # Classes | Notes |
|-------|-----------|-------|
| Tomato | 10 | Most disease variety |
| Apple | 4 | Including healthy |
| Grape | 4 | Including healthy |
| Corn | 4 | Including healthy |

---

## Healthy vs Diseased Distribution

| Category | Count | Percentage |
|----------|-------|------------|
| **Diseased** | ~46,000 | ~85% |
| **Healthy** | ~8,000 | ~15% |

### Insight:
Dataset is heavily weighted toward diseased samples, which is beneficial for training disease detection models.

---

## Image Properties

- **Consistent dimensions:** 256×256 pixels (standardized dataset)
- **Color mode:** RGB (3 channels)
- **File format:** JPG
- **Average file size:** ~20-30 KB

### Color Analysis:
- Healthy leaves show higher green channel values
- Diseased leaves often show:
  - Brown/yellow discoloration (lower green)
  - Spots and lesions (irregular color patterns)

---

## EDA Recommendations Applied

1. ✅ **Weighted Loss Function** - CrossEntropyLoss with class weights
2. ✅ **Data Augmentation** - Flips, rotations, color jitter
3. ✅ **Stratified Splits** - Maintained class ratios
4. ✅ **Transfer Learning** - EfficientNet-B0 pretrained on ImageNet
5. ✅ **Macro F1 Score** - Primary metric for imbalanced data

---

# 3. Model Architectures

---

## Model 1: EfficientNet-B0 (Transfer Learning)
### Notebook: `02_train_efficientnet.ipynb`

**Architecture:**
- Pre-trained on ImageNet (1.2M images, 1000 classes)
- Compound scaling (depth, width, resolution)
- ~4M parameters

**Training Strategy:** Progressive Unfreezing
1. **Phase 1 (5 epochs):** Freeze backbone, train classifier only
2. **Phase 2 (25 epochs):** Unfreeze last 3 blocks, fine-tune with differential learning rates

| Component | Learning Rate |
|-----------|---------------|
| Backbone (frozen layers) | 1e-4 |
| Classifier head | 1e-3 |

---

## Model 2: Custom CNN (From Scratch)
### Notebook: `03_train_cnn.ipynb`

**Architecture:** 5 Convolutional Blocks

```
Input (224×224×3)
    ↓
Conv Block 1: 32 filters → BatchNorm → ReLU → MaxPool
    ↓
Conv Block 2: 64 filters → BatchNorm → ReLU → MaxPool
    ↓
Conv Block 3: 128 filters → BatchNorm → ReLU → MaxPool
    ↓
Conv Block 4: 256 filters → BatchNorm → ReLU → MaxPool
    ↓
Conv Block 5: 512 filters → BatchNorm → ReLU → MaxPool
    ↓
Global Average Pooling
    ↓
FC (512 → 256) → Dropout(0.5) → ReLU
    ↓
FC (256 → 38 classes)
```

**Parameters:** ~6.5M trainable parameters

---

## Training Configuration (Both Models)

| Hyperparameter | Value |
|----------------|-------|
| **Optimizer** | AdamW |
| **Weight Decay** | 0.01 |
| **Scheduler** | CosineAnnealingWarmRestarts |
| **Batch Size** | 32 |
| **Image Size** | 224×224 |
| **Early Stopping** | 5 epochs patience |
| **Loss Function** | Weighted CrossEntropyLoss |

### Data Augmentation:
- Random horizontal/vertical flips
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Random resized crop

---

# 4. Training & Results

---

## EfficientNet-B0 Training Curves

### Phase 1: Classifier Warmup (Epochs 1-5)
- Backbone frozen
- Rapid accuracy improvement
- Training accuracy: ~90%+

### Phase 2: Fine-tuning (Epochs 6-30)
- Last 3 blocks unfrozen
- Slower, steady improvement
- Final validation accuracy: **~99%**

### Key Observations:
- ✅ Minimal overfitting (train-val gap < 1%)
- ✅ Healthy F1 progression throughout
- ✅ Stable training with smooth curves

---

## Custom CNN Training Curves

### Training (30 epochs)
- Initial rapid learning
- Peak performance around epoch 25-30
- Final validation accuracy: **~96%**

### Key Observations:
- ⚠️ Moderate overfitting after epoch 25
- Training accuracy reaches ~99% while validation plateaus at ~96%
- Could benefit from:
  - Earlier stopping (epoch 25-30)
  - Stronger regularization
  - More aggressive dropout

---

## Model Comparison

| Metric | EfficientNet-B0 | Custom CNN |
|--------|-----------------|------------|
| **Test Accuracy** | ~99% | ~96% |
| **F1 Score (Macro)** | ~0.99 | ~0.96 |
| **F1 Score (Weighted)** | ~0.99 | ~0.96 |
| **Parameters** | ~4M | ~6.5M |
| **Training Time** | Faster (pretrained) | Slower |
| **Overfitting** | Minimal | Moderate |

### Winner: **EfficientNet-B0** 🏆

---

## Why Transfer Learning Won

1. **Pre-trained Features:** ImageNet features include low-level patterns (edges, textures) useful for plant images

2. **Better Generalization:** Pre-training provides strong regularization

3. **Faster Convergence:** Already knows useful visual features

4. **More Efficient:** Fewer parameters needed for same/better performance

5. **Progressive Unfreezing:** Controlled fine-tuning prevents catastrophic forgetting

---

# 5. Model Interpretability
## Notebook: `04_evaluation.ipynb`

---

## Grad-CAM Visualization

**Gradient-weighted Class Activation Mapping (Grad-CAM)**

Shows which regions of an image the model focuses on to make predictions.

### How it works:
1. Forward pass through the model
2. Compute gradients of target class w.r.t. last conv layer
3. Weight activations by gradient importance
4. Create heatmap overlay on original image

### Insights:
- Models correctly focus on **disease lesions** and **affected areas**
- Healthy predictions focus on **overall leaf structure**
- Helps identify potential failure modes

---

## Per-Class Analysis

### Best Performing Classes (F1 > 0.99):
- Well-separated visual features
- Distinctive disease patterns
- Sufficient training samples

### Challenging Classes (Lower F1):
- Similar visual appearance to other classes
- Smaller sample sizes
- Subtle disease symptoms

### Common Confusions:
- Similar diseases on same plant species
- Early-stage vs late-stage disease

---

# 6. Conclusions

---

## Key Achievements

✅ **High Accuracy:** 99% with EfficientNet-B0

✅ **Robust Training:** Minimal overfitting with transfer learning

✅ **38-Class Classification:** Successfully distinguishes all disease types

✅ **Interpretable Results:** Grad-CAM shows meaningful focus regions

✅ **Production-Ready:** Models exported for deployment

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| **Framework** | PyTorch 2.9.1 |
| **GPU** | NVIDIA RTX 3080 (CUDA 12.6) |
| **Models** | EfficientNet-B0, Custom CNN |
| **Metrics** | Accuracy, F1, Precision, Recall |
| **Logging** | TensorBoard |
| **Visualization** | Matplotlib, Seaborn |

---

## Future Work

1. **Real-time Inference**
   - Mobile deployment (TensorFlow Lite, ONNX)
   - Edge device optimization

2. **Model Improvements**
   - Ensemble methods
   - Larger EfficientNet variants (B1-B7)
   - Vision Transformers

3. **Dataset Expansion**
   - More plant species
   - Real-world field images
   - Different lighting conditions

4. **Application Development**
   - Mobile app for farmers
   - Web API for integration
   - Recommendation system for treatments

---

# Thank You!

## Questions?

---

## Appendix: Notebook Summary

| Notebook | Purpose | Key Outputs |
|----------|---------|-------------|
| `01_eda.ipynb` | Exploratory Data Analysis | Class distribution charts, image statistics, recommendations |
| `02_train_efficientnet.ipynb` | Transfer Learning Training | EfficientNet model, training curves, checkpoints |
| `03_train_cnn.ipynb` | Custom CNN Training | Custom CNN model, training curves, checkpoints |
| `04_evaluation.ipynb` | Model Comparison | Metrics, confusion matrices, Grad-CAM, conclusions |

---

## Appendix: Project Structure

```
plant-disease-project/
├── data/
│   ├── raw/                 # Original PlantVillage dataset
│   └── processed/           # Train/Val/Test splits
│       ├── train/
│       ├── val/
│       └── test/
├── ml/
│   ├── notebooks/           # Jupyter notebooks (01-04)
│   ├── src/                 # Source code modules
│   │   ├── data/            # Dataset utilities
│   │   ├── models/          # Model architectures
│   │   └── utils/           # Training utilities
│   └── outputs/             # Figures, CSVs
├── models_exported/         # Saved model checkpoints
├── runs/                    # TensorBoard logs
└── requirements.txt         # Dependencies
```
