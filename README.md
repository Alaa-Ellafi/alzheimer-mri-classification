# Alzheimer's Disease Detection from MRI — 2D & 3D CNNs

**Multi-class classification of Alzheimer's disease stages** from brain MRI scans using 2D and 3D convolutional neural networks.  
**Best model: DenseNet121 — Balanced Accuracy: 94.88% · AUC: 0.97 · Sensitivity: 98.89%**

---

## Problem

Alzheimer's disease affects ~35 million people worldwide. Only 5% of early-stage cases receive a timely diagnosis. This project builds CNN-based classifiers to automatically detect and stage Alzheimer's disease from brain MRI scans, comparing lightweight custom architectures against pretrained models in both 2D and 3D.

**Dataset:** OASIS-1 — brain MRI scans from 416 subjects (18–96 years), including demographic and clinical data.  
**Classes:** Non-Demented · Very Mild Dementia · Mild Dementia · Moderate Dementia  
**Total images:** 86,426 JPG slices extracted from MRI volumes (224×224px)

---

## Approach

### Data Preprocessing
- MRI slices extracted at indices 60–80 and 100–120 (informative brain regions)
- **Subject-level train/test split** to prevent data leakage from correlated slices
- Class imbalance addressed via augmentation: rotations, edge enhancement, detexturisation
- 3D volumes processed with the Clinica platform; grey matter segmentation; focused on right hippocampus (30×40×30)

### Models Compared

**2D Architectures:**
| Model | Accuracy | Balanced Accuracy | AUC |
|-------|----------|-------------------|-----|
| BrainNet2D (custom) | — | — | — |
| VGG16 (pretrained) | 0.607 | **0.759** | — |
| DenseNet121 (pretrained) | **85.38%** | **94.88%** | **0.97** |
| InceptionV3 (pretrained) | — | — | — |

**3D Architectures** (trained on right hippocampus volumes):
- BrainNet3D (baseline)
- BrainNet3D Modified (7 conv blocks, custom MaxPool3D, Dropout)
- ResNet-50 3D · ResNet-101 3D
- DenseNet-169 3D

### Best Result — DenseNet121 (2D)
- Accuracy: **85.38%**
- Balanced Accuracy: **94.88%**
- Sensitivity (Recall): **98.89%** — very few missed Alzheimer cases
- Specificity: **90.88%**
- AUC: **~0.97**

---

## Key Findings

- Pretrained models (DenseNet121, VGG16) consistently outperformed custom architectures due to richer feature representations from ImageNet pretraining
- Very mild dementia cases were hardest to classify — visually similar to healthy brains, especially at the hippocampal level
- Mild-to-moderate cases showed clearer hippocampal atrophy, leading to higher classification accuracy (~85.7%)
- 3D ResNet configurations outperformed BrainNet3D, thanks to residual connections reducing gradient issues
- Subject-level data splitting was critical to avoid artificially inflated performance

---

## Stack

- Python · PyTorch · TensorFlow · Keras
- DenseNet121 · VGG16 · InceptionV3 · ResNet-50/101 3D · DenseNet-169 3D
- Clinica · NumPy · Pandas · Scikit-learn · Matplotlib

---

## Structure

```
alzheimer-mri-classification/
├── Copy_of_P2M.ipynb      # Full pipeline notebook
├── report.pdf             # Detailed methodology and results (French)
└── requirements.txt
```

---

## Dataset

OASIS-1: [https://www.kaggle.com/datasets/ninadaithal/imagesoasis](https://www.kaggle.com/datasets/ninadaithal/imagesoasis)

---

*Project supervised by Mr. Riadh Abdelfattah, Sup'Com. Developed jointly with Siwar Amri.*
