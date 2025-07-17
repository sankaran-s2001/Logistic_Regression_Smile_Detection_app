# 😊 SmileScan AI - Smile Detection App

![App Screenshot](./image.png) 

A machine learning web application that detects smiles in facial images using logistic regression, trained on facial expression data.

## 🚀 Features
- **Instant Prediction**: Classifies images as "Smiling" (1) or "Not Smiling" (0)
- **Confidence Scores**: Displays prediction probabilities (0-100%)
- **Example Testing**: Pre-loaded sample images for quick validation
- **File Upload**: Supports JPG, JPEG, PNG formats
- **Mobile-Friendly**: Responsive Streamlit interface

## 📊 Dataset
**Source**: [Smiling or Not Face Dataset](https://www.kaggle.com/datasets/chazzer/smiling-or-not-face-data) on Kaggle

### Composition
| Class | Samples | Image Type |
|-------|---------|------------|
| Smiling | 600 | Close-up face crops |
| Non-Smiling | 603 | Close-up face crops |
| Test (Unlabeled) | ~12,000 | Varied facial expressions |

**Key Characteristics**:
- 64×64 pixel RGB images
- Front-facing portraits with varied lighting
- Balanced classes (600 vs 603 samples)
- Additional 12K unlabeled test images available

### Preprocessing Pipeline
1. **Grayscale Conversion**: RGB → 1-channel (Luminosity)
2. **Resizing**: Standardized to 64×64 pixels
3. **Normalization**: Pixel values scaled to [0,1]
4. **Flattening**: 2D image → 1D array (4096 features)

## 🛠️ Technical Implementation

### Model Architecture
```mermaid
graph LR
A[Input Image] --> B[Resize to 64x64]
B --> C[Convert to Grayscale]
C --> D[Flatten to 4096 features]
D --> E[Standard Scaling]
E --> F[Logistic Regression]
F --> G[Prediction]
