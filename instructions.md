# Reproducibility Instructions — March Mania 2026

Follow these steps to reproduce the "Modern Era Calibrated Ensemble" results.

## 1. Environment Setup
Ensure you have Python 3.10+ installed. Install dependencies:
```bash
pip install -r requirements.txt
```

## 2. Data Preparation
Place the Kaggle Competition Stage 2 data in the `data/raw/` directory.

## 3. Training
Run the training pipeline. This will:
- Process features (Elo, Four Factors, Bradley-Terry).
- Train XGBoost (Cauchy), LightGBM, and Leaf Embedding models.
- Optimize ensemble weights.
- Perform Isotonic Calibration & Sharpening.
```bash
python train.py
```

## 4. Prediction
Generate the competition submission files.
```bash
python predict.py
```
This will output `submission_optimized.csv`, `submission_defensive.csv`, and `submission_aggressive.csv` in the `outputs/` folder.

## Hardware Used
- Hardware: Standard Kaggle CPU/GPU Instance (or local machine with 16GB+ RAM)
- Training Time: ~5-10 minutes
