# Reproducibility & Technical Architecture — March Mania 2026

This document provides a full technical breakdown and step-by-step reproduction guide for our "Modern Era Calibrated Ensemble."

## 🏗️ Technical Architecture
Our system utilizes a multi-stage ensemble designed to neutralize the high variance of the NCAA tournament.

### 1. Preprocessing & Feature Engineering
- **Dynamic Elo**: 1500-base ratings with a custom margin-of-victory multiplier.
- **Bradley-Terry**: Logit-scale pairwise strengths derived from the full regular-season matrix.
- **Four Factors**: Aggregate efficiency metrics (eFG%, TOV%, ORB%, FTR).
- **Conference Normalization**: Team metrics are adjusted by the `NetEff` of their conference to normalize strength-of-schedule.

### 2. Modeling & Loss Functions
We blend three learners to ensure stability:
- **XGBoost (Cauchy Loss)**: We use the Cauchy objective because it is robust to outliers (large score margins) and provides a more stable gradient than standard MSE or LogLoss during tournament upsets.
- **LightGBM**: Optimized for high-fidelity feature interactions.
- **Leaf Embedding**: A second-stage Logistic Regression that uses the leaf indices of our LGBM model as categorical features.

### 3. Hyperparameters
- **XGB**: `eta: 0.01`, `max_depth: 4`, `num_boost_round: 1000`, `objective: cauchy` (parameter `c=10`).
- **LGBM**: `learning_rate: 0.01`, `max_depth: 4`, `n_estimators: 1000`.
- **Calibration**: Isotonic Regression with out-of-bounds clipping.
- **Sharpening**: Temperature scaling (optimized via `scipy.optimize.minimize`).

## 🚀 Reproduction Steps
1. **Environment Setup**: Python 3.10+
   ```bash
   pip install -r requirements.txt
   ```
2. **Data Placement**: Place Kaggle Stage 2 raw CSVs in `data/raw/`.
3. **Execution**:
   ```bash
   python train.py
   python predict.py
   ```

## 📊 Evaluation Logic
We utilize a **Rolling Cross-Validation** strategy across the seasons 2021-2025. This ensures the model is not overfitted to any specific "era" or outlier season.

**Code Repository**: [github.com/aryanmehra0/march-mania-2026](https://github.com/aryanmehra0/march-mania-2026)
