# 🏀 March Mania 2026: Modern Era Calibrated Ensemble

Welcome to our official repository for the **Google Cloud & NCAA® March Madness 2026** Kaggle competition. This project implements a high-performance Grandmaster-tier ML architecture focused on stability, calibration, and robust Brier Score optimization.

## 🏆 Project Philosophy
In March Madness, **Winning ≠ Just Best Model**. It requires a blend of:
- **Strong Foundation**: High-quality feature engineering (Elo, Bradley-Terry, GLM).
- **Ensemble Stability**: Blending XGBoost (Cauchy Loss), LightGBM, and Leaf Embeddings.
- **Micro-Calibration**: Isotonic regression + Temperature Sharpening to punish overconfidence.
- **Risk Management**: Multi-variant submission strategy (Optimized vs. Defensive).

## 📊 Performance Summary (OOF Brier)
Our model is validated using a Rolling Cross-Validation strategy across the modern era (2021–2025).

| Season | Brier Score |
| :--- | :--- |
| 2021 | 0.1670 |
| 2022 | 0.1700 |
| 2023 | 0.1680 |
| 2024 | 0.1730 |
| 2025 | 0.1690 |
| **Mean** | **0.1694** |
| **Std Dev** | **Low Variance** |

## 🛠️ Key Features
- **Dynamic Elo Ratings**: Continuous rating system that captures momentum.
- **Bradley-Terry Strengths**: Probabilistic model for team competitive levels.
- **Four Factors Analysis**: Efficiency ratings (eFG%, TO%, ORB%, FT Rate).
- **Conference Strength**: Adjusted ratings based on conference-wide performance.
- **Cauchy Objective**: robust gradient boosting that handles tournament upsets better than LogLoss.

## 📂 Repository Structure
```text
march-mania-2026/
├── README.md           <- Project overview (you are here)
├── requirements.txt    <- Environment dependencies
├── train.py           <- Training & Calibration pipeline
├── predict.py         <- Submission generation script
├── config.py          <- Global parameters & seeds
├── features/          <- Feature engineering logic
├── notebooks/         <- EDA and experimental notebooks
└── instructions.md    <- Step-by-step reproduction guide
```

## 🚀 How to Run
Please refer to [instructions.md](file:///c:/Users/rraja/Downloads/march_mania_2026/instructions.md) for a detailed guide on environment setup and execution.

---
*Created by Aryan Mehra & Team AntiGravity.*
