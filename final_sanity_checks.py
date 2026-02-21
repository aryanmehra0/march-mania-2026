import os
import pandas as pd
import numpy as np
from sklearn.metrics import brier_score_loss
from config import *

# 1. LOAD OOF DATA
oof_path = os.path.join(DATA_PROCESSED, "oof_preds.csv")
if os.path.exists(oof_path):
    oof_df = pd.read_csv(oof_path)
    print("--- 2021-2025 OOF Brier Check ---")
    for s in [2021, 2022, 2023, 2024, 2025]:
        mask = oof_df['Season'] == s
        if mask.any():
            brier = brier_score_loss(oof_df[mask]['Result'], oof_df[mask]['oof_pred'])
            print(f"Season {s}: Brier = {brier:.5f}")
else:
    print("OOF file not found.")

# 2. VARIANT CORRELATION & DISTRIBUTION
vars = ['optimized', 'defensive', 'aggressive']
preds = {}

print("\n--- Variant Correlation & Distribution ---")
for v in vars:
    path = os.path.join(OUTPUT_DIR, f"submission_{v}.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        preds[v] = df['Pred'].values
        print(f"{v.capitalize()} - Mean: {np.mean(preds[v]):.4f}, Std: {np.std(preds[v]):.4f}, Min: {np.min(preds[v]):.4f}, Max: {np.max(preds[v]):.4f}")
    else:
        print(f"Submission {v} not found.")

if len(preds) == 3:
    print("\nCorrelations:")
    print(f"Optimized vs Defensive: {np.corrcoef(preds['optimized'], preds['defensive'])[0,1]:.4f}")
    print(f"Optimized vs Aggressive: {np.corrcoef(preds['optimized'], preds['aggressive'])[0,1]:.4f}")
    print(f"Defensive vs Aggressive: {np.corrcoef(preds['defensive'], preds['aggressive'])[0,1]:.4f}")
