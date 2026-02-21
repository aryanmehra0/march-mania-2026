import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from config import *
from features.feature_engineering import *

os.makedirs(OUTPUT_DIR, exist_ok=True)

###########################################
# LOAD MODELS & TRANSFORMERS
###########################################

xgb_cauchy = joblib.load(os.path.join(MODEL_DIR, "xgb_cauchy.pkl"))
lgb_model = joblib.load(os.path.join(MODEL_DIR, "lgb_model.pkl"))
leaf_model = joblib.load(os.path.join(MODEL_DIR, "leaf_model.pkl"))
ohe = joblib.load(os.path.join(MODEL_DIR, "ohe.pkl"))
poly = joblib.load(os.path.join(MODEL_DIR, "poly.pkl"))
iso_model = joblib.load(os.path.join(MODEL_DIR, "iso_model.pkl"))
best_weights = joblib.load(os.path.join(MODEL_DIR, "ensemble_weights.pkl"))
sharpen_params = joblib.load(os.path.join(MODEL_DIR, "sharpen_params.pkl"))

glm_strengths = joblib.load(os.path.join(MODEL_DIR, "glm_strengths.pkl"))
bt_strengths = joblib.load(os.path.join(MODEL_DIR, "bt_strengths.pkl"))
elo_ratings = joblib.load(os.path.join(MODEL_DIR, "elo_ratings.pkl"))

stats_path = os.path.join(DATA_PROCESSED, "team_stats.csv")
stats = pd.read_csv(stats_path)
sample_orig = pd.read_csv(os.path.join(DATA_RAW, "SampleSubmissionStage2.csv"))

def mc_smooth(p, n=5000):
    return np.mean(np.random.binomial(1, p, n))

def build_submissions():
    sample = sample_orig.copy()
    sample[['Season','Team1','Team2']] = sample['ID'].str.split('_', expand=True)
    sample['Season'] = sample['Season'].astype(int)
    sample['Team1'] = sample['Team1'].astype(int)
    sample['Team2'] = sample['Team2'].astype(int)

    temp = sample.merge(stats, left_on=['Season','Team1'], right_on=['Season','TeamID'], how='left')
    temp = temp.merge(stats, left_on=['Season','Team2'], right_on=['Season','TeamID'], how='left', suffixes=('_1','_2'))

    temp['Net_diff'] = temp['NetEff_1'] - temp['NetEff_2']
    temp['Conf_diff'] = temp['Conf_Strength_1'] - temp['Conf_Strength_2']
    temp['Elo_diff'] = temp['Team1'].map(elo_ratings).fillna(1500) - temp['Team2'].map(elo_ratings).fillna(1500)

    temp['GLM_diff'] = temp.apply(lambda r: glm_strengths.get(r['Season'], {}).get(r['Team1'], 0) - glm_strengths.get(r['Season'], {}).get(r['Team2'], 0), axis=1)
    temp['BT_diff'] = temp.apply(lambda r: bt_strengths.get(r['Season'], {}).get(r['Team1'], 0) - bt_strengths.get(r['Season'], {}).get(r['Team2'], 0), axis=1)

    features = ['Net_diff', 'Conf_diff', 'Elo_diff', 'GLM_diff', 'BT_diff']
    X = temp[features].fillna(0)
    X_poly = pd.DataFrame(poly.transform(X), columns=poly.get_feature_names_out(features))

    import xgboost as xgb
    dtest = xgb.DMatrix(X_poly)
    p_xgb = 1 / (1 + np.exp(-xgb_cauchy.predict(dtest) / 10))
    p_lgb = lgb_model.predict_proba(X_poly)[:,1]
    
    leaf_indices = lgb_model.predict(X_poly, pred_leaf=True)
    p_leaf = leaf_model.predict_proba(ohe.transform(leaf_indices))[:,1]

    # Dynamically Weighted Blend
    p_blend = best_weights[0]*p_xgb + best_weights[1]*p_lgb + best_weights[2]*p_leaf
    p_calibrated = iso_model.transform(p_blend)
    
    # VARIANT GENERATION
    variants = [
        {'name': 'optimized', 't': sharpen_params['t'], 's': sharpen_params['s']},
        {'name': 'defensive', 't': sharpen_params['t'] * 0.95, 's': 0.90}, # Lower temp, fixed 0.90 shrink
        {'name': 'aggressive', 't': sharpen_params['t'] * 1.05, 's': 0.95}, # Higher temp, closer to raw
    ]

    for var in variants:
        t, s = var['t'], var['s']
        p_temp = 1 / (1 + np.exp(-t * (p_calibrated - 0.5) / 0.1))
        p_final = 0.5 + s * (p_temp - 0.5)
        
        sample_var = sample.copy()
        sample_var['Pred'] = [mc_smooth(p) for p in p_final]
        
        out_path = os.path.join(OUTPUT_DIR, f"submission_{var['name']}.csv")
        sample_var[['ID','Pred']].to_csv(out_path, index=False)
        print(f"Generated {var['name']} submission.")

    # Also save the main one as submission.csv (the optimized one)
    optimized_path = os.path.join(OUTPUT_DIR, "submission_optimized.csv")
    import shutil
    shutil.copyfile(optimized_path, os.path.join(OUTPUT_DIR, "submission.csv"))

if __name__ == "__main__":
    build_submissions()
