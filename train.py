import os
import pandas as pd
import numpy as np
import joblib
from scipy.optimize import minimize, minimize_scalar

from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

from config import *
from features.feature_engineering import *
from features.strength_models import *

os.makedirs(DATA_PROCESSED, exist_ok=True)
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)

###########################################
# CUSTOM LOSS: CAUCHY
###########################################

def cauchy_obj(preds, dtrain):
    c = 10
    labels = dtrain.get_label()
    residual = preds - labels
    grad = 2 * residual / (1 + (residual/c)**2)
    hess = 2 / (1 + (residual/c)**2)
    return grad, hess

###########################################
# LOAD DATA
###########################################

def load_data():
    M_reg = pd.read_csv(os.path.join(DATA_RAW, "MRegularSeasonDetailedResults.csv"))
    W_reg = pd.read_csv(os.path.join(DATA_RAW, "WRegularSeasonDetailedResults.csv"))
    M_tour = pd.read_csv(os.path.join(DATA_RAW, "MNCAATourneyDetailedResults.csv"))
    W_tour = pd.read_csv(os.path.join(DATA_RAW, "WNCAATourneyDetailedResults.csv"))
    M_conf = pd.read_csv(os.path.join(DATA_RAW, "MTeamConferences.csv"))
    W_conf = pd.read_csv(os.path.join(DATA_RAW, "WTeamConferences.csv"))
    conf_df = pd.concat([M_conf, W_conf], ignore_index=True)
    return M_reg, W_reg, M_tour, W_tour, conf_df

###########################################
# TRAINING PIPELINE
###########################################

def train_model():
    M_reg, W_reg, M_tour, W_tour, conf_df = load_data()

    M_reg = add_four_factors(add_possessions(M_reg))
    W_reg = add_four_factors(add_possessions(W_reg))
    stats = pd.concat([team_season_stats(M_reg), team_season_stats(W_reg)], ignore_index=True)
    stats = add_conference_strength(stats, conf_df)

    reg_combined = pd.concat([M_reg, W_reg], ignore_index=True)
    elo_ratings, elo_history = dynamic_elo(reg_combined)

    tour = pd.concat([M_tour, W_tour], ignore_index=True)
    tour['Team1'] = tour[['WTeamID','LTeamID']].min(axis=1)
    tour['Team2'] = tour[['WTeamID','LTeamID']].max(axis=1)
    tour['Result'] = (tour['WTeamID']==tour['Team1']).astype(int)
    tour['ScoreDiff'] = tour['WScore'] - tour['LScore']
    tour.loc[tour['Result'] == 0, 'ScoreDiff'] = -tour['ScoreDiff']

    tour = tour.merge(stats, left_on=['Season','Team1'], right_on=['Season','TeamID'], how='left')
    tour = tour.merge(stats, left_on=['Season','Team2'], right_on=['Season','TeamID'], how='left', suffixes=('_1','_2'))

    temp_elo_map = elo_ratings
    tour['Net_diff'] = tour['NetEff_1'] - tour['NetEff_2']
    tour['Conf_diff'] = tour['Conf_Strength_1'] - tour['Conf_Strength_2']
    tour['Elo_diff'] = tour['Team1'].map(temp_elo_map).fillna(1500) - tour['Team2'].map(temp_elo_map).fillna(1500)
    
    glm_strengths = {}
    bt_strengths = {}
    for s in tour['Season'].unique():
        season_reg = reg_combined[reg_combined['Season'] == s]
        glm_strengths[s] = compute_glm_strength(season_reg)
        bt_strengths[s] = compute_bt_strength(season_reg)

    tour['GLM_diff'] = tour.apply(lambda r: glm_strengths[r['Season']].get(r['Team1'], 0) - glm_strengths[r['Season']].get(r['Team2'], 0), axis=1)
    tour['BT_diff'] = tour.apply(lambda r: bt_strengths[r['Season']].get(r['Team1'], 0) - bt_strengths[r['Season']].get(r['Team2'], 0), axis=1)

    features = ['Net_diff', 'Conf_diff', 'Elo_diff', 'GLM_diff', 'BT_diff']
    X = tour[features].fillna(0)
    y = tour['Result']
    y_diff = tour['ScoreDiff']
    seasons = tour['Season']

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    X_poly = pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out(features))

    unique_seasons = sorted(seasons.unique())
    
    # Storage for individual model OOFs
    oof_xgb = np.zeros(len(tour))
    oof_lgb = np.zeros(len(tour))
    oof_leaf = np.zeros(len(tour))

    for test_season in unique_seasons[-6:]:
        train_idx = seasons < test_season
        val_idx = seasons == test_season
        if not train_idx.any() or not val_idx.any(): continue

        X_train, y_train, y_diff_train = X_poly[train_idx], y[train_idx], y_diff[train_idx]
        X_val, y_val = X_poly[val_idx], y[val_idx]
        
        if len(np.unique(y_train)) < 2: continue

        # EXPONENTIAL SEASON WEIGHTING
        s_min = seasons[train_idx].min()
        weights = np.exp(0.15 * (seasons[train_idx] - s_min)) # Micro-edge weighting

        # XGB Cauchy
        dtrain = xgb.DMatrix(X_train, label=y_diff_train, weight=weights)
        dval = xgb.DMatrix(X_val)
        xgb_model = xgb.train({'eta': 0.01, 'max_depth': 4}, dtrain, obj=cauchy_obj, num_boost_round=1000)
        oof_xgb[val_idx] = 1 / (1 + np.exp(-xgb_model.predict(dval) / 10))

        # LGBM
        lgb_model = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.01, max_depth=4, random_state=SEED, verbose=-1)
        lgb_model.fit(X_train, y_train, sample_weight=weights)
        oof_lgb[val_idx] = lgb_model.predict_proba(X_val)[:,1]

        # Leaf Embedding
        leaf_indices = lgb_model.predict(X_train, pred_leaf=True)
        ohe = OneHotEncoder(handle_unknown='ignore')
        leaf_features_train = ohe.fit_transform(leaf_indices)
        leaf_model = LogisticRegression(max_iter=1000)
        leaf_model.fit(leaf_features_train, y_train, sample_weight=weights)
        
        leaf_indices_val = lgb_model.predict(X_val, pred_leaf=True)
        leaf_features_val = ohe.transform(leaf_indices_val)
        oof_leaf[val_idx] = leaf_model.predict_proba(leaf_features_val)[:,1]

    mask = oof_xgb > 0
    y_target = y[mask]

    # CORRELATION ANALYSIS
    print("\n--- Model Correlation Analysis ---")
    corr_matrix = np.corrcoef([oof_xgb[mask], oof_lgb[mask], oof_leaf[mask]])
    print(f"XGB vs LGB: {corr_matrix[0,1]:.4f}")
    print(f"XGB vs Leaf: {corr_matrix[0,2]:.4f}")
    print(f"LGB vs Leaf: {corr_matrix[1,2]:.4f}")

    # WEIGHT OPTIMIZATION
    def brier_objective(weights):
        w = np.exp(weights) / np.sum(np.exp(weights))
        combined = w[0]*oof_xgb[mask] + w[1]*oof_lgb[mask] + w[2]*oof_leaf[mask]
        return brier_score_loss(y_target, combined)

    res = minimize(brier_objective, [0, 0, 0])
    best_weights = np.exp(res.x) / np.sum(np.exp(res.x))
    print(f"Optimal Weights: XGB={best_weights[0]:.3f}, LGB={best_weights[1]:.3f}, Leaf={best_weights[2]:.3f}")

    oof_blend = best_weights[0]*oof_xgb + best_weights[1]*oof_lgb + best_weights[2]*oof_leaf
    
    # CALIBRATION + MICRO-TUNING
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(oof_blend[mask], y[mask])
    p_calibrated = iso.transform(oof_blend[mask])

    # Optimize Temperature (t) and Shrink (s)
    def sharpen_objective(params):
        t, s = params
        p = 1 / (1 + np.exp(-t * (p_calibrated - 0.5) / 0.1))
        p_adj = 0.5 + s * (p - 0.5)
        return brier_score_loss(y_target, p_adj)

    res_sh = minimize(sharpen_objective, [1.02, 0.92], bounds=[(0.5, 2.0), (0.7, 1.0)])
    best_t, best_s = res_sh.x
    print(f"Optimal Sharpening: Temperature={best_t:.4f}, Shrink={best_s:.4f}")

    oof_final = np.zeros_like(oof_blend)
    p_final = 1 / (1 + np.exp(-best_t * (p_calibrated - 0.5) / 0.1))
    oof_final[mask] = 0.5 + best_s * (p_final - 0.5)

    # TOURNAMENT PERFORMANCE
    print("\n--- Tournament Performance ---")
    tourney_brier = brier_score_loss(y[mask], oof_final[mask])
    print(f"Tournament-only OOF Brier: {tourney_brier:.5f}")

    # STABILITY ANALYSIS
    print("\n--- Seasonal Stability Analysis ---")
    for s in unique_seasons[-6:]:
        s_mask = (seasons == s)
        if s_mask.any() and oof_final[s_mask].any():
            s_brier = brier_score_loss(y[s_mask], oof_final[s_mask])
            print(f"Season {s}: Brier = {s_brier:.5f}")

    # SAVE EVERYTHING
    oof_df = tour[mask].copy()
    oof_df['oof_pred'] = iso.transform(oof_final[mask])
    oof_df.to_csv(os.path.join(DATA_PROCESSED, "oof_preds.csv"), index=False)
    
    joblib.dump(best_weights, os.path.join(MODEL_DIR, "ensemble_weights.pkl"))
    joblib.dump({ 't': best_t, 's': best_s }, os.path.join(MODEL_DIR, "sharpen_params.pkl"))
    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgb_cauchy.pkl"))
    joblib.dump(lgb_model, os.path.join(MODEL_DIR, "lgb_model.pkl"))
    joblib.dump(leaf_model, os.path.join(MODEL_DIR, "leaf_model.pkl"))
    joblib.dump(ohe, os.path.join(MODEL_DIR, "ohe.pkl"))
    joblib.dump(poly, os.path.join(MODEL_DIR, "poly.pkl"))
    joblib.dump(iso, os.path.join(MODEL_DIR, "iso_model.pkl"))
    joblib.dump(glm_strengths, os.path.join(MODEL_DIR, "glm_strengths.pkl"))
    joblib.dump(bt_strengths, os.path.join(MODEL_DIR, "bt_strengths.pkl"))
    joblib.dump(elo_ratings, os.path.join(MODEL_DIR, "elo_ratings.pkl"))
    stats.to_csv(os.path.join(DATA_PROCESSED, "team_stats.csv"), index=False)

    print(f"\nOverall Sharpened OOF Brier: {brier_score_loss(y[mask], oof_final[mask]):.5f}")

if __name__ == "__main__":
    train_model()
