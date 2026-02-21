import os
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb

from config import *

# Load models & transformers
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

# Load stats
stats_path = os.path.join(DATA_PROCESSED, "team_stats.csv")
stats = pd.read_csv(stats_path)

def predict_matchup(season, team1_id, team2_id, mode='optimized'):
    t1_stats = stats[(stats['Season'] == season) & (stats['TeamID'] == team1_id)]
    t2_stats = stats[(stats['Season'] == season) & (stats['TeamID'] == team2_id)]

    if t1_stats.empty or t2_stats.empty:
        return "Stats not found."

    features = ['Net_diff', 'Conf_diff', 'Elo_diff', 'GLM_diff', 'BT_diff']
    X = pd.DataFrame({
        'Net_diff': [t1_stats['NetEff'].values[0] - t2_stats['NetEff'].values[0]],
        'Conf_diff': [t1_stats['Conf_Strength'].values[0] - t2_stats['Conf_Strength'].values[0]],
        'Elo_diff': [elo_ratings.get(team1_id, 1500) - elo_ratings.get(team2_id, 1500)],
        'GLM_diff': [glm_strengths.get(season, {}).get(team1_id, 0) - glm_strengths.get(season, {}).get(team2_id, 0)],
        'BT_diff': [bt_strengths.get(season, {}).get(team1_id, 0) - bt_strengths.get(season, {}).get(team2_id, 0)]
    })

    X_poly = pd.DataFrame(poly.transform(X), columns=poly.get_feature_names_out(features))

    dtest = xgb.DMatrix(X_poly)
    p_xgb = 1 / (1 + np.exp(-xgb_cauchy.predict(dtest) / 10))[0]
    p_lgb = lgb_model.predict_proba(X_poly)[:,1][0]
    
    leaf_indices = lgb_model.predict(X_poly, pred_leaf=True)
    p_leaf = leaf_model.predict_proba(ohe.transform(leaf_indices))[:,1][0]

    p_blend = best_weights[0]*p_xgb + best_weights[1]*p_lgb + best_weights[2]*p_leaf
    p_calibrated = iso_model.transform([p_blend])[0]
    
    t, s = sharpen_params['t'], sharpen_params['s']
    if mode == 'defensive':
        t, s = t * 0.95, 0.90
    elif mode == 'aggressive':
        t, s = t * 1.05, 0.95
    
    p_temp = 1 / (1 + np.exp(-t * (p_calibrated - 0.5) / 0.1))
    p_final = 0.5 + s * (p_temp - 0.5)
    
    return p_final

if __name__ == "__main__":
    available_seasons = sorted(stats['Season'].unique())
    test_season = available_seasons[-1]
    season_teams = stats[stats['Season'] == test_season]['TeamID'].unique()
    
    if len(season_teams) >= 2:
        t1, t2 = season_teams[0], season_teams[1]
        print(f"Matchup: {t1} vs {t2} in {test_season}")
        print(f"  - Optimized:  {predict_matchup(test_season, t1, t2, 'optimized'):.4f}")
        print(f"  - Defensive:  {predict_matchup(test_season, t1, t2, 'defensive'):.4f}")
        print(f"  - Aggressive: {predict_matchup(test_season, t1, t2, 'aggressive'):.4f}")
    else:
        print("Not enough team data.")
