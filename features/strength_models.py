import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression

def compute_glm_strength(df):
    teams = pd.unique(df[["WTeamID","LTeamID"]].values.ravel())
    team_map = {team:i for i,team in enumerate(teams)}

    rows = []
    targets = []

    for _, row in df.iterrows():
        vec = np.zeros(len(teams))
        vec[team_map[row["WTeamID"]]] = 1
        vec[team_map[row["LTeamID"]]] = -1
        rows.append(vec)
        targets.append(row["WScore"] - row["LScore"])

    X_glm = np.array(rows)
    y_glm = np.array(targets)

    # Ridge removes schedule bias via regularization
    model = Ridge(alpha=10)
    model.fit(X_glm, y_glm)

    strengths = {team: model.coef_[team_map[team]] for team in teams}
    return strengths

def compute_bt_strength(df):
    teams = pd.unique(df[["WTeamID","LTeamID"]].values.ravel())
    team_map = {team:i for i,team in enumerate(teams)}

    rows = []
    targets = []

    for _, row in df.iterrows():
        # Win observation
        vec_w = np.zeros(len(teams))
        vec_w[team_map[row["WTeamID"]]] = 1
        vec_w[team_map[row["LTeamID"]]] = -1
        rows.append(vec_w)
        targets.append(1)
        
        # symmetric loss observation (essential for LogisticRegression to see 2 classes)
        vec_l = np.zeros(len(teams))
        vec_l[team_map[row["LTeamID"]]] = 1
        vec_l[team_map[row["WTeamID"]]] = -1
        rows.append(vec_l)
        targets.append(0)

    X_bt = np.array(rows)
    y_bt = np.array(targets)

    # Bradley-Terry captures probabilistic strength
    model = LogisticRegression(max_iter=1000)
    model.fit(X_bt, y_bt)

    strengths = {team: model.coef_[0][team_map[team]] for team in teams}
    return strengths
