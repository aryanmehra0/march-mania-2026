import pandas as pd
import numpy as np

############################################
# POSSESSION FEATURES
############################################

def add_possessions(df):
    df['WPoss'] = df['WFGA'] - df['WOR'] + df['WTO'] + 0.475*df['WFTA']
    df['LPoss'] = df['LFGA'] - df['LOR'] + df['LTO'] + 0.475*df['LFTA']

    df['WOffEff'] = df['WScore'] / df['WPoss']
    df['LOffEff'] = df['LScore'] / df['LPoss']
    return df

def add_four_factors(df):
    df["eFG_W"] = (df["WFGM"] + 0.5 * df["WFGM3"]) / df["WFGA"]
    df["eFG_L"] = (df["LFGM"] + 0.5 * df["LFGM3"]) / df["LFGA"]

    df["TOV_W"] = df["WTO"] / (df["WFGA"] + 0.44*df["WFTA"] + df["WTO"])
    df["TOV_L"] = df["LTO"] / (df["LFGA"] + 0.44*df["LFTA"] + df["LTO"])

    df["ORB_W"] = df["WOR"] / (df["WOR"] + df["LDR"])
    df["ORB_L"] = df["LOR"] / (df["LOR"] + df["WDR"])

    df["FTR_W"] = df["WFTM"] / df["WFGA"]
    df["FTR_L"] = df["LFTM"] / df["LFGA"]

    return df

############################################
# TEAM SEASON AGGREGATES
############################################

def team_season_stats(df):
    wins = df.groupby(['Season','WTeamID']).agg({
        'WOffEff':'mean',
        'eFG_W':'mean',
        'TOV_W':'mean',
        'ORB_W':'mean',
        'FTR_W':'mean'
    }).reset_index()

    losses = df.groupby(['Season','LTeamID']).agg({
        'LOffEff':'mean',
        'eFG_L':'mean',
        'TOV_L':'mean',
        'ORB_L':'mean',
        'FTR_L':'mean'
    }).reset_index()

    wins.columns = ['Season','TeamID','OffEff', 'eFG', 'TOV', 'ORB', 'FTR']
    losses.columns = ['Season','TeamID','DefEff', 'eFG_opp', 'TOV_opp', 'ORB_opp', 'FTR_opp']

    stats = pd.merge(wins, losses,
                     on=['Season','TeamID'],
                     how='outer').fillna(0)

    stats['NetEff'] = stats['OffEff'] - stats['DefEff']
    return stats

def add_conference_strength(df, seeds_df):
    # Join with seeds to get Conference info
    # Ensure no duplicates in seeds_df for (Season, TeamID)
    seeds_dedup = seeds_df[['Season', 'TeamID', 'ConfAbbrev']].drop_duplicates(['Season', 'TeamID'])
    df = df.merge(seeds_dedup, on=['Season', 'TeamID'], how='left')
    
    # Calculate avg NetEff per Conference
    conf_strength = df.groupby(['Season', 'ConfAbbrev'])['NetEff'].mean().reset_index()
    conf_strength.columns = ['Season', 'ConfAbbrev', 'Conf_Strength']
    
    df = df.merge(conf_strength, on=['Season', 'ConfAbbrev'], how='left')
    return df

############################################
# ELO RATING
############################################

def margin_multiplier(margin, elo_diff):
    return (20 * (margin+3)**0.8) / (7.5 + 0.006 * elo_diff)

def dynamic_elo(df, base_k=20):
    ratings = {}
    
    # Store history for slope/trend
    elo_history = [] 

    df = df.sort_values(['Season','DayNum'])

    for _, row in df.iterrows():
        t1, t2 = row['WTeamID'], row['LTeamID']
        margin = row['WScore'] - row['LScore']

        r1 = ratings.get(t1, 1500)
        r2 = ratings.get(t2, 1500)

        # Clip elo_diff to prevent overflow in exponent
        elo_diff = np.clip(r2 - r1, -1000, 1000)
        expected = 1/(1+10**(elo_diff/400))
        
        # Advanced Elo with margin multiplier
        k = base_k * margin_multiplier(margin, r1 - r2)

        r1_new = r1 + k*(1-expected)
        r2_new = r2 - k*(1-expected)

        ratings[t1] = r1_new
        ratings[t2] = r2_new
        
        elo_history.append({
            'Season': row['Season'],
            'DayNum': row['DayNum'],
            'TeamID': t1,
            'Elo': r1_new
        })
        elo_history.append({
            'Season': row['Season'],
            'DayNum': row['DayNum'],
            'TeamID': t2,
            'Elo': r2_new
        })

    return ratings, pd.DataFrame(elo_history)
