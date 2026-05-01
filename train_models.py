import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

print("Loading data...")
try:
    matches_df = pd.read_csv("matches.csv")
    points_df = pd.read_csv("points_table.csv")
    batting_df = pd.read_csv("batting_stats.csv")
    bowling_df = pd.read_csv("bowling_stats.csv")
except Exception as e:
    print(f"Error reading datasets: {e}")
    exit()

print("Calculating team stats...")
team_mapping = {
    'Punjab Kings': 'PBKS',
    'Royal Challengers Bengaluru': 'RCB',
    'Sunrisers Hyderabad': 'SRH',
    'Rajasthan Royals': 'RR',
    'Gujarat Titans': 'GT',
    'Chennai Super Kings': 'CSK',
    'Delhi Capitals': 'DC',
    'Kolkata Knight Riders': 'KKR',
    'Mumbai Indians': 'MI',
    'Lucknow Super Giants': 'LSG'
}
points_df['team'] = points_df['team'].map(team_mapping)
team_stats = {}

# Aggregate batting stats
batting_grouped = batting_df.groupby('team')['strike_rate'].mean().reset_index()
batting_grouped.rename(columns={'strike_rate': 'team_batting_sr'}, inplace=True)

# Aggregate bowling stats
bowling_grouped = bowling_df.groupby('team')['economy'].mean().reset_index()
bowling_grouped.rename(columns={'economy': 'team_bowling_eco'}, inplace=True)

for team in team_mapping.values():
    points = points_df[points_df['team'] == team]['points'].values
    nrr = points_df[points_df['team'] == team]['nrr'].values
    sr = batting_grouped[batting_grouped['team'] == team]['team_batting_sr'].values
    eco = bowling_grouped[bowling_grouped['team'] == team]['team_bowling_eco'].values
    
    team_stats[team] = {
        'points': points[0] if len(points) > 0 else 0,
        'nrr': nrr[0] if len(nrr) > 0 else 0.0,
        'batting_sr': sr[0] if len(sr) > 0 else 130.0,
        'bowling_eco': eco[0] if len(eco) > 0 else 8.5
    }

print("Preprocessing data for Match Winner Model...")
# Keep relevant columns
relevant_cols = ['venue', 'team1', 'team2', 'toss_winner', 'toss_decision', 'match_winner']
df = matches_df[relevant_cols].copy()
df = df.dropna()

# Map stats to dataframe
df['team1_points'] = df['team1'].map(lambda x: team_stats.get(x, {}).get('points', 0))
df['team1_nrr'] = df['team1'].map(lambda x: team_stats.get(x, {}).get('nrr', 0))
df['team1_batting_sr'] = df['team1'].map(lambda x: team_stats.get(x, {}).get('batting_sr', 130))
df['team1_bowling_eco'] = df['team1'].map(lambda x: team_stats.get(x, {}).get('bowling_eco', 8.5))

df['team2_points'] = df['team2'].map(lambda x: team_stats.get(x, {}).get('points', 0))
df['team2_nrr'] = df['team2'].map(lambda x: team_stats.get(x, {}).get('nrr', 0))
df['team2_batting_sr'] = df['team2'].map(lambda x: team_stats.get(x, {}).get('batting_sr', 130))
df['team2_bowling_eco'] = df['team2'].map(lambda x: team_stats.get(x, {}).get('bowling_eco', 8.5))

# Encode categorical variables
le_venue = LabelEncoder()
le_team = LabelEncoder()
le_toss_decision = LabelEncoder()

all_teams = pd.concat([df['team1'], df['team2'], df['toss_winner'], df['match_winner']]).unique()
le_team.fit(all_teams)

df['venue'] = le_venue.fit_transform(df['venue'])
df['team1'] = le_team.transform(df['team1'])
df['team2'] = le_team.transform(df['team2'])
df['toss_winner'] = le_team.transform(df['toss_winner'])
df['match_winner'] = le_team.transform(df['match_winner'])
df['toss_decision'] = le_toss_decision.fit_transform(df['toss_decision'])

# Prepare features (X) and target (y)
feature_cols = [
    'venue', 'team1', 'team2', 'toss_winner', 'toss_decision',
    'team1_points', 'team1_nrr', 'team1_batting_sr', 'team1_bowling_eco',
    'team2_points', 'team2_nrr', 'team2_batting_sr', 'team2_bowling_eco'
]
X = df[feature_cols]
y = df['match_winner']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42, min_samples_split=5)
clf.fit(X_train, y_train)

score = clf.score(X_test, y_test)
print(f"Model Test Accuracy: {score * 100:.2f}%")

print("Saving models and encoders...")
os.makedirs("models", exist_ok=True)
joblib.dump(clf, 'models/winner_model.pkl')
joblib.dump(le_venue, 'models/le_venue.pkl')
joblib.dump(le_team, 'models/le_team.pkl')
joblib.dump(le_toss_decision, 'models/le_toss_decision.pkl')
joblib.dump(team_stats, 'models/team_stats.pkl')

print("Data preprocessing and model training completed successfully.")
