# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:27:49 2023

@author: Julius
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pybaseball import statcast, cache
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

cache.enable()

data = statcast(start_dt="2022-03-24", end_dt="2023-03-25")

#%%

""" #FIP scoring - not really ML
pitcher_list = data["pitcher"].unique()
fip_list = []


def calc_fip(dataframe, pitcher):
    data_pitcher = dataframe[dataframe["pitcher"] == pitcher]    
    innings_pitched = data_pitcher ['inning'].max()
    strikeouts = data_pitcher ['events'].str.contains('strikeout').sum()
    walks = data_pitcher ['events'].str.contains('walk').sum()
    home_runs = data_pitcher ['events'].str.contains('home_run').sum()
    hit_by_pitch = data_pitcher['events'].str.contains('hit_by_pitch').sum()
    return ((13*home_runs) + (3 * (walks + hit_by_pitch)) - (2 * strikeouts)) / innings_pitched

data_pitcher = data[data["pitcher"] == 593576]

for pitcher in pitcher_list:
    fip_list.append((calc_fip(data, pitcher), pitcher, data[data["pitcher"] == pitcher]["player_name"].iloc[0]))
    
pitcher_df = pd.DataFrame(fip_list)
pitcher_df.columns = ["FIP", "ID", "Name"]
    
best_pitcher = pitcher_df[pitcher_df['FIP'] == pitcher_df['FIP'].min()]
"""
#%%

pitch_variables = ['release_speed', 'release_pos_x', 'release_pos_z', 'zone', "p_throws", 'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'vx0',
 'vy0', 'vz0', 'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'effective_speed', 'release_spin_rate', 'release_extension',
 'release_pos_y', 'spin_axis', "description"]

label_encoder = LabelEncoder()
pitching_data = data[pitch_variables].dropna()  

pitching_data["description"] = [label if label in ["ball", "hit_into_play", "called_strike", "foul", "swinging_strike", "blocked_ball"] else "other" for label in pitching_data["description"]]

# Create data and target variable
X = pitching_data.loc[:, pitching_data.columns != "description"]
X["p_throws"] = label_encoder.fit_transform(X["p_throws"])
y = label_encoder.fit_transform(pitching_data['description'])

# Train Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Plotting to make sure that there is no bias in dropped NaN's
value_counts_raw = data["description"].value_counts()[:7]
value_counts_dropped = pitching_data["description"].value_counts()[:7]

# Plot histogram to see no bias drop
plt.figure(figsize = (10, 5))
plt.subplot(1, 1, 1)
plt.bar(value_counts_raw.index, value_counts_raw.values)
plt.xlabel('Values')
plt.ylabel('Counts')
plt.title('Histogram of Raw Value Counts')


#%%


train_data = lgb.Dataset(X_train.astype(float), label=y_train)
class_labels = label_encoder.inverse_transform(np.arange(7))  


params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y_train)),
    'metric': 'multi_logloss'
}

# Train the LightGBM model
model = lgb.train(params, train_data)

# Make predictions on the test set
y_pred = model.predict(X_test.astype(float))
y_pred_class = y_pred.argmax(axis=1)
y_pred_train = model.predict(X_train.astype(float))

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_class)
classification = classification_report(y_test, y_pred_class, target_names = class_labels)

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("Classification Report:\n", classification)


# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred_class)


# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels = class_labels, yticklabels = class_labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

lgb.plot_importance(model)
#%%
# Mapping of score labels to their respective values
score_mapping = {
    'ball': 5,
    'blocked_ball': 5,
    'called_strike': 8,
    'foul': 5,
    'hit_into_play': 6,
    'other': 6,
    'swinging_strike': 8
}

# Convert the score_mapping dictionary to a score_vector array
score_vector = np.array([score_mapping[key] for key in score_mapping])

# Calculate the sample scores for the test and train sets
sample_score_test = np.dot(y_pred, score_vector)
sample_score_train = np.dot(y_pred_train, score_vector)

# Combine the test and train sets and assign pitch scores
X_test["pitch_score"] = sample_score_test
X_train["pitch_score"] = sample_score_train
pitch_score_combined = pd.concat([X_test, X_train], axis=0, ignore_index=True)
data["pitch_score"] = pitch_score_combined["pitch_score"]

# Group the data by player name
grouped = data.groupby("player_name")

# Calculate the size of each group (number of occurrences per player)
group_counts = grouped.size()

# Filter groups based on a minimum count threshold (e.g., 300)
filtered_groups = group_counts[group_counts >= group_counts.mean()/2]

# Filter the data based on the player names in filtered_groups
filtered_df = data[data['player_name'].isin(filtered_groups.index)]

# Group the filtered data by player name
group_player_filter = filtered_df.groupby(["player_name"])

# Calculate the mean pitch score for each player
pitch_score_player = group_player_filter["pitch_score"].mean()

# Sort the players based on their mean pitch scores in descending order
sorted_series = pitch_score_player.sort_values(ascending=False)

# Print the sorted series of players and their mean pitch scores
print(sorted_series)

#%%

# Convert the 'date' column to datetime type
data['date'] = pd.to_datetime(data['game_date'])

# Reference date (day 0)
reference_date = data["date"].min()

# Calculate the number of days since the reference date
data['days_since_day0'] = (data['date'] - reference_date).dt.days

avg_score_player_game_day = data[data["player_name"] == "Paxton, James"].groupby("days_since_day0").mean()["pitch_score"]
pitches_in_one_game = data[data["player_name"] == "Paxton, James"].groupby("game_date")["pitch_score"]

mask = (data["player_name"] == "Paxton, James") & (["days_since_day0"] == 40)


plt.figure()
plt.title("Gerret Cole")
plt.ylabel("Pitch Score TM")
plt.xlabel("days since season start")
plt.ylim(5, 7)
plt.plot(avg_score_player_game_day, 'o')
