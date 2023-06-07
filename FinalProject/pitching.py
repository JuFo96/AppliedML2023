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

data = statcast(start_dt="2020-03-24", end_dt="2021-03-25")

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

fip_score = 

