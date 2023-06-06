# -*- coding: utf-8 -*-
"""
Created on Wed May 24 17:27:49 2023

@author: Julius
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster 
import pandas as pd
from pybaseball import statcast



data = statcast(start_dt="2020-03-24", end_dt="2021-03-25")

pitch_cluster_variables = ["release_speed", "release_pos_x", "release_pos_z", "zone", "vx0", "vy0", "vz0", "launch_speed", "spin_axis"]

#%%
subset_data = data[pitch_cluster_variables].dropna()

cluster_model = cluster.KMeans(n_clusters = 10).fit(subset_data)

labels = cluster_model.labels_
#%%
# Plot the data points with colors representing the clusters
plt.scatter(subset_data.iloc[:, 4], subset_data.iloc[:, 5], c=labels, cmap='viridis')
plt.title('KMeans Clustering')
plt.xlabel(subset_data.columns[4])
plt.ylabel(subset_data.columns[5])
plt.show()


#%%

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

#%%

pitcher_vars = ["pitcher", "inning", "events", "player_name"]

new_data = data[pitcher_vars]

event_vars = ["strikeout", "walk", "home__run", "hit_by_pitch"]
event_sum = 0

for event_vars in data["events"]:
    event_sum += 1
    
print(event_sum, )
    