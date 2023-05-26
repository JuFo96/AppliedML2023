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

data = statcast(start_dt="2020-05-24", end_dt="2021-06-25")

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
