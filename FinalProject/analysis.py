#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 24 10:14:45 2023

@author: jufo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pybaseball import statcast, lahman, cache

cache.enable()

data = statcast(start_dt="2021-05-24", end_dt="2021-06-25")

# https://github.com/robotallie/baseball-injuries/blob/master/injuries.csv
injuries = pd.read_csv("injuries.csv")

salary = lahman.salaries()

bark_mask = salary["playerID"] == "kershcl01"
top_salary = salary["salary"] == np.max(salary["salary"])
injury_mask = injuries["Injury_Type"] != "unknown"



#print(salary[top_salary])
#print(salary[bark_mask])
injury_types = injuries[injury_mask]["Injury_Type"]
unique_injury_types = set(injury_types)
#plt.plot(salary[bark_mask]["yearID"],salary[bark_mask]["salary"])

arm_injuries = injuries[injuries["Injury_Type"].str.contains("arm")]

# Pitcher