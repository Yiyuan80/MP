import numpy as np
import pandas as pd
import os
import itertools
from matplotlib import pyplot as plt
# from datetime import datetime



def filter_data():
    """Filter days with less than 120 (2 hours) missing 
    and participants with at least 5 consecutive days."""

    filtered_id =[]

    filedir = '/home/DHE/yy289/projects/Pro00101670-DIBS/Actigraph Data/Cleaned Actigraph Data'
    filenames = [f[:-16] for f in os.listdir(filedir) if f[-5:] == 'w.csv']

    for file in filenames:

        print('parsing:',file)

        df = pd.read_csv(os.path.join(filedir, (file + ' cleaned raw.csv')),skiprows=3) # load actigraphy

        days = df['Date'].unique() 

        df_excluded = df[df['Date'].isin(days)]
        
        complete_day = []
        for i in range(len(days)):
            if detect_na(df_excluded['Axis1'].loc[df_excluded['Date']==days[i]].values.tolist(), num=120):
                complete_day.append(i)


        # if len(complete_day)>=6:
        #     filtered_id.append(file)
        if detect_consecutive(complete_day, n=5):
            filtered_id.append(file)

    with open('//home/DHE/yy289/PACE_Home_Drive/yiyuan/filtered_id.txt', "w") as f:
        for row in filtered_id:
            s = "".join(map(str, row))
            f.write(s+'\n')

    print(len(filtered_id))

def detect_na(df, num=120):
    count_dup = [sum(1 for _ in group) for _,group in itertools.groupby(df)]
    # print(max(count_dup))
    if max(count_dup) >= num:
        print(max(count_dup))
        return False
    return True
    
def detect_consecutive(days, n=5):
  """check whether input data contains at least n consecutive days."""
  diff_days = list(np.diff(days)==1)
  runs = [len(list(g)) for _,g in itertools.groupby(diff_days)]
  if len(runs)==0 or max(runs) < n-1:
    return False
  elif max(runs) >= n-1: # test differences between days contain consecutive n-1 
    return True

# filter_data()

# load filtered data
file = open('//home/DHE/yy289/PACE_Home_Drive/yiyuan/filtered_id.txt')
lines = file.read().split()
filtered_list = []
for i in lines:
  filtered_list.append(i) # convert txt file to a list
# print(len(filtered_list))
