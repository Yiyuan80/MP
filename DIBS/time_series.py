import tsfel
import pandas as pd
import os
import numpy as np
import itertools

def select_conse(complete_day):
    for i in complete_day:
        conse = list(range(i,i+5))
        if all(item in complete_day for item in conse):
            return conse

def detect_na(df, num=120):
    count_dup = [sum(1 for _ in group) for _,group in itertools.groupby(df)]
    # print(max(count_dup))
    if max(count_dup) >= num:
        print(max(count_dup))
        return False
    return True

def time_series():

    id_file = open('//home/DHE/yy289/PACE_Home_Drive/yiyuan/filtered_id.txt')
    lines = id_file.read().split()
    filtered_list = []
    for i in lines:
        filtered_list.append(i) # convert txt file to a list

    # Retrieves a pre-defined feature configuration file to extract all available features
    cfg = tsfel.get_features_by_domain()

    time_series = pd.DataFrame([])

    for file in filtered_list:

        print('parsing:',file)

        filedir = '/home/DHE/yy289/projects/Pro00101670-DIBS/Actigraph Data/Cleaned Actigraph Data'
        df = pd.read_csv(os.path.join(filedir, (file + ' cleaned raw.csv')),skiprows=3) # load actigraphy
        df['Sleep or Awake?'].replace(('W', 'S'), (1, 0), inplace=True) # convert W/S To 1/0
        df_summary = pd.read_csv(os.path.join(filedir, (file+' cleaned summary.csv')),skiprows=5)


        # select 5 consecutive days
        complete_day = []
        days = df['Date'].unique()[1:-1] 
        df_excluded = df[df['Date'].isin(days)]
        for i in range(len(days)):
            if detect_na(df_excluded['Axis1'].loc[df_excluded['Date']==days[i]].values.tolist(), num=120):
                complete_day.append(i)
        
        selected_days = list(map(days.__getitem__, select_conse(complete_day)))

        df_excluded = df_excluded[df_excluded['Date'].isin(selected_days)]
        activity = np.array(df_excluded['Axis1']/1931.1)

        X = tsfel.time_series_features_extractor(cfg, activity)
        # Insert id
        X.insert(0,'Participant id',file)
        # Combine features of all the ids
        time_series = pd.concat([time_series,X])

    # save to csv
    time_series.to_csv('/home/DHE/yy289/PACE_Home_Drive/yiyuan/time_series.csv')
    return time_series

time_series()
