import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import itertools

def normalize_act(filedir,id_list):
    """Calculate mean/std of activity data."""
    all_act = []

    for file in id_list:
        print('parsing:',file)

        df = pd.read_csv(os.path.join(filedir, (file + ' cleaned raw.csv')),skiprows=3) # load actigraphy

        days = df['Date'].unique()[0:-1] # exclude last day

        df_excluded = df[df['Date'].isin(days)]
        
        complete_day = []
        for i in range(len(days)):
            if detect_na(df_excluded['Axis1'].loc[df_excluded['Date']==days[i]].values.tolist(), num=120):
                complete_day.append(days[i])
        df_complete = df[df['Date'].isin(complete_day)]
        act = df_complete['Axis1'].values.tolist()
        all_act = all_act + act
    # calculate mean and std
    all_act = np.array(all_act)
    mean_act = np.mean(all_act)
    std_act = np.std(all_act)

    return mean_act, std_act, all_act

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

def data_indices():

    id =[]
    total_sleep = []
    SRI = []
    midpoint = []
    waso = []
    mean_activity = []
    sd_activity = []

    # load filtered data
    file = open('//home/DHE/yy289/PACE_Home_Drive/yiyuan/filtered_id.txt')
    lines = file.read().split()
    filtered_list = []
    for i in lines:
        filtered_list.append(i) # convert txt file to a list

    
    filenames = [f[:-16] for f in os.listdir(filedir) if f[-5:] == 'w.csv']

    for file in filtered_list:

        print('parsing:',file)

        id.append(file)
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


        # convert sleep and activity to numpy arraies
        sleep = df_excluded['Sleep or Awake?'].to_numpy()
        activity = np.array(df_excluded['Axis1']/1931.1)

        # TST
        ttl_sleep = df_excluded['Sleep or Awake?'].sum()
        ttl_ep = len(days)*1440
        aver_total_sleep = (1-float(ttl_sleep)/float(ttl_ep))*1440 # compute the average total sleep time
        total_sleep.append(aver_total_sleep)

        # SRI
        sleep_reg = sleep_reg_index(sleep, epoch = 1440)
        SRI.append(sleep_reg)

        # Midpoint
        midpoint.append(mdpt(sleep,start=0))

        # Wake after sleep onsite 
        ## Can I just take the average of waso in summary file?
        waso_index = df_summary['Wake After Sleep Onset (WASO)'].mean()
        waso.append(waso_index)

        # activity mean and sd
        mean_act, sd_act = mean_sd_activity(activity)
        mean_activity.append(mean_act)
        sd_activity.append(sd_act)

    indices = {
        'Participant id': id,
        'total_sleep': total_sleep,
        'SRI': SRI,   
        'sleep_midpoint': midpoint,
        'waso': waso,
        'mean_activity': mean_activity,
        'sd_activity': sd_activity
    }
    indices = pd.DataFrame(indices)
    indices.to_csv('//home/DHE/yy289/PACE_Home_Drive/yiyuan/indices.csv')
    print(indices)


def mdpt(sleep,start=0):
    '''Circular mean:

    Note that sleep==1 -> sleep, sleep==0 -> wake''' 

    sleep_mat = np.reshape(sleep,(1440,-1),order='F') 

    cosines = np.expand_dims(np.cos(np.arange(1440)*2*np.pi/1440),axis=1)

    sines = np.expand_dims(np.sin(np.arange(1440)*2*np.pi/1440),axis=1) 
    
    tm = 1440*np.arctan2(np.nansum(sines*sleep_mat),np.nansum(cosines*sleep_mat))/np.pi 

    return (tm+start)%1440


def sleep_reg_index(sleep, epoch = 1440):
    """Calculate SRI."""
    sleep_arr = sleep.reshape(-1,1440)
    diff = np.array([], dtype=np.int64).reshape(0, 1440)
    for i in range(sleep_arr.shape[0]-1):
        diff = np.concatenate((diff, (sleep_arr[i,:] == sleep_arr[i+1,:]).astype(int).reshape(-1,1440)))
    return float(np.sum(diff)*200) / float(((sleep_arr.shape[0]-1) * epoch)) - 100.0
       
def mean_sd_activity(activity):
    """Calculate mean and sd of activity."""
    activity = np.nan_to_num(activity) # replace nan with 0
    return activity.mean(),activity.std()


data_indices()  
