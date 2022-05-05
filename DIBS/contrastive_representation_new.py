import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten, Conv2D,  MaxPool3D, Reshape
import numpy as np
import pandas as pd
import os
import itertools

from tensorflow.keras import Model



def return_vec(x, model):
    '''return contrastive represention.'''
    x = model.conv1(x)
    x = model.maxpool1(x)
    x = model.conv2(x)
    x = model.maxpool2(x)
    x = model.conv3(x)
    x = model.d1(x)
    x = tf.reduce_max(x,[2,3])

    return x.numpy()

def contra_representation(model):

    contra_rep = pd.DataFrame([])
    # load filtered data
    file = open('//home/DHE/yy289/PACE_Home_Drive/yiyuan/filtered_id.txt')
    lines = file.read().split()
    filtered_list = []
    for i in lines:
        filtered_list.append(i)


    for file in filtered_list:
      
        print('parsing:',file)

        # id.append(file)

        filedir = '/home/DHE/yy289/projects/Pro00101670-DIBS/Actigraph Data/Cleaned Actigraph Data'

        df = pd.read_csv(os.path.join(filedir, (file + ' cleaned raw.csv')),skiprows=3) # load actigraphy
        df['Sleep or Awake?'].replace(('W', 'S'), (1, 0), inplace=True) # convert W/S To 1/0  

        participant_id = {'Participant id': [file]}
        participant_id = pd.DataFrame(participant_id)

        # select 5 consecutive days
        complete_day = []
        days = df['Date'].unique()[1:-1] 
        df_excluded = df[df['Date'].isin(days)]
        for i in range(len(days)):
            if detect_na(df_excluded['Axis1'].loc[df_excluded['Date']==days[i]].values.tolist(), num=120):
                complete_day.append(i)
        
        selected_days = list(map(days.__getitem__, select_conse(complete_day)))


        df_excluded = df_excluded[df_excluded['Date'].isin(selected_days)]

        # duplicate df_excluded to have 2880 epochs
        df_excluded = df_excluded.loc[df_excluded.index.repeat(2)]

        repre_all = []
        for i in range(3):

            wake = df_excluded['Sleep or Awake?'].values.astype(float).reshape(5,2880)[i:(i+3),:]
            # concat wake to a batch size that can be transformed by trained model
            batch_list = []
            for j in range(64):
                batch_list.append(np.stack([wake, wake]))

            batch = np.stack(batch_list)[:, :, :, :, tf.newaxis]
            repre_i = return_vec(batch, model)[0,0,:]
            repre_all.append(repre_i)
        repre_avg = pd.DataFrame(np.mean(np.stack(repre_all),axis=0)).T

        # convert to numpy array
        # return trained contrastive representation of the trunk
        representation = pd.concat([participant_id,repre_avg],axis=1)
        contra_rep = pd.concat([contra_rep,representation])

    # save representation to csv file
    contra_rep.to_csv('/home/DHE/yy289/PACE_Home_Drive/yiyuan/contra_representation_new.csv')
    return contra_rep


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



new_model =  keras.models.load_model('/home/DHE/yy289/PACE_Home_Drive/my_model_new')
contra_representation(new_model)
