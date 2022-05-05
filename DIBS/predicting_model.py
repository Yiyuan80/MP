import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier


# load outcomes
outcome_var = ['Participant ID','CBCL_External_T','CBCL_Internal_T','YSR_External_T','YSR_Internal_T','PROMIS_SD_Child_T','PROMIS_SD_Parent_T','PROMIS_SRI_Child_T','PROMIS_SRI_Parent_T']
outcome_variables = ['CBCL_External_T','CBCL_Internal_T','YSR_External_T','YSR_Internal_T','PROMIS_SD_Child_T','PROMIS_SD_Parent_T','PROMIS_SRI_Child_T','PROMIS_SRI_Parent_T']
outcome_variables_all = ['CBCL_External_T','CBCL_Internal_T','YSR_External_T','YSR_Internal_T','PROMIS_SD_Child_T','PROMIS_SD_Parent_T','PROMIS_SRI_Child_T','PROMIS_SRI_Parent_T','Age','BMI','Gender']
# outcome_variables = ['CBCL_External_T','YSR_External_T','PROMIS_SD_Child_T','PROMIS_SD_Parent_T','PROMIS_SRI_Parent_T']
outcome_var_add = ['Participant ID','Height','Weight','Sex','Age - Years','Body Mass Index >95th percentile for age and gender']
outcome = pd.read_csv('/home/DHE/yy289/projects/Pro00101670-DIBS/20201028_feature_engineered_master_dataset.csv')
outcome_add = pd.read_csv('/home/DHE/yy289/projects/Pro00101670-DIBS/master_analysis_db.csv')
bmi_percentile = pd.read_csv('/home/DHE/yy289/projects/Pro00101670-DIBS/dibs_bmi_percentiles.csv')
bmi_percentile = bmi_percentile[['Participant ID','85th Percentile BMI Value']]
outcome = outcome[outcome_var]
outcome_add = outcome_add[outcome_var_add]
outcome_add['Sex'].replace(('Female','Male'),(1,0),inplace=True)
outcome_add['Body Mass Index >95th percentile for age and gender'].replace(('Yes','No'),(1,0),inplace=True)
outcome_add['BMI']=outcome_add['Weight']/((outcome_add['Height']/100)**2)
outcome = pd.merge(outcome,outcome_add,how='left',on='Participant ID')
outcome = pd.merge(outcome,bmi_percentile,how='left',on='Participant ID')
outcome = outcome.rename(
    columns = {'Participant ID':'Participant id',
    'Sex':'Gender',
    'Age - Years':'Age',
    'Body Mass Index >95th percentile for age and gender':'Obesity'}
)

# load contrastice representations
contra = pd.read_csv('/home/DHE/yy289/PACE_Home_Drive/yiyuan/contra_representation_new.csv',usecols=range(1,34))
# load indices
indices = pd.read_csv('//home/DHE/yy289/PACE_Home_Drive/yiyuan/indices.csv',usecols=range(1,8))
# load time series
time_series = pd.read_csv('/home/DHE/yy289/PACE_Home_Drive/yiyuan/time_series.csv',usecols=range(1,392))
# merge data 
df = pd.merge(indices, time_series, how='left', on='Participant id')
df = pd.merge(df, contra, how='left', on='Participant id') 
df = pd.merge(df, outcome, how='left', on='Participant id')
# print(df.iloc[:,397:429])

# dichotomize CBCL YSR by 65th percentile
for i in outcome_variables:
    cut_i = np.percentile(df[i],65)
    # print(i,cut_i)
    min_i = df[i].min()
    max_i = df[i].max()
    df[i]=pd.cut(df[i],bins=[min_i-1,cut_i,max_i+1],labels=[0,1]).astype(float).astype("Int64")

    # print(df[i].value_counts())
# dichotomize age by median
age_min = df['Age'].min()
age_median = df['Age'].median()
age_max = df['Age'].max()
df['Age']=pd.cut(df['Age'],bins=[age_min-1,age_median,age_max+1],labels=[0,1]).astype(float).astype("Int64")
df['BMI']=np.where(df['BMI']>df['85th Percentile BMI Value'],1,0)


def predicting_model(df, outcome_variables, model=LogisticRegressionCV(penalty='l2',random_state=0,max_iter=100, cv=15)):

    # roc_auc = pd.DataFrame([])
    auc_all = {'6 indices':[],
            'time_series':[],
            'contra':[]}
 

    for i in outcome_variables:
        
        Y = df[i].astype(np.int8).to_numpy()

        X1 = df.iloc[:,1:7].to_numpy()
        X2 = df.iloc[:,7:397].to_numpy()
        X3 = df.iloc[:,397:429].to_numpy()

        auc_1 = []
        auc_2 = []
        auc_3 = [] 

        all_prob1 = []
        all_prob2 = []
        all_prob3 = []

        all_y = []

        loo = LeaveOneOut()
        
        for train_index, test_index in loo.split(X1):
            print("TRAIN:", train_index, "TEST:", test_index)
            X1_train, X1_test = X1[train_index], X1[test_index]
            X2_train, X2_test = X2[train_index], X2[test_index]
            X3_train, X3_test = X3[train_index], X3[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            print(Y_test)
            
            all_prob1.append(model.fit(X1_train, Y_train).predict_proba(X1_test)[:,1])
            all_prob2.append(model.fit(X2_train, Y_train).predict_proba(X2_test)[:,1])
            all_prob3.append(model.fit(X3_train, Y_train).predict_proba(X3_test)[:,1])
            
            all_y.append(Y_test[0])

        all_y = np.array(all_y)
        all_prob1 = np.array(all_prob1)
        all_prob2 = np.array(all_prob2)
        all_prob3 = np.array(all_prob3)

        fpr1, tpr1, thresholds1 = roc_curve(all_y,all_prob1)
        fpr2, tpr2, thresholds2 = roc_curve(all_y,all_prob2)
        fpr3, tpr3, thresholds3 = roc_curve(all_y,all_prob3)

        print(fpr1, tpr1, thresholds1) #For validation
        roc_auc1 = auc(fpr1, tpr1)
        roc_auc2 = auc(fpr2, tpr2)
        roc_auc3 = auc(fpr3, tpr3)

        auc_all['6 indices'].append(roc_auc1)

        auc_all['time_series'].append(roc_auc2)

        auc_all['contra'].append(roc_auc3)
            

    auc_all = pd.DataFrame(auc_all)
    auc_all.index = outcome_variables

    print(auc_all) 
