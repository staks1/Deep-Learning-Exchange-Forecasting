#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 17:15:48 2023

@author: st_ko
"""

import glob
import keras as ks
import pandas as pd
import numpy as np
from collections import namedtuple
import tensorflow as tf
import datetime as dt
import os
from scipy import stats
#from utilities import *
#from model_definitions import *
from build_model import *
from utils import *
import matplotlib.pyplot as plt 


# function to plot all history  + prediction using subplots 
def visualize(data,predicted_data):
    
    
    # total plots 
    fig, axes = plt.subplots(nrows = 9, ncols=2, figsize=(25, 40))  # 3x4 grid for subplots
    #select currency from dataset 
    
    
    # for each month 
    
    for cur in range(18):
        row = (cur-1) // 2  # Determine row index
        col = (cur-1) % 2
        
        # pick year and currency 
        currency_history = data.iloc[cur-1,:]
        currency = list(data.index)[cur-1]
        currency_predictions = predicted_data.loc[currency,:]
        
        ax = axes[row,col]
        # rotate txt 
        ax.xaxis.set_tick_params(rotation=60)
        
        # 
        ax.plot(currency_history,'o-',color ='green',label='original history')
        ax.plot(currency_predictions,'o-',color='purple',label = 'predictions')
        ax.set_title(f"{currency}")
        
    #general title for whole plot     
    plt.suptitle(f"Time Series Observations for Years", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # close plot 
    #plt.close()






# true values have the "V" column index but the predicted have the "F" so we rename the true values 
# to be able to plot corresponding values more easily  
def mapName(df,freq_name):
    if(freq_name == 'Y'):   
        fat = 'Y'
    elif(freq_name == 'Q'):
        fat = 'Q'
    elif(freq_name == 'M'):
        fat = 'M'
    elif(freq_name == 'W'):
        fat = 'W'
    elif(freq_name == 'D'):
        fat = 'D'
    else:
        fat = 'H'
        
    column_mapping = {f'V{i}': f'V{i-1}' for i in range(1, len(df.columns) + 2)}
    # Rename the DataFrame columns using the mappings
    df.rename(columns=column_mapping, inplace=True)
    column_mapping2 = {f'V{i}': f'F{i}' for i in range(1, len(df.columns) + 1)}
    df.rename(columns=column_mapping2, inplace=True)
    # replace row index to be the same as with our model's predictions 
    custom_index =  [ f'{fat}{i}' for i in range(1,len(df)+1 )]
    df.set_index(pd.Index(custom_index), inplace=True)
    
    return df




R = 'W'   
if(R == 'D'):
    label = "Daily Predictions" 
    horizon = 14
    #y_orig_freq = y_orig.loc[R+'1':R+'4227',:].loc[R+'1',:'F'+str(horizon)]
    #df = pd.read_csv(os.path.join('data', 'Daily-train.csv'), header=0, index_col=0)
    
    # y_path 
    y_path = glob.glob(os.path.join('../predictions','daily','*.csv'))
    y_path = "".join(y_path)
    
    y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
    y_freq_true = pd.read_csv(os.path.join('data', 'Daily-test.csv'), header=0).iloc[:,1:]
    y_freq_true= mapName(y_freq_true,R)
    y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
        
    
elif(R=='W'):
    label = "Weekly Predictions" 
    horizon = 13
    #df = pd.read_csv(os.path.join('data', 'Weekly-train.csv'), header=0, index_col=0)
    #y_orig_freq = y_orig.loc[R+'1':R+'359',:].loc[R+'1',:'F'+str(horizon)]
    y_path = glob.glob(os.path.join('../predictions','weekly','*.csv'))
    
    

    for p in y_path :
        
        # read predictions 
        y_model_1 = pd.read_csv(p, header=0, index_col=0).loc[:,'F1':'F'+str(horizon)]
        
        # let's pick a period of the last 3*horizon values to show as a history for each series  
        # customizable to show more or less historic values depending on the model series_length S
        # for example for bigger series_length models we could have more history
        weekly_history = pd.read_csv(os.path.join('../dataset', 'weekly.csv'), header=0,index_col=0).transpose().iloc[:,-3*horizon:]
        weekly_history = weekly_history.iloc[:,-3 * horizon:]
        
        
        # transpose both dataframes
        #weekly_history = weekly_history.transpose()
        #y_model_1 = y_model_1.transpose()
        
        visualize(weekly_history,y_model_1)
        
        
        
        # create new figure 
        plt.figure()
        for i in range(len(weekly_history)):
            plt.annotate(weekly_history.index[i], (i, weekly_history.values[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center',fontsize=8,rotation = 45)
            
        plt.title(label)
        plt.xticks(rotation = 90)
        plt.xlabel("Time Steps : History + Horizon",weight = 'bold',size = 12)
        plt.ylabel("Time Series Observations",weight= 'bold',size = 12)

        # history values before horizon 
        plt.plot(weekly_history, 'o-',color='blue',label = 'History ')
        # prediction of our model 
        plt.plot(y_model_1, color='green', label='Our Model Predictions')
        
        
        
elif(R=='M'):
    label = "Monthly Predictions" 
    horizon = 18
    y_orig_freq = y_orig.loc[R+'1':R+'48000',:].loc[R+'1',:'F'+str(horizon)]
    df = pd.read_csv(os.path.join('data', 'Monthly-train.csv'), header=0, index_col=0)
    
    y_path = glob.glob(os.path.join('predictions','Monthly','Submission_fc_*Monthly.csv'))
    y_path = "".join(y_path)        
    
    y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
    y_freq_true = pd.read_csv(os.path.join('data', 'Monthly-test.csv'), header=0).iloc[:,1:]
    y_freq_true= mapName(y_freq_true,R)
    y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
    
    
elif(R=='Q'):
    label = "Quarterly Predictions" 
    horizon = 8
    df = pd.read_csv(os.path.join('data', 'Quarterly-train.csv'), header=0, index_col=0)
    y_orig_freq = y_orig.loc[R+'1':R+'24000',:].loc[R+'1',:'F'+str(horizon)]
    
    y_path = glob.glob(os.path.join('predictions','Quarterly','Submission_fc_*Quarterly.csv'))
    y_path = "".join(y_path)        
    
    y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
    y_freq_true = pd.read_csv(os.path.join('data', 'Quarterly-test.csv'), header=0).iloc[:,1:]
    y_freq_true= mapName(y_freq_true,R)
    y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
    
    
else :
    label = "Yearly Predictions" 
    horizon = 6
    df = pd.read_csv(os.path.join('data', 'Yearly-train.csv'), header=0, index_col=0)
    y_orig_freq = y_orig.loc[R+'1':R+'23000',:].loc[R+'1',:'F'+str(horizon)]
    
    y_path = glob.glob(os.path.join('predictions','Yearly','Submission_fc_*Yearly.csv'))
    y_path = "".join(y_path)
    
    y_model_1= pd.read_csv(y_path, header=0, index_col=0).loc[R+'1',:'F'+str(horizon)]
    y_freq_true = pd.read_csv(os.path.join('data', 'Yearly-test.csv'), header=0).iloc[:,1:]
    y_freq_true= mapName(y_freq_true,R)
    y_freq_true = y_freq_true.loc[R+'1',:'F'+str(horizon)]
    

# READ ONLY THE LAST 100 VALUES FROM HISTORY 








# the final data for plotting 
model_final_1 = pd.concat([x_history,y_model_1])
model_final_2 = pd.concat([x_history,y_orig_freq])
model_final_3 = pd.concat([x_history,y_freq_true])




# PLOT PARAMETERS 
plt.title(label)
plt.xticks(rotation = 90)
plt.xlabel("Time Steps : History + Horizon",weight = 'bold',size = 12)
plt.ylabel("Time Series Observations",weight= 'bold',size = 12)

# history values before horizon 
plt.plot(x_history, 'o-',color='blue',label = 'History ')
# prediction of our model 
plt.plot(model_final_1.iloc[-horizon:], color='green', label='Our Model Predictions')
# prediction of the author's models 
plt.plot(model_final_2.iloc[-horizon:], color='red', label="Author's Model Predictions")
#plot the true values 
plt.plot(model_final_3[-horizon:],color='skyblue', label='True Horizon Values')

# add legend
plt.legend()
    
    
    