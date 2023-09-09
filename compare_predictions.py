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
#from build_model import *
#from utils import *
import matplotlib.pyplot as plt 
import glob


# function to plot all history  + prediction using subplots 
def visualize(data,predicted_data,label,save=False):
    
    
    # total plots 
    fig, axes = plt.subplots(nrows = 9, ncols=2, figsize=(25, 40))  # 3x4 grid for subplots
    #select currency from dataset 
    # just for testing 
    # let's only compare the 3 first currencies 
    # ---I COMMENT OUT THE RANGE TO TEST ONLY THE FIRST 3 NOW --- #  
    for cur in range(18):
    #for cur in range(3):
    
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
    plt.suptitle(f"Forecasting with type : {label} ", fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.show()
  






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





    
    
# give 'weekly'/'daily' ... horizon_number to plot history vs predictions 
def history_vs_prediction(label,horizon,save=None):
    #title = label + " Predictions" 
    horizon = horizon
    # path of the different models for this frequency 
    y_path = glob.glob(os.path.join('../predictions', label  ,"*.csv"))
    
    for p in y_path :
        
        # read predictions 
        y_model_1 = pd.read_csv(p, header=0, index_col=0).loc[:,'F1':'F'+str(horizon)]
        # i will pick for history a history of 3*horizons (can also be customized )
        history = pd.read_csv(os.path.join('../dataset',f"{label}.csv"), header=0,index_col=0).transpose().iloc[:,-3*horizon:]
        #history = history.iloc[:,-3 * horizon:]
        visualize(history,y_model_1,label)
        
        
      


# run as main 
if __name__=="__main__":
    # history_vs_prediction('weekly',13)
    #history_vs_prediction('daily',14)
    
    
    
    # Let's now read all the different frequency - datasets 
    # for each frequency we take the end of the month 
    frequencies = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q',
        'yearly': 'Y'
    }
    
    
    # Create the output folder if it does not exist
    # also make generator for later to read each dataset sequentially 
    # add to dictionary the datasets 

    for freq_name, freq_code in frequencies.items():
        data = pd.read_csv(f"/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/dataset/{freq_name}.csv",index_col='Date')
        frequencies[freq_name] = (frequencies[freq_name],data)
        
        
    # let's only test the 3 first currencies 
    
    data = frequencies['daily'][1].transpose()
    
    
    
    
    # for daily series 
    horizon = 14
    
    
    # path of the different models for this frequency 
    # here we will read all the series length models for each currency 
    # let's pick ud first 
    currency = 'USD'
    freq_name = 'daily'
    
    # pick all the series_length models for this currency and this freq_name 
    y_path = glob.glob(f"../predictions/multi_step/sliding/{currency}/{freq_name}/*/*.csv",recursive=True)
    
    
    for p in y_path :
        
        # let's get the series_length from the glob path 
        series_length = p.split("/")[-2]
        
        # read predictions 
        #y_model_1 = pd.read_csv(p, header=0, index_col=0).loc[:,'F1':'F'+str(horizon)].transpose()
        
        y_model_1 = pd.read_csv(p, header=0, index_col=0).loc[:,:].transpose()
        # i will pick for history a history of 3*horizons (can also be customized )
        # or maybe i will plot (series_length + horizon of the history)
        history = pd.read_csv(os.path.join('../dataset',f"{freq_name}.csv"), header=0,index_col=0).transpose().iloc[:,-(3*int(series_length)):]
        #history = history.iloc[:,-3 * horizon:]
        #visualize(history,y_model_1,freq_name)
        
        
        # we will now pick only the currency we are interested in to plot 
        history = history.loc[currency,:].transpose()
        
    
        #-------------------------------------#
        
        fig = plt.figure()
        plt.xticks(rotation = 60)
        plt.plot(history,'o-',color ='green',label='original history')
        plt.plot(y_model_1,'o-',color='purple',label = 'predictions')
        plt.title(f"FORECASTING, CURRENCY : {currency} , FREQUENCY : {freq_name} , TRAINING LENGTH : {series_length}")
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    