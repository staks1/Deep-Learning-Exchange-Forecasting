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
from utils import *
from build_multistep_model_2 import *
from data_analysis_3c import *
from sklearn.preprocessing import MinMaxScaler




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
def history_vs_prediction():
   

# add the list of currencies that we want to plot here 
 for cur in ['USD','JPY']:
    
    y_path = glob.glob(os.path.join('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/predictions/scnn/multi_step/slide', cur ,"**/*.csv"),recursive=True)
   
    
    
    for p in y_path :
        #get the frequenxy so i know which dataset to plot with 
        freq = p.split('/')[-3]
        series_length = p.split('/')[-2]
        
        # do we need the horizon ?? here 
        y_model_1 = pd.read_csv(p, header=0, index_col=0).loc[:,'F1':].transpose()
        
        
        # i will pick for history a history of 3*horizons (can also be customized )
        # DOES IT NEED TO BE TRANSPOSED ? 
        history = pd.read_csv(os.path.join('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/dataset',f"{freq}.csv"), 
                              header=0,index_col=0).loc[:,cur][-30:].transpose()
    
        fig = plt.figure()
        plt.plot(history,'o-',color ='green',label='original history')
        plt.plot(y_model_1,'o-',color='purple',label = 'predictions')
        plt.title(f'FORECASTING with currency : {cur} , frequency : {freq}, training_length : {series_length}' )
        plt.legend()
        
        # save images 
        if not (os.path.exists('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/visualizations')):
                os.makedirs('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/visualizations')
        plt.savefig(os.path.join('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/visualizations', 
                                 cur +'_' + freq + '_' + str(series_length) + '.png'))
        plt.close()
        
        
        
      


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
    
    for freq_name, freq_code in frequencies.items():
        data = pd.read_csv(f"/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/dataset/{freq_name}.csv",index_col='Date')
        frequencies[freq_name] = (frequencies[freq_name],data)
        
        
    # plot the predictions against the history 
    history_vs_prediction()
    
    