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
# for future data 
def visualize_future_horizon(curs):
   


# add the list of currencies that we want to plot here 
 for cur in curs:
    
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
                              header=0,index_col=0).loc[:,cur][-1000:].transpose()
    
        fig = plt.figure()
        plt.plot(history,'o-',color ='green',label='original history')
        plt.plot(y_model_1,'o-',color='purple',label = 'predictions')
        plt.title(f'FORECASTING with currency : {cur} , frequency : {freq}, training_length : {series_length}' )
        plt.xticks(rotation=90)
        plt.legend()
        
        # save images 
        if not (os.path.exists('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/visualizations/scnn/multi_step/slide')):
                os.makedirs('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/visualizations/scnn/multi_step/slide')
        plt.savefig(os.path.join('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/visualizations/scnn/multi_step/slide', 
                                 cur +'_' + freq + '_' + str(series_length) + '.png'))
        plt.close()
        
        
# for already existent data (plot predictions vs true data)
def visualize_past_horizon(frequencies):
    
    
    
    #------------------ DATA FOR EVALUATION -------------------------#
    # from 2009 but i plot from 2010 just for observation 
    dseries = pd.DataFrame(frequencies['daily'][1].loc['2009-12-31':])
    wseries = pd.DataFrame(frequencies['weekly'][1].loc['2009-12-27':])
    mseries = pd.DataFrame(frequencies['monthly'][1].loc['2009-12-31':])
    qseries = pd.DataFrame(frequencies['quarterly'][1].loc['2009-12-31':])
    yseries = pd.DataFrame(frequencies['yearly'][1].loc['2009-12-31':])
    #------------------------------------------------------------------#
 
    # Begin by selecting 20% of the datasets for each different frequency 
    # create a dictionary to retrieve each series for prediction , it also has the the horizon so we know how many values to plot/visualize
    # in general we plot the actual values vs predicted horizon values 
    
    pred_dict = {'daily' : (dseries, 14),
                 'weekly' : (wseries,13),
                 'monthly' : (mseries,18),
                 'quarterly' : (qseries,8),
                 'yearly' : (yseries,6) }
    
    
    
    # define the currencies
    series = frequencies['daily'][1]
    curs = series.columns
    

    for cur in curs:
       
       y_path = glob.glob(os.path.join('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/predictions/scnn/multi_step/slide', cur ,"**/*.csv"),recursive=True)
      
       
       
       for p in y_path :
           #get the frequenxy so i know which dataset to plot with 
           freq = p.split('/')[-3]
           series_length = p.split('/')[-2]
           
         
           # i will pick for history a history of 3*horizons (can also be customized )
           # DOES IT NEED TO BE TRANSPOSED ? 
           horizon = pred_dict[freq][1]
           history = pred_dict[freq][0][cur][: horizon * 2].transpose()
           selected_index = history[:horizon].index 
           
           
           # read model predictions and set the predictions index same as the data index to align the true values with the predictions 
           y_model_1 = pd.read_csv(p, header=0, index_col=0).iloc[:,: horizon].transpose()
           y_model_1.index = selected_index
           
           
       
           fig = plt.figure()
           plt.plot(history,'o-',color ='green',label='original history')
           plt.plot(y_model_1,'o-',color='purple',label = 'predictions')
           plt.title(f'FORECASTING with currency : {cur} , frequency : {freq}, training_length : {series_length}' )
           plt.xticks(rotation=90)
           plt.legend()
           
           # save images 
           if not (os.path.exists('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/visualizations/scnn/multi_step/slide')):
                   os.makedirs('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/visualizations/scnn/multi_step/slide')
           plt.savefig(os.path.join('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/visualizations/scnn/multi_step/slide', 
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
        
        
    #-------------- PLOT UNKNOWN FUTURE VALUES -----------------#
    #visualize_future_horizon(frequencies['daily'][1].columns)
    
    #-------------- PLOT EVALUATION PLOTS, PREDICTED DATA VS ACTUAL DATA-----#
    visualize_past_horizon(frequencies)
    