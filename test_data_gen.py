#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 14:35:14 2023

@author: st_ko
"""


import keras as ks
import pandas as pd
import numpy as np
#import metrics
from collections import namedtuple
import tensorflow as tf
import datetime as dt
import os
import tensorflow as tf
from build_multistep_model_2 import *
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from data_analysis_3c import * 
from keras.utils import Sequence


# class sequence 
class CustomDataGen(Sequence):
    def __init__(self, data, batch_size,series_length,horizon):
        # feed the smoothed dataframe here
        self.data = data
        self.batch_size = batch_size
        self.series_length = series_length
        self.horizon = horizon
        # construct a list of all the horizon values 
        self.labels = []
        for i in range(len(self.data)//self.series_length):
            temp = self.data[self.series_length * i : self.series_length * (i+1) + self.horizon]
            self.labels.append(temp[-horizon:])
        self.labels = np.array(self.labels).reshape((-1,1))
             
            
        
    # return number of batches 
    # actually we want a size of a batch to be batch_size  * series_length 
    def __len__(self):
        return len(self.data) // (self.series_length * self.batch_size)
    

    # get the next item from the generator 
    # actually we want each x to be [idx * batch_size : (idx + series_length) * batch_size]
    # 0:2*364 
    # 2* 364:2* 2 * 364
    def __getitem__(self, idx):
        
        # pick batch_x , batch_y
        batch_x = self.data[idx*(self.batch_size * self.series_length) :idx * (self.batch_size*self.series_length) + self.batch_size*self.series_length]
        # let's keep the batch only if it is of size = series_length else we throw it away
        # also we throw away the y if the last batch is less than the series_length
        # if the last batch is > 300 (of course not a integer factor ) , we throw away the remaining part to keep 300 
        if(len(batch_x) < self.batch_size * self.series_length):
    
            if(len(batch_x) > self.series_length ):
                batch_x = batch_x[:self.series_length]
                #batch_y = self.labels[idx : idx + self.horizon]
                batch_y = self.labels[idx*self.batch_size * self.horizon : idx * self.batch_size*self.horizon + self.batch_size * self.horizon].reshape((-1,self.horizon))
                
                # reshape into tensor
                batch_x = np.array(batch_x).reshape((-1,self.series_length))
                return (batch_x,batch_y)
            
            
            elif(len(batch_x) ==300):
                #batch_x = batch_x
                batch_x = np.array(batch_x).reshape((-1,self.series_length))
                batch_y = self.labels[idx*self.batch_size * self.horizon : idx * self.batch_size*self.horizon + self.batch_size * self.horizon].reshape((-1,self.horizon))
                
                return (batch_x,batch_y)
            
            else: 
                # what should we return if batch size is smaller ???  # 
                # TODO : CHECK IF THIS WORKS IN THE TRAINING ALGORITHM 
                raise StopIteration
        else :
            batch_x = np.array(batch_x).reshape((-1,self.series_length))
            batch_y = self.labels[idx*self.batch_size * self.horizon : idx * self.batch_size*self.horizon + self.batch_size * self.horizon].reshape((-1,self.horizon))
            return (batch_x,batch_y)
        

                
        
    
    
    
if __name__ == "__main__":
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
        
    
    
   
    #-------------- BEGIN TRAINING MULTI STEP SLIDING WINDOW, ONE MODEL PER CURRENCY---#
    #slide_train_multi_step_one(frequencies)
    #----------------------------------------------------------------------------------#
    
    ############################################
    # implementing exponential smoothing + cnn #
    ############################################
    
    #-------------let's do it for the daily series------------------------------------#
    # 0) first we only keep the data from 2013 onwards (2013-2023)
    horizon  = 14
    dataset = frequencies['daily'][1].loc['2013-01-07':]
    # we pick close to 80% of the dataset for training 
    # 0.8 * 2729 ~= 2183 , i also subtract the last 3 values since they are at the start if the next week 
    # here we will give just one currency 
    train_dataset = dataset.iloc[:2180]['USD']
    # and 20% for testing (start at the start of the week) and inlcude all data until (-horizon) since we want the y true values of the horizon
    test_dataset = dataset.iloc[2180:-horizon]['USD']
    
    a = CustomDataGen(train_dataset, 2,300, 14)
    x,y = a.__getitem__(0)
    #number_batches = a.__len__()
    #batch_size = 2
    #series_length = 10
    #currency_size = 17
    #k4 = a.__getitem__(3)
    a=iter(a)
    b = next(a)
    #c = a.__next__()
    #d = a.__next__()
    #e = a.__next__()
    print("hello")
    for x in a :
        print(x)
    