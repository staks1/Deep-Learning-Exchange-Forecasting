#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 01:44:26 2023

@author: st_ko
"""

import os 
import matplotlib.pyplot as plt
import pandas as pd



# standardize values 
# takes shape of [observations,series]
# returns shape of [series,observations]
def standardize(data):
    standardized = []    
    # save the old mean, std to destandardize later 
    means = []
    stds = []
    
    
    for col in data.columns:
        stdc = data[col].std()
        meanc = data[col].mean()
        # append the means/stds
        means.append(meanc)
        stds.append(stdc)
        
        standardized.append( (data[col] - meanc )/ stdc)
        
    # transpose to the original form 
    standardized_data = pd.DataFrame(standardized)
    return standardized_data , means , stds 




# destandardize to get the original data 
# takes shape of [series,observations]
# returns shape of [series,observations]
def destandardize(data,means,stds):
    # get the original orientation 
    data = data.transpose()
    
    destandardized = [] 
    
    for i,col in enumerate(data.columns):
        destandardized.append(data[col]*stds[i] + means[i])
        
    destandardized_data = pd.DataFrame(destandardized)
        
    #standardized_data = pd.DataFrame(standardized).transpose()
    return destandardized_data






def plotLoss(history,horizon_step,series_length,freq_name):
    # create the plots directories 
    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig=plt.figure()
    plt.plot(history.history['loss'],'o-')
    plt.plot(history.history['val_loss'],'o-')
    plt.title('Loss : ' + ' '+freq_name +' '+ str(series_length) +' ' + str(horizon_step) )
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    
    # pick the last trained epoch as key
    #key = history.epoch[-1]
    #value = history.history['val_loss'][key]
    
    #plt.annotate(value, (key,value ),textcoords="offset points", xytext=(0, 10), ha='center',fontsize=8,rotation = 90)
    
    
    # or annotate all the epochs
    # every 10 epochs we annotate 
    for i in range(0,len(history.epoch),10):
        key = history.epoch[i]
        value = history.history['val_loss'][key]
        
        # maybe annotate only every 5 or 10 epochs 
        plt.annotate(value, (key,value ),textcoords="offset points", xytext=(0, 10), ha='center',fontsize=8,rotation = 90)
    
    
    #plt.xticks(np.arange())
    if not (os.path.exists(os.path.join('plots','Loss',freq_name ,str(series_length)) ) ):
        os.makedirs(os.path.join('plots','Loss',freq_name ,str(series_length)))
    plt.savefig(os.path.join('plots','Loss',freq_name ,str(series_length)) +'/'+ str(horizon_step)+ '.png')
    plt.close(fig)
    #plt.show()


