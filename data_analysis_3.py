#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 23:00:15 2023

@author: st_ko
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import os 
import keras as ks
from sklearn.metrics import mean_absolute_error
from fetch_data import download_zip, unzip_and_rename
from process_data import clean_data, resample_data
from utils import *
from build_model import *
from collections import namedtuple
import tensorflow as tf 
from IPython.display import clear_output
from sklearn.utils import resample
#from sklearn import cross_validation
from sklearn.model_selection import GroupShuffleSplit
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.metrics import mean_squared_error


# function to plot resampled momthly (mean of all month) observations for all years for all currencies 
# for each currency draw different plot 
def plot_all_currencies_monthly(data,size,num_currencies,ann=False,save=False):
    # annotate a subpart of the series 
    for col in data.columns[0:num_currencies]:
        fig=plt.figure()
        for i in range(size):
            if(ann):
                plt.annotate(data[col][i], (data[col].index[i], data[col].values[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center',fontsize=8,rotation = 45)
        plt.plot(data[col][:size],'o-',label=col)
        plt.legend(loc='lower right',bbox_to_anchor=(1.05,0.2))
        plt.title("Monthly sampled All years " + col )
        if(save):
            # 1000 means all years but monthly
            # just a random chosen number 
            save_image(fig, col,None)
            plt.close()
        


# plot the currencies within a given range of observations 
# added option to select currencies subrange as well
# this plots chosen subset of series and observations all together in one plot  
def plot_Series(data,size,num_currencies,ann=False):
    # annotate a subpart of the series 
    plt.figure()
    for col in data.columns[0:num_currencies]:
        for i in range(size):
            if(ann):
                plt.annotate(data[col][i], (data[col].index[i], data[col].values[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center',fontsize=8,rotation = 45)
        plt.plot(data[col][:size],'o-',label=col)
        plt.legend(loc='lower right',bbox_to_anchor=(1.05,0.2))



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




# ! More cuztomized 
# create a nested dictionary of year -> dictionary of months --> observations of this month in this year 

def create_nested_dict(data):
        total_time= [x for x in data.index]
        year_dict = {x: {x2 : [] for x2 in iter(range(1,13)) } for x in range(data.index.min().year,data.index.max().year+ 1)}
        m_step = data.index.min().month
        y_step = data.index.min().year
        y=True
        m = True
        
        # create nested dictionary 
        # i will turn this later into a function 
        for i in range(len(total_time)-1) :
            if(total_time[i].year == total_time[i+1].year):
                y = True
                if(total_time[i].month == total_time[i+1].month):
                    year_dict[y_step][m_step].append(total_time[i])
                    m = True
                else :
                    if(m==True):
                        year_dict[y_step][m_step].append(total_time[i])
                        m=False
                    m_step +=1   
            else :
                # reset months to set the last observation 
                m_step = 12
                # add the remaining same year observation 
                year_dict[y_step][m_step].append(total_time[i]) 
                # reset month index 
                m_step=data.index.min().month
                y_step+=1
                y=False
        return year_dict    
                
        
        
# for a certain year , for a certain currency plot the monthy series of this currency (one subplot per month of the year)
def plot_one_year_one_currency(year_dict,data,year,currency,save=False):
    # total plots 
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))  # 3x4 grid for subplots
    #select currency from dataset 
    
    
    # for each month 
    for month in range(1, 13):
        row = (month - 1) // 4  # Determine row index
        col = (month - 1) % 4   # Determine column index
        
        #pick year and currency 
        dollar_month = data.loc[year_dict[year][month],currency]
        ax = axes[row,col]
        # rotate txt 
        ax.xaxis.set_tick_params(rotation=60)
        ax.plot(dollar_month,'o-',label= currency + ' : ' + str(month))   
        ax.set_title(f"{year}-{month}-{currency}")
        
    #general title for whole plot     
    plt.suptitle(f"Time Series Observations for Year {year}", fontsize=16)
    plt.tight_layout()
    #plt.show()
    # close plot 
    #plt.close()
    
    
    
    # save image (optional)
    if(save):
        save_image(fig,currency,year)


# same as above but do it for all currencies (17 currencies -> 17 plots )
def plot_one_year_all_currencies(year_dict,data,year,save=False):
    for cu in data.columns:
        plot_one_year_one_currency(year_dict,data,year,cu,save)
    


# plot all years one currency 
def plot_all_years_one_currency(year_dict,data,currency,save=False):
    # total plots 
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 10))  # 3x4 grid for subplots
    #select currency from dataset 
    
    
    # for each month 
    inyear = data.index.min().year
    for year in range(1, 26):
        row = (year - 1) // 5  # Determine row index
        col = (year - 1) % 5   # Determine column index
        
        
        #f1=[x for x in y_dict[inyear].values()]
        #f2 = [k for k2 in f1 for k in k2]
        
        # take the yearly indices 
        indices = [k for k2 in [x for x in y_dict[inyear].values()] for k in k2]
        
        
        dollar_month = data.loc[ indices ,currency]
        ax = axes[row,col]
        # rotate txt 
        ax.xaxis.set_tick_params(rotation=60)
        ax.plot(dollar_month,'o-',label= currency + ' : ' + str(inyear))   
        ax.set_title(f"{inyear}-{currency}")
        inyear += 1
        
    #general title for whole plot     
    plt.suptitle(f"Time Series Observations for All years - {currency}", fontsize=16)
    plt.tight_layout()
    #plt.show()
    #plt.close()
    
    if(save):
        # i use 25 to denote that we plot all the years 
        save_image(fig,currency,25)
    
    # save image (optional)
    
    

# function to plot all years for all currencies 
def plot_all_years_all_currencies(year_dict,data,save=False):
    for cur in data.columns:
            plot_all_years_one_currency(year_dict, data, cur,save)
    
    


# function to save images 
def save_image(fig,currency,year):
    if not (os.path.exists('plots')):
            os.makedirs('plots')
    plt.savefig(os.path.join('plots', currency + ' ' +str(year) +  '-forecast.png'))
    plt.close()
    
'''
class CustomKerasRegressor(KerasRegressor):
    def fit(self, X, y, **kwargs):
        
        self.history = self.model.fit(X, y, **kwargs)
        #self.fit(X,y)
'''

# only called as main for testing 
if __name__=="__main__" :
       
    
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
            
        
        #-------------- BEGIN TRAINING SINGLE STEP---------------------------------------#
        #train_models_single_step(frequencies)
        #--------------------------------------------------------------------------------#
        
        
        
        #-------------- BEGIN TRAINING MULTI STEP----------------------------------------#
        train_models_multi_step(frequencies)
        #--------------------------------------------------------------------------------#
        
                
                    
                  
               
                    
                    
                    
                    
                   