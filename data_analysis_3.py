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

# import statsmodels as well
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose  # to split the time series into components
import statsmodels.tsa.api as smt

from fetch_data import download_zip, unzip_and_rename
from process_data import clean_data, resample_data
from utils import *
from build_model import *
from collections import namedtuple
import tensorflow as tf 
from IPython.display import clear_output


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
'''
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
    standardized_data = pd.DataFrame(standardized).transpose()
    return standardized_data , means , stds 



# destandardize to get the original data 
def destandardize(data,means,stds):
    # get the original orientation 
    data = data.transpose()
    
    destandardized = [] 
    
    for i,col in enumerate(data.columns):
        destandardized.append(data[col]*stds[i] + means[i])
        
    #standardized_data = pd.DataFrame(standardized).transpose()
    return destandardized
        
'''


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
    




# only called as main for testing 
if __name__=="__main__" :
        # read data from csv 
        data = pd.read_csv('../dataset/processed_dataset.csv', index_col=0, parse_dates=True)
        
        
        
        '''
        # standardize 
        #standardized_data = standardize(data)
        
        #create nested dict
        # run with data or standardized_data
        #y_dict = create_nested_dict(data)
        
        
        #run plot_one_year_all_currencies to plot for a certain year all the currencies monthly plots 
        #plot_one_year_all_currencies(y_dict,standardized_data,1999,save=False)
        
        #------------------------------------------------------#
        
        #plot_one_year_all_currencies(y_dict,data,2018,save=True)
        #plot_one_year_all_currencies(y_dict,data,2019,save=True)
        #plot_one_year_all_currencies(y_dict,data,2020,save=True)
        #plot_one_year_all_currencies(y_dict,data,2021,save=True)
        #plot_one_year_all_currencies(y_dict,data,2022,save=True)
        #plot_one_year_all_currencies(y_dict,data,2023,save=True)
        
        #-----------------------------------------------------#

        
        #plot all years for all currencies  
        #plot_all_years_all_currencies(y_dict,data,True)
      
        
        
        
        
        # let's transpose the series to make them more compatible with the cnn models created 
        #all_series_T = standardized_data.transpose()
        
        # transpose data series 
        #if we need to feed them to the models later 
        #data_T = data.transpose()
        
        
        # let's try to do some resampling into the data 
        # instead of all days for each year 
        # it calculates the 1st day of each month for all years 
        # by using the resample function 
        
        
        # --- resample the data to get monthly/daily/weekly ---
        #all_series_resampled = data.resample('MS').mean()
        
        # plot a subset of the new monthly sampled observations for all years 
        #plot_Series(all_series_resampled,296,1,False)
        
        
        # plot all currencies for all years (but sampling one day for each month of the year)
        # --- comment to skip plots ---
        #plot_all_currencies_monthly(all_series_resampled,296,17,False,True)
        
        
        
        #TODO : maybe using resample i calculate one value for each month/week/quarter/year
        # so i build the monthly/weekly/quarterly/yearly/ series of the dataset 
        # and then build different models for eacg frequency based on the cnn models i have built
        # i can start with the daily model , because we already have the original daily series of all years and currencies 
        '''
        
    
        
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
            
            
        # let's start with the daily dataset 
        daily_series = frequencies['daily'][1]
        
        
        # transpose to fit into model
        daily_series_T = daily_series.transpose()
        
        # it works but i need to have years on the ticks 
        #plot_all_currencies_monthly(daily_series, 6000, 17,False,False)
        
        
        #-------------- BEGIN TRAINING ---------------------------------------#
        
        
        
        # define callback 
        # to save only the best model (with min loss)
        # this hiners training so i deactivated it for now
        checkpoint_filepath = '../checkpoint'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
        
        
        # let's create the logs to get the loss updates 
        logs = {
            'loss' : 0.1
            }
        
        # function not created yet 
        # to create the models for each frequency 
        models = build_Model()
        
        
        ###############################################
        #------------ BEGIN TRAINING LOOP ------------#
        ###############################################
        
        for m in models :
            
            # read series corresponding to frequency 
            series = frequencies[m.freq_name][1]
            # standardize is used 
            series,_,_ = standardize(series)
            
            #series = series.transpose()
            
            
            # train for all the training_lengths set on the models #
            # one model for each training_length #
            for series_length in m.training_lengths:
                
        
                ################################################
                #------------ SPLITTING DATASET ---------------#
                ################################################
                # create the array to keep all the train series
                # create the new array of series 
                all_series = np.zeros((series.shape[0], series_length + m.horizon ))
        
        
                #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
                # fill all the training_series with the data from the original series 
                # i make here the assumption that we have : number_of_available_values_in_series > series_length  + horizon 
                # which makes sense since here we have around 6000 daily observations ~ around 16.3 years which is plenty 
                # if for some reason i need to set a very high series_length + horizon and it so happens that number_of_available_values_in_series > series_length  + horizon 
                # i will use the methods of btrotta to extend each series in the all_series array with data from the original series picked from the corresponding period tp fill the missing values 
                # but for now i assume we have enough series_length + horizon values from all currency series 
                # so we fill each series in the all_series array with series_length + horizon values , starting from the end of the series to get the latest data of course 
                # Also i will probably add the option to augment (add more rows for each series ) later, in case we don't have enough data 
                #------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
                
                
                
                
                # fill new array of series 
                all_series[:,:] = series.iloc[:, -(series_length + m.horizon):]
                
                # set x,y train (training + horizon )        
                x_train = all_series[:,:-m.horizon]
                y_train = all_series[:,-m.horizon:]
                
                # train for each horizon 
                # should we use range here ?? 
                for horizon_step in range(m.horizon) :
                    
                    # read only one horizon step as y_train
                    cur_y_train = y_train[:,horizon_step]
                    
                    # set up tensorflow 
                    # clear session and reset default graph, as suggested here, to speed up training
                    # https://stackoverflow.com/questions/45796167/training-of-keras-model-gets-slower-after-each-repetition
                    ks.backend.clear_session()
                    #tf.reset_default_graph()
                    tf.compat.v1.reset_default_graph()
    
                    # set random seeds as described here:
                    # https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
                    
                    # create keras session #
                    np.random.seed(0)
                    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                    tf.compat.v1.set_random_seed(0)
                    tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
                    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_conf))
                    
                    ###########################
                    #--instantiate the model--# 
                    ###########################
            
                    
                    # for daily use these parameters for now !
                    # of course i will test many parameters 
                    
                    mod=m.model_constructor
                    # call daily model for now 
                    # build different model based on the frequency 
                    
                    #----------------------------------------------------------------------#
                    if(m.freq_name=='daily'):
                        # test -- > set batch = 1 
                        cur_model,epochs,batch_size = mod(series_length,7,3,250,400,20)
                    elif(m.freq_name == 'weekly'):
                        cur_model,epochs,batch_size = mod(series_length,52,4,52,250,20)
                    elif(m.freq_name == 'monthly'):
                        cur_model,epochs,batch_size = mod(series_length,12,6,50,250,20)
                    elif(m.freq_name == 'quarterly'):
                        cur_model,epochs,batch_size = mod(series_length,4,4,50,65,20)
                    else :
                        # 'Yearly'
                        cur_model,epochs,batch_size = mod(series_length,2,4,20,250,1000)
                    #-----------------------------------------------------------------------#
        
                    # train 
                    # set up parameters 
                    # checkpoint --> save best model 
                    # early_stopping 
                    
                    #pl1 = PlotLearning()
                    # maybe add pl1 in callbacks
                    # maybe add model_checkpoint_callback
                    # try with one series each time 
                    # -----------------------------------------------------------------------------------------------------------#
                    # ------------------ I NEED TO CHANGE VALIDATION_SPLIT TO 0.2 ~ 0.3 -----------------------------------------#
                    #------------------------------------------------------------------------------------------------------------#
                    # or set batch_size = 1 
                    history = cur_model.fit(x_train, cur_y_train, epochs=epochs, batch_size=20, shuffle=True, validation_split=0.4,
                        callbacks=[ ks.callbacks.EarlyStopping(monitor='val_loss', patience=100)])
                    
                    
                    # plot Loss for this horizon step 
                    plotLoss(history,horizon_step,series_length,m.freq_name)
                    
                    #########################
                    #----save the model ----#
                    #########################
                    if not os.path.exists('../trained_models'+'/'+ m.freq_name):
                            os.mkdir('../trained_models' + '/'+ m.freq_name)
                            
                    model_file = os.path.join('../trained_models',m.freq_name,
                                                  '{}_length_{}_step_{}.h5'.format(m.freq_name, series_length,
                                                                                     horizon_step))
                    # save model 
                    cur_model.save(model_file)
        
        
        