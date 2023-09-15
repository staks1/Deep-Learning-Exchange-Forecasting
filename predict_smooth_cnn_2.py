#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 20:51:37 2023

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
from scipy import stats
#from build_model import *
from utils import *
from build_multistep_model_2 import *
from data_analysis_3c import *
from sklearn.preprocessing import MinMaxScaler


###### SECOND VERSION TO DO THE EXPONENTIAL SMOOTHING AFTER THE MIN MAX SCALER 
# function to predict the future 
def predict_smooth_cnn(frequencies):
    

    # define the currencies
    series = frequencies['daily'][1]
    currencies = series.columns
    

    # dictionary of predictions
    prediction = { x : {} for x in currencies}


    #we need to read all the models
    models = sCnn()


   
    
    for cur in prediction.keys():

        
            
            for m in models:
                
                
                # SHOULDI DO THE EXPONENTIAL SMOOTHING AND THE NORMALIZATION ONLY ON THE SUBPARTS I WANT TO TRAIN OR NOT ??
                series = pd.DataFrame(frequencies[m.freq_name][1].loc['2010-01-04':][cur])
                
                # pick index frequency , business frequency 
                f_s = frequencyCalc(m.freq_name)
                series.index = pd.DatetimeIndex(series.index)
                series = series.to_period(f_s)
                
                
                series_n = np.reshape(series,(-1,1))
                
                
                
                # fit 
                minmax = MinMaxScaler().fit(series_n)
                series_norm = minmax.transform(series_n)
                
                
                # fit 
                series_norm = pd.DataFrame(series_norm, index=series.index, columns = series.columns) 
                
                
                # find the optimum alpha for this currency and apply the exponential smoothing 
                
                optimum_a = optimum_al(series_norm)
                _,smoothed_series = exponential_smooth(series_norm,optimum_a,m.freq_name,Hw=True)
                
                
                
                prediction[cur][m.freq_name] = {}

                
                print(f"\n------------------PREDICTING CURRENCY : {cur} --------------------\n")

                # for all training lengths
                for series_length in m.training_lengths:


                    # let's normalize all the currencies and apply the exponential smoothing
                    

                     # pick the history we need to use to predict the next horizon
                     # maybe it needs reshaping into (1,series_length) ??
                     x_train = smoothed_series.iloc[-series_length:,:]
                    
                     
                     # initialize dict 
                     prediction[cur][m.freq_name][series_length] = np.zeros([1,m.horizon])
                     
                     # pointer 
                     curr_prediction = prediction[cur][m.freq_name][series_length]


                     # ---------- TF KERAS SET UP ------------------------------------------------------#
                     ks.backend.clear_session()
                     tf.compat.v1.reset_default_graph()
                     # load the particular model
                     model_file = os.path.join(f'../trained_models/scnn/multi_step/slide/{cur}/{m.freq_name}/{series_length}',
                                               '{}_length_{}.h5'.format(m.freq_name, series_length))
                     est = ks.models.load_model(model_file)
                     # ---------------------------------------------------------------------------------#
                     
                     # turn x_train into array to feed to model 
                     x_train = np.array(x_train)
                     x_train = np.reshape(x_train,(1,-1))

                     # fill prediction 
                     curr_prediction[:,:m.horizon] = est.predict(x_train)



                     # we want to fit the inverse transform here to destandardize the data 
                     prediction_df  =  pd.DataFrame(curr_prediction.copy())
                     prediction_denorm = np.array(prediction_df)
                     prediction_denorm = np.reshape(prediction_denorm,(-1,1))
                     # denormalize 
                     prediction_denorm = minmax.inverse_transform(prediction_denorm)
                     final_prediction = np.reshape(prediction_denorm,(1,-1))
                     prediction[cur][m.freq_name][series_length]= final_prediction
                     



                     #------------ save prediction csvs for all models ------------------#
                     output = pd.DataFrame(index=[0], columns=['F' + str(i) for i in range(1, 49)])
                     output.index.name = 'id'
                    
                     # fill dataframe with prediction
                     output.iloc[:, : m.horizon] = prediction[cur][m.freq_name][series_length]
                    
                     if not (os.path.exists(os.path.join('../predictions','scnn/multi_step/slide',cur,m.freq_name,str(series_length )))):
                       os.makedirs(os.path.join('../predictions','scnn/multi_step/slide',cur,m.freq_name,str(series_length) ))
                     output.to_csv(os.path.join('../predictions','scnn/multi_step/slide',cur,m.freq_name,str(series_length) , 'prediction.csv'))
                     #--------------------------------------------------------------------#



# function to predict older horizons to compare with the original series 
# and see how close we were to reality
def multi_single_step_one_old(frequencies,k):
    

    # define the currencies
    series = frequencies['daily'][1]
    currencies = series.columns
    currencies_size = len(currencies)




    # build the dictionary
    # dictionary of predictions
    prediction = { x : {} for x in currencies}


    #we need to read all the models
    models = build_m_Model()


    # for each currency
    # for the test we are going to use just 3 currencies 
    # USD,JPY,CZK
    # and we are going to use k = 1 to predict the future 
    # then we will predict a horizon back ( the last known values ---> (:-(series_length + horizon ))
    
    #for cur in prediction.keys():
    for cur in ['USD','JPY','CZK']:

        
            # for each model of different frequency
            for m in models:

                # read the series to make the predictions
                # convert to dataframe so we can standardize it 
                series = frequencies[m.freq_name][1]
                series = pd.DataFrame(series.loc[:,cur])

                # reshape so we can still view it as a matrix (actually it is a vector)
                #series = np.reshape(series,(series.shape[0],1))
                prediction[cur][m.freq_name] = {}

                
                print(f"\n------------------PREDICTING CURRENCY : {cur} --------------------\n")

                # for all training lengths
                for series_length in m.training_lengths:


                    # pick older series length and predict the horizon 
                    # each time i go k * series_length and try to predict using the last series_length
                    #new_len = series.shape[0] - k * series_length
                    
                    x_train = series.iloc[: (series.shape[0] - k * series_length),:]
                    x_train = x_train.iloc[-series_length:,:]
                    x_train,means,stds = standardize(x_train)



                    # we need to find the next horizon dates-indices after the training history 
                    # so we can align our predictions with our history 
                    #---------- THIS IS THE DATES THAT WE NEED TO ALIGN OUR PREDICTIONS WITH THE HISTORY ------#
                    ind = series.loc[ x_train.columns[-1] : ]
                    ind = ind.iloc[1: m.horizon + 1]
                    ind = ind.index
                    #------------------------------------------------------------------------------------------#


                    # initialize each one with empty values
                    #prediction[cur][m.freq_name][series_length] = np.zeros([len(x_train), m.horizon])
                    # we need 1D array now right ?
                    prediction[cur][m.freq_name][series_length] = np.zeros([1,m.horizon])
                    
                    curr_prediction = prediction[cur][m.freq_name][series_length]

                    # ---------- TF KERAS SET UP ------------------------------------------------------#
                    ks.backend.clear_session()
                    tf.compat.v1.reset_default_graph()
                    # load the particular model
                    model_file = os.path.join(f'../trained_models/multi_step/sliding/{cur}/{m.freq_name}/{series_length}',
                                              '{}_length_{}.h5'.format(m.freq_name, series_length))
                    est = ks.models.load_model(model_file)
                    # ---------------------------------------------------------------------------------#
                    
                    

                    # make prediction
                    #curr_prediction[:,:m.horizon] = est.predict(x_train).flatten()
                    curr_prediction[:,:m.horizon] = est.predict(x_train).flatten()



                    # fill array of values for this dictionary
                    prediction_df  =  pd.DataFrame(curr_prediction.copy())
                    prediction[cur][m.freq_name][series_length]= destandardize(prediction_df, means,stds)



                    #------------ save prediction csvs for all models ------------------#
                    # set for the predictions the corresponding series dates so the predictions get aligned #
                    output = pd.DataFrame(index=[0], columns=ind)
                    output.index.name = 'id'
                    # fill dataframe with prediction
                    output.iloc[:, : m.horizon] = prediction[cur][m.freq_name][series_length]
                    if not (os.path.exists(os.path.join('../predictions','multi_step/sliding',cur,m.freq_name,str(series_length )))):
                        os.makedirs(os.path.join('../predictions','multi_step/sliding',cur,m.freq_name,str(series_length) ))
                    output.to_csv(os.path.join('../predictions','multi_step/sliding',cur,m.freq_name,str(series_length) , 'prediction.csv'))
                    #--------------------------------------------------------------------#




# LET'S DO SOME TEST PREDICTIONS USING ONLY 2 CURRENCIES 
# Let's now read all the different frequency - datasets 
# for each frequency we take the end of the month 
if __name__ == "__main__":
    
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
            
            
        predict_smooth_cnn(frequencies)