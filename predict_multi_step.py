#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 17:04:00 2023

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
from build_multistep_model import *



def multi_single_step_one(frequencies,k):
    
    # just for testing
    k=1

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


                    # pick the history we need to use to predict the nex horizon
                    x_train = series.iloc[-(k *series_length):,:]
                    x_train,means,stds = standardize(x_train)



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
                    output = pd.DataFrame(index=[0], columns=['F' + str(i) for i in range(1, 49)])
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
            
        multi_single_step_one(frequencies,1)