#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 21:54:59 2023

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
from build_model import *
from utils import *






if __name__== "__main__":
    # build all the models for all frequencies 
    models = build_Model()
    
    
    # for each model of different frequency 
    for m in models:
    
        # read data
        series = pd.read_csv(os.path.join('../dataset', '{}.csv'.format(m.freq_name)), header=0, index_col=0)
        
        # standardize series 
        #series,means,stds = standardize(series)
        
    
        # transpose series 
        series = series.transpose()
    
        # dictionary of predictions
        prediction = dict()
    
    
        
        # for all training lengths 
        for series_length in m.training_lengths:
            
            
        
            # pick the latest series_length values (from the end) of the series to predict the next unknown horizon      
            
            x_train = series.iloc[:,-series_length:]
            x_train,means,stds = standardize(x_train.transpose())
            
            
            
        
            # initialise array for forecast
            prediction[series_length] = np.zeros([len(x_train), m.horizon])
            curr_prediction = prediction[series_length]
            curr_prediction[:] = np.nan
    
    
    
    
            # different models for each future time step
            for horizon_step in range(m.horizon):
                
                print(m.freq_name, series_length, horizon_step)
    
                # clear session and reset default graph, as suggested here, to speed up prediction
                # https://stackoverflow.com/questions/45796167/training-of-keras-model-gets-slower-after-each-repetition
                ks.backend.clear_session()
                tf.compat.v1.reset_default_graph()
    
    
                # load model and predict
                # load each model of different frequency 
                # load each model to predict each time step 
                model_file = os.path.join(f'../trained_models/{m.freq_name}',
                                          '{}_length_{}_step_{}.h5'.format(m.freq_name, series_length,
                                                                           horizon_step))
                
                # load models from 'trained_models/freq_name' directory 
                est = ks.models.load_model(model_file)
                
                
                
                # make prediction 
                curr_prediction[:, horizon_step] = est.predict(x_train).flatten()
                
                
            # denormalise and get the actual values 
            # fill the list with the results 
            # turn each prediction into dataframe before feeding it to destandardize 
            
            prediction_df  =  pd.DataFrame(curr_prediction.copy())
            # destandardize prediction to get actual currency values and save them into the prediction dictionary 
            prediction[series_length] = destandardize(prediction_df, means,stds)
            
            
        #------------ save prediction csvs for all models ------------------#
        # create directories for predictions 
        if not (os.path.exists(os.path.join('../predictions',m.freq_name ))):
            os.makedirs(os.path.join('../predictions',m.freq_name ))
    
    
        for series_length in m.training_lengths:
                ## we need different csv files for each model ## 
                    # create the Dataframe 
                    output = pd.DataFrame(index=series.index, columns=['F' + str(i) for i in range(1, 49)])
        
                    # index of frequency/series type 
                    output.index.name = 'id'
        
                    # fc,lower,upper intervals 
        
                    output.iloc[:, : m.horizon] = prediction[series_length]
            
        
                    # write output to csv for each model 
                    output.to_csv( os.path.join('../predictions',m.freq_name + '/' + f'Predictions_{series_length}.csv'))
        
