#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 04:16:07 2023

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


def slide_train_multi_step(frequencies):
    # call build model to create the models 
    # probably we need new model scrip here
    # which will have the multi step models only
    
    models = sCnn()
    
    
    # add the other frequencies as well (i have only built with daily for now)
    for m in models :
                
            # pick each frequency dataset (but after 2013-...)
            # normalize and find optimum 'alpha' for exponential smoothing 
            series = frequencies[m.freq_name][1].loc['2013-01-07':]
            series_norm = MinMaxScaler().fit_transform(series)
            series_norm = pd.DataFrame(series_norm, index=series.index , columns = series.columns) 
            optimum_a = optimum_al(series_norm)
            smoothed_series = exponential_smooth(series_norm, optimum_a)
            
                
            for series_length in m.training_lengths:
                
               
                ks.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                np.random.seed(0)
                session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                tf.compat.v1.set_random_seed(0)
                tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
                tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_conf))
                
                
                #for all currencies 
                for cur in series.columns:
                
                    # call the model constructor
                    mod=m.model_constructor
                    #------------ I can also test the same model with all the other frequencies !
                    if(m.freq_name=='daily'):
                        cur_model,epochs,batch_size = mod(series_length,1,m.horizon)
                    
                   
                    
                    #------------- HOW DO WE SPLIT FOR TRAINING ?? IT IS NOT KNOWN FROM THE PAPER ----------#
                    # WHAT HORIZON DO WE TRAIN FOR . IT IS ALSO NOT KNOWN FROM THE PAPER -------------------#
                    # TODO : SHOULD TEST SLIDING WINDOW TRAINING OR DIFFERENT RANGES TO TRAIN 
                    # I WILL PICK FOR NOW FROM THE LATEST YEAR 2022 : 2022-01-03 + history(364) days , 91 days for validation (20%)
                    # 2022-01-03
                    
                    # for now just test using 364 for training 
                    x_train = smoothed_series.iloc[-500:-136,:]
                    y_train = smoothed_series.iloc[-136:-122,:]
                    
                    # pick current validation set 
                    x_val = smoothed_series.iloc[-122:-31,:]
                    y_val = smoothed_series.iloc[-31:-17,:]
                    
                    # pick only 1d series slice for each currency to train 
                    # maybe also i should transpose it ! 
                    x_train = x_train[cur].transpose()
                    y_train = y_train[cur].transpose()
                    x_val = x_val[cur].transpose()
                    y_val = y_val[cur].transpose()
                
                    
                    #-----------------------------------------------------------------------#
                    # set batch size = 20 
                    history = cur_model.fit(x_train, y_train, epochs=epochs, 
                    validation_data = (x_val,y_val),
                    callbacks=[ ks.callbacks.EarlyStopping(monitor='val_loss', patience=800)])
                    
                    
                    # plot Loss for this horizon step 
                    #plotLoss(history,horizon_step,series_length,m.freq_name)
                    
                    #plot_multi_step_Loss(history,m.horizon,series_length,m.freq_name)
                                
                    
                    # plot the losses 
                    '''
                    plot_multi_step_Loss(history,m.horizon,series_length,m.freq_name,True)
                    
                    
                    #########################
                    #----save the model ----#
                    #########################
                    
                    #TODO : SHOULD FIX BUG HERE 
                    # IT CREATES ANOTHER DIRECTORY WITH CONCATENATED THE FREQUENCY AND 'MULTI_STEP'
                    if not os.path.exists(os.path.join('../trained_models', 'multi_step','sliding',m.freq_name)):
                            os.makedirs(os.path.join('../trained_models', 'multi_step','sliding', m.freq_name))
                            
                    model_file = os.path.join('../trained_models','multi_step','sliding',m.freq_name,
                                                  '{}_length_{}.h5'.format(m.freq_name, series_length))
                    # save model 
                    cur_model.save(model_file) 
                    '''
                    
                    
################### ONLY AS MAIN ###################             
if __name__ =="__main__":
    
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
        
        
    #slide_train_multi_step(frequencies)
    models = sCnn()
    
    
    # add the other frequencies as well (i have only built with daily for now)
    for m in models :
                
            # pick each frequency dataset (but after 2013-...)
            # normalize and find optimum 'alpha' for exponential smoothing 
            series = frequencies[m.freq_name][1].loc['2013-01-07':]
            series_norm = MinMaxScaler().fit_transform(series)
            series_norm = pd.DataFrame(series_norm, index=series.index , columns = series.columns) 
            optimum_a = optimum_al(series_norm)
            smoothed_series = exponential_smooth(series_norm, optimum_a)
            
                
            for series_length in m.training_lengths:
                
               
                ks.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                np.random.seed(0)
                session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                tf.compat.v1.set_random_seed(0)
                tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
                tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_conf))
                
                
                #for all currencies 
                for cur in series.columns:
                
                    # call the model constructor
                    mod=m.model_constructor
                    #------------ I can also test the same model with all the other frequencies !
                    if(m.freq_name=='daily'):
                        cur_model,epochs,batch_size = mod(series_length,1,m.horizon)
                    
                   
                    
                    #------------- HOW DO WE SPLIT FOR TRAINING ?? IT IS NOT KNOWN FROM THE PAPER ----------#
                    # WHAT HORIZON DO WE TRAIN FOR . IT IS ALSO NOT KNOWN FROM THE PAPER -------------------#
                    # TODO : SHOULD TEST SLIDING WINDOW TRAINING OR DIFFERENT RANGES TO TRAIN 
                    # I WILL PICK FOR NOW FROM THE LATEST YEAR 2022 : 2022-01-03 + history(364) days , 91 days for validation (20%)
                    # 2022-01-03
                    
                    # for now just test using 364 for training 
                    # TODO : SHOULD CREATE A DATA GENERATOR - BATCH GENERATOR THAT YIELDS EACH TIME A BATCH OF SIZE : SERIES_LENGTH
                    x_train = smoothed_series.iloc[-500:-136,:][cur]
                    y_train = smoothed_series.iloc[-136:-122,:][cur]
                    
                    ## pick current validation set 
                    #x_val = smoothed_series.iloc[-122:-31,:][cur]
                    #y_val = smoothed_series.iloc[-31:-17,:][cur]
                    
                    # pick only 1d series slice for each currency to train 
                    # maybe also i should transpose it ! first turn into array
                    x_train = np.array(x_train).reshape((1,-1))
                    #x_val = np.array(x_val).reshape((1,-1,1))
                    y_train = np.array(y_train).reshape((1,-1))
                    #y_val = np.array(y_val).reshape((1,-1,1))
                   
                    #inside history --> validation_data = (x_val,y_val),
                    history = cur_model.fit(x_train, y_train, epochs=epochs,batch_size =1 ,
                                            callbacks=[ ks.callbacks.EarlyStopping(monitor='val_loss', patience=800)])
                    