#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 21:56:11 2023

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
#from build_multistep_model_2 import *
from data_analysis_3c import *
from sklearn.preprocessing import MinMaxScaler
from build_multistep_model_2 import *




# simple smape metric 
def smape(history, predictions):
    return 1/len(history) * np.sum(2 * np.abs(history-predictions) / (np.abs(history) + np.abs(predictions))*100)




# simple (no seasonlaity) mase
def mase(history,predictions):

    mae = np.mean(np.abs(history - predictions))

    # Calculate the mean absolute error of a naive forecast (using the previous value)
    naive_forecast = np.roll(history, 1)
    naive_forecast[0] = history[0]  # Set the first value to match the history
    mae_naive = np.mean(np.abs(history - naive_forecast))

    # Calculate the MASE
    mase = mae / mae_naive
    return mase 
    
    
    

# mase for seasonal series 
#m is the seasonality (the number of time periods in a seasonal cycle).
# depends on frequency 
def mase_seasonal(history,predictions,seasonality):
    mae_forecast = np.mean(np.abs(history - predictions))

    # Calculate the mean absolute error (MAE) for the seasonal naive forecast
    mae_naive = np.mean(np.abs(history - np.roll(history, seasonality)))

    # Calculate the MASE
    mase = mae_forecast / mae_naive

    return mase
    







def predict_on_history(frequencies):
    # TODO : NOW i have to
    # 1) make predictions on horizons chosen from the train dataset , for each currency series ,each frequency and for each series_length 
    # 2) begin by making predictions for the last horizon values in the dataset (maybe keep 20% percent of the dataset and make predictions in 
    # consecutive windows of those horizons
    # 3) then calculate MASE , SMAPE for all those horizons using all series_lengths 
    # 4) ideally we do not want to use a set that has been used in training 
    # so i will take the previous years ...-2009 and use the history to predict the next horizons 
    # test for already seen data just for observation purposes 
    #--------------------- data for evaluation ---------------------------#
    dseries = pd.DataFrame(frequencies['daily'][1].loc[:'2009-12-31'])
    wseries = pd.DataFrame(frequencies['weekly'][1].loc[:'2009-12-27'])
    mseries = pd.DataFrame(frequencies['monthly'][1].loc[:'2009-12-31'])
    qseries = pd.DataFrame(frequencies['quarterly'][1].loc[:'2009-12-31'])
    yseries = pd.DataFrame(frequencies['yearly'][1].loc[:'2009-12-31'])
    #---------------------------------------------------------------------#
    
    
    # Begin by selecting 20% of the datasets for each different frequency 
    # create a dictionary to retrieve each series for prediction 
    
    pred_dict = {'daily' : dseries, 
                 'weekly' : wseries,
                 'monthly' : mseries,
                 'quarterly' : qseries,
                 'yearly' : yseries}
    
    # define the currencies
    series = frequencies['daily'][1]
    currencies = series.columns
    
    

    # dictionary of predictions
    prediction = { x : {} for x in currencies}


    #we need to read all the models
    models = sCnn()

     
    for cur in prediction.keys():

        
            
            for m in models:
                
                
                # we will pick the dataset that we have not observed so , 2008 - 2010 
                series = pd.DataFrame(pred_dict[m.freq_name][cur])
                
                
                
                
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
                _,smoothed_series = exponential_smooth(series_norm,optimum_a,m.freq_name,Hw=False)
                
                
                
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


# function to evaluate the predictions 
# against the true values (for past data , where the values are available)
def evaluate(frequencies):
    
    # create dataframe for results
    metrics = np.zeros((1,3))
    
    #metrics["smape"]=2
    
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
           history = pred_dict[freq][0][cur][: horizon ].transpose()
           #selected_index = history[:horizon].index 
           
           
           # read model predictions and set the predictions index same as the data index to align the true values with the predictions 
           y_model_1 = pd.read_csv(p, header=0, index_col=0).iloc[:,: horizon].transpose()
           #y_model_1.index = selected_index
           
           
           # transform into arrays to calculate the metrics 
           y_model_1 = np.array(y_model_1)
           history = np.array(history)
           
           # calculate the metrics and save them into a csv file for each training_length for each frequency for each currency
           mase_result = mase(history,y_model_1)
           smape_result = smape(history,y_model_1)
           
           # ------- NOT SURE IF I AM GOING TO USE IT IN THE END ------------------------------#
           # also calculate seasonal smape using the calculated seasonalities for each frequency 
           seasonality = seasonals(freq)
           seasonal_smape_result = mase_seasonal(history, y_model_1, seasonality)
           
           
           # create dataframe of results 
           metrics = pd.DataFrame ( {"mase":mase_result,"smape":smape_result,
                                     "seasonal_smape":seasonal_smape_result},index = [0])
           
           
           # save csv of metric to cur-> frequency -> series_length directory 
           if not (os.path.exists(os.path.join('../evaluation','scnn/multi_step/slide',cur,freq,str(series_length )))):
             os.makedirs(os.path.join('../evaluation','scnn/multi_step/slide',cur,freq,str(series_length) ))
           metrics.to_csv(os.path.join('../evaluation','scnn/multi_step/slide',cur,freq,str(series_length) , 'evaluation.csv'))
           
           
       
      



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
        
        
        
    '''    
    # let's take a predictions csv file ( for daily , for usd , for series_length 14 )
    # read the csv , read the true values for the horizon window  and calculate the smape , mase 
    # since the predictions we made are for a future window we will just use as true values the last 14 days of the history of the dataset
    # and so the errors will be defined between the last 14 days in the history and the 14 predictions for our series length and the currency 'USD'
    cur = 'GBP'
    freq_name = 'daily'
    horizon = 14
    predictions = pd.read_csv('/home/st_ko/Desktop/Deep_Learning_Project/neural-networks-project/predictions/scnn/multi_step/slide/simple_exponential_smoothing_after_2010/USD/daily/14/prediction.csv',index_col=0)
    predictions = predictions.iloc[:,:horizon]
    
   
    # history 
    # We will pick the range we want , here we want to take the final horizon values from the dataset as the y true to compare with the 14 predictions
    history = frequencies[freq_name][1]['USD'][-horizon:]
   
    
    # to calculate the differences it is better to transform both of them into arrays
    # also we need to reshape history (N,1)---> (1,N)
    predictions = np.array(predictions)
    history = np.array(history)
    
   
    
    # calculate the smape 
    smape_result = smape(history,predictions)
    mase_result = mase(history,predictions)
    mase_result2 = mase_seasonal(history, predictions, 5)
    '''
    
    
    #----- use this to predict on past data so we can compare with the true data (for available true history) ---#
    #predict_on_history(frequencies)
    #------------------------------------------------------------------------------------------------------------#
    evaluate(frequencies)
    
    