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
from keras.utils import Sequence


class CustomDataGen(Sequence):
    def __init__(self, data, batch_size,series_length,horizon,epochs):
        # feed the smoothed dataframe here
        self.data = data
        self.batch_size = batch_size
        self.series_length = series_length
        self.horizon = horizon
        self.epochs = epochs
        self.current_epoch = 0
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
                #return None
                raise StopIteration
        else :
            batch_x = np.array(batch_x).reshape((-1,self.series_length))
            batch_y = self.labels[idx*self.batch_size * self.horizon : idx * self.batch_size*self.horizon + self.batch_size * self.horizon].reshape((-1,self.horizon))
            return (batch_x,batch_y)

    
    # define the other 2 methods to loop back to the beginning of the generator
    def on_epoch_end(self):
        self.current_epoch += 1

    def __iter__(self):
        while self.current_epoch < self.epochs:
            self.on_epoch_end()
            for i in range(len(self)):
                yield self[i]
            self.current_epoch += 1
    



# function to plot multi step function 
def plot_scnn_Loss(history,horizon,series_length,freq_name,cur,s=False):
    # create the plots directories 
    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig=plt.figure()
    plt.plot(history.history['loss'],'o-')
    plt.plot(history.history['val_loss'],'o-')
    plt.title('Loss : ' + ' '+freq_name +' '+ str(series_length) +' ' + str(horizon) + ' ' + cur)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    # or annotate all the epochs
    # every 10 epochs we annotate 
    #for i in range(0,len(history.epoch),10):
    #    key = history.epoch[i]
    #    value = history.history['val_loss'][key]
        
    #     # maybe annotate only every 5 or 10 epochs 
    #    #plt.annotate(value, (key,value ),textcoords="offset points", xytext=(0, 10), ha='center',fontsize=8,rotation = 90)
    
    if(s==False):
    
        #plt.xticks(np.arange())
        if not (os.path.exists(os.path.join('plots','Loss','scnn','multi_step',cur,freq_name ,str(series_length)) ) ):
            os.makedirs(os.path.join('plots','Loss','scnn','multi_step',cur,freq_name ,str(series_length)))
        plt.savefig(os.path.join('plots','Loss','scnn','multi_step',cur,freq_name ,str(series_length)) +'/'+ str(horizon)+ '.png')
        plt.close(fig)
    else:
        if not (os.path.exists(os.path.join('plots','Loss','scnn','multi_step','slide',cur,freq_name ,str(series_length)) ) ):
            os.makedirs(os.path.join('plots','Loss','scnn','multi_step','slide',cur,freq_name ,str(series_length)))
        plt.savefig(os.path.join('plots','Loss','scnn','multi_step','slide',cur,freq_name ,str(series_length)) +'/'+ str(horizon)+ '.png')
        plt.close(fig)
        #plt.show()
    

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
                    
                    # Here we wil call the datagenerator #
                    
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
                    
              
# CREATE THE TRAINING AND VALIDATION GENERATORS 


           
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
                        cur_model,epochs,batch_size = mod(series_length,2,m.horizon)
                    
                    # create the 2 datasets
                    train_dataset = smoothed_series.iloc[:2180][cur]
                    val_dataset = smoothed_series.iloc[2180:][cur]
                    
                    # create the new generators 
                    # choose batch size here 
                    train_gen = iter(CustomDataGen(train_dataset, 2, series_length,m.horizon ,epochs))
                    val_gen = iter(CustomDataGen(val_dataset, 1, series_length,m.horizon,epochs ))
                    
                    x_train,y_train = next(train_gen)
                    
                    # TODO : TRAIN WITH VALIDATION SPLIT = 0.2 AND NOT WITH VALIDATION GENERATOR #
                    # ALSO SHOULD TEST WITH NO GENERATOR JUST TO BE MORE CERTAIN --> TRAIN WITH A SIMPLE TRAINING LOOP
                    # TRAIN[:2180] , VALIDATION [2180:] AND SPLIT VALIDATION_SIZE = 0.2
                    
                    #inside history --> validation_data = (x_val,y_val),
                    # for now i train with validation data from random split
                    # i should fix the validation generator 
                    #try:
                    #    x_train,y_train = next(train_gen)
                    #    #x_val,y_val = next(val_gen)
                    #except StopIteration:
                    #    break
                
                    history = cur_model.fit( x_train,y_train,epochs=epochs,
                                            validation_data = next(val_gen),
                                            callbacks=[ ks.callbacks.EarlyStopping(monitor='val_loss', patience=800)])
                    
                    # PLOT LOSS FOR THIS MODEL
                    plot_scnn_Loss(history,m.horizon,series_length,m.freq_name,cur,True)
                    
                    
                    #########################
                    #----save the model ----#
                    #########################
                    
                    #TODO : SHOULD FIX BUG HERE 
                    # IT CREATES ANOTHER DIRECTORY WITH CONCATENATED THE FREQUENCY AND 'MULTI_STEP'
                    
                    if not os.path.exists(os.path.join('../trained_models','scnn', 'multi_step','slide',cur,m.freq_name,str(series_length) )):
                            os.makedirs(os.path.join('../trained_models','scnn', 'multi_step','slide', cur,m.freq_name,str(series_length) ))
                            
                    model_file = os.path.join('../trained_models','scnn','multi_step','slide',cur,m.freq_name,str(series_length),
                                                  '{}_length_{}.h5'.format(m.freq_name, series_length))
                    # save model 
                    cur_model.save(model_file)
                    