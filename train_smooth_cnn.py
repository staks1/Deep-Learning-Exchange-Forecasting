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
            
            
            elif(len(batch_x) ==self.series_length):
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
    


def dataset_picker(smoothed_series,freq_name,frequencies,cur):
    # pick after 2010 or whole dataset for yearly 
    if(freq_name != 'yearly'):
        dataset2 = smoothed_series.loc['2010-01-04':]
    # re normalize and smooth the new yearly data 
    else:   
        dataset2 = frequencies[freq_name][1]
        series_norm = MinMaxScaler().fit_transform(dataset2)
        series_norm = pd.DataFrame(series_norm, index=dataset2.index , columns = dataset2.columns) 
        optimum_a = optimum_al(series_norm)
        dataset2 = exponential_smooth(series_norm, optimum_a)
        
    # initial dataset 
    # customize each frequency
    if(freq_name =="daily"):    
        train_dataset = dataset2.iloc[:2802][cur]
        val_dataset = dataset2.iloc[2802:][cur]
    elif(freq_name =="weekly"):    
         train_dataset = dataset2.iloc[:570][cur]
         val_dataset = dataset2.iloc[570:][cur]
    elif(freq_name =="monthly"):    
         train_dataset = dataset2.iloc[:115][cur]
         val_dataset = dataset2.iloc[115:][cur]
    elif(freq_name =="quarterly"):    
         train_dataset = dataset2.iloc[:35][cur]
         val_dataset = dataset2.iloc[35:][cur]
    # i split 50 % 50 (probably not good but we dont have enough data)
    # should try yearly with random splittingas well 
    # TODO :TRAIN YEARLY WITH AUGMENTATION OR RANDOM SPLITTING SINCE WE DON'T HAVE ENOUGH DATA FOR HISTORY AND PREDICTIONS AND VALIDATION DATA 
    elif(freq_name =="yearly"):    
         train_dataset = dataset2.iloc[:13][cur]
         val_dataset = dataset2.iloc[13:][cur]

    return (train_dataset,val_dataset)









def train_smooth_cnn(frequencies):
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
                
            # pick each frequency dataset (but after 2010-...)
            # normalize and find optimum 'alpha' for exponential smoothing 
            #series = frequencies[m.freq_name][1].loc['2013-01-07':]
            series = frequencies[m.freq_name][1].loc['2010-01-04':]
            f_s = frequencyCalc(m.freq_name)
            series.index = pd.DatetimeIndex(series.index).to_period(f_s)
            
            
            # normalize
            series_norm = MinMaxScaler().fit_transform(series)
            series_norm = pd.DataFrame(series_norm, index=series.index , columns = series.columns) 
            # exponential smoothing
            optimum_a = optimum_al(series_norm)
            _,smoothed_series = exponential_smooth(series_norm, optimum_a)
            
                
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
                    epochs = 5000
                
                    # call the model constructor
                    mod=m.model_constructor
                    #------------ I can also test the same model with all the other frequencies !
                    if(m.freq_name=='daily'):
                        cur_model,epochs,batch_size = mod(series_length,2,m.horizon,epochs=epochs)
                    elif(m.freq_name == 'weekly'):
                        cur_model,epochs,batch_size = mod(series_length,2,m.horizon,epochs=epochs)
                    elif(m.freq_name == 'monthly'):
                        cur_model,epochs,batch_size = mod(series_length,2,m.horizon,epochs=epochs)
                    elif(m.freq_name == 'quarterly'):
                        cur_model,epochs,batch_size = mod(series_length,2,m.horizon,epochs=epochs)
                    else :
                        cur_model,epochs,batch_size = mod(series_length,1,m.horizon)   
                    
                    
                    # pick dataset depending on frequency 
                    train_dataset,val_dataset = dataset_picker(smoothed_series,m.freq_name,frequencies,cur)
                    
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

################### ONLY AS MAIN ###################             
if __name__ =="__main__":
    
    frequencies = {
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M',
        'quarterly': 'Q',
        'yearly': 'Y'
    }
    
    # train the smooth cnn 
    # 1) exponential smooth
    # 2) training 
    train_smooth_cnn(frequencies)
    
    
    
    # predict 
    
    
                    