#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 01:44:26 2023

@author: st_ko
"""

import os 
import matplotlib.pyplot as plt
import pandas as pd
import glob
from  build_model import *
from build_multistep_model import * 



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





# function to plot the loss for each series_length 
# sigle step model
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


# function for training 

#-------------- BEGIN TRAINING ---------------------------------------#


# function not created yet 
# to create the models for each frequency 


###############################################
#------------ BEGIN TRAINING LOOP ------------#
###############################################

def train_models_single_step(frequencies):
    
        # call build model to create the models 
        models = build_Model()
        
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
                    ks.backend.clear_session()
                    tf.compat.v1.reset_default_graph()
                    # create keras session #
                    np.random.seed(0)
                    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                    tf.compat.v1.set_random_seed(0)
                    tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
                    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_conf))
                    mod=m.model_constructor
                    
                    #------------Pick models' parameters-------------------------------------#
                    if(m.freq_name=='daily'):
                        cur_model,epochs,batch_size = mod(series_length,7,3,250,400,20)
                    elif(m.freq_name == 'weekly'):
                        cur_model,epochs,batch_size = mod(series_length,52,4,52,250,20)
                    elif(m.freq_name == 'monthly'):
                        cur_model,epochs,batch_size = mod(series_length,12,6,50,250,20)
                    elif(m.freq_name == 'quarterly'):
                        cur_model,epochs,batch_size = mod(series_length,4,4,50,65,20)
                    else :
                        # 'Yearly'
                        cur_model,epochs,batch_size = mod(series_length,2,4,20,400,20)
                    #-----------------------------------------------------------------------#
                    # set batch size = 20 
                    history = cur_model.fit(x_train, cur_y_train, epochs=epochs, batch_size=5, shuffle=True, validation_split=0.3,
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
                    
                    
                    
# function to train with cross validation , for single step models 
# TODO : SHOULD FIX THE FINAL LOSSES LISTS 
# SO WE GET ONE LIST OF THE K-FOLD ERRORS FOR EACH TIME STEP FOR EACH MODEL
def train_single_step_cv(frequencies):
        # build models 
        models = build_Model()
        
        
        for m in models :
            
            # read series corresponding to frequency 
            series = frequencies[m.freq_name][1]
            # standardize is used 
            series,_,_ = standardize(series)
            series_length= m.training_lengths[-1]
            
            
            
            
            all_series = np.zeros((series.shape[0], series_length + m.horizon ))
    
            
            # fill new array of series 
            all_series[:,:] = series.iloc[:, -(series_length + m.horizon):]
            
            # set x,y train (training + horizon )   
            
            # create groups of series for k fold cross validation
            
            x_train = all_series[:,:-m.horizon]
            y_train = all_series[:,-m.horizon:]
            
            # pick a random indices split so i can get some  pseudorandom splits in each fold 
            groups = [1, 1, 2, 2, 3, 3, 3, 4,5,5,5,6,6,6,6,6,6]
            gss = GroupShuffleSplit(n_splits=5, train_size=0.85 )
            
        
            results = []
            
            
            # define model losses list
            model_losses = []
            
            # for each horizon step 
            for horizon_step in range(m.horizon) :
                
                    mod = m.model_constructor
                    
                    # call each model 
                    # here i run the yearly only for example 
                    # i should customize it with all the frequencies 
                    cur_model,epochs,batch_size = mod(series_length,2,4,20,400,20)
                    
                    # wrap keras model around our model to apply cross validation 
                    keras_model = KerasRegressor(cur_model)
                
                    # Perform grouped shuffle split cross-validation
                    for train_idx, test_idx in gss.split(x_train, y_train, groups=groups):
                            print(train_idx)
                            print(test_idx)
                            
                            
                            X_train, X_test = x_train[train_idx], x_train[test_idx]
                            Y_train, Y_test = y_train[train_idx,horizon_step], y_train[test_idx,horizon_step]
                            
                            
                            #keras_model.reset_states()
                            
                            # Fit the Keras model on the training data
                            keras_model.fit(X_train, Y_train)
                            
                            # Get predictions on the test data
                            Y_pred = keras_model.predict(X_test)
                            
                            # Calculate the mean squared error (MSE) as an example of an evaluation metric
                            mse = mean_squared_error(Y_test, Y_pred)
                            model_losses.append(mse)
                            
                            
                     
                            
                     
# function to train multi time-step models
# -> one single model predicts the whole horizon each time

def train_models_multi_step(frequencies):
    
        # call build model to create the models 
        # probably we need new model scrip here
        # which will have the multi step models only
        
        models = build_m_Model()
        
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
        
                
                # fill new array of series 
                all_series[:,:] = series.iloc[:, -(series_length + m.horizon):]
                
                # set x,y train (training + horizon )        
                x_train = all_series[:,:-m.horizon]
                y_train = all_series[:,-m.horizon:]
                
            
               
                ks.backend.clear_session()
                tf.compat.v1.reset_default_graph()
                # create keras session #
                np.random.seed(0)
                session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
                tf.compat.v1.set_random_seed(0)
                tf_session = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
                tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=session_conf))
                
                # call the model constructor
                mod=m.model_constructor
                
                #------------Pick models' parameters-------------------------------------#
                if(m.freq_name=='daily'):
                    cur_model,epochs,batch_size = mod(series_length,7,3,250,400,20,m.horizon)
                elif(m.freq_name == 'weekly'):
                    cur_model,epochs,batch_size = mod(series_length,52,4,52,250,20,m.horizon)
                elif(m.freq_name == 'monthly'):
                    cur_model,epochs,batch_size = mod(series_length,12,6,50,250,20,m.horizon)
                elif(m.freq_name == 'quarterly'):
                    cur_model,epochs,batch_size = mod(series_length,4,4,50,65,20,m.horizon)
                else :
                    # 'Yearly'
                    cur_model,epochs,batch_size = mod(series_length,2,4,20,400,20,m.horizon)
                #-----------------------------------------------------------------------#
                # set batch size = 20 
                history = cur_model.fit(x_train, y_train, epochs=epochs, batch_size=20, shuffle=True, validation_split=0.3,
                    callbacks=[ ks.callbacks.EarlyStopping(monitor='val_loss', patience=100)])
                
                
                # plot Loss for this horizon step 
                #plotLoss(history,horizon_step,series_length,m.freq_name)
                
                #########################
                #----save the model ----#
                #########################
                #TODO : SHOULD FIX BUG HERE 
                # IT CREATES ANOTHER DIRECTORY WITH CONCATENATED THE FREQUENCY AND 'MULTI_STEP'
                if not os.path.exists('../trained_models'+ '/multi_step'+ m.freq_name):
                        os.mkdir('../trained_models'+ '/multi_step'+ m.freq_name)
                        
                model_file = os.path.join('../trained_models','multi_step',m.freq_name,
                                              '{}_length_{}.h5'.format(m.freq_name, series_length))
                # save model 
                cur_model.save(model_file) 
                        
    