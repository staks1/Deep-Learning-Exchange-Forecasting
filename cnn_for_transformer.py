#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 23:10:17 2023

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
from keras_nlp.layers import PositionEmbedding
from tensorflow import keras
from tensorflow.keras import layers
from keras.utils import plot_model 





# function to try to feed the embeddings to the transformer 
# and pass them through , then provide a mlp head to make the predictions 
# I TRY TO FOLLOW THE IDEA FROM THE PAPER 
# 1) 1 NORMALIZATION 
# 2) 1 MULTI HEAD ATTENTION HEAD
# 3) OUTPUT OF MULTI HEAD + RESIDUAL FROM NORMALIZATION LAYER(1)
# 4) NORMALIZATION
# 5) MLP FROM (4) AND REDISUAL FROM (3)
# THE PAPER USES DEPTH = 4 , 4 SELF ATTENTION HEADS , EMBEDDING DIMENSION OF 128  , DROPOUT = 0.3

# head size = embedding dimension ?? in paper 128 is used 
# num_heads = 4 
def transformer_encoder(x, embedding_dimension=128,dropout=0.3, num_heads=4):
    
    
    # x dim (batch,number,features)
    # 1st normalization layer 
    x1 = layers.LayerNormalization(epsilon=1e-6)(x)
    
    
    # MULTI HEAD ATTENTION 
    # key_dim = query and key dimension 
    x2 = layers.MultiHeadAttention(
        key_dim=embedding_dimension, num_heads=4, dropout=dropout
    )(x1, x1)
    
    
    # add residual 
    x3 = x + x2
    
    # 2nd normalization layer 
    x4 = layers.LayerNormalization(epsilon=1e-6)(x3)
    
    # add dense layer here 
    # i skipped the MLP HERE > HOW CAN MLP BE IMPLEMENTED . SHOULD WE THEN RESHAPE INTO (BATCH,NUMBER,FEATURES) ?? 
    #TODO : MAYBE I SHOULD ADD THE MLP HERE 
    #x6 = ks.layers.Dense(x5.shape[0]*x5.shape[1])(x5)
    
    # ADD DROPOUT
    x5 = layers.Dropout(dropout)(x4)
    
    return x5 


# implement the stacking of transformer blocks 
# define the shape of input here 
# i use dim since we try to predict one time step 
# i will change dim to horizon if we want to create a multi step model 
def transformer_model(inputs,num_transformer_blocks=4,dim=1,dropout=0.3,mlp_dropout=0.3,epochs=200):
    
    #inputs = keras.Input(shape=input_shape)
    
    x = inputs
    
    # for all stacked transformer blocks 
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x=x, embedding_dimension=128, dropout=0.3)

    # pool for all time steps in the batch
    # for example if i have 5 timesteps in the batch --> shape (batch_size,5,feature_dimensions)
    # after the global pooling 1d i will have --> (batch_size,1,feature_dimensions) --> it takes the average across all time steps in each feature dimension
    # NOT SURE IF WE NEED TO REDUCE THE DIMENSIONS TO A POOLED (BATCH_SIZE,1,AVERAGED_FEATURES )
    #x = tf.reshape(x,(x.sa,x.shape[1],x.shape[2]))
    x = layers.GlobalAveragePooling1D()(x)
    
    # try to predict the time step ?? 
    # try to predict the next time step only (single step model ???)
    x = layers.Dropout(mlp_dropout)(x)
    
    outputs = layers.Dense(dim, activation="relu")(x)
    
        
    
    return outputs



# this model should create the embedding ofthe times series 
# based on paper cnn kernel_size = 16 , stride = 8 
# for series_length here we will feed the full time_series 

# pick dimensionality of splitting (how many subseries to we have ??)
# FOR NOW I WILL PICK 6320 / 40 = 158 , SO I GET 158 SUBSERIES , EACH OF DIMENSION 40 
# SO I CAN SLIDE THE PAPER DEFINED KERNEL INTO EACH SUBSERIES


################## TODO : TRY THE FOLLOWING SPLIT ############################
# THIS WAY WE KIND OF TRY TO FIND THE SHORT TERM PATTERS INSIDE THIS FREQUENCY SUB SERIES 
# E.X INSIDE EACH WEEK , OR IN A MONTH 
# TODO : SHOULD TRY THE FREQUENCY AS A NUMBER TO SPLIT THE SERIES 
# FOR THE DAILY SERIES WE HAVE 6320 AND IF WE ASSUME A PERIOD OF 5 DAYS 
# WE CAN SPLIT INTO A NUMBER-GROUPS OF WEEKS (5 DAYS)
# SO I WILL USE THE FREQUENCY  OF 5 BUT DOUBLE IT TO GET SMALLER EMBEDDINGS 
# 6320  / 5 = 1264
# SO WE HAVE 1264 EMBEDDINGS , EACH OF DIMENSION  1264
# NEXT WE WILL SLIDE THE KERNEL DEFINED IN THE PAPER IN EACH GROUP OF 
##############################################################################

# THE FOLLOWING MODEL ASSUMES WE HAVE SERIES LENGTH THE WHOLE SERIES = 6320 (or + -1 element to make the series appropriate to split )
# divisor will be 40 here 
def cnnt_model(series_length,divisor,bs=2,epochs=1000):
    
    # TODO : CHECK AT ALL TIMES THAT THE VALUES ARE < 1 (AND FLOAT)
    # BEACUSE WE NEED THEM TO BE FLOATS TO BE ADDED WITH THE POSITIONAL EMBEDDINGS
    
    input = ks.layers.Input((series_length,1))
    
    input_t = tf.reshape(input,(1,series_length//divisor,divisor,1))
    # or  try valid padding ?
    conv1 = tf.keras.layers.Conv2D(filters = 1 ,kernel_size=(1,16) ,strides = (1,8),padding='same')(input_t)
    #output = ks.layers.Dense(horizon,activation='relu')(flatten1)
    # not sure if this reshaping is needed
    token_embeddings = tf.reshape(conv1,(conv1.shape[1],conv1.shape[2]))
    
    
    
    
    ##### let's create the positional embeddings for the series of 158 subseries ###
    position_embeddings = PositionEmbedding(sequence_length=token_embeddings.shape[0])(token_embeddings)
    outputs = token_embeddings + position_embeddings
    # reshape again into (batch,time_steps, features)
    outputs = tf.reshape (outputs,(1,outputs.shape[0],outputs.shape[1]))
    
    
    #------- call the transformer function to implement the transformer model ------------------------------------------------#
    outputs = transformer_model(outputs,num_transformer_blocks=4,dim=1,dropout=0.3,mlp_dropout=0,epochs=200)
    #-------------------------------------------------------------------------------------------------------------------------#
    
    
    est = ks.Model(inputs=input, outputs=outputs,name='cnnt')
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    epochs = epochs
    batch_size =  bs
    return est, epochs, batch_size
    
    





#################################### build the models #################################################
def Cnnt():
    Model = namedtuple('Model', ['freq_name', 'horizon', 'freq', 'model_constructor', 'training_lengths',
                                 'cycle_length', 'augment'])
    daily = Model('daily', 14, 1, cnnt_model, [14,20,240], 7, True)
    #weekly = Model('weekly', 13, 1, cnnt_model, [52,13,26], 52, True)
    #monthly = Model('monthly', 18, 12, cnnt_model, [24,18], 12, False)
    #quarterly = Model('quarterly', 8, 4, cnnt_model, [12,8], 4, False)
    #yearly = Model('yearly', 6, 1, cnnt_model, [6], 1, False)
    
    
    return [daily]
    
######################### ONLY AS MAIN ################################################################

if __name__ == "__main__":
    model1 = Cnnt()
    m = model1[0].model_constructor
    m1,ep,bs = m(6320,40)
    #print(m1.summary())
    #plot_model(m1,to_file='/home/st_ko/Desktop/model.jpg')
    print(m1.summary())
    