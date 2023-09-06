#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 01:56:15 2023

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


def quarterly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):
    input = ks.layers.Input((series_length,))
    weekly_input = ks.layers.Reshape((-1,series_length,1))(input)

    # 4 quarters
    if(series_length == 4):
            conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
            fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))

            #adding pairs of 2 successive months history
            conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
            conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
            output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv3))

    elif(series_length == 8):
            conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
            fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))

            #adding pairs of 2 successive months history
            conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
            conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
            conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
            conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
            output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv5))
    else:
            # 12,24,36
            conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
            fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))

            #adding pairs of 2 successive months history
            conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
            # stack on top convolutional of 3 successive days after conv2
            conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
            # addd conv, 2 successive pairs with stride = 1 on top
            # or maybe add kernel 1,3 instead again
            conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
            conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
            output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv5))


    # concatenate both outputs
    # we get shape (None,2)
    comb = tf.keras.layers.Concatenate()([fc_1,output1])
    # weighted combination of both the outputs
    output = tf.keras.layers.Dense(1, input_shape=(None, comb.shape[-1]))(comb)
    
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    
    epochs = epochs
    batch_size =  bs
    return est, epochs, batch_size
    
    
'''
def quarterly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):
    
    
    
    input = ks.layers.Input((series_length,))
    quarterly_input = ks.layers.Reshape((series_length, 1))(input)

    # instead of taking the average of year let's take the average of each pair of 2 quarters
    quarterly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='valid')(quarterly_input)


    # naive prediction will be taken into account
    # WE WILL TEST BY ADDING THE NAIVE PREDICTIONS AS WELL (IN A LESSER EXTENT)
    # AND ALSO WITHOUT THE NAIVE
    naive_1 = tf.roll(input, axis=0, shift=1)
    # feed naive into a fcn
    naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive_1))
    naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
    naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)
    


    quarterly_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(quarterly_avg))
    quarterly_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(quarterly_hidden1)
    quarterly_output = ks.layers.Dense(units=1, activation='linear')(quarterly_hidden2)


    # ?? size = 4 ? or 2
    quarterly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(quarterly_avg)


    periodic_diff = ks.layers.Subtract()([input, ks.layers.Flatten()(quarterly_avg_up)])

    periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)





    # change convolutional filters
    periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),
                                     padding='valid')(periodic_input)

    # change units
    periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
    periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
    periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)

    # TODO : TEST WITH AND WITHOUT THE NAIVE ADDED
    
    #output = ks.layers.Add()([quarterly_output, periodic_output,naive_output ])
    
    # i also combine the naive solution and feed all the outputs into an average aggregate 
    output = tf.keras.layers.Average()([quarterly_output, periodic_output,naive_output ])


    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.001), loss='mse', metrics=['mse','accuracy'])
    epochs = epochs
    batch_size = bs
    return est, epochs, batch_size
'''


# old model
'''
def weekly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):

    # if we make the assumption of full year (with february + 1 day)
    if series_length == 52:
        
        
        # Reshape into (X,1)
        input = ks.layers.Input((series_length,1))

    
        # average of input for one year 
        # maybe pool every 2 weeks and not from all the 52 weeks 
        yearly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=1, padding='valid')(input)
        # reshape into (1,1)
        yearly_avg2 = ks.layers.Reshape((1,1))(yearly_avg)
        
        
        
        # usampling # 
        yearly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(yearly_avg)
        
        
        periodic_diff = ks.layers.Subtract()([input, yearly_avg_up])
        
        
        periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)


        periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),padding='valid')(periodic_input)
        
        periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
        periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
        periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)


        # pass the naive predictions through a fcn
        naive_1 = tf.roll(input, axis=0, shift=1)
        naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive_1))
        naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
        naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)
        
        # final output 
        output = ks.layers.Add()([periodic_output, naive_output])
        
    
        est = ks.Model(inputs=input, outputs=output)
        est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse','accuracy'])
        epochs = epochs
        batch_size = bs


    else:

        #
        input = ks.layers.Input((series_length,))
        yearly_input = ks.layers.Reshape((series_length, 1))(input)

        # calculate the naive solution
        naive_1 = tf.roll(input, axis=0, shift=1)

        # yearly average
        yearly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='valid')(yearly_input)
        yearly_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(yearly_avg))
        yearly_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(yearly_hidden1)
        yearly_output = ks.layers.Dense(units=1, activation='linear')(yearly_hidden2)


        # usampling # 
        yearly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(yearly_avg)


        periodic_diff = ks.layers.Subtract()([input, ks.layers.Flatten()(yearly_avg_up)])


        periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)


        periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),
                                         padding='valid')(periodic_input)
        
        periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
        periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
        periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)
        
        
        
        # pass the naive predictions through a fcn
        naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive_1))
        naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
        naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)
        
        output = ks.layers.Add()([yearly_output, periodic_output])
        #output = tf.keras.layers.Average()([yearly_output, periodic_output, naive_output ])
        
        est = ks.Model(inputs=input, outputs=output)
        est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse','accuracy'])
        epochs = 250
        batch_size = 1000
    return est, epochs, batch_size
'''


# new weekly model 
def weekly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):

    # if we make the assumption of full year (with february + 1 day)
    if series_length == 13:
        
        
        # Reshape into (X,1)
        input = ks.layers.Input((series_length,))
        weekly_input = ks.layers.Reshape((-1,series_length,1))(input)


        # ------ Try only with a convolutional model -----------#
        
        # both 2 last weeks influence 
        # we get (None,1,2,32)  // how much each week influences the result
        # convolutional on  groups of 4 weeks (in each group we take the last week as well)
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(weekly_input)
        fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))
        
    
        #adding pairs of 2 successive weeks history 
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        
        
        
        # stack on top convolutional of 3 successive weeks after conv2
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        
        
        # addd conv, 2 successive pairs with stride = 1 on top 
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        
        

        # add conv , 3 successive pairs with stride = 1 on top
        conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
        
        
        
        # now we have 3 units 
        # we will flatten them and feed to a fully connected network
        output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv5))
        
        
        # concatenate both outputs 
        # we get shape (None,2)
        comb = tf.keras.layers.Concatenate()([fc_1,output1])
        
        # weighted combination of both the outputs 
        #comb_output = Weighted_add(1)(comb)
        output = tf.keras.layers.Dense(1, input_shape=(None, comb.shape[-1]))(comb)
        
    elif (series_length == 26) :
        
        
        # Reshape into (X,1)
        input = ks.layers.Input((series_length,))
        weekly_input = ks.layers.Reshape((-1,series_length,1))(input)

    
        # ------ Try only with a convolutional model -----------#
        
        # groups of 4 weeks --> first output 
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(weekly_input)
        conv1b = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(conv1)
        fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1b))
        
        # second convolutional model 
        #adding pairs of 2 successive weeks history 
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        # addd conv, 2 successive pairs with stride = 1 on top 
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        
        # add conv , 3 successive pairs with stride = 1 on top
        conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
        conv6 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,3),padding = 'valid' ,use_bias = True)(conv5)
        conv7 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv6)
        output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv7))
        
        comb = tf.keras.layers.Concatenate()([fc_1,output1])
        
        # weighted combination of both the outputs 
        #comb_output = Weighted_add(1)(comb)
        
        output = tf.keras.layers.Dense(1, input_shape=(None, comb.shape[-1]))(comb)
        
    elif (series_length == 52) :
        

        # Reshape into (X,1)
        input = ks.layers.Input((series_length,))
        weekly_input = ks.layers.Reshape((-1,series_length,1))(input)


        # ------ Try only with a convolutional model -----------#
        
        # groups of 4 weeks --> first output 
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(weekly_input)
        conv1b = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,3), padding = 'valid',use_bias=True,)(conv1)
        conv1c = ks.layers.Conv2D(filters = 32,kernel_size=(1,3),strides = (1,1), padding = 'valid',use_bias=True,)(conv1b)
        fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1c))
        
        
        # second convolutional model 
        #adding pairs of 2 successive weeks history 
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        # addd conv, 2 successive pairs with stride = 1 on top 
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        
        # add conv , 3 successive pairs with stride = 1 on top
        conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
        conv6 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,3),padding = 'valid' ,use_bias = True)(conv5)
        conv7 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv6)
        conv8 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,3),padding = 'valid' ,use_bias = True)(conv7)
        #conv9 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv8)
        
        
        output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv8))
        
        comb = tf.keras.layers.Concatenate()([fc_1,output1])
        
        # weighted combination of both the outputs 
        #comb_output = Weighted_add(1)(comb)
        
        output = tf.keras.layers.Dense(1, input_shape=(None, comb.shape[-1]))(comb)    
    
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    epochs = 250
    batch_size = 1000    
 
        
    return est, epochs, batch_size



# i start from daily model 
'''  old model (sos)
def daily_model(series_length,yearly_count,filter_count,units_count,epochs,bs):
    
    
    
    input = ks.layers.Input((series_length,))
    weekly_input = ks.layers.Reshape((series_length,1))(input)
    
    
    
    
    # naive model 
    naive_1 = tf.roll(input, axis=0, shift=1)
    # feed naive into a fcn
    naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive_1))
    naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
    naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)
    
    
    
    
    
    # add simple regression parameters (single layer perceptron)
    # maybe test higher powers 
    y_linear_reg = ks.layers.Dense(units=1,activation = 'linear')(input)
    
    y_linear_reg = ks.layers.BatchNormalization()(y_linear_reg)
    
    # weekly avg # 
    weekly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='valid')(weekly_input)
    
    
    weekly_avg = ks.layers.BatchNormalization()(weekly_avg)
    
    #weekly_hidden1 = ks.layers.Dense(units=series_length, activation='relu')(ks.layers.Flatten()(weekly_avg))
    weekly_hidden1 = ks.layers.Dense(units=series_length, activation='relu')(weekly_avg)
    weekly_output = ks.layers.Dense(units=1, activation='linear')(weekly_hidden1)
    # average upsampling 
    weekly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(weekly_avg)
    
    

    # subtract from the naive solution the average and not from the original series 
    periodic_diff = ks.layers.Subtract()([weekly_input,weekly_avg_up])
    
    periodic_diff = ks.layers.BatchNormalization()(periodic_diff)
    
    
    # tensor and convolution 
    periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)
    
    # convolutional 
    periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),padding='valid')(periodic_input)
    periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
    periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
    periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)
    
    #periodic_output = ks.layers.BatchNormalization()(periodic_output)
    
    
    #output = ks.layers.Add()([weekly_output, periodic_output,naive_output])
    #output = tf.keras.layers.Average()([weekly_output, periodic_output,y_linear_reg])
    output = tf.keras.layers.Add()([weekly_output , naive_output])
    
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    
    
    
    epochs = epochs
    batch_size =  bs
    return est, epochs, batch_size
'''

# new convolutional daily model 
def daily_model(series_length,yearly_count,filter_count,units_count,epochs,bs):
    
    
    
    input = ks.layers.Input((series_length,))
    weekly_input = ks.layers.Reshape((-1,series_length,1))(input)
    
    
    
    # ------ Try only with a convolutional model -----------#
    
    # both 2 last weeks influence 
    # we get (None,1,2,32)  // how much each week influences the result
    conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,7),strides = (1,7), padding = 'valid',use_bias=True,)(weekly_input)
    fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))
    
    
    #adding pairs of 2 successive days history 
    conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
    
    # stack on top convolutional of 3 successive days after conv2
    conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
    
    
    # addd conv, 2 successive pairs with stride = 1 on top 
    conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
    
    
    # add conv , 3 successive pairs with stride = 1 on top
    conv5 = ks.layers.Conv2D(filters = 128, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
    
    # now we have 3 units 
    # we will flatten them and feed to a fully connected network
    output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv5))
    
    
    # concatenate both outputs 
    # we get shape (None,2)
    comb = tf.keras.layers.Concatenate()([fc_1,output1])
    
    # weighted combination of both the outputs 
    #comb_output = Weighted_add(1)(comb)
    
    output = tf.keras.layers.Dense(1, input_shape=(None, comb.shape[-1]))(comb)
    
    
    #output = ks.layers.Add()([weekly_output, periodic_output,naive_output])
    #output = tf.keras.layers.Average()([weekly_output, periodic_output,y_linear_reg])
    #output = tf.keras.layers.Add()([weekly_output , naive_output])
    
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    
    
    
    epochs = epochs
    batch_size =  bs
    return est, epochs, batch_size


'''
def monthly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):


    input = ks.layers.Input((series_length,))
    yearly_input = ks.layers.Reshape((series_length, 1))(input)


    # pool average of 1 year
    # 12 months
    yearly_avg = ks.layers.AveragePooling1D(pool_size=yearly_count, strides=yearly_count, padding='valid')(yearly_input)

    # 50 units fcn
    yearly_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(yearly_avg))
    yearly_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(yearly_hidden1)

    # output 1 unit fcn
    yearly_output = ks.layers.Dense(units=1, activation='linear')(yearly_hidden2)




    # from each month subtract the average of the corresponding year
    yearly_avg_up = ks.layers.UpSampling1D(size=yearly_count)(yearly_avg)
    # upsample and create the differences
    periodic_diff = ks.layers.Subtract()([input, ks.layers.Flatten()(yearly_avg_up)])


    # create tensor for cnn
    periodic_input = ks.layers.Reshape((series_length // yearly_count, yearly_count, 1))(periodic_diff)


    periodic_conv = ks.layers.Conv2D(filters=filter_count, kernel_size=(1, yearly_count), strides=(1, yearly_count),
                                     padding='valid')(periodic_input)


    periodic_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(periodic_conv))
    periodic_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(periodic_hidden1)
    periodic_output = ks.layers.Dense(units=1, activation='linear')(periodic_hidden2)


    # we are going to use each k month
    # and use the value for the k+1 month as an additional term to be added in the output
    # (naive method term)
    # substitute each (k+1) month with k month
    # first shift each month value towards bigger index (each value moves one time step forward)
    # THE LAST MONTH WILL GO CYCLICALLY AND REPLACE THE 1ST MONTH !!! THIS IS NOT A GOOD IDEA
    # SINCE WE LOSE THE 1ST MONTH
    # MAYBE WE SHOULD COPY THE 1ST MONTH VALUE (WHICH WAS SHIFTED TO THE 3ND VALUE)
    # BACK TO THE 1ST VALUE AS WELL
    naive= tf.roll(input,shift=1,axis=0)
    # pass the naive predictions through a fcn
    naive_hidden1 = ks.layers.Dense(units=units_count, activation='relu')(ks.layers.Flatten()(naive))
    naive_hidden2 = ks.layers.Dense(units=units_count, activation='relu')(naive_hidden1)
    naive_output = ks.layers.Dense(units=1, activation='linear')(naive_hidden2)

    # output 1 unit fcn
    #yearly_output = ks.layers.Dense(units=1, activation='linear')(yearly_hidden2)


    # i also add the naive output in a lesser extent
    #output = ks.layers.Add()([yearly_output, periodic_output,0.4 * naive_output])
    output = tf.keras.layers.Average()([yearly_output, periodic_output,naive_output ])


    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse','accuracy'])
    epochs = 250
    batch_size = 1000
    return est, epochs, batch_size
'''



def monthly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):

    input = ks.layers.Input((series_length,))
    weekly_input = ks.layers.Reshape((-1,series_length,1))(input)


    # how much each month influences the result
    if(series_length == 6):

        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,6),strides = (1,6), padding = 'valid',use_bias=True,)(weekly_input)
        fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))

        #adding pairs of 2 successive months history
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        # stack on top convolutional of 3 successive days after conv2
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        # addd conv, 2 successive pairs with stride = 1 on top
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv4))


    elif(series_length == 8):
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,8),strides = (1,8), padding = 'valid',use_bias=True,)(weekly_input)
        fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))

        #adding pairs of 2 successive months history
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv4))

    else:
        # 12,24,36
        conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,6),strides = (1,6), padding = 'valid',use_bias=True,)(weekly_input)
        fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))

        #adding pairs of 2 successive months history
        conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
        # stack on top convolutional of 3 successive days after conv2
        conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
        # addd conv, 2 successive pairs with stride = 1 on top
        conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,3),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
        conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
        # *** maybe add again the same layers as in 6 months case here

        output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv5))


    # concatenate both outputs
    # we get shape (None,2)
    comb = tf.keras.layers.Concatenate()([fc_1,output1])
    # weighted combination of both the outputs
    output = tf.keras.layers.Dense(1, input_shape=(None, comb.shape[-1]))(comb)
    
    est = ks.Model(inputs=input, outputs=output)
    est.compile(optimizer=ks.optimizers.Adam(lr=0.01), loss='mse', metrics=['mse'])
    
    epochs = epochs
    batch_size =  bs
    return est, epochs, batch_size



# new quarterly model
def yearly_model(series_length,yearly_count,filter_count,units_count,epochs,bs):
        input = ks.layers.Input((series_length,))
        weekly_input = ks.layers.Reshape((-1,series_length,1))(input)
        # 4 quarters
        if(series_length == 2):
                fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(weekly_input))
    
                #adding pairs of 2 years
                # maybe we dont need this since it is still the same fcn from above -- > conv 2->1 or fcn 2 -> 1 ? they are the same
                # conv1 and fc_1 here are the same
                
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,2),strides = (1,1), padding = 'valid',use_bias=True)(weekly_input)
                output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))
                comb = tf.keras.layers.Concatenate()([fc_1,output1])
    
    
        elif(series_length == 3):
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,3),strides = (1,3), padding = 'valid',use_bias=True)(weekly_input)
                fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))
    
    
                #adding pairs of 2 successive months history
                conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
                conv3 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
                output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv3))
                comb = tf.keras.layers.Concatenate()([fc_1,output1])
                
        elif(series_length == 6):
                # using half a year weighting
                # maybe use strides 3 instead of 6 , here
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,6),strides = (1,6), padding = 'valid',use_bias=True,)(weekly_input)
                fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))
    
                #adding pairs of 2 successive months history
                conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
                # stack on top convolutional of 3 successive days after conv2
                conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
                # addd conv, 2 successive pairs with stride = 1 on top
                # or maybe add kernel 1,3 instead again
                conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
                conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
                output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv5))
                comb = tf.keras.layers.Concatenate()([fc_1,output1])
    
        elif(series_length == 8):
                # weighting of groups of 4 years ! maybe use 8 years grouping instead
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
                fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))
    
                #adding pairs of 2 successive months history
                conv2 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
                # stack on top convolutional of 3 successive days after conv2
                conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv2)
                # addd conv, 2 successive pairs with stride = 1 on top
                # or maybe add kernel 1,3 instead again
                conv4 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
                # maybe change the stride here 
                conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,1),padding = 'valid' ,use_bias = True)(conv4)
                output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv5))
                comb = tf.keras.layers.Concatenate()([fc_1,output1])
        else:
                # 12 years
                # weighting of groups of 4 years ! maybe use 8 years grouping instead
                conv1 = ks.layers.Conv2D(filters = 32,kernel_size=(1,4),strides = (1,4), padding = 'valid',use_bias=True,)(weekly_input)
                fc_1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv1))
    
                conv2 = ks.layers.Conv2D(filters = 32,kernel_size=(1,6),strides = (1,6), padding = 'valid',use_bias=True,)(weekly_input)
                fc_2 = ks.layers.Dense(1)(ks.layers.Flatten()(conv2))
    
                #adding pairs of 2 successive months history
                conv3 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(weekly_input)
                # stack on top convolutional of 3 successive days after conv2
                conv4 = ks.layers.Conv2D(filters = 32, kernel_size = (1,2),strides=(1,1),padding = 'valid' ,use_bias = True)(conv3)
                # addd conv, 2 successive pairs with stride = 1 on top
                # or maybe add kernel 1,3 instead again
                conv5 = ks.layers.Conv2D(filters = 64, kernel_size = (1,4),strides=(1,2),padding = 'valid' ,use_bias = True)(conv4)
                conv6 = ks.layers.Conv2D(filters = 64, kernel_size = (1,2),strides=(1,2),padding = 'valid' ,use_bias = True)(conv5)
                output1 = ks.layers.Dense(1)(ks.layers.Flatten()(conv6))
                comb = tf.keras.layers.Concatenate()([fc_1,fc_2,output1])
    
        # weighted combination of both the outputs
        
        output = tf.keras.layers.Dense(1, input_shape=(None, comb.shape[-1]))(comb)
        
        est = ks.Model(inputs=input, outputs=output)
        est.compile(optimizer=ks.optimizers.Adam(lr=0.001), loss='mse', metrics=['mse'])
        
        epochs = epochs
        batch_size =  bs
        return est, epochs, batch_size




# build the models
def build_Model():
    Model = namedtuple('Model', ['freq_name', 'horizon', 'freq', 'model_constructor', 'training_lengths',
                                 'cycle_length', 'augment'])
    
    
    # just for testing i comment out the other models to run only the daily model 
    # we give those training lengths for training 
    # use training length --> 1 week,2 weeks,3 weeks,1 month, 2 months, 3 months ,1 year 
    daily = Model('daily', 14, 1, daily_model, [7,14,21,28,56,84,364], 7, True)
    #daily = Model('daily', 14, 1, daily_model, [7], 7, True)
    
    # we rougly predict 3 months so 
    # use training length --> 4 weeks, 8 weeks , 13 weeks, 52 weeks , 
    weekly = Model('weekly', 13, 1, weekly_model, [13,26,52], 52, True)
    # why 48 , 120 , 240 
    monthly = Model('monthly', 18, 12, monthly_model, [6, 8, 12,24,36], 12, False)
    yearly = Model('yearly', 6, 1, yearly_model, [2, 3, 6, 8, 12,18,24], 1, False)
    quarterly = Model('quarterly', 8, 4, quarterly_model, [4, 8, 12], 4, False)
    
    # return [daily,monthly,quarterly,weekly,hourly,yearly]
    return [yearly]