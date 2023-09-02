#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  3 01:44:26 2023

@author: st_ko
"""

import os 
import matplotlib.pyplot as plt


def plotLoss(history,horizon_step,series_length,freq_name):
    # create the plots directories 
    if not os.path.exists('plots'):
        os.mkdir('plots')
    fig=plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss : ' + ' '+freq_name +' '+ str(series_length) +' ' + str(horizon_step) )
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    if not (os.path.exists(os.path.join('plots','Loss',freq_name ,str(series_length)) ) ):
        os.makedirs(os.path.join('plots','Loss',freq_name ,str(series_length)))
    plt.savefig(os.path.join('plots','Loss',freq_name ,str(series_length)) +'/'+ str(horizon_step)+ '.png')
    plt.close(fig)
    #plt.show()
