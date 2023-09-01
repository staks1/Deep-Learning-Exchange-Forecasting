#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:34:40 2023

@author: st_ko
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import os 



# plot the currencies within a given range of observations 
def plot_Series(data,size):
    # annotate a subpart of the series 
    plt.figure()
    for col in data.columns:
        for i in range(size):
            plt.annotate(data[col][i], (data[col].index[i], data[col].values[i]),
                         textcoords="offset points", xytext=(0, 10), ha='center',fontsize=8,rotation = 45)
        plt.plot(data[col][:size],'o-',label=col)
        plt.legend(loc='lower right',bbox_to_anchor=(1.05,0.2))


# standardize values 
def standardize(data):
    standardized = []    
    # dol = data['USD']
    for col in data.columns:
        stdc = data[col].std()
        meanc = data[col].mean()
        standardized.append( (data[col] - meanc )/ stdc)
    standardized_data = pd.DataFrame(standardized).transpose()
    return standardized_data   



# ! More cuztomized 
# create a nested dictionary of year -> dictionary of months --> observations of this month in this year 

def create_nested_dict(data):
        total_time= [x for x in data.index]
        year_dict = {x: {x2 : [] for x2 in iter(range(1,13)) } for x in range(data.index.min().year,data.index.max().year+ 1)}
        m_step = data.index.min().month
        y_step = data.index.min().year
        y=True
        m = True
        
        # create nested dictionary 
        # i will turn this later into a function 
        for i in range(len(total_time)-1) :
            if(total_time[i].year == total_time[i+1].year):
                y = True
                if(total_time[i].month == total_time[i+1].month):
                    year_dict[y_step][m_step].append(total_time[i])
                    m = True
                else :
                    if(m==True):
                        year_dict[y_step][m_step].append(total_time[i])
                        m=False
                    m_step +=1   
            else :
                # reset months to set the last observation 
                m_step = 12
                # add the remaining same year observation 
                year_dict[y_step][m_step].append(total_time[i]) 
                # reset month index 
                m_step=data.index.min().month
                y_step+=1
                y=False
        return year_dict    
                
        
        
# for a certain year , for a certain currency plot the monthy series of this currency (one subplot per month of the year)
def plot_one_year_one_currency(year_dict,data,year,currency,save=False):
    # total plots 
    fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(15, 10))  # 3x4 grid for subplots
    #select currency from dataset 
    
    
    # for each month 
    for month in range(1, 13):
        row = (month - 1) // 4  # Determine row index
        col = (month - 1) % 4   # Determine column index
        
        #pick year and currency 
        dollar_month = data.loc[year_dict[year][month],currency]
        ax = axes[row,col]
        # rotate txt 
        ax.xaxis.set_tick_params(rotation=60)
        ax.plot(dollar_month,'o-',label= currency + ' : ' + str(month))   
        ax.set_title(f"{year}-{month}-{currency}")
        
    #general title for whole plot     
    plt.suptitle(f"Time Series Observations for Year {year}", fontsize=16)
    plt.tight_layout()
    #plt.show()
    # close plot 
    #plt.close()
    
    
    
    # save image (optional)
    if(save):
        save_image(fig,currency,year)


# same as above but do it for all currencies (17 currencies -> 17 plots )
def plot_one_year_all_currencies(year_dict,data,year,save=False):
    for cu in data.columns:
        plot_one_year_one_currency(year_dict,data,year,cu,save)
    



# plot all years one currency 

def plot_all_years_one_currency(year_dict,data,currency,save=False):
    # total plots 
    fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 10))  # 3x4 grid for subplots
    #select currency from dataset 
    
    
    # for each month 
    inyear = data.index.min().year
    for year in range(1, 26):
        row = (year - 1) // 5  # Determine row index
        col = (year - 1) % 5   # Determine column index
        
        
        #f1=[x for x in y_dict[inyear].values()]
        #f2 = [k for k2 in f1 for k in k2]
        
        # take the yearly indices 
        indices = [k for k2 in [x for x in y_dict[inyear].values()] for k in k2]
        
        
        dollar_month = data.loc[ indices ,currency]
        ax = axes[row,col]
        # rotate txt 
        ax.xaxis.set_tick_params(rotation=60)
        ax.plot(dollar_month,'o-',label= currency + ' : ' + str(inyear))   
        ax.set_title(f"{inyear}-{currency}")
        inyear += 1
        
    #general title for whole plot     
    plt.suptitle(f"Time Series Observations for All years - {currency}", fontsize=16)
    plt.tight_layout()
    #plt.show()
    #plt.close()
    
    if(save):
        # i use 25 to denote that we plot all the years 
        save_image(fig,currency,25)
    
    # save image (optional)
    
    

# function to plot all years for all currencies 
def plot_all_years_all_currencies(year_dict,data,save=False):
    for cur in data.columns:
            plot_all_years_one_currency(year_dict, data, cur,save)
    
    


# function to save images 
def save_image(fig,currency,year):
    if not (os.path.exists('plots')):
            os.makedirs('plots')
    plt.savefig(os.path.join('plots', currency + ' ' +str(year) +  '-forecast.png'))
    plt.close()
    




# only called as main for testing 
if __name__=="__main__" :
        # read data from csv 
        data = pd.read_csv('../dataset/processed_dataset.csv', index_col=0, parse_dates=True)
        
        
        
        
        # standardize 
        standardized_data = standardize(data)
        
        #create nested dict
        # run with data or standardized_data
        y_dict = create_nested_dict(data)
        
        
        #run plot_one_year_all_currencies to plot for a certain year all the currencies monthly plots 
        #plot_one_year_all_currencies(y_dict,standardized_data,1999,save=False)
        
        #------------------------------------------------------#
        '''
        plot_one_year_all_currencies(y_dict,data,2018,save=True)
        plot_one_year_all_currencies(y_dict,data,2019,save=True)
        plot_one_year_all_currencies(y_dict,data,2020,save=True)
        plot_one_year_all_currencies(y_dict,data,2021,save=True)
        plot_one_year_all_currencies(y_dict,data,2022,save=True)
        plot_one_year_all_currencies(y_dict,data,2023,save=True)
        '''
        #-----------------------------------------------------#
        
        # it is easy to plot all the years and all the currencies as well 
        # but we get 17 * 25 = 425 plots so not really feasible 
        # i believe with the above we can focus on the years we need 
        # TODO : create another function that plots all the years for one currency , if we need to focus and study a currency in particular 
        
        
        
        #plot all years for all currencies  
        plot_all_years_all_currencies(y_dict,data,True)
      
        
        
        
        
        # let's transpose the series to make them more compatible with the cnn models created 
        all_series = standardized_data.transpose()
        