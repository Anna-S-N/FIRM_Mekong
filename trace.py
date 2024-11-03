# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 14:54:34 2024

@author: u6942852
"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as pltd
import warnings
from datetime import datetime as dt

from Input import *
    
sns.set_theme()
cp = sns.color_palette()


def plottrace_network(timeseries, startidx=None, endidx=None, pltkwargs={}):
    fig, ax = plt.subplots(**pltkwargs)
    
    time = np.arange(len(timeseries))[startidx:endidx]
    #TODO convert to dates (being aware of leap?)
    
    neg_cols = [-timeseries[startidx:endidx, n] for n in (1,10,2)]
    neg_col_names = ['Pumped Hydro Storage', 'Battery Storage', 'Spillage']
    neg_col_colors = [9, 2, 6]
    
    pos_cols = [timeseries[startidx:endidx, n] for n in (8, 9, 3, 4, 5, 11, 7)]
    pos_col_names = ['Hydro', 'Hydro', 'Solar PV', 'Wind', 'Pumped Hydro Storage', 'Battery Storage', 'Deficit']
    pos_col_colors = [0, 0, 1, 7, 9, 2]
    
    demand = timeseries[startidx:endidx, 0]
    
    # plot negative values
    plot = ax.stackplot( 
        time, 
        neg_cols,
        labels=neg_col_names,
        )
    for i, color in enumerate(neg_col_colors):
        plot[i].set_color(cp[color])

    # plot positive values
    plot = ax.stackplot(
        time, 
        pos_cols,
        labels=pos_col_names,
        )
    for i, color in enumerate(pos_col_colors):
        plot[i].set_color(cp[color])
    
    ax.plot( # fix extra ring of colour
        time, 
        sum(neg_cols),
        color="#EAEAF2",
        )
    ax.plot( # fix extra ring of colour
        time, 
        sum(pos_cols),
        color="#EAEAF2",
        )
    
    ax.plot(
        time, 
        demand, 
        color = 'black',
        label='demand',
        linewidth=1.75,
        )
    
    # When time is made into times 
    # ax.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d'))
    
    
    ax.set_ylabel('Power (MW)')
    ax.set_xlabel('Date and Time')
    ax.set_xticks([])
    
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*0.95, pos.height*1.0])
    lns, labs = ax.get_legend_handles_labels()
    # deduplicate legend
    leg = dict(zip(labs, lns))
    
    ax.legend(leg.values(), leg.keys(), bbox_to_anchor=(  1.15, 0.5)) 
    # if legend is badly placed,  probably need to adjust ^^^^ that number


def plottrace_node(timeseries, startidx=None, endidx=None, pltkwargs={}):
    fig, ax = plt.subplots(**pltkwargs)
    
    time = np.arange(len(timeseries))[startidx:endidx]
    #TODO convert to dates (being aware of leap?)
    
    neg_cols = [-timeseries[startidx:endidx, n] for n in (12, 13, 11, 9)]
    neg_col_names = ['Pumped Hydro Storage', 'Battery Storage', 'Transmission', 'Spillage']
    neg_col_colors = [9, 2, 4, 6]
    
    pos_cols = [timeseries[startidx:endidx, n] for n in (3, 4, 5, 1, 2, 7, 14, 10, 6, 8)]
    pos_col_names = ['Hydro', 'Hydro', 'Hydro', 'Solar PV', 'Wind', 'Pumped Hydro Storage', 'Battery Storage', 'Transmission', 'Import', 'Deficit']
    pos_col_colors = [0, 0, 0, 1, 7, 9, 2, 4, 5]
    
    demand = timeseries[startidx:endidx, 0]
    
    # plot negative values
    plot = ax.stackplot( 
        time, 
        neg_cols,
        labels=neg_col_names,
        )
    for i, color in enumerate(neg_col_colors):
        plot[i].set_color(cp[color])

    # plot positive values
    plot = ax.stackplot(
        time, 
        pos_cols,
        labels=pos_col_names,
        )
    for i, color in enumerate(pos_col_colors):
        plot[i].set_color(cp[color])
    
    ax.plot( # fix extra ring of colour
        time, 
        sum(neg_cols),
        color="#EAEAF2",
        )
    ax.plot( # fix extra ring of colour
        time, 
        sum(pos_cols),
        color="#EAEAF2",
        )
    
    ax.plot(
        time, 
        demand, 
        color = 'black',
        label='demand',
        linewidth=1.75,
        )
    
    # When time is made into times 
    # ax.xaxis.set_major_formatter(pltd.DateFormatter('%b-%d'))
    
    
    ax.set_ylabel('Power (MW)')
    ax.set_xlabel('Date and Time')
    ax.set_xticks([])
    
    pos = ax.get_position()
    ax.set_position([pos.x0, pos.y0, pos.width*0.95, pos.height*1.0])
    lns, labs = ax.get_legend_handles_labels()
    # deduplicate legend
    leg = dict(zip(labs, lns))
    
    ax.legend(leg.values(), leg.keys(), bbox_to_anchor=(  1.15, 0.5)) 
    # if legend is badly placed,  probably need to adjust ^^^^ that number














#%%

if __name__=='__main__':

    behaviour = np.genfromtxt('Results/TimeSeries_{}_{}_{}_{}_{}_{}_{}_NETWORK.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario), delimiter=',', skip_header=1, dtype=float)

    plottrace_network(
        behaviour, 
        0, 
        24*7, 
        {'figsize':(12, 5),
         'dpi':1250,#needs to be quite high to avoid graininess around edge
         },
        )
    
    for nodej in Nodel:
        behaviour = np.genfromtxt('Results/TimeSeries_{}_{}_{}_{}_{}_{}_{}_{}.csv'.format(node, percapita, iterations, population, nuclear_scenario, hydro_scenario, battery_scenario,nodej), delimiter=',', skip_header=1, dtype=float)
        
        plottrace_node(
            behaviour, 
            0, 
            24*7, 
            {'figsize':(12, 5),
             'dpi':1250,#needs to be quite high to avoid graininess around edge
             },
            )
        



