#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 11:11:52 2021
'''make plots of mean, median, mode values for a night
@author: rbrown
"""

import numpy as np
import pandas as pd
from astropy.time import Time
import matplotlib.pyplot as plt

#%%
#one night version

scope = 'r'   #telescope (r, g, b)
nights = ['20210804', '202108013', '20210922']  #list of nights to plot
num_nights = len(nights)

#name of telescope
if scope == 'b':
    scopeName = 'Blue'
elif scope == 'r':
    scopeName = 'Red'
elif scope == 'g':
    scopeName = 'Green'
else:
    scopeName = scope

data = {}       #dictionary to hold data for each night for plotting

#load in dataframes for each night
for night in nights:
    data[night] = {'df': pd.read_csv('./BiasTests/' + night + '_bias_temp_stats_' + scope + '.txt', delim_whitespace = True)}

#get upper/lower limits for plots, make day and hour columns
for night in nights:
     data[night]['lower_v'] = min([min(data[night]['df']['med']), min(data[night]['df']['mean']), min(data[night]['df']['mode'])]),
     data[night]['upper_v'] = max([max(data[night]['df']['med']), max(data[night]['df']['mean']), max(data[night]['df']['mode'])]),
     data[night]['lower_t'] = min([min(data[night]['df']['baseTemp']), min(data[night]['df']['FPGAtemp'])]),
     data[night]['upper_t'] = min([max(data[night]['df']['baseTemp']), max(data[night]['df']['FPGAtemp'])]),
     data[night]['df'][['day','hour']] = data[night]['df']['time'].str.split('T', expand = True)
     
     folders = data[night]['df']['filename'].str.split('\\', expand = True)[1]
     folders, data[night]['index'] = np.unique(folders, return_index = True)
     labels = data[night]['df']['hour'][data[night]['index']]
     data[night]['labels'] = labels.str.split('.', expand = True)[0]
     
   

#%%
#get axes for each plot
axes = []   #list of axes labels for night

for i in range(0, num_nights):
    val_ax = 'ax'+str(2*i+1)
    temp_ax = 'ax'+str(2*i+2)
    axes.append(val_ax) 
    axes.append(temp_ax)
    #data[nights[i]]['val_ax'] = val_ax
    #data[nights[i]]['temp_ax'] = temp_ax
    
#%% set up figure with multiple panels
rows = 2
num_panels = rows*num_nights

fig = plt.figure(1, figsize = (16,5))
position = range(1, num_panels+1)

for k in range(num_panels):
    axes[k] = fig.add_subplot(rows, num_nights, position[k])
    
for i in range(num_nights):
    data[nights[i]]['val_ax'] = axes[i]
    data[nights[i]]['temp_ax'] = axes[i+rows]
    data[nights[-1]]['temp_ax'] = axes[-1]
    
lv = 96
hv = 99
lt = 25
ht = 55

for key, v in data.items():
    v['val_ax'].scatter(v['df']['hour'], v['df']['med'], label = 'median', s = 2)
    v['val_ax'].scatter(v['df']['hour'], v['df']['mean'], label = 'mean', s = 2)
    v['val_ax'].scatter(v['df']['hour'], v['df']['mode'], label = 'mode', s = 2)
    
    v['val_ax'].set_title(scopeName + ' biases - ' + v['df'].loc[0]['day'])
    v['val_ax'].set_ylabel('image pixel value')
    #v['val_ax'].vlines(v['index'], v['lower_v'][0], v['upper_v'][0], color = 'black', linewidth = 1)
    v['val_ax'].vlines(v['index'], lv, hv, color = 'black', linewidth = 1)
    v['val_ax'].set_xticks([])
    v['val_ax'].set_xticklabels([])
    #v['val_ax'].set_ylim(v['lower_v'][0], v['upper_v'][0])
    v['val_ax'].set_ylim(lv, hv)
    

    
    v['temp_ax'].scatter(v['df']['hour'], v['df']['baseTemp'], label = 'Base temp', s = 2)
    v['temp_ax'].scatter(v['df']['hour'], v['df']['FPGAtemp'], label = 'FGPA temp', s = 2)

   # v['temp_ax'].vlines(v['index'], v['lower_t'][0], v['upper_t'][0], color = 'black', linewidth = 1)
    v['temp_ax'].vlines(v['index'], lt, ht, color = 'black', linewidth = 1)
    v['temp_ax'].set_xlabel('time')
    v['temp_ax'].set_ylabel('Temperature (C)')
    v['temp_ax'].set_xticks(v['index'])
    v['temp_ax'].set_xticklabels(v['labels'],rotation=20)
    #v['temp_ax'].set_ylim(v['lower_t'][0], v['upper_t'][0])
    v['temp_ax'].set_ylim(lt, ht)

    
axes[0].legend()
axes[num_nights].legend()   
axes[num_nights-1].set_yticks([])
axes[num_nights-1].set_ylabel('')
axes[num_nights+1].set_yticks([])
axes[num_nights+1].set_ylabel('')

plt.subplots_adjust(wspace=0.05, hspace=0.1)
    

#plt.savefig('./BiasTests/bias_stats_' + scope + '.png')
plt.show()
plt.close()
    
    
