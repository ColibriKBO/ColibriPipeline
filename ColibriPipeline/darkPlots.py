# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 19:36:31 2022

@author: Roman A. used Rachel's dark graphing scripts

script to quickly see dark and sensor temperature values for specific night

It is suposed to run on Green only 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


night = '2022-08-06' #night you want to analyze


#ColibriArchive directories of 3 telescopes
Green_data=Path('/','D:','/ColibriArchive',night.replace('-', '')+'_diagnostics','Dark_Stats')
Red_data=Path('/','Y:',night.replace('-', '')+'_diagnostics','Dark_Stats')
Blue_data=Path('/','Z:',night.replace('-', '')+'_diagnostics','Dark_Stats')

#read txt files with dark and temperature stats using pandas
Green_dark = pd.read_csv(Green_data.joinpath(night+'_stats.txt'), delim_whitespace = True)
Red_dark = pd.read_csv(Red_data.joinpath(night+'_stats.txt'), delim_whitespace = True)
Blue_dark = pd.read_csv(Blue_data.joinpath(night+'_stats.txt'), delim_whitespace = True)

#locate and write time columns
Green_dark[['day','hour']] = Green_dark['time'].str.split('T', expand = True)
Red_dark[['day','hour']] = Red_dark['time'].str.split('T', expand = True)
Blue_dark[['day','hour']] = Blue_dark['time'].str.split('T', expand = True)

#find time breaks in data (when a new folder was created)

#get list of different dark folders, the indices of where these start in the data frame
Green_folders = Green_dark['filename'].str.split('\\', expand = True)[1]
Red_folders = Red_dark['filename'].str.split('\\', expand = True)[1]
Blue_folders = Blue_dark['filename'].str.split('\\', expand = True)[1]

Green_minutes = Green_dark['hour'].str.split(':', expand = True)[1]
Red_minutes = Red_dark['hour'].str.split(':', expand = True)[1]
Blue_minutes = Blue_dark['hour'].str.split(':', expand = True)[1]

Green_folders, Green_index = np.unique(Green_minutes, return_index = True)
Red_folders, Red_index = np.unique(Red_minutes, return_index = True)
Blue_folders, Blue_index = np.unique(Blue_minutes, return_index = True)

Green_labels = Green_dark['hour'][Green_index]
Red_labels = Red_dark['hour'][Red_index]
Blue_labels = Blue_dark['hour'][Blue_index]

Green_labels = Green_labels.str.split('.', expand = True)[0]
Red_labels = Red_labels.str.split('.', expand = True)[0]
Blue_labels = Blue_labels.str.split('.', expand = True)[0]

    #%%
'''-----------------------------------------dark median and temperature graph--------------------------------------------------'''
#plot with single nights data with temperature
fig, [[ax11, ax21, ax31], [ax12, ax22, ax32]] = plt.subplots(ncols=3, nrows=2, sharex = True, figsize = (22,8), gridspec_kw=dict(hspace = 0.15))

#minimum and maximum value in one night
lower_m = -4#data['mean'].min()
upper_m = 4 #data['mean'].max()

ax11.scatter(Green_dark['hour'], np.mean(Green_dark['mean']) - Green_dark['mean'], label = 'mean', s = 2)

ax11.set_title('Green' + ' Dark Levels - ' + Green_dark.loc[0]['day'])
ax11.set_ylabel('(Overall mean) - (mean dark pixel value)')
ax11.vlines(Green_index, lower_m, upper_m, color = 'black', linewidth = 1)
ax11.set_xticks([])
ax11.set_xticklabels([])
ax11.set_ylim(lower_m, upper_m)

#ax1.legend()

lower_t = -3 #48
upper_t = 3 #52
#ax2.scatter(data['hour'], data['baseTemp'], label = 'Base temp', s = 2)
ax12.scatter(Green_dark['hour'], np.mean(Green_dark['FPGAtemp']) - Green_dark['FPGAtemp'], label = 'FGPA temp', s = 2)

ax12.vlines(Green_index, lower_t, upper_t, color = 'black', linewidth = 1)
ax12.set_xlabel('Time')
ax12.set_ylabel('(mean temperature) - temperature (C)')
ax12.set_xticks(Green_index)
ax12.set_xticklabels(Green_labels,rotation=90)
ax12.set_ylim(lower_t, upper_t)

#ax2.legend()------------------------------------------------------

ax21.scatter(Red_dark['hour'], np.mean(Red_dark['mean']) - Red_dark['mean'], label = 'mean', s = 2)

ax21.set_title('Red' + ' Dark Levels - ' + Red_dark.loc[0]['day'])
ax21.set_ylabel('(Overall mean) - (mean dark pixel value)')
ax21.vlines(Red_index, lower_m, upper_m, color = 'black', linewidth = 1)
ax21.set_xticks([])
ax21.set_xticklabels([])
ax21.set_ylim(lower_m, upper_m)

ax22.scatter(Red_dark['hour'], np.mean(Red_dark['FPGAtemp']) - Red_dark['FPGAtemp'], label = 'FGPA temp', s = 2)

ax22.vlines(Red_index, lower_t, upper_t, color = 'black', linewidth = 1)
ax22.set_xlabel('Time')
ax22.set_ylabel('(mean temperature) - temperature (C)')
ax22.set_xticks(Red_index)
ax22.set_xticklabels(Red_labels,rotation=90)
ax22.set_ylim(lower_t, upper_t)

#ax2.legend()------------------------------------------------------

ax31.scatter(Blue_dark['hour'], np.mean(Blue_dark['mean']) - Blue_dark['mean'], label = 'mean', s = 2)

ax31.set_title('Blue' + ' Dark Levels - ' + Blue_dark.loc[0]['day'])
ax31.set_ylabel('(Overall mean) - (mean dark pixel value)')
ax31.vlines(Blue_index, lower_m, upper_m, color = 'black', linewidth = 1)
ax31.set_xticks([])
ax31.set_xticklabels([])
ax31.set_ylim(lower_m, upper_m)

ax32.scatter(Blue_dark['hour'], np.mean(Blue_dark['FPGAtemp']) - Blue_dark['FPGAtemp'], label = 'FGPA temp', s = 2)

ax32.vlines(Blue_index, lower_t, upper_t, color = 'black', linewidth = 1)
ax32.set_xlabel('Time')
ax32.set_ylabel('(mean temperature) - temperature (C)')
ax32.set_xticks(Blue_index)
ax32.set_xticklabels(Blue_labels,rotation=90)
ax32.set_ylim(lower_t, upper_t)

#ax2.legend()------------------------------------------------------

plt.savefig(Green_data.joinpath( night + 'meanofmean_dark_stats_Green-Red_Blue.png'),bbox_inches = "tight",dpi=300)
plt.show()
plt.close()
    

    #%%
'''------------------------------------dark median and mean graph----------------------------------------'''
lower = 70
upper = 78

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharey = False, figsize = (24,10), gridspec_kw=dict(hspace = 0.38))

ax1.scatter(Green_dark['hour'], Green_dark['med'], label = 'median', s = 2)
ax1.scatter(Green_dark['hour'], Green_dark['mean'], label = 'mean', s = 2)
#plt.scatter(data['hour'], data['mode'], label = 'mode', s = 2)

ax2.scatter(Red_dark['hour'], Red_dark['med'], label = 'median', s = 2)
ax2.scatter(Red_dark['hour'], Red_dark['mean'], label = 'mean', s = 2)
#plt.scatter(data['hour'], data['mode'], label = 'mode', s = 2)

ax3.scatter(Blue_dark['hour'], Blue_dark['med'], label = 'median', s = 2)
ax3.scatter(Blue_dark['hour'], Blue_dark['mean'], label = 'mean', s = 2)
#plt.scatter(data['hour'], data['mode'], label = 'mode', s = 2)



ax1.set_title('Green darks - ' + Green_dark.loc[0]['day'])
ax1.set_ylabel('image pixel value')
ax1.vlines(Green_index, np.min(Green_dark['med'])-0.5, np.amax(Green_dark['med'])+0.5, color = 'black', linewidth = 1)
#ax1.set_xlabel('time')
ax1.set_xticks(Green_index)
ax1.set_xticklabels(Green_labels,rotation=20,fontsize=10)
#ax1.ylim(lower-0.2, upper+0.2)

ax2.set_title('Red darks - ' + Red_dark.loc[0]['day'])
ax2.set_ylabel('image pixel value')
ax2.vlines(Red_index, np.min(Red_dark['med'])-0.5, np.amax(Red_dark['med'])+0.5, color = 'black', linewidth = 1)
#ax2.set_xlabel('time')
ax2.set_xticks(Red_index)
ax2.set_xticklabels(Red_labels,rotation=20,fontsize=10)
#ax1.ylim(lower-0.2, upper+0.2)

ax3.set_title('Blue  darks - ' + Blue_dark.loc[0]['day'])
ax3.set_ylabel('image pixel value')
ax3.vlines(Blue_index, np.min(Blue_dark['med'])-0.5, np.amax(Blue_dark['med'])+0.5, color = 'black', linewidth = 1)
#ax3.set_xlabel('time')
ax3.set_xticks(Blue_index)
ax3.set_xticklabels(Blue_labels,rotation=20,fontsize=10)
#ax1.ylim(lower-0.2, upper+0.2)

plt.legend()

plt.savefig(Green_data.joinpath('Green-Red-Blue' + '.png'),dpi=300)
plt.show()
plt.close()
