# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 19:36:31 2022

@author: GreenBird
"""

import numpy as np

import pandas as pd




import matplotlib.pyplot as plt
from pathlib import Path

#scope = 'Green'
night = '2022-08-06'
#night=obs_date
Green_data=Path('/','D:','/ColibriArchive',night.replace('-', '')+'_diagnostics','Bias_Stats')
Red_data=Path('/','Y:',night.replace('-', '')+'_diagnostics','Bias_Stats')
Blue_data=Path('/','Z:',night.replace('-', '')+'_diagnostics','Bias_Stats')

Green_bias = pd.read_csv(Green_data.joinpath(night+'_stats.txt'), delim_whitespace = True)
Red_bias = pd.read_csv(Red_data.joinpath(night+'_stats.txt'), delim_whitespace = True)
Blue_bias = pd.read_csv(Blue_data.joinpath(night+'_stats.txt'), delim_whitespace = True)

Green_bias[['day','hour']] = Green_bias['time'].str.split('T', expand = True)
Red_bias[['day','hour']] = Red_bias['time'].str.split('T', expand = True)
Blue_bias[['day','hour']] = Blue_bias['time'].str.split('T', expand = True)

#find time breaks in data (when a new folder was created)

#get list of different bias folders, the indices of where these start in the data frame
Green_folders = Green_bias['filename'].str.split('\\', expand = True)[1]
Red_folders = Red_bias['filename'].str.split('\\', expand = True)[1]
Blue_folders = Blue_bias['filename'].str.split('\\', expand = True)[1]

Green_minutes = Green_bias['hour'].str.split(':', expand = True)[1]
Red_minutes = Red_bias['hour'].str.split(':', expand = True)[1]
Blue_minutes = Blue_bias['hour'].str.split(':', expand = True)[1]

Green_folders, Green_index = np.unique(Green_minutes, return_index = True)
Red_folders, Red_index = np.unique(Red_minutes, return_index = True)
Blue_folders, Blue_index = np.unique(Blue_minutes, return_index = True)

Green_labels = Green_bias['hour'][Green_index]
Red_labels = Red_bias['hour'][Red_index]
Blue_labels = Blue_bias['hour'][Blue_index]

Green_labels = Green_labels.str.split('.', expand = True)[0]
Red_labels = Red_labels.str.split('.', expand = True)[0]
Blue_labels = Blue_labels.str.split('.', expand = True)[0]

    #%%
'''-----------------------------------------bias median and temperature graph--------------------------------------------------'''
#plot with single nights data with temperature
fig, [[ax11, ax21, ax31], [ax12, ax22, ax32]] = plt.subplots(ncols=3, nrows=2, sharex = True, figsize = (22,8), gridspec_kw=dict(hspace = 0.15))

#minimum and maximum value in one night
lower_m = -1 #data['mean'].min()
upper_m = 1 #data['mean'].max()

ax11.scatter(Green_bias['hour'], np.mean(Green_bias['mean']) - Green_bias['mean'], label = 'mean', s = 2)

ax11.set_title('Green' + ' Bias Levels - ' + Green_bias.loc[0]['day'])
ax11.set_ylabel('(Overall mean) - (mean bias pixel value)')
ax11.vlines(Green_index, lower_m, upper_m, color = 'black', linewidth = 1)
ax11.set_xticks([])
ax11.set_xticklabels([])
ax11.set_ylim(lower_m, upper_m)

#ax1.legend()

lower_t = -3 #48
upper_t = 3 #52
#ax2.scatter(data['hour'], data['baseTemp'], label = 'Base temp', s = 2)
ax12.scatter(Green_bias['hour'], np.mean(Green_bias['FPGAtemp']) - Green_bias['FPGAtemp'], label = 'FGPA temp', s = 2)

ax12.vlines(Green_index, lower_t, upper_t, color = 'black', linewidth = 1)
ax12.set_xlabel('Time')
ax12.set_ylabel('(mean temperature) - temperature (C)')
ax12.set_xticks(Green_index)
ax12.set_xticklabels(Green_labels,rotation=90)
ax12.set_ylim(lower_t, upper_t)

#ax2.legend()------------------------------------------------------

ax21.scatter(Red_bias['hour'], np.mean(Red_bias['mean']) - Red_bias['mean'], label = 'mean', s = 2)

ax21.set_title('Red' + ' Bias Levels - ' + Red_bias.loc[0]['day'])
ax21.set_ylabel('(Overall mean) - (mean bias pixel value)')
ax21.vlines(Red_index, lower_m, upper_m, color = 'black', linewidth = 1)
ax21.set_xticks([])
ax21.set_xticklabels([])
ax21.set_ylim(lower_m, upper_m)

ax22.scatter(Red_bias['hour'], np.mean(Red_bias['FPGAtemp']) - Red_bias['FPGAtemp'], label = 'FGPA temp', s = 2)

ax22.vlines(Red_index, lower_t, upper_t, color = 'black', linewidth = 1)
ax22.set_xlabel('Time')
ax22.set_ylabel('(mean temperature) - temperature (C)')
ax22.set_xticks(Red_index)
ax22.set_xticklabels(Red_labels,rotation=90)
ax22.set_ylim(lower_t, upper_t)

#ax2.legend()------------------------------------------------------

ax31.scatter(Blue_bias['hour'], np.mean(Blue_bias['mean']) - Blue_bias['mean'], label = 'mean', s = 2)

ax31.set_title('Blue' + ' Bias Levels - ' + Blue_bias.loc[0]['day'])
ax31.set_ylabel('(Overall mean) - (mean bias pixel value)')
ax31.vlines(Blue_index, lower_m, upper_m, color = 'black', linewidth = 1)
ax31.set_xticks([])
ax31.set_xticklabels([])
ax31.set_ylim(lower_m, upper_m)

ax32.scatter(Blue_bias['hour'], np.mean(Blue_bias['FPGAtemp']) - Blue_bias['FPGAtemp'], label = 'FGPA temp', s = 2)

ax32.vlines(Blue_index, lower_t, upper_t, color = 'black', linewidth = 1)
ax32.set_xlabel('Time')
ax32.set_ylabel('(mean temperature) - temperature (C)')
ax32.set_xticks(Blue_index)
ax32.set_xticklabels(Blue_labels,rotation=90)
ax32.set_ylim(lower_t, upper_t)

#ax2.legend()------------------------------------------------------

plt.savefig(Green_data.joinpath( night + 'meanofmean_bias_stats_Green-Red_Blue.png'),bbox_inches = "tight",dpi=300)
plt.show()
plt.close()
    

    #%%
'''------------------------------------bias median and mean graph----------------------------------------'''
lower = 70
upper = 78

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey = False, figsize = (24,9), gridspec_kw=dict(wspace = 0.1))

ax1.scatter(Green_bias['hour'], Green_bias['med'], label = 'median', s = 2)
ax1.scatter(Green_bias['hour'], Green_bias['mean'], label = 'mean', s = 2)
#plt.scatter(data['hour'], data['mode'], label = 'mode', s = 2)

ax2.scatter(Red_bias['hour'], Red_bias['med'], label = 'median', s = 2)
ax2.scatter(Red_bias['hour'], Red_bias['mean'], label = 'mean', s = 2)
#plt.scatter(data['hour'], data['mode'], label = 'mode', s = 2)

ax3.scatter(Blue_bias['hour'], Blue_bias['med'], label = 'median', s = 2)
ax3.scatter(Blue_bias['hour'], Blue_bias['mean'], label = 'mean', s = 2)
#plt.scatter(data['hour'], data['mode'], label = 'mode', s = 2)



ax1.set_title('Green biases - ' + Green_bias.loc[0]['day'])
ax1.set_ylabel('image pixel value')
ax1.vlines(Green_index, np.min(Green_bias['med'])-0.5, np.amax(Green_bias['med'])+0.5, color = 'black', linewidth = 1)
ax1.set_xlabel('time')
ax1.set_xticks(Green_index)
ax1.set_xticklabels(Green_labels,rotation=20,fontsize=10)
#ax1.ylim(lower-0.2, upper+0.2)

ax2.set_title('Red biases - ' + Red_bias.loc[0]['day'])
ax2.set_ylabel('image pixel value')
ax2.vlines(Red_index, np.min(Red_bias['med'])-0.5, np.amax(Red_bias['med'])+0.5, color = 'black', linewidth = 1)
ax2.set_xlabel('time')
ax2.set_xticks(Red_index)
ax2.set_xticklabels(Red_labels,rotation=20,fontsize=10)
#ax1.ylim(lower-0.2, upper+0.2)

ax3.set_title('Blue  biases - ' + Blue_bias.loc[0]['day'])
ax3.set_ylabel('image pixel value')
ax3.vlines(Blue_index, np.min(Blue_bias['med'])-0.5, np.amax(Blue_bias['med'])+0.5, color = 'black', linewidth = 1)
ax3.set_xlabel('time')
ax3.set_xticks(Blue_index)
ax3.set_xticklabels(Blue_labels,rotation=20,fontsize=10)
#ax1.ylim(lower-0.2, upper+0.2)

plt.legend()


plt.savefig(Green_data.joinpath('Green-Red-Blue' + '.png'),dpi=300)
plt.show()
plt.close()