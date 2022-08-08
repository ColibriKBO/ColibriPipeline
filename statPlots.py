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

scope = 'Green'
night = '20220719'

#data = pd.read_csv('./meanTests/' + night + '_mean_' + scope + '.txt', delim_whitespace = True, nrows = 16764)
data = pd.read_csv('./Green/ElginfieldGreen/20220719_diagnostics/Bias_Stats/2022-07-19_stats.txt', delim_whitespace = True, nrows = 16764)
data[['day','hour']] = data['time'].str.split('T', expand = True)

#find time breaks in data (when a new folder was created)

#get list of different bias folders, the indices of where these start in the data frame
folders = data['filename'].str.split('\\', expand = True)[1]
minutes = data['hour'].str.split(':', expand = True)[1]
#folders, index = np.unique(folders, return_index = True)
folders, index = np.unique(minutes, return_index = True)
labels = data['hour'][index]
labels = labels.str.split('.', expand = True)[0]

#%%
#plot with single nights data with temperature
fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (16,8), gridspec_kw=dict(hspace = 0.05))

#minimum and maximum value in one night
lower_m = -1 #data['mean'].min()
upper_m = 1 #data['mean'].max()

ax1.scatter(data['hour'], np.mean(data['mean']) - data['mean'], label = 'mean', s = 2)

ax1.set_title(scope + ' Bias Levels - ' + data.loc[0]['day'])
ax1.set_ylabel('(Overall mean) - (mean bias pixel value)')
ax1.vlines(index, lower_m, upper_m, color = 'black', linewidth = 1)
ax1.set_xticks([])
ax1.set_xticklabels([])
ax1.set_ylim(lower_m, upper_m)

#ax1.legend()

lower_t = -3 #48
upper_t = 3 #52
#ax2.scatter(data['hour'], data['baseTemp'], label = 'Base temp', s = 2)
ax2.scatter(data['hour'], np.mean(data['FPGAtemp']) - data['FPGAtemp'], label = 'FGPA temp', s = 2)

ax2.vlines(index, lower_t, upper_t, color = 'black', linewidth = 1)
ax2.set_xlabel('Time')
ax2.set_ylabel('(mean temperature) - temperature (C)')
ax2.set_xticks(index)
ax2.set_xticklabels(labels,rotation=90)
ax2.set_ylim(lower_t, upper_t)

#ax2.legend()

plt.savefig('./Green/ElginfieldGreen/20220719_diagnostics/Bias_Stats/' + night + 'meanofmean_bias_stats_' + scope + '.png',bbox_inches = "tight")
plt.show()
plt.close()
#%%
#get data for each night
scope = 'Red'

data_04 = pd.read_csv('./Red/ElginfieldRed/20220621_diagnostics/Bias_Stats/2022-06-21_stats.txt', delim_whitespace = True)
data_13 = pd.read_csv('./Red/ElginfieldRed/20220719_diagnostics/Bias_Stats/2022-07-19_stats.txt', delim_whitespace = True)

data_04[['day','hour']] = data_04['time'].str.split('T', expand = True)  
data_13[['day','hour']] = data_13['time'].str.split('T', expand = True)  

#%%
#find time breaks in data (when a new folder was created)

folders_04 = data_04['filename'].str.split('\\', expand = True)[1]
minutes_04 = data_04['hour'].str.split(':', expand = True)[1]
#folders, index = np.unique(folders, return_index = True)
folders_04, index_04 = np.unique(minutes_04, return_index = True)
labels_04 = data_04['hour'][index_04]
labels_04 = labels_04.str.split('.', expand = True)[0]

folders_13 = data_13['filename'].str.split('\\', expand = True)[1]
minutes_13 = data_13['hour'].str.split(':', expand = True)[1]
#folders, index = np.unique(folders, return_index = True)
folders_13, index_13 = np.unique(minutes_13, return_index = True)
labels_13 = data_13['hour'][index_13]
labels_13 = labels_13.str.split('.', expand = True)[0]



#get list of different bias folders, the indices of where these start in the data frame
# folders_04 = data_04['filename'].str.split('\\', expand = True)[1]
# folders_04, index_04 = np.unique(folders_04, return_index = True)
# labels_04 = data_04['hour'][index_04]
# labels_04 = labels_04.str.split('.', expand = True)[0]


# folders_13 = data_13['filename'].str.split('\\', expand = True)[1]
# folders_13, index_13 = np.unique(folders_13, return_index = True)
# labels_13 = data_13['hour'][index_13]
# labels_13 = labels_13.str.split('.', expand = True)[0]

#minimum and maximum value in one night
lower_04 = np.min(data_04.min(axis = 1))
upper_04 = np.max(data_04.max(axis = 1))

#minimum and max value in other night
lower_13 = np.min(data_13.min(axis = 1))
upper_13 = np.max(data_13.max(axis = 1))

#min and max value for both nights
#lower = np.min([lower_13, lower_04])
lower = 0
upper = 200
#upper = np.max([upper_13, upper_13])


#%%
#make plot with 2 nights data in side by side panels

fig, (ax1, ax2) = plt.subplots(1, 2, sharey = True, figsize = (12,4), gridspec_kw=dict(wspace = 0.05))


#ax1.scatter(data_04['hour'], data_04['med'], label = 'median', s = 2)
ax1.scatter(data_04['hour'], data_04['mean'], label = 'mean', s = 2)
#ax1.scatter(data_04['hour'], data_04['mode'], label = 'mode', s = 2)

ax1.set_title(scope + ' biases - ' + data_04.loc[0]['day'])
ax1.set_ylabel('image pixel value')
ax1.vlines(index_04, lower-0.2, upper+0.2, color = 'black', linewidth = 1)
ax1.set_xlabel('time')
ax1.set_xticks(index_04)
ax1.set_xticklabels(labels_04,rotation=20)
ax1.set_ylim(lower-0.2, upper+0.2)


#ax2.scatter(data_13['hour'], data_13['med'], label = 'median', s = 2)
ax2.scatter(data_13['hour'], data_13['mean'], label = 'mean', s = 2)
#ax2.scatter(data_13['hour'], data_13['mode'], label = 'mode', s = 2)

ax2.set_title(scope + ' biases - ' + data_13.loc[0]['day'])
ax2.vlines(index_13, lower-0.2, upper+0.2, color = 'black', linewidth = 1)
ax2.set_xlabel('time')
ax2.set_xticks(index_13)
ax2.set_xticklabels(labels_13,rotation=20)
#ax2.set_ylim(lower_13-0.2, upper_13+0.2)

#ax2.legend()

#plt.savefig('./Red/ElginfieldRed/bias_compare_0621-0719.png')
plt.show()
plt.close()
#%%
#plot with single nights data
plt.scatter(data_04['hour'], data_04['med'], label = 'median', s = 2)
plt.scatter(data_04['hour'], data_04['mean'], label = 'mean', s = 2)
plt.scatter(data_04['hour'], data_04['mode'], label = 'mode', s = 2)



plt.title(scope + ' biases - ' + data_04.loc[0]['day'])
plt.ylabel('image pixel value')
plt.vlines(index_04, lower_04-0.2, upper_04+0.2, color = 'black', linewidth = 1)
plt.xlabel('time')
plt.xticks(index_04, labels_04,rotation=20)
plt.ylim(lower_04-0.2, upper_13+0.2)

plt.legend()


plt.savefig('./imageStats/20210804_bias_stats_' + scope + '.png')
plt.show()
plt.close()

#%%

plt.scatter(data_13['hour'], data_13['med'], label = 'median', s = 2)
plt.scatter(data_13['hour'], data_13['mean'], label = 'mean', s = 2)
plt.scatter(data_13['hour'], data_13['mode'], label = 'mode', s = 2)

plt.vlines(index_13, lower_13-0.2, upper_13+0.2, color = 'black', linewidth = 1)

plt.title(scope + ' biases - ' + data_13.loc[0]['day'])
plt.ylabel('image pixel value')

plt.xlabel('time')
plt.xticks(index_13, labels_13, rotation=20)
plt.ylim(lower_13-0.2, upper_13+0.2)

plt.legend()


plt.savefig('./imageStats/202108013_bias_stats_' + scope + '.png')
plt.show()
plt.close()






