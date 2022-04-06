#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 15:45:35 2022

@author: rbrown
"""

import numpy as np
import pandas as pd
import pathlib
import matplotlib.pyplot as plt
import datetime


base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri')  #path to main directory
kernel_set = np.loadtxt(base_path.joinpath('kernels_040622.txt'))

param_filename = base_path.joinpath('params_kernels_040622.txt')
kernel_params = pd.read_csv(param_filename, delim_whitespace = True)

kernel_indices = list(range(len(kernel_set)))

kernel_plots = {}


for i in kernel_indices:
    kernel_plots[i] = {}
    kernel_plots[i]['flux'] = kernel_set[i]
    kernel_plots[i]['Freq'] = kernel_params.iloc[i]['samplingFreq']
    kernel_plots[i]['O_r'] = kernel_params.iloc[i]['ObjectRadius']
    kernel_plots[i]['S_d'] = kernel_params.iloc[i]['StellarDiameter']
    kernel_plots[i]['b'] = kernel_params.iloc[i]['ImpactParameter']
    kernel_plots[i]['shift'] = kernel_params.iloc[i]['ShiftAdjustment']
    
#reorder sets for plotting
kernel_groups = []

def getKernels(objectR, starD):
    '''get kernels that match the given parameters and add to dictionary for plotting '''
    
    kernel_dicts = []
    
    #find dictionarys containing kernel params that match
    for i, data in kernel_plots.items():
        if data['O_r'] == objectR:
            if data['S_d'] == starD:
                           
                kernel_dicts.append(data)
            
    return kernel_dicts

objectRs = [750, 1375, 2500]
starDs = [0.01, 0.03, 0.08]
shifts = [-0.25, 0, 0.25, 0.5]

for r in objectRs:
    for d in starDs:
        kernel_groups.append(getKernels(r, d))
        
#make plots

for group in kernel_groups:
    
    objectR = group[0]['O_r']
    Bs = [0.0, objectR/2, objectR]
    

     
    fig, ax_array = plt.subplots(4,3, sharex = True, sharey = True, figsize = (10,15), gridspec_kw = {'wspace':0, 'hspace':0})

    
    for i,ax_row in enumerate(ax_array):
        for j,axes in enumerate(ax_row):
            
            shift = [shifts[i]]

            filtered_group = list(filter(lambda x: x['shift'] in shift, group))
            
            b = [Bs[j]]
            kernel_dict = list(filter(lambda d: d['b'] in b, filtered_group))[0]

            
            axes.set_xlim(0, 32)
            axes.set_ylim(0, 1.25)
            
            axes.set_yticks(ticks = [0.5, 1.0])
            
            axes.set_xticks(ticks = [10, 20, 30])
            
            axes.tick_params(direction = 'inout', right = True, top = True, length = 10, labelsize = 15)
         #   if j == 0:
                
          #      axes.set_yticks(ticks = [0.5, 1.0])
                
           # if j != 0:

            #    axes.set_yticks(ticks = [0.5, 1.0])
                
            if j == 2:
                axes.set_ylabel(shift[0], labelpad = 65, size = 20, rotation = 0)
                axes.yaxis.set_label_position('right')
                
            if i == 0:
                axes.set_xlabel(b[0], labelpad = 35, size = 20)
                axes.xaxis.set_label_position('top')
            
            
            axes.plot(kernel_dict['flux'], label = '%f %f' %(kernel_dict['b'], kernel_dict['shift']))
              
    
    ax = fig.add_subplot(111, frame_on = False)

    ax.grid(False) 
    ax.set_title('Object r: ' + str(objectR) + ' m, Stellar d: ' + str(group[0]['S_d']) + ' mas', pad = 85, size = 25)
    ax.set_xlabel('Time [25 ms frames]', size = 20, labelpad = 18)
    ax.set_ylabel('Normalized Flux [counts]', size = 20, labelpad = 18)
    
    ax.text(-0.2, 1.02, '    Impact\n Parameter:\n       [m]', size = 20 )
    ax.text(1.0, 1.02, '       Shift\n Adjustment:\n    [frames]', size = 20)
    
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
#    plt.show()
    plt.savefig('../kernel_plots_040622/R-'+str(objectR) + '_d-' + str(group[0]['S_d']) + '.png', bbox_inches = 'tight')
    plt.close()
    



                
    

    