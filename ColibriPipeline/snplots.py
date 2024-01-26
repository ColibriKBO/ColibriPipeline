
# -*- coding: utf-8 -*-
"""
Created on Fri May 28 11:42:12 2021
Update: Jan. 24, 2022, 11:25am

@author: Rachel A. Brown
"""

from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import sep
import os
import pandas as pd
import linecache
from glob import glob


def snr_single(flux_dir):
    '''calculates med, std, snr for star in a .txt file with light curve
    input: path to directory containing flux files (path object)
    returns: x, y, med, std, snr and saves .txt file with this info '''
    
    #make lists to hold calculations
    snr = []
    med_list = []
    std_list = []
    x = []
    y = []

    #directory containing .txt files
    directory = flux_dir

    #get a list of star lightcurve files and sort properly
    files = os.listdir(directory)
    files = [f for f in files if '.txt' in f]
    files = [f for f in files if 'snr' not in f]
#    files = sorted(files, key = lambda x: float(x.split('_')[0].split('-')[1]))
    files = sorted(files, key = lambda x: float(x.split('_')[0].split('star')[1]))

    #loop through each star
    for filename in files:
            
            coords = linecache.getline(str(directory.joinpath(filename)),6).split(': ')[1].split(' ')    #star coordinates
            x.append(float(coords[0]))          #append X coord to list
            y.append(float(coords[1]))          #append Y coord to list
            
            #dataframe containing filenames, times, and flux values for star lightcurve
            flux = pd.read_csv(str(directory.joinpath(filename)), delim_whitespace = True, 
                               names = ['filename', 'time', 'flux'], comment = '#')
            
            med = np.median(flux['flux'])       #star median flux
            med_list.append(med)
            std = np.std(flux['flux'])          #standard deviation of star flux
            std_list.append(std)
            snr.append(med/std)                 #star SNR (median/stddev)
            
    
    savefile = flux_dir.joinpath('stars_snr.txt')       #save star fluxes etc to file
    
    with open(savefile, 'w') as filehandle:
        filehandle.write('#x    y    median   stddev   snr\n')
        for i in range(0, len(x)):
            filehandle.write('%f %f %f %f %f\n' %(x[i], y[i], med_list[i], std_list[i], snr[i]))
            
    stars = np.array([x, y, med_list, std_list, snr]).transpose()
            
    return stars
        
