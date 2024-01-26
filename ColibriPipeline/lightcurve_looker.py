# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 15:14:51 2021
Update: Jan. 24, 2022, 11:40

@author: Rachel A. Brown
Make plot of star light curves from .txt files with flux values
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pathlib
import linecache
from astropy.io import fits

def plot_wholecurves(directory):
    '''make .png with lightcurves from entire minute
    input: directory containing lightcurve text files (path object)
    returns: saves .png files for each star with light curve'''
    
    lightcurves = {}        #dictionary to save lightcurves in


    #get a list of files and sort properly
    files = os.listdir(directory)
    files = [x for x in directory.iterdir() if x.is_file()]
    files = [x for x in directory.iterdir() if 'stars_snr.txt' not in x.name]

    #loop through each star
    for filename in files:
        if filename.suffix == ".txt": 
        
            #make dataframe containing image name, time, and star flux value for the star
            flux = pd.read_csv(directory.joinpath(filename), delim_whitespace = True, 
                   names = ['filename', 'time', 'flux'], comment = '#')

            #star X, Y coordinates
            coords = linecache.getline(str(directory.joinpath(filename)),6).split(': ')[1].split(' ')
            

            starnum = filename.name.split('star')[1].split('_')[0]     #star number
            med = np.median(flux['flux'])                           #median flux
            std = np.std(flux['flux'])                              #standard deviation of flux
            SNR = med/std                                           #Star Signal/Noise = median/stddev
        
            #add star to dictionary with star number as key
            lightcurves[starnum] = {'flux': flux, 'coords': coords,
                               'median': med, 'std': std, 'SNR': SNR}
        else:
            continue
        
    
    #loop through each star
    for key, info in lightcurves.items():
        
        #get star info
        med = info['median']
        std = info['std']
        flux = info['flux']
        coords = info['coords']
        SNR = info['SNR']
         
        #fix time to account for minute rollover
        seconds = []    #list of times since first frame
        t0 = flux['time'][0]    #time of first frame
        
        for t in flux['time']:
    
            if t < t0:          #check if time has gone back to 0
                t = t + 60.
            
            if t != t0:         #check if minute has rolled over
                if t - t0 < seconds[-1]:
                    t = t + 60.
            
            seconds.append(t - t0)
                
        #make plot
        fig, ax1 = plt.subplots()

      #  biassub = './ColibriArchive/biassubtracted/202106023/20210623_00.52.14.567/sub_field1_25ms-E_0000002.fits'
      #starimage = fits.getdata(eventfile)
      #  starimage = fits.getdata(biassub)
  #  pad = 100
    
    # Define your sub-array
  #  subarray = starimage[int(float(coords[0]))-pad:int(float(coords[0]))+pad,int(float(coords[1]))-pad:int(float(coords[1]))+pad]

#    left, bottom, width, height = [0.95, 0.6, 0.2, 0.2]
#    ax2 = fig.add_axes([left, bottom, width, height])
#    ax2.imshow(subarray, vmin = 0, vmax = med)
    
        ax1.plot(seconds, flux['flux'])
        ax1.hlines(med, min(seconds), max(seconds), color = 'black', label = 'median: %i' % med)                                                                                                 
        ax1.hlines(med + std, min(seconds), max(seconds), linestyle = '--', color = 'black', label = 'stddev: %.3f' % std)
        ax1.hlines(med - std, min(seconds), max(seconds), linestyle = '--', color = 'black')
       
        ax1.set_xlabel('time (seconds)')
        ax1.set_ylabel('Counts/circular aperture')
        ax1.set_title('Star #%s [%.1f, %.1f], SNR = %.2f' %(key, float(coords[0]), float(coords[1]), SNR))
        ax1.legend()

        plt.savefig(directory.joinpath('star' + key + '.png'), bbox_inches = 'tight')
     #   plt.show()
        plt.close()
        
def plot_event(directory, starData, eventFrame, starNum, starCoords, eventType):
    '''make .png with lightcurves from entire minute
    input: directory containing lightcurve text files (path object)
    returns: saves .png files for each star with light curve'''
    
    lightcurve = starData['flux']        #dictionary to save lightcurves in
    times = starData['time']
    
    eventDate = pathlib.Path(starData['filename'][0]).parent.name
    
    med = np.median(lightcurve)
    std = np.std(lightcurve)
    SNR = med/std
         
    #fix time to account for minute rollover
    seconds = []    #list of times since first frame
    t0 = times[0]    #time of first frame
        
    for t in times:
    
        if t < t0:          #check if time has gone back to 0
            t = t + 60.
            
        if t != t0:         #check if minute has rolled over
            if t - t0 < seconds[-1]:
                t = t + 60.
            
        seconds.append(t - t0)
        
    eventTime = seconds[eventFrame]
    
    #make plot
    fig, ax1 = plt.subplots()

    
    ax1.plot(seconds, lightcurve)
    ax1.hlines(med, min(seconds), max(seconds), color = 'black', label = 'median: %i' % med)                                                                                                 
    ax1.hlines(med + std, min(seconds), max(seconds), linestyle = '--', color = 'black', label = 'stddev: %.3f' % std)
    ax1.hlines(med - std, min(seconds), max(seconds), linestyle = '--', color = 'black')
    
    ax1.vlines(eventTime, min(lightcurve), max(lightcurve), color = 'red', label = 'event: %f' %eventTime)
    ax1.set_xlabel('time (seconds)')
    ax1.set_ylabel('Counts/circular aperture')
    ax1.set_title('Star #%s [%.1f, %.1f], SNR = %.2f, Type = %s' %(starNum, float(starCoords[0]), float(starCoords[1]), SNR, eventType))
    ax1.legend()


    plt.savefig(directory.joinpath('star' + starNum + '_' + eventDate + '.png'), bbox_inches = 'tight')
   # plt.show()
    plt.close()
        
        
'''for single use'''
#plot_wholecurves(pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri', 'ColibriArchive', 'Red', '2022-03-01'))
    

