# -*- coding: utf-8 -*-
"""
Created on 20.04.2023

@author: Roman A.

Create and save plots of photometry performed on 1-minute stacked frames

"""




import matplotlib.pyplot as plt
import pathlib
import numpy as np
import pandas as pd
import linecache
import matplotlib.dates as mdates
from tqdm import trange
from matplotlib.dates import DateFormatter
from astropy import wcs, stats

def get_Lightcurve(file):
    ''' retrieve lightcurve data from .txt files
    input: file name (pathlib.Path object)
    output: lightcurve (dict) '''
    # print(file)
    lightcurve = {}

    flux = pd.read_csv(file, delim_whitespace = True, names = ['filename', 'time', 'flux', 'flux_err', 'rel_flux', 'x', 'y'], comment = '#')

    coords = linecache.getline(str(file),6).split(': ')[1].split(' ')
    # med = np.median(flux['rel_flux'])                                   #median flux
    # std = np.std(flux['rel_flux'])
    
    med=stats.sigma_clipped_stats(flux['rel_flux'])[1]
    std=stats.sigma_clipped_stats(flux['rel_flux'])[2]
    try:
        SNR = med/std
    except:
        SNR=0

    lightcurve = {'flux': flux, 'coords': coords, 'median': med, 'std': std, 'SNR': SNR}
    return lightcurve

def lookLightcurve(star, lightcurve, save_path):
    ''' used to plot lightcurve against time  
    input: star number (str), star lightcurve (dict), 
     
    output: None'''

    #get star info
    med = lightcurve['median']
    std = lightcurve['std']
    flux = lightcurve['flux']
    coords = lightcurve['coords']
    SNR = lightcurve['SNR']
    
    seconds=flux['time']

    fig, ax1 = plt.subplots()
    
    ax1.scatter(seconds, flux['rel_flux']) #can also plot raw flux
    ax1.hlines(med, min(seconds), max(seconds), color = 'black', label = 'median: %.5f' % med)
    ax1.hlines(med + std, min(seconds), max(seconds), linestyle = '--', color = 'black', label = 'stddev: %.5f' % std)
    ax1.hlines(med - std, min(seconds), max(seconds), linestyle = '--', color = 'black')
    ax1.set_xlabel('time (UT)')
    ax1.set_ylabel('relative counts')
    ax1.set_title('Star #%s [%.1f, %.1f], SNR = %.2f' %(star, float(coords[0]), float(coords[1]), SNR))
    xfmt = DateFormatter('%H:%M') #time format works even for JD
    ax1.xaxis.set_major_formatter(xfmt)
    ax1.legend()
    directory = save_path
    plt.savefig(directory.joinpath(str(star) + '_'+telescope+'_rel.png'), bbox_inches = 'tight',dpi=500)
    # plt.show()
    plt.close()

    

if __name__ == '__main__':
    base_path=pathlib.Path('/','E:')
    field_name='field6'
    obs_date='2023-04-10'
    telescope='Green'
    lc_path=base_path.joinpath('/StackedData',field_name,obs_date, telescope, 'lightcurves')#path where photometry results are
    stars=[f for f in lc_path.iterdir() if '.txt' in f.name]#list of txts with photometry data
    rel_SNRs=[]#for statistics only
    rel_MEDs=[]#for statistics only
    for i in trange(len(stars)):
        
        lightcurve = get_Lightcurve(stars[i]) #dictionary with photometric results
        # flux = lightcurve['flux']
        # clipped_flux=stats.sigma_clip(flux['flux'],masked=False,axis=0)
        # rel_SNRs.append(lightcurve['SNR'])
        # rel_MEDs.append(lightcurve['median'])
        lookLightcurve(stars[i].name.split('_')[0], lightcurve, lc_path) #plot each star
        
