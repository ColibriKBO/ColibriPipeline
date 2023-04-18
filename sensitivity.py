#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" 
Created on Mon Nov 22 11:05:06 2021
Update: Jan. 24, 2022, 11:20

@author: Rachel A Brown
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime, date
import astropy
import astropy.stats
from astropy.io import fits
from astropy import wcs
from astropy.time import Time 
import getRAdec
import pathlib
import snplots
import lightcurve_maker
import lightcurve_looker
import read_npy
import VizieR_query
import astrometrynet_funcs
import sys, os, time
import argparse


def match_RADec(data, gdata, SR):
    '''matches list of found stars with Gaia catalog by RA/dec to get magnitudes
    input: pandas Dataframe of detected star data {x, y, RA, dec}, dataframe of Gaia data {ra, dec, magnitudes....}, search radius [deg]
    returns: original star data frame with gaia magnitude columns added where a match was found'''
    
    #from Mike Mazur's 'MagLimitChecker.py' script--------------------
    
    match_counter = 0
    
    for i in range(len(data.index)):            #loop through each detected star

        RA = data.loc[i,'ra']                  #RA coord for matching
        DEC = data.loc[i,'dec']                #dec coord for matching
         
        RA_SR = SR/np.cos(np.radians(DEC))     #adjustment for RA

        df = gdata[(gdata['RA_ICRS'] <= (RA+RA_SR))]      #make new data frame with rows from Gaia df that are within upper RA limit
        df = df[(df['RA_ICRS']  >= (RA-RA_SR))]            #only include rows from this new df that are within lower RA limit
        df = df[(df['DE_ICRS'] <= (DEC+SR))]          #only include rows from this new df that are within upper dec limit
        df = df[(df['DE_ICRS'] >= (DEC-SR))]          #only include rows from this new df that are withing lower dec limit
         
         #RAB - uncomment lines below to match based on smallest distance
        #df['diff'] = np.sqrt(((df.RA_ICRS - RA)**2*np.cos(np.radians(df.DE_ICRS))) + (df.DE_ICRS - DEC)**2)
        #df.sort_values(by=['diff'], ascending=True, inplace = True)
       
        #RAB - uncomment line below to match based on brightest magnitude
        df.sort_values(by=["Gmag"], ascending=True, inplace=True) #sort matches by brightness (brightest at top)
        
        #reset index after sorting
        df.reset_index(drop=True, inplace=True)

        #if matches are found, add corresponsing magnitude columns from brightest Gaia match to original star dataframe
        if len(df.index)>=1:   #RAB - added >= sign
             data.loc[i,'GMAG'] = df.loc[0]['Gmag']
             data.loc[i,'Gaia_RA'] = df.loc[0]['RA_ICRS']
             data.loc[i,'Gaia_dec'] = df.loc[0]['DE_ICRS']
             data.loc[i, 'Gaia_B-R'] = df.loc[0]['BP-RP']

             
       #end of Mike's section -------------------------------------------
             match_counter +=1 
    
    print('Number of Gaia matches: ', match_counter)
    
    return data
             

def match_XY(mags, snr, SR):
    '''matches two data frames based on X, Y 
    input: pandas dataframe of star data with Gaia magnitudes {x, y, RA, dec, 3 magnitudes} 
    dataframe of star data with SNR {x, y, med, std, snr}, search radius [px]
    returns: original star data frame with SNR data added where a match was found'''
    
    match_counter = 0
    
    #modified from Mike Mazur's code in 'MagLimitChecker.py'------------------
    SR = 0.1 # Search distance in pixels
    for i in range(len(mags.index)):                    #loop through each detected star
         X = mags.loc[i,'X']                            #X coord from star magnitude table
         Y = mags.loc[i,'Y']                            #Y coord from star magnitude table

         df = snr[(snr.X < (X+SR))]                     #make new data frame with rows from SNR df that are within upper X limit
         df = df[(df.X > (X-SR))]                       #only include rows from this new df that are within lower X limit
         df = df[(df.Y < (Y+SR))]                       #only include rows from this new df that are within lower Y limit
         df = df[(df.Y > (Y-SR))]                       #only include rows from this new df that are within lower Y limit
         df.sort_values(by=["SNR"], ascending=True, inplace=True)       #sort matches by SNR
         df.reset_index(drop=True, inplace=True)

         #if matches are found, add corresponsing med, std, SNR columns from match to original star magnitude dataframe
         if len(df.index)>=1:
             mags.loc[i,'med'] = df.loc[0]['med']
             mags.loc[i,'std'] = df.loc[0]['std']
             mags.loc[i,'SNR'] = df.loc[0]['SNR']
             
       #end of Mike's section -------------------------------------------
             match_counter +=1 
    
    print('Number of SNR matches: ', match_counter)
    return mags

def getAirmass(time, RA, dec):
    '''get airmass of the field at the given time
    input: time [isot format string], field coordinates [RA, Dec]
    returns: airmass, altitude [degrees], and azimuth [degrees]'''
    
    #latitude and longitude of Elginfield
    siteLat = 43.1933116667
    siteLong = -81.3160233333
    
    #get J2000 day
    
    #get local sidereal time
    LST = Time(time, format='isot', scale='utc').sidereal_time('mean', longitude = siteLong)

    #get hour angle of field
    HA = LST.deg - RA
    
    #convert angles to radians for sin/cos funcs
    dec = np.radians(dec)
    siteLat = np.radians(siteLat)
    HA = np.radians(HA)
    
    #get altitude (elevation) of field
    alt = np.arcsin(np.sin(dec)*np.sin(siteLat) + np.cos(dec)*np.cos(siteLat)*np.cos(HA))
    
    #get azimuth of field
    A = np.degrees(np.arccos((np.sin(dec) - np.sin(alt)*np.sin(siteLat))/(np.cos(alt)*np.cos(siteLat))))
    
    if np.sin(HA) < 0:
        az = A
    else:
        az = 360. - A
    
    alt = np.degrees(alt)
    
    #get zenith angle
    ZA = 90 - alt
    
    #get airmass
    airmass = 1./np.cos(np.radians(ZA))
    
    return airmass, alt, az

def RAdec_diffPlot(matched):
    '''makes plot of X difference vs Y differences, calculates mean and stddev
    input: dataframe containing columns 'x_diff' and 'y_diff'
    returns: displays and saves plot of xdifferences and y differences, prints out mean and stddev'''
    
    clip_sigma = 2
    clipped_radiff = astropy.stats.sigma_clip(matched['ra_diff'], sigma = clip_sigma, cenfunc = 'mean')
    clipped_decdiff = astropy.stats.sigma_clip(matched['dec_diff'], sigma = clip_sigma, cenfunc = 'mean')
    
    ra_mean = np.mean(clipped_radiff)*3600
    ra_std = np.std(clipped_radiff)*3600
    dec_mean = np.mean(clipped_decdiff)*3600
    dec_std = np.std(clipped_decdiff)*3600
    
    #print('RA mean: ', ra_mean)
    #print('RA stddev: ', ra_std)
    #print('dec mean:  ', dec_mean)
    #print('dec stddev: ', dec_std)
    
    plt.figure(figsize=(10, 10), dpi=100)
    
    plt.scatter(matched['ra_diff'].multiply(3600), matched['dec_diff'].multiply(3600), s = 8)
    plt.scatter(0, 0, color = 'black', marker = '+', s = 50)
    
    plt.text(-14, 14, 'RA mean difference: %.3f +/- %.3f "' %(ra_mean, ra_std), fontsize = 18)
    plt.text(-14, 12, 'Dec mean difference: %.3f +/- %.3f "' %(dec_mean, dec_std), fontsize = 18)
    plt.text(-14, 10, '(%.0f$\sigma$ clipped)' %(clip_sigma), fontsize = 18)

    #plt.xlim(-0.004*3600, 0.004*3600)
    plt.xlim(-7*2.4, 7*2.4)
    plt.ylim(-7*2.4, 7*2.4)

    plt.title(polynom_order + ' order, ' + telescope + ', ' + str(field_centre) + ', ' + str(obs_date) + ', ' + str(obs_time), fontsize = 20)
    plt.xlabel('(SEP Measured RA - Gaia RA) [arcsec]', fontsize = 18)
    plt.ylabel('(SEP Measured Dec - Gaia Dec) [arcsec]', fontsize = 18)
    plt.yticks(fontsize = 15)
    plt.xticks(fontsize = 15)
    plt.gca().set_aspect('equal', adjustable='box')

    plt.savefig(save_path.joinpath('RA-dec_diff_' + gain + '_' + str(detect_thresh) + 'sig_' +  telescope + '_' + str(field_centre) + '_' + str(obs_date) + '_' + str(obs_time).replace(':','.') + '.png'))
    
'''---------------------------------SCRIPT STARTS HERE--------------------------------------------'''

'''------------set up--------------------'''
print('setting up')
#time and date of observations/processing
#MODIFY LINES BELOW BEFORE RUNNING
arg_parser = argparse.ArgumentParser(description=""" Run secondary Colibri processing
    Usage:

    """,
    formatter_class=argparse.RawTextHelpFormatter)

arg_parser.add_argument('-b', '--basedir', help='Base directory for data (typically d:)', default='d:')
arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')
arg_parser.add_argument('-t', '--threshold', help='Star detection threshold.', default='4')
arg_parser.add_argument('-m', '--minute', help='hh.mm.ss to process.')
arg_parser.add_argument('-l', '--lightcurve', help='Star detection threshold.', default=True)

cml_args = arg_parser.parse_args()
obsYYYYMMDD = cml_args.date
obs_date = date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))

cml_args = arg_parser.parse_args()

base_path = pathlib.Path(cml_args.basedir)
data_path = base_path.joinpath('/ColibriData', str(obs_date).replace('-', ''))    #path to data


minute_dirs=[f.name for f in data_path.iterdir() if ('Bias' not in f.name and '.txt' not in f.name)]




if not cml_args.minute:
    obs_time=minute_dirs[int(len(minute_dirs) / 2)].split('_')[1][:-4]
else:
    try:
        desired_time = str(cml_args.minute)
        #desired_time =datetime.strptime(desired_time,"%H.%M.%S").time()
        for i in range(len(minute_dirs)):
            # print(minute_dirs[i])
            # if (minute_names[i]>det_time and i==0):
            #     print('search lightcurve in ',minute_names[i])
            minute_time=datetime.strptime(minute_dirs[i].split('_')[1],"%H.%M.%S.%f").time()
            
            
            if desired_time[:-2] in minute_dirs[i]:
                obs_time=str(minute_time).replace(':','.')[:-4]
                print(obs_time)
                break
    except:
        obs_time=minute_dirs[int(len(minute_dirs) / 2)].split('_')[1][:-4]
        

cml_args = arg_parser.parse_args()
detect_thresh = int(cml_args.threshold)


# obs_date = datetime.date(2022, 8, 12)           #date of observation
# obs_time = datetime.time(3, 42, 28)             #time of observation (to the second)
#image_index = '2'                               #index of image to use (if uploading to astrometry.net manually)
polynom_order = '4th'                           #order of astrometry.net plate solution polynomial
ap_r = 3                                        #radius of aperture for photometry
gain = 'high'                                   #which gain to take from rcd files ('low' or 'high')
global telescope
telescope = os.environ['COMPUTERNAME']       #identifier for telescope                              #telescope identifier
#field_name = 'field1'                           #name of field observed
                             #detection threshold

#paths to required files
#MODIFY BASE PATH BEFORE RUNNING
#base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri', telescope)  
# base_path = pathlib.Path('D:\\')                            #path to main directory


#get exact name of desired minute directory
#subdirs = [f.name for f in data_path.iterdir() if f.is_dir()]                   #all minutes in night directory (comment out if setting minutedir explicitly)
minute_dir = [f for f in minute_dirs if obs_time in f][0]    #minute we're interested in
#minute_dir = '20220518_05.40.22.844'                                           #minute label (if don't have data)

#path to output files
save_path = base_path.joinpath('/ColibriArchive', str(obs_date).replace('-', '') + '_diagnostics', 'Sensitivity', minute_dir)       #path to save outputs in
#save_path = base_path.joinpath('Elginfield' + telescope, str(obs_date).replace('-', '') + '_diagnostics', 'Sensitivity', minute_dir)       #path to save outputs in


lightcurve_path = save_path.joinpath(gain + '_' + str(detect_thresh) +  'sig_lightcurves')          #path that light curves are saved to

#make directory to hold results in if doesn't already exist
save_path.mkdir(parents=True, exist_ok=True)


'''-------------make light curves of data----------------------'''
if cml_args.lightcurve==True:
    print('making light curve .txt files')
    lightcurve_maker.getLightcurves(data_path.joinpath(minute_dir), save_path, ap_r, gain, telescope, detect_thresh)   #save .txt files with times|fluxes

#save .png plots of lightcurves
print('saving plots of light curves')
#lightcurve_looker.plot_wholecurves(lightcurve_path)

'''--------upload image to astrometry.net for plate solution------'''
median_image = save_path.joinpath('high_medstacked.fits')     #path to median combined file for astrometry solution
median_str="/mnt/d/"+str(median_image).replace('d:', '').replace('\\', '/') #10-12 Roman A.
median_str=median_str.lower()
transform_file = save_path.joinpath(minute_dir + '_' + polynom_order + '_wcs.fits') #path to save WCS header file in
transform_str=str(transform_file).split('\\')[-1]

#check if the tranformation has already been calculated and saved
if transform_file.exists():
            
    #open file and get transformation
    wcs_header = fits.open(transform_file)
    transform = wcs.WCS(wcs_header[0])

else:
    #get solution from astrometry.net
    wcs_header = astrometrynet_funcs.getLocalSolution(median_str, transform_str, int(polynom_order[0]))
        
    #calculate coordinate transformation
    transform = wcs.WCS(wcs_header)

'''-----------get [x,y] to [RA, Dec] transformation------------'''
print('RA to dec transformation')

#get filepaths
#transform_file = sorted(save_path.glob('*' + image_index + '_new-image_' + polynom_order + '.fits'))[0]     #output from astrometry.net
star_pos_file = sorted(save_path.glob('*' + gain + '*' + str(detect_thresh) + 'sig' + '*.npy'))[0]          #star position file from lightcurve_maker
star_pos_ds9 = save_path.joinpath(star_pos_file.name.replace('.npy', '_ds9.txt'))                           #star positions in ds9 format (for marking regions)
RADec_file = save_path.joinpath('XY_RD_' + gain + '_' + str(detect_thresh) + 'sig_' + polynom_order + '_' + minute_dir + '.txt')  #file to save XY-RD in


#get dataframe of {x, y, RA, dec}
#coords = getRAdec.getRAdecfromFile(transform_file, star_pos_file, RADec_file)
coords = getRAdec.getRAdec(transform, star_pos_file, RADec_file)
coords_df = pd.DataFrame(coords, columns = ['X', 'Y', 'ra', 'dec'])         #pandas dataframe containing coordinate info

#save star coords in .txt file using ds9 format (can be marked on ds9 image using 'regions' tool)
read_npy.to_ds9(star_pos_file, star_pos_ds9)


'''---------read in Gaia coord file to get magnitudes----------'''
print('getting Gaia data')

field_centre = [round(coords_df['ra'].median(), 4), round(coords_df['dec'].median(), 4)]  #take field centre to be med star coords
print('Field centre: ', field_centre[0], field_centre[1])
gaia_SR = 1.5         #search radius for query in degrees
gaia = VizieR_query.makeQuery(field_centre, gaia_SR)        #get dataframe of Gaia results {RA, dec, Gmag}


'''----------calculate altitude, azimuth, airmass of field-----------------------'''
#get timestamp for beginning of minute
starTxtFile = sorted(lightcurve_path.glob('*.txt'))[0]

#get header info from file
with starTxtFile.open() as f:
        
    #loop through each line of the file
    for i, line in enumerate(f):
            
        #get event time
        if i == 6:
            timestamp = line.split(' ')[-1].split('\n')[0]

airmass, altitude, azimuth = getAirmass(timestamp, *field_centre)


'''----------------------get star SNRs------------------------'''
print('calculating SNRS')
#dataframe of star info {x, y, median, stddev, median/stddev}
stars = pd.DataFrame(snplots.snr_single(lightcurve_path), columns = ['X', 'Y', 'med', 'std', 'SNR'])


'''----------------matching tables----------------------------'''
print('matching tables')
# 1: match (RA, dec) from light curves with (RA, dec) from Gaia to get magnitudes
SR = 1/3600   #half side length of search box in degrees (1 arcsec)

rd_mag = match_RADec(coords_df, gaia, SR)       #dataframe of Colibri detections with Gaia magnitudes {X, Y, ra, dec, gmag}

# 2: match (X,Y) from light curves with (X,Y) from position file to get (RA, dec)
final = match_XY(rd_mag, stars, SR)      #dataframe of Colibri detections with SNR and Gaia magnitudes {X, Y, ra, dec, GMAG, med, std, SNR}

#filter out bad regions of image
#for Blue: filter out stars that are in striped regions (right and left sides)
#final = final.drop(final[(final.X < 450) | (final.X > 1750)].index)
#For Jan. Red data: filter out stars in striped regions (right side only)
#final = final.drop(final[(final.X > 1750)].index)    
final_sort = final.sort_values('GMAG')
#save text file version of this table for later reference
final.to_csv(save_path.joinpath('starTable_' + minute_dir + '_' + gain + '_' + polynom_order + '_' + telescope + '_' + str(detect_thresh) +'_'+str(airmass)+'_' + 'sig.txt'), sep = ' ', na_rep = 'nan')

'''---------------------make plot of mag vs snr-----------------'''
print('making mag vs snr plot')
plt.scatter(final['GMAG'], final['SNR'], s = 5, label = 'Airmass: %.2f\nAlt, Az: (%.2f, %.2f)' % (airmass, altitude, azimuth))

#plt.ylim(0, 16)
#plt.title('ap r = ' + str(ap_r) + ', ' + str(field_centre) + ', ' + telescope + ', ' + str(obs_date) + ', ' + str(obs_time) + ', ' + gain)
plt.title('ap r = ' + str(ap_r) + ', ' + str(field_centre) + ', ' + telescope + ', ' + str(obs_date) + ', ' + str(obs_time) + ', ' + gain+
    ', stars:'+ str(np.sum(~np.isnan(np.array(final['GMAG'])))))
#plt.title('Sensitivity limits for "%s" telescope at airmass %.1f' %(telescope, airmass))
plt.xlabel('Gaia g-band magnitude')
#plt.ylabel('Star signal-to-noise ratio')
plt.ylabel('Signal-to-Noise Ratio (star median/stddev)')
plt.grid()

plt.legend()
plt.savefig(save_path.joinpath('magvSNR_' + gain  + '_' + minute_dir + '_' + str(detect_thresh) +'.png'), bbox_inches = 'tight')#, dpi = 3000)
plt.close()


'''----------------------make plot of mag vs mean value-----------'''
print('making mag vs mean plot')
plt.scatter(final['GMAG'], -2.5*np.log(final['med']), s = 10, label = 'Airmass: %.2f\nAlt, Az: (%.2f, %.2f)' % (airmass, altitude, azimuth))

plt.title('ap r = ' + str(ap_r) + ', ' + str(field_centre) + ', ' + telescope + ', ' + str(obs_date) + ', ' + str(obs_time) + ', ' + gain)
plt.xlabel('Gaia g-band magnitude')
plt.ylabel('-2.5*log(median)')

plt.grid()
plt.legend()
plt.savefig(save_path.joinpath('magvmed_' + gain + '_' + minute_dir + '_' + str(detect_thresh) + '.png'), bbox_inches = 'tight')
plt.close()
'''----------------Print Statements-----------------'''
#get number of stars with SNR greater than 10
high_SNR = final[final['SNR'] >= 10.]   #dataframe of results with SNR >= 10
final_dropna = final.dropna()           #dataframe excluding unmatched stars
print('SNR results: ')
print('Number of stars detected: ', rd_mag.shape[0])
print('Final # of stars matched: ', final_dropna.shape[0])
print('Number of stars with SNR >= 10: ', high_SNR.shape[0])


'''-----------find unmatched rows--------------'''
#is_NaN = final.isnull()

#row_has_NaN = is_NaN.any(axis=1)

#unmatched = final[row_has_NaN]

final['ra_diff'] = (final['ra'] - final['Gaia_RA'])*np.cos(np.radians(final['dec']))
final['dec_diff']  = (final['dec'] - final['Gaia_dec'])

RAdec_diffPlot(final)
