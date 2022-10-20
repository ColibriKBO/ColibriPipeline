# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 16:58:29 2022

@author: Roman A.
"""



from pathlib import Path
import pandas as pd
from astropy import wcs
from astropy.io import fits
import astrometrynet_funcs
import getRAdec
import sys
from datetime import date
import argparse

def getTransform(date):                                     #redifinition from Colibri Pipeline's function
    '''get astrometry.net transform for a given minute'''

    #if transformation has already been calculated, get from dictionary
    if date in transformations:
        return transformations[date]
    
    #calculate new transformation from astrometry.net
    else:
        #get median combined image
        median_image = [f for f in median_combos if date in f.name][0]
        median_str="/mnt/d/"+str(median_image).replace('D:', '').replace('\\', '/') #10-12 Roman A.
        median_str=median_str.lower()
        
        #get name of transformation header file to save
        transform_file = median_image.with_name(median_image.name.strip('_medstacked.fits') + '_wcs.fits')
        
        transform_str=str(transform_file).split('\\')[-1] #10-12 Roman A.
        
        #check if the tranformation has already been calculated and saved
        if transform_file.exists():
            
            #open file and get transformation
            wcs_header = fits.open(transform_file)
            transform = wcs.WCS(wcs_header[0])

        #calculate new transformation
        else:
            #get WCS header from astrometry.net plate solution
            soln_order = 4
            
            wcs_header = astrometrynet_funcs.getLocalSolution(median_str, transform_str, soln_order) #10-12 Roman A.
        
            #calculate coordinate transformation
            transform = wcs.WCS(wcs_header)
        
        #add to dictionary
        transformations[date] = transform
        
        return transform

def readFile(filepath):                                      #redifinition from Colibri Pipeline's function
    '''read in a .txt detection file and get information from it'''
    
    #make dataframe containing image name, time, and star flux value for the star
    starData = pd.read_csv(filepath, delim_whitespace = True, 
           names = ['filename', 'time', 'flux','conv_flux'], comment = '#')

    first_frame = int(starData['filename'][0].split('_')[-1].split('.')[0])
    
    #get header info from file
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            

            #get event frame number
            if i == 4:
                event_frame = int(line.split('_')[-1].split('.')[0])

            #get star coords
            elif i == 5:
                star_coords = line.split(':')[1].split(' ')[1:3]
                star_x = float(star_coords[0])
                star_y = float(star_coords[1])
            
            #get event time
            elif i == 7:
                event_time = line.split('T')[2].split('\n')[0]
                
            elif i == 10:
                event_type = line.split(':')[1].split('\n')[0].strip(" ")
                
            elif i == 11:
                star_med = line.split(':')[1].split('\n')[0].strip(" ")
                
            elif i == 12:
                star_std = line.split(':')[1].split('\n')[0].strip(' ')
                
        #reset event frame to match index of the file
        event_frame = event_frame - first_frame

    #return all event data from file as a tuple    
    return (starData, event_frame, star_x, star_y, event_time, event_type, star_med, star_std)


if len(sys.argv) > 2:
    arg_parser = argparse.ArgumentParser(description=""" Run secondary Colibri processing
        Usage:
    
        """,
        formatter_class=argparse.RawTextHelpFormatter)
    
    arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')
    # arg_parser.add_argument('-p', '--procdate', help='Processing date.', default=obs_date)
    cml_args = arg_parser.parse_args()
    obsYYYYMMDD = cml_args.date
    obs_date = date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
else:
    obs_date='2022-09-30'
    
print(obs_date)
data_path=Path('/','D:','/ColibriArchive',str(obs_date))

detect_files = [f for f in data_path.iterdir() if 'det' in f.name] #list of time matches

''' get astrometry.net plate solution for each median combined image (1 per minute with detections)'''
median_combos = [f for f in data_path.iterdir() if 'medstacked' in f.name]

#dictionary to hold WCS transformations for each transform file
transformations = {}

for filepath in detect_files:
    
    

    
    #read in file data as tuple containing (star lightcurve, event frame #, star x coord, star y coord, event time, event type, star med, star std)
    eventData = readFile(filepath)

    #get corresponding WCS transformation
    date = Path(eventData[0]['filename'][0]).parent.name.split('_')[1]

   
    transform = getTransform(date)
    
    
    #get star coords in RA/dec
    star_wcs = getRAdec.getRAdecSingle(transform, (eventData[2], eventData[3]))
    star_RA = star_wcs[0]
    star_DEC = star_wcs[1]
    
    #write a line with RA dec in each file
    with open(filepath, 'r') as filehandle:
        lines = filehandle.readlines()
        lines[6] = '#    RA Dec Coords: %f %f\n' %(star_RA, star_DEC)
    with open(filepath, 'w') as filehandle:
        filehandle.writelines( lines )