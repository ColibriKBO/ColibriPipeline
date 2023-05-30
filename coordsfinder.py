# -*- coding: utf-8 -*-
"""
Filename:   coordsfinder.py
Author(s):  Roman Akhmetshyn
Contact:    roman_akhmetshyn@univ.kiev.ua
Created:    Thu Oct 20 16:58:29 2022
Updated:    Thu Oct 20 16:58:29 2022
    
Usage: python coordsfinder.py [-d]

Description: writes Ra Dec coordinates into dip detection txt file by performing AstrometryNet transformation
"""

# Module Imports

import argparse
import pandas as pd
from pathlib import Path
from astropy import wcs
from astropy.io import fits
from datetime import date

# Custom Script Imports
import astrometrynet_funcs
import getRAdec



#--------------------------------functions------------------------------------#

def getTransform(timestamp, median_stacks):
    """
    Finds median image that best fits for the time of the detection and uses it to get Astrometry solution.
    Required to have a list of median-combined images (median_combos)

    Parameters
    ----------
    timestamp : str
        Time of the detection file HH.MM.SS.ms

    Returns
    -------
    transform : file headers
        Headers of the WCS file that contain transformation info.

    """
    

    #if transformation has already been calculated, get from dictionary
    if timestamp in transformations:
        return transformations[timestamp]
    
    #calculate new transformation from astrometry.net
    else:
        #get median combined image
        median_image = [f for f in median_stacks if timestamp in f.name][0] #this is to run web version of Astrometry
        #10-12 Roman A. to run astrometry localy using Linux subsystem
        median_str="/mnt/d/"+str(median_image).replace('D:', '').replace('\\', '/')
        median_str=median_str.lower()
        
        #get name of transformation header file to save
        #this is to run web version of Astrometry
        transform_file = median_image.parent / (median_image.name.replace('medstacked', 'wcs'))
        
        #10-12 Roman A. to run astrometry localy using Linux subsystem
        transform_str=str(transform_file).split('\\')[-1]
        
        #check if the tranformation has already been calculated and saved
        if transform_file.exists():
            
            #open file and get transformation
            wcs_header = fits.open(transform_file)
            transform = wcs.WCS(wcs_header[0])

        #calculate new transformation
        else:
            #get WCS header from astrometry.net plate solution
            soln_order = 4
            
            try:
                #try if local Astrometry can solve it
                wcs_header = astrometrynet_funcs.getLocalSolution(median_str, transform_str, soln_order) #10-12 Roman A.
            except:
                wcs_header = astrometrynet_funcs.getSolution(median_image, transform_file, soln_order)
        
            #calculate coordinate transformation
            transform = wcs.WCS(wcs_header)
        
        #add to dictionary
        transformations[timestamp] = transform
        
        return transform


def readFile(filepath):
    """
    read in a .txt detection file and get information from it

    Parameters
    ----------
    filepath : path-like obj.
        Path of the detection .txt.

    Returns
    -------
    starData : dataframe
        Dataframe containing image name, time, and star flux value for the star.
    event_frame : TYPE
        DESCRIPTION.
    star_x : float
        Star coordinates in XY.
    star_y : float
        Star coordinates in XY.
    event_time : str
        Time of the dip event.
    event_type : str
        Significance of the event?.
    star_med : float
        Median of the lightcurve.
    star_std : float
        Std of the lightcurve.

    """
    
    
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



#-----------------------------------main--------------------------------------#

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description="Run secondary Colibri processing",
                                        formatter_class=argparse.RawTextHelpFormatter)
    arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')
    # arg_parser.add_argument('-p', '--procdate', help='Processing date.', default=obs_date)

    cml_args = arg_parser.parse_args()
    obsYYYYMMDD = cml_args.date
    obs_date = date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
        
    print(obs_date)
    data_path=Path('/','D:/','ColibriArchive',str(obs_date)) #path to archive with dip detection txts

    detect_files = [f for f in data_path.iterdir() if 'det' in f.name] #list of txts

    ''' get astrometry.net plate solution for each median combined image (1 per minute with detections)'''
    median_combos = [f for f in data_path.iterdir() if 'medstacked' in f.name] #list of median combined images to use for WCS

    #dictionary to hold WCS transformations for each transform file
    transformations = {}

    for filepath in detect_files:
        
        #read in file data as tuple containing (star lightcurve, event frame #, star x coord, star y coord, event time, event type, star med, star std)
        eventData = readFile(filepath)

        #get corresponding WCS transformation
        timestamp = Path(eventData[0]['filename'][0]).parent.name.split('_')[1]

    
        transform = getTransform(timestamp, median_combos)
        
        
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
