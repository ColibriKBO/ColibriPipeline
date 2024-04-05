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
import numpy as np
from pathlib import Path
from astropy import wcs
from astropy.io import fits
from datetime import date

# Custom Script Imports
import astrometrynet_funcs
#import getRAdec



#--------------------------------functions------------------------------------#

def getTransform(timestamp, median_stacks, transformations, return_transformations=False):
    """
    Finds median image that best fits for the time of the detection and uses it to get Astrometry solution.
    Required to have a list of median-combined images (median_combos)

    Parameters
    ----------
    timestamp : str
        Time of the detection file HH.MM.SS.ms
    median_stacks: list
        A list object containing either a single or multiple median stack objects.
        Matches name to timestamp if multiple options are given

    Returns
    -------
    transform : file headers
        Headers of the WCS file that contain transformation info.

    """
    

    #if transformation has already been calculated, get from dictionary
    if timestamp in transformations:
        if return_transformations:
            return transformations[timestamp], transformations
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
        
        if return_transformations:
            return transform, transformations
        else:
            return transform


def updateNPY_RAdec(transform, star_pos_file, savefile=None):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file (path object), star position file (.npy path object), filename to save to (path object)
    returns: coordinate transform'''
    
    #load in transformation information
#    transform_im = fits.open(transform_file)
#    transform = wcs.WCS(transform_im[0].header)
    
    #get star coordinates from observation image (.npy file)
    star_pos = np.load(star_pos_file)
    
    #get transformation
    world = transform.all_pix2world(star_pos[:,[0,1]], 0,ra_dec_order=True)[:,[0,1]] #2022-07-21 Roman A. changed solution function to fit SIP distortion
   # print(world)
   # px = transform.wcs_world2pix(world, 0)
   # print(px)
    
    #optional: save text file with transformation
    if savefile is not None:
        with open(savefile, 'w') as filehandle:
            filehandle.write('#\n#\n#\n#\n#X  Y  RA  Dec\n')
    
            for i in range(0, len(star_pos)):
                #output table: | x | y | RA | Dec | 
                filehandle.write('%f %f %f %f\n' %(star_pos[i][0], star_pos[i][1], star_pos[i][2], world[i][0], world[i][1]))
      
    coords = np.hstack((star_pos, world))
    return coords


def getRAdec_arrays(transform, star_pos):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file (path object), star position file (.npy path object), filename to save to (path object)
    returns: coordinate transform'''
    
    #load in transformation information
#    transform_im = fits.open(transform_file)
#    transform = wcs.WCS(transform_im[0].header)
    
    #get transformation
    world = transform.all_pix2world(star_pos, 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion
                
    # output table: | x | y | RA | Dec | 
    #coords = np.array([star_pos[:,0], star_pos[:,1], world[:,0], world[:,1]]).transpose()
    coords = np.hstack((star_pos[:,0], star_pos[:,1], world[:,0], world[:,1]))
    return coords


def getRAdecfromFile(transform_file, star_pos_file, savefile):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file (path object), star position file (.npy path object), filename to save to (path object)
    returns: coordinate transform'''
    
    #load in transformation information
    transform_im = fits.open(transform_file)
    transform = wcs.WCS(transform_im[0].header)
    
    #get star coordinates from observation image (.npy file)
    star_pos = np.load(star_pos_file)
    
    #get transformation
    world = transform.all_pix2world(star_pos, 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion
   # print(world)
   # px = transform.wcs_world2pix(world, 0)
   # print(px)
    
    #optional: save text file with transformation
    with open(savefile, 'w') as filehandle:
        filehandle.write('#\n#\n#\n#\n#X  Y  RA  Dec\n')
    
        for i in range(0, len(star_pos)):
            #output table: | x | y | RA | Dec | 
            filehandle.write('%f %f %f %f\n' %(star_pos[i][0], star_pos[i][1], world[i][0], world[i][1]))
      
    coords = np.array([star_pos[:,0], star_pos[:,1], world[:,0], world[:,1]]).transpose()
    return coords


def getXY(transform_file, star_pos_file, savefile=None):
    '''get WCS transform ([RA,dec] -> [X,Y]) from astrometry.net header
    input: astrometry.net output file, star position file (.npy)
    returns: coordinate transform'''
    
    #load in transformation information
    transform_im = fits.open(transform_file)
    transform = wcs.WCS(transform_im[0].header)
    
    #get star coordinates from observation image (.npy file)
    star_pos = np.load(star_pos_file)
    
    #get transformation
    px = transform.wcs_world2pix(star_pos, 0)
   # print(px)
    
    #optional: save text file with transformation
    if savefile != None:
        with open(savefile, 'w') as filehandle:
            filehandle.write('#\n#\n#\n#\n#X  Y  RA  Dec\n')
    
            for i in range(0, len(star_pos)):
                #output table: | x | y | RA | Dec | 
                filehandle.write('%f %f %f %f\n' %(star_pos[i][0], star_pos[i][1], px[i][0], px[i][1]))
      
    coords = np.array([star_pos[:,0], star_pos[:,1], px[:,0], px[:,1]]).transpose()
    return coords


def getRAdecSingle(transform, star_pos):
    '''get WCS transform from astrometry.net header for a single star
    input: astrometry.net transformation (object), star position (X, Y)
    returns: star position in RA/dec'''
        
    #get transformation
    star_pos = np.array([[star_pos[0], star_pos[1]]])
    
    world = transform.all_pix2world(np.array(star_pos), 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion
  #  star_pos = np.array([star_pos[0], star_pos[1]])
   # world = transform.pixel_to_world(np.array(star_pos))
    
    return world[0]

def getXYSingle(transform, star_pos):
    '''get WCS transform from astrometry.net header for a single star
    input: astrometry.net transformation (object), star position (RA, dec)
    returns: star position in X/Y'''
        
    #get transformation
    star_pos = np.array([[star_pos[0], star_pos[1]]])
    
    px = transform.wcs_world2pix(np.array(star_pos), 0)
    
    return px[0]


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

    
        transform,transformations = getTransform(timestamp, median_combos, 
                                                 transformations, return_transformations=True)
        
        
        #get star coords in RA/dec
        star_wcs = getRAdecSingle(transform, (eventData[2], eventData[3]))
        star_RA = star_wcs[0]
        star_DEC = star_wcs[1]
        
        #write a line with RA dec in each file
        with open(filepath, 'r') as filehandle:
            lines = filehandle.readlines()
            lines[6] = '#    RA Dec Coords: %f %f\n' %(star_RA, star_DEC)
        with open(filepath, 'w') as filehandle:
            filehandle.writelines( lines )
