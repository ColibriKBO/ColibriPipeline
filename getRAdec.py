# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 14:51:30 2021
Update: Jan. 24, 2022 11:20

@author: Rachel A Brown
"""

import astropy
from astropy.io import fits
import numpy as np
import pathlib
import datetime
from astropy import wcs


#--------------------------------functions------------------------------------#

def getRAdec(transform, star_pos_file, savefile):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file (path object), star position file (.npy path object), filename to save to (path object)
    returns: coordinate transform'''
    
    #load in transformation information
#    transform_im = fits.open(transform_file)
#    transform = wcs.WCS(transform_im[0].header)
    
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
    
    


