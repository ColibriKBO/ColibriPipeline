#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 13:32:08 2022

@author: Roman A.

Perform multiaperture photometry of 1-minute stacked frames
"""

import sep
import numpy as np
from glob import glob
from astropy.io import fits
from astropy.time import Time
from copy import deepcopy
import pathlib
import datetime
import os
from tqdm import trange
import sys
import math
from astropy import wcs, stats

def getXY(transform_header, ra_dec):
    '''get WCS transform ([RA,dec] -> [X,Y]) from astrometry.net header
    input: astrometry.net output file, star position file (.npy)
    returns: coordinate transform'''
    
    #load in transformation information

    transform = wcs.WCS(transform_header)
    
    ra=ra_dec[:,0]
    dec=ra_dec[:,1]

    px = transform.all_world2pix(ra, dec, 0, ra_dec_order=True,quiet=True)
   # print(px)
    
      
    
    return px[0], px[1]

def getRAdec(transform, xy):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file (path object), star position file (.npy path object), filename to save to (path object)
    returns: coordinate transform'''
    
    #load in transformation information
#    transform_im = fits.open(transform_file)
#    transform = wcs.WCS(transform_im[0].header)
    
    #get star coordinates from observation image (.npy file)
    x=xy[:,0]
    y=xy[:,1]
    
    
    
    #get transformation
    world = transform.all_pix2world(x,y, 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion
    
   # px = transform.wcs_world2pix(world, 0)
   # print(px)
    
      
    coords = np.array([world[0], world[1]]).transpose()
    
    return coords



def initialFindFITS(data, detect_thresh):
    """ Locates the stars in the initial time slice 
    input: flux data in 2D array for a fits image, star detection threshold (float)
    returns: [x, y, half light radius] of all stars in pixels"""

    ''' Background extraction for initial time slice'''
    data_new = deepcopy(data)           #make copy of data
    bkg = sep.Background(data_new)      #get background array
    bkg.subfrom(data_new)               #subtract background from data
    thresh = detect_thresh * bkg.globalrms      # set detection threshold to mean + 3 sigma


    ''' Identify stars in initial time slice '''
    objects = sep.extract(data_new, thresh)#, deblend_nthresh = 1)


    ''' Characterize light profile of each star '''
    halfLightRad = np.sqrt(objects['npix'] / np.pi) / 2.  # approximate half light radius as half of radius

    
    ''' Generate tuple of (x,y,r) positions for each star'''
    positions = zip(objects['x'], objects['y'], halfLightRad, objects['npix'])
    

    return positions


def refineCentroid(data, time, coords, sigma):
    """ Refines the centroid for each star for an image based on previous coords, used for tracking
    input: flux data in 2D array for single fits image, header time of image, 
    coord of stars in previous image, weighting (Gauss sigma)
    returns: new [x, y] positions, header time of image """

    '''initial x, y positions'''
    x_initial = [pos[0] for pos in coords]
    y_initial = [pos[1] for pos in coords]
    
    '''use an iterative 'windowed' method from sep to get new position'''
    new_pos = np.array(sep.winpos(data, x_initial, y_initial, sigma, subpix=5))[0:2, :]
    x = new_pos[:][0].tolist()
    y = new_pos[:][1].tolist()
    
    '''returns tuple x, y (python 3: zip(x, y) -> tuple(zip(x,y))) and time'''
    return x,y



def clipCutStars(x, y, x_length, y_length):
    """ When the aperture is near the edge of the field of view sets flux to zero to prevent 
    fadeout
    input: x coords of stars, y coords of stars, length of image in x-direction, 
    length of image in y-direction
    returns: indices of stars to remove"""

    edgeThresh = 20.          #number of pixels near edge of image to ignore
    
    '''make arrays of x, y coords'''
    xeff = np.array(x)
    yeff = np.array(y) 
    
    '''get list of indices where stars too near to edge'''
    ind = np.where(edgeThresh > xeff)
    ind = np.append(ind, np.where(xeff >= (x_length - edgeThresh)))
    ind = np.append(ind, np.where(edgeThresh > yeff))
    ind = np.append(ind, np.where(yeff >= (y_length - edgeThresh)))
    
    return ind



def getSizeFITS(filenames):
    """ gets dimensions of fits 'video' 
    input: list of filenames in directory
    returns: width, height of fits image, number of images in directory, 
    list of header times for each image"""
    
    '''get names of first and last image in directory'''
    filename_first = filenames[0]
    frames = len(filenames)     #number of images in directory

    '''get width/height of images from first image'''
    file = fits.open(filename_first)
    header = file[0].header
    width = header['NAXIS1']
    height = header['NAXIS2']

    return width, height, frames


def importFramesFITS(parentdir, filenames, start_frame, num_frames):
    """ reads in frames from fits files starting at frame_num
    input: parent directory (minute), list of filenames to read in, starting frame number, how many frames to read in, 
    dark image (2D array of fluxes)
    returns: array of image data arrays, array of header times of these images"""

    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]

    '''get data from each file in list of files to read, subtract dark frame'''
    for filename in files_to_read:
        file = fits.open(filename)
        
        header = file[0].header
        
        ''' Calibration frame correction '''
        data = (file[0].data) #- dark #- dark)/flat 
        headerTime = header['DATE-OBS']
            
        file.close()

        imagesData.append(data)
        imagesTimes.append(headerTime)
         
    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData, imagesTimes

 

base_path = pathlib.Path('/', 'E:')
telescope='Green'
obs_date='2023-04-10'
field_name='field6'
data_path=base_path.joinpath('/StackedData',field_name, obs_date, telescope) #path for mean-stacked frames
savefolder=base_path.joinpath('/StackedData',field_name, obs_date, telescope, 'lightcurves') #folder to save results in txts
if not os.path.exists(savefolder):
    os.makedirs(savefolder)


''' get list of image names to process'''
filenames = sorted(data_path.glob('*.fits'))#list of mean-stacked frames


detect_thresh=4 #star finding threshold
global inner_annulus
inner_annulus = 5 #photometry bkg inner anulus
global outer_annulus 
outer_annulus = 8 #photometry bkg outer anulus
global Edge_buffer
Edge_buffer=10 #to remove stars that are close to edge, I think this should be more than 20px

ap_r=3 #aperture radius for photometry

''' get 2d shape of images, number of image in directory'''
x_length, y_length, num_images = getSizeFITS(filenames)

print (datetime.datetime.now(), "Imported", num_images, "frames")


''' load/create star positional data'''

first_frame = importFramesFITS(data_path, filenames, 0, 1)      #data and time from 1st image
headerTimes = [first_frame[1]] #list of image header times

star_find_results = tuple(initialFindFITS(first_frame[0], detect_thresh)) #find stars on first frame

if len(star_find_results)<20: #quit when no stars
    print("Not good image! ", filenames[0])
    sys.exit()

    
#remove stars where centre is too close to edge of frame
before_trim = len(star_find_results)
star_find_results = tuple(x for x in star_find_results if x[0] + Edge_buffer < x_length and x[0] - Edge_buffer > 0)
star_find_results = tuple(y for y in star_find_results if y[1] + Edge_buffer < x_length and y[1] - Edge_buffer > 0)
after_trim = len(star_find_results)

print('Number of stars cut because too close to edge: ', before_trim - after_trim)
    
star_find_results = np.array(star_find_results)
radii = star_find_results[:,-2] #get radius of stars in px
num_pixels= star_find_results[:,-1] #get number of pixels each star occupies
prev_star_pos = star_find_results[:,:-1] #initial star position in x and y

initial_positions = prev_star_pos   
	
#remove stars that have drifted out of frame (i dont think this work)
initial_positions = initial_positions[(x_length >= initial_positions[:, 0])]
initial_positions = initial_positions[(y_length >= initial_positions[:, 1])]


num_stars = len(initial_positions)      #number of stars in image
print(datetime.datetime.now(), 'number of stars found: ', num_stars) 



''' flux and time calculations with optional time evolution '''
hdu = fits.open(filenames[0]) #open fits file
transform = wcs.WCS(hdu[0]) #open wcs from headers
hdu.close()
ra_dec=getRAdec(transform, initial_positions)  #get ra dec of each star 

data = np.empty([num_images, num_stars], dtype=(np.float64, 4)) #array to store data

#get first image data from initial star positions
bkg = sep.Background(first_frame[0]) #create background profile for error estimation
bkg_rms = bkg.rms()

data[0] = tuple(zip(initial_positions[:,0], 
                    initial_positions[:,1], 
                    #sum_flux(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r),
                    (sep.sum_circle(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r, bkgann = (inner_annulus, outer_annulus))[0]).tolist(), 
                    (sep.sum_circle(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r, err=bkg_rms, bkgann = (inner_annulus, outer_annulus))[1]).tolist()
                    # np.ones(np.shape(np.array(initial_positions))[0]) * (Time(first_frame[1], precision=9, format = 'fits').unix)))
                    ))
GaussSigma = np.mean(radii * 2. / 2.35) #only for refining centroid which works awfully

for t in trange(1, num_images):#read through all images

    imageFile = importFramesFITS(data_path, filenames, t, 1)
    headerTimes.append(imageFile[1])  #add header time to list
    
    # frame_time = Time(t, precision=9, format = 'fits').unix   #current frame time from file header (unix)
    hdu = fits.open(filenames[t])
    try:
        x,y=getXY(hdu[0], ra_dec)#open fits file, get its wcs and get x y having initial ra dec list
    except ValueError:
        print('no wcs')
        continue
    hdu.close()
    '''add up all flux within aperture'''
    bkg = sep.Background(imageFile[0]) #create background profile
    bkg_rms = bkg.rms()
    
    fluxes, flux_err, flags = sep.sum_circle(imageFile[0], x, y, ap_r, err=bkg_rms, bkgann = (inner_annulus, outer_annulus))
    
    # newx, newy = refineCentroid(*imageFile, [[x,y]], GaussSigma)
    
    # fluxes, flux_err, flags = sep.sum_circle(imageFile[0], newx[0], newy[0], ap_r, err=bkg_rms, bkgann = (inner_annulus, outer_annulus))
    

    data[t] = tuple(zip(x, y, fluxes, flux_err))
    # data[t] = tuple(zip(newx[0], newy[0], fluxes, flux_err))



# SNRs=[]
# MEDs=[]


#%% this part is only for statistics
# for star in range(0, num_stars):

#     star_fluxes.append(data[:,star,2])
#     weights.append((stats.sigma_clipped_stats(data[:,star,2])[1]/stats.sigma_clipped_stats(data[:,star,2])[2])**2)
# #     # SNRs.append(stats.sigma_clipped_stats(data[:,star,2])[1]/stats.sigma_clipped_stats(data[:,star,2])[2])
# #     # MEDs.append(stats.sigma_clipped_stats(data[:,star,2])[1])
    
# mean_lc=np.average(star_fluxes,weights=weights, axis=0)
#%%
rel_lcs=[]  
for measured_star in range(0, num_stars): #loop through each star
    star_fluxes=[] #for mean lightcurve
    weights=[] #for weighted average
    
    #%% DIFFERENT METHODS TO MAKE MEAN LIGHTCURVES FOR RELATIVE PHOTOMETRY
    
    # mean_lc=[]
    # for frame in range(0, num_images):
    #     stars_flux=[x for i,x in enumerate(data[frame,:,2]) if i!=star]
    #     w=[1/x for i,x in enumerate(data[frame,:,2]) if i!=star]
    #     mean_lc.append(np.average(stars_flux,weights=w))
    
    # star_fluxes=[]
    # weights=[]

    # for star in range(0, num_stars):

    #     star_fluxes.append(data[:,star,2])
    #     weights.append((stats.sigma_clipped_stats(data[:,star,2])[1]/stats.sigma_clipped_stats(data[:,star,2])[2])**2)
        
    # mean_lc=np.average(star_fluxes,weights=weights, axis=0)
    
    # measured_star_pix=num_pixels[measured_star]
    # stars_pix_diff=[(i,abs(x-measured_star_pix)) for i,x in enumerate(num_pixels) if i!=measured_star]
    # stars_pix_diff.sort(key = lambda x: x[1])
    # for star in stars_pix_diff[0:10]:
    #     star_fluxes.append(data[:,star[0],2])
    #     weights.append((stats.sigma_clipped_stats(data[:,star[0],2])[1]/stats.sigma_clipped_stats(data[:,star[0],2])[2])**2)
    
    # mean_lc=np.average(star_fluxes,weights=weights, axis=0)   
    # mean_lc=np.average(star_fluxes, axis=0)  
    
    # star_fluxes=[]
    # weights=[]
    # star_xy=[initial_positions[measured_star][0],initial_positions[measured_star][1]]
    # close_stars=[i for i,x in enumerate((initial_positions)) if (star_xy[0]-200<x[0]<star_xy[0]+200 and star_xy[1]-200<x[1]<star_xy[1]+200)]
    # for star in close_stars[:10]:
    #     star_fluxes.append(data[:,star,2])
    #     weights.append((stats.sigma_clipped_stats(data[:,star,2])[1]/stats.sigma_clipped_stats(data[:,star,2])[2])**2)
    # mean_lc=np.average(star_fluxes,weights=weights, axis=0)

#%%

    chi_stars=[]#list with stars and their chi^2 value
    star_flux=[x/np.average(data[:,measured_star,2]) for x in data[:,measured_star,2]] #normalized lightcurve of target star
    # star_flux=[x for x in data[:,measured_star,2]]
    for star in range(0, num_stars): #loop through all stars
        if star!=measured_star: #but not our
            ref_flux=[x/np.average(data[:,star,2]) for x in data[:,star,2]] #get their relative flux
            # ref_flux=[x for x in data[:,star,2]]
            # residual=data[:,star,2]-star_flux
            residual=np.subtract(ref_flux,star_flux) #substract target and reference lightcurve
            residual_sq=[x**2 for x in residual] #get square of residuals
            chi2=np.sum(np.array(residual_sq)/np.array(star_flux)) #chi^2
            chi_stars.append([star, chi2])
    chi_stars.sort(key = lambda x: x[1]) #sort in descending order
    for star in chi_stars[0:10]: #choose 10 stars that have least difference with target lightcurve
        star_fluxes.append(data[:,star[0],2]) #append those lightcurves
        try: 
            weights.append((stats.sigma_clipped_stats(data[:,star[0],2])[1]/stats.sigma_clipped_stats(data[:,star[0],2])[2])**2)
        except ZeroDivisionError:
            weights.append(1)
            
    try: #get weighted average lightcurve
        mean_lc=np.average(star_fluxes,weights=weights, axis=0)
    except ZeroDivisionError:
        mean_lc=np.average(star_fluxes, axis=0)
    rel_lc=data[:,measured_star,2]/mean_lc #get relative lightcurve
    rel_lcs.append(rel_lc)
    

results = []
for star in range(0, num_stars): #append results [flux, star, relative flux, flux error, x, y]
    results.append((data[:, star, 2], star, rel_lcs[star], data[:, star, 3] )+(data[:, star, 0], data[:, star, 1]))
    
results = np.array(results, dtype = object)


''' data archival '''

#make directory to save lightcurves in
lightcurve_savepath = savefolder
if not lightcurve_savepath.exists():
    lightcurve_savepath.mkdir()      #make folder to hold master dark images in

headerJD=[t[0] for t in headerTimes]
headerJD=Time(headerJD,format='isot', scale='utc')
headerJD.format='jd'

for row in results:  # loop through each detected event
    star_coords = initial_positions[row[1]]     #coords of occulted star
    
    savefile = lightcurve_savepath.joinpath('star' + str(row[1]) + "_" + str(obs_date) + '_' + telescope + "_"+  str(int(star_coords[0])) + "-" + str(int(star_coords[1])) + ".txt")
    
    #open file to save results
    with open(savefile, 'w') as filehandle:
        
        #file header
        filehandle.write('#\n#\n#\n#\n')
        filehandle.write('#    First Image File: %s\n' %(filenames[0]))
        filehandle.write('#    Star Coords: %f %f\n' %(star_coords[0], star_coords[1]))
        filehandle.write('#    Ra Dec: %f %f\n' %(ra_dec[row[1]][0], ra_dec[row[1]][1]))
        filehandle.write('#    DATE-OBS (JD): %s\n' %(headerTimes[0]))
        filehandle.write('#    Telescope: %s\n' %(telescope))
        filehandle.write('#    Field: %s\n' %(field_name))
        # filehandle.write('#    Error: %s\n' %(error))
        filehandle.write('#\n#\n#\n')
        filehandle.write('#filename     time      flux    flux_err    rel_flux    x    y\n')
      
        
        data_err=row[3]
        relative_lightcurve=row[2]
        x=row[4]
        y=row[5]
        
        # SNRs.append(stats.sigma_clipped_stats(data)[1]/stats.sigma_clipped_stats(row[3])[0])
        # MEDs.append(stats.sigma_clipped_stats(data)[1])
    
        files_to_save = filenames
        star_save_flux = row[0]            #part of light curve to save
        
        
  
        #loop through each frame to be saved
        for i in range(0, len(files_to_save)):  
            filehandle.write('%s %f  %.6f  %f  %f  %f  %f\n' % (files_to_save[i], float(headerJD[i].value), float(star_save_flux[i]), float(data_err[i]), float(relative_lightcurve[i]), float(x[i]), float(y[i])))


print ("\n")
