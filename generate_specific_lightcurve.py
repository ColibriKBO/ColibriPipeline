"""
Filename:   generate_specific_lightcurve.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Thu May 25, 2022
Updated:    Thu May 25, 2022
    
Usage: python generate_specific_lightcurve.py <obs_date> <time_of_interest> [RA,DEC]
"""

# Module Imports
import os,sys
import argparse
import pathlib
import multiprocessing
import datetime
import gc
import sep
import numpy as np
import time as timer
from astropy.convolution import RickerWavelet1DKernel
from astropy.time import Time
from copy import deepcopy
from multiprocessing import Pool

# Custom Script Imports
import colibri_image_reader as cir
import colibri_photometry as cp

# Disable Warnings
import warnings
import logging
#warnings.filterwarnings("ignore",category=DeprecationWarning)
#warnings.filterwarnings("ignore",category=VisibleDeprecationWarning)

#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = pathlib.Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'

# Processing parameters
POOL_SIZE  = multiprocessing.cpu_count() - 2  # cores to use for multiprocessing
CHUNK_SIZE = 10  # images to process at once

# Photometry parameters
EXCLUDE_IMAGES = 1
STARFINDING_STACK = 9
APERTURE_RADIUS = 3.0
STAR_DETEC_THRESH = 4.0

# Drift limits
DRIFT_TOLERANCE = 1.0  # maximum drift (in px/s) without throwing an error
DRIFT_THRESHOLD = 0.025  # maximum drif (in px/s) without applying a drift correction


#--------------------------------functions------------------------------------#

def trackStar(minuteDir, MasterBiasList, kernel, exposure_time, sigma_threshold,
              base_path, obs_date, telescope='TEST', RCDfiles = True, gain_high = True)

    print (f"{datetime.datetime.now()} Opening: {minuteDir}")

    ## Define adjustable parameters
    ap_r = 3.  # radius of aperture for flux measuremnets
    detect_thresh = 4.  # threshold for star detection

    ## Create folder for results
    savefolder = base_path.joinpath('ColibriArchive', str(obs_date))
    if not savefolder.exists():
        savefolder.mkdir()      

    ## Select and load the master bias closest in time to the current minute
    bias = cir.chooseBias(minuteDir, MasterBiasList, obs_date)

    ## Get a sorted list of images in this minute directory, ignoring the first
    ## due to vignetting
    imagePaths = sorted(minuteDir.glob('*.rcd'))     
    del imagePaths[0]
        
    ## Idenitify important characteristics from name and length of images list
    field_name = imagePaths[0].name.split('_')[0]  #which of 11 fields are observed
    x_length, y_length, num_images = cir.getSizeRCD(imagePaths)

    print(datetime.datetime.now(), "Imported", num_images, "frames")
    
    ## Check if there are enough images in the current directory 
    minNumImages = len(kernel.array)*3         #3x kernel length
    if num_images < minNumImages:
        print(datetime.datetime.now(), "Insufficient number of images, skipping...")
        return minuteDir.name, 0
    

###########################
## Star Identification
###########################
    
    #print (datetime.datetime.now(), field_name, 'starfinding...',)
    print(f"Starfinding in {field_name}...")
    
    ## Create median combined image for star finding
    startIndex = 1          # which image to start stack at (vignetting in 0th image)
    numtoStack = 9          # number of images to include in stack
    scaledThres = detect_thresh*numtoStack**0.1 # detection threshold scaled to number of images stacked
    stacked = cir.stackImages(minuteDir, savefolder, startIndex, numtoStack, bias)

    ## Make list of star coords and half light radii using a conservative
    ## threshold scaled to the number of images stacked
    star_find_results = tuple(cp.initialFind(stacked, scaledThres))

    ## Remove stars where centre is too close to edge of frame
    edge_buffer = 10     #number of pixels between edge of star aperture and edge of image
    #star_find_results = tuple(x for x in star_find_results if x[0] + ap_r + edge_buffer < x_length and x[0] - ap_r - edge_buffer > 0)
    #star_find_results = tuple(y for y in star_find_results if y[1] + ap_r + edge_buffer < x_length and y[1] - ap_r - edge_buffer > 0)
    star_find_results = tuple(x for x in star_find_results if x[0] + edge_buffer < x_length and x[0] - edge_buffer > 0)
    star_find_results = tuple(y for y in star_find_results if y[1] + edge_buffer < x_length and y[1] - edge_buffer > 0)
            
    ## Enforce a minimum number of visible stars in each image
    min_stars = 30
    if len(star_find_results) < min_stars:
        print(f"Insufficient stars in minute: {minuteDir}")
        print (f"{datetime.datetime.now()} Closing: {minuteDir}")
        print ("\n")
        return minuteDir.name, len(star_find_results)
        
    
    ## Save the array of star positions as an .npy file
    ## Format: x  |  y  | half light radius
    star_pos_file = base_path.joinpath('ColibriArchive', str(obs_date), minuteDir.name + '_' + str(detect_thresh) + 'sig_pos.npy')
    if star_pos_file.exists():
        star_pos_file.unlink()
    np.save(star_pos_file, star_find_results)
        
    ## Seperate radii and positions in separate arrays
    star_find_results = np.array(star_find_results)
    radii = star_find_results[:,-1]
    initial_positions = star_find_results[:,:-1]

    ## Number of stars identified in image
    num_stars = len(initial_positions)
    
    
###########################
## Drift Calculations
########################### 

    ## Initialize centroid refining parameters
    drift_pos = np.empty([2, num_stars], dtype = (np.float64, 2))  #array to hold first and last positions
    GaussSigma = np.mean(radii * 2. / 2.35)  # calculate gaussian sigma for each star's light profile
    drift_multiple = 1  # relaxation factor on our centroid finding
    print(f"Weighting Radius for Starfinding (GaussSigma) = {GaussSigma}")

    ## Import the first frame and last frame for starfinding purposes
    fframe_data,fframe_time = cir.importFramesRCD(imagePaths, startIndex, 1, bias)   #import first image
    lframe_data,lframe_time = cir.importFramesRCD(imagePaths, len(imagePaths)-1, 1, bias)  #import last image
    headerTimes = [fframe_time] #list of image header times

    ## Refine star positions for first and last image (also returns frame times)
    first_drift = cp.refineCentroid(fframe_data,fframe_time[0], initial_positions, GaussSigma)
    last_drift = cp.refineCentroid(lframe_data,lframe_time[0], first_drift[0], GaussSigma*drift_multiple)

    ## Organize frame times and centroid positions into containers
    drift_pos[0] = first_drift[0]  # first frame positions
    drift_pos[1] = last_drift[0]  # last frame positions
    drift_times = [first_drift[1], last_drift[1]]  # frame times
    drift_times = Time(drift_times, precision=9).unix  # convert times to unix

    ## Calculate median drift rate [px/s] in x and y over the minute
    x_drift, y_drift = cp.averageDrift(drift_pos[0],drift_pos[1], drift_times[0],drift_times[1])
    print(f"Drift (x,y): {x_drift} px/s, {y_drift} px/s")
    
    ## Decide whether to apply drift modifications to frame analysis using a
    ## tolerence threshold. If the drift values are too large, disregard this
    ## minute as a tracking error. Rates in px/s.
    driftErrorThresh = 1        # threshold for drift that is manageable
    driftTolerance = 2.5e-2     # threshold for drift compensation
    if abs(x_drift) > driftErrorThresh or abs(y_drift) > driftErrorThresh:
        print (f"{datetime.datetime.now()} Significant drift. Skipping {minuteDir}...") 
        return minuteDir.name, num_stars
    elif abs(x_drift) > driftTolerance or abs(y_drift) > driftTolerance:
        drift = True # variable to check whether stars have drifted since last frame
    else:
        drift = False


###########################
## Flux Measurements
########################### 

    ## Define the container containing the flux measurements of all images
    ## with each frame holding a ( num_stars * 4 ) sized array.
    ## [0] x coorinate; [1] y coordinate; [2] flux; [3] frame time
    starData = np.empty((num_images, num_stars, 4), dtype=np.float64)
    
    ## Fill the first data frame with information from the first image
    starData[0] = tuple(zip(initial_positions[:,0], 
                        initial_positions[:,1], 
                        #sumFlux(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r),
                        # (sep.sum_circle(fframe_data, initial_positions[:,0], initial_positions[:,1], ap_r)[0]).tolist(),
                        (sep.sum_circle(fframe_data, initial_positions[:,0], initial_positions[:,1], ap_r)[0]).tolist(),
                        np.ones(np.shape(np.array(initial_positions))[0]) * (Time(fframe_time, precision=9).unix)))


    ## Iteratively process the images and put the results into starData
    if drift:  # If drift compensation is required, enter this loop
    
        print(f"{minuteDir} Drifted - applying drift to photometry {x_drift} {y_drift}")
        
        # Loop through each image in the minute-long dataset in chunks
        chunk_size = 10
        residual   = (num_images-1)%chunk_size
        for i in range((num_images-1)//chunk_size):
            # Read in the images in a given chunk
            imageFile,imageTime = cir.importFramesRCD(imagePaths,chunk_size*i+1, chunk_size, bias)
            headerTimes = headerTimes + imageTime
            for j in range(chunk_size):
                # Process the images in the current chunk
                starData[chunk_size*i+j+1] = cp.timeEvolve(imageFile[j],
                                                            deepcopy(starData[chunk_size*i+j]),
                                                            imageTime[j],
                                                            ap_r,
                                                            num_stars,
                                                            (x_length, y_length),
                                                            (x_drift, y_drift))
                gc.collect()
            
        # Read in the remaining images
        imageFile,imageTime = cir.importFramesRCD(imagePaths,
                                                  num_images-residual,
                                                  residual,
                                                  bias)
        headerTimes = headerTimes + imageTime
        
        # Process the remaining images
        for i in range(residual+1)[:0:-1]:
            starData[num_images-i] = cp.timeEvolve(imageFile[residual-i],
                                                    deepcopy(starData[num_images-i-1]),
                                                    imageTime[residual-i],
                                                    ap_r,
                                                    num_stars,
                                                    (x_length, y_length),
                                                    (x_drift, y_drift))        
        gc.collect()
        
            
    else:  # if there is not significant drift, don't account for drift  in photometry
        
        print(f'{minuteDir} No drift')
        
        # Loop through each image in the minute-long dataset in chunks
        chunk_size = 10
        residual   = (num_images-1)%chunk_size
        for i in range((num_images-1)//chunk_size):
            # Read in the images in a given chunk
            imageFile,imageTime = cir.importFramesRCD(imagePaths,chunk_size*i+1, chunk_size, bias)
            headerTimes = headerTimes + imageTime
            # Process the the current chunk all at once
            starData[chunk_size*i+1:chunk_size*(i+1)+1] = cp.getStationaryFlux(imageFile,
                                                                               deepcopy(starData[chunk_size*i]),
                                                                               imageTime,
                                                                               ap_r,
                                                                               num_stars,
                                                                               (x_length, y_length))
                
            gc.collect()
            
        # Read in the remaining images
        imageFile,imageTime = cir.importFramesRCD(imagePaths,
                                                  num_images-(num_images-1)%chunk_size,
                                                  (num_images-1)%chunk_size,
                                                  bias)
        headerTimes = headerTimes + imageTime
        # Process the residual chunk
        starData[num_images-residual:] = cp.getStationaryFlux(imageFile,
                                                              deepcopy(starData[num_images-residual-1]),
                                                              imageTime,
                                                              ap_r,
                                                              num_stars,
                                                              (x_length, y_length))
                
        gc.collect()


###########################
## Dip Detection
###########################
   
    #loop through each detected object
    dipResults = np.array([cp.dipDetection(starData[:, starNum, 2], kernel, starNum, sigma_threshold) for starNum in range(0,num_stars)],
                          dtype=object)
   
    
    event_frames = dipResults[:,0]         #array of event frames (-1 if no event detected, -2 if incomplete data)
    light_curves = dipResults[:,1]         #array of light curves (empty if no event detected)
    
    conv_flux=dipResults[:,2]              #array of convolved light curves (empty if no event detected)
    lightcurve_std=dipResults[:,3]          #std of minute-long lightcurve
    lightcurve_mean= dipResults[:,4]        #mean of minute-long lightcurve
    Bkg_std= dipResults[:,5]                #std of convolved flux background
    Bkg_mean= dipResults[:,6]               #mean of convolved flux background
    conv_min= dipResults[:,7]               #minimum value of the convolution
    significance=dipResults[:,8]          #significance of the event x*sigma

    ''' data archival '''
    
    secondsToSave =  1.0    #number of seconds on either side of event to save 
    save_frames = event_frames[np.where(event_frames > 0)]  #frame numbers for each event to be saved
    save_chunk = int(round(secondsToSave / exposure_time))  #save certain num of frames on both sides of event
    save_curves = light_curves[np.where(event_frames > 0)]  #light curves for each star to be saved
    #save_types = dip_types[np.where(event_frames > 0)]
    
    save_conv_flux=conv_flux[np.where(event_frames > 0)]    #save convolved flux for each event
    save_lc_std=lightcurve_std[np.where(event_frames > 0)]  #save lightcurve std for each event
    save_lc_mean=lightcurve_mean[np.where(event_frames > 0)]#save lightcurve mean
    save_bkg_std=Bkg_std[np.where(event_frames > 0)]        #save background std of convolved lightcurve
    save_bkg_mean=Bkg_mean[np.where(event_frames > 0)]      #save background mean of convolved lightcurve
    save_minVal=conv_min[np.where(event_frames > 0)]        #save minimum value of the convolution
    save_sigma=significance[np.where(event_frames > 0)]     #save significance of each event

    #loop through each detected event
    j=0
    for f in save_frames:
        
        date = headerTimes[f].split('T')[0]                                 #date of event
        time = headerTimes[f].split('T')[1].split('.')[0].replace(':','')   #time of event
        star_coords = initial_positions[np.where(event_frames == f)[0][0]]     #coords of occulted star
        mstime = headerTimes[f].split('T')[1].split('.')[1]                 # micros time of event
       # print(datetime.datetime.now(), ' saving event in frame', f)
        
        star_all_flux = save_curves[np.where(save_frames == f)][0]  #total light curve for current occulted star
        star_all_conv=save_conv_flux[np.where(save_frames == f)][0] #total convolved light curve for current occulted star
        #text file to save results in
        #saved file format: 'det_date_time_star#_telescope.txt'

        savefile = base_path.joinpath('ColibriArchive', str(obs_date), ''.join(('det_', date, '_', time, '_', mstime, '_star', str(np.where(event_frames == f)[0][0]), '_', telescope, '.txt')))
        #columns: fits filename and path | header time (seconds) |  star flux
        
        #open file to save results
        # with open(savefile, 'w') as filehandle:
            
        #     #file header
        #     filehandle.write('#\n#\n#\n#\n')
        #     filehandle.write('#    Event File: %s\n' %(imagePaths[f]))
        #     filehandle.write('#    Star Coords: %f %f\n' %(star_coords[0], star_coords[1]))
        #     filehandle.write('#    DATE-OBS: %s\n' %(headerTimes[f]))
        #     filehandle.write('#    Telescope: %s\n' %(telescope))
        #     filehandle.write('#    Field: %s\n' %(field_name))
        #     filehandle.write('#    Dip Type: %s\n' %(save_types[np.where(save_frames == f)][0]))
        #     filehandle.write('#    Median Flux: %.2f\n' %(np.median(star_all_flux)))
        #     filehandle.write('#    Stddev Flux: %.3f\n' %(np.std(star_all_flux)))
        #     filehandle.write('#\n#\n#\n')
        #     filehandle.write('#filename     time      flux\n')
        
        with open(savefile, 'w') as filehandle:
            
            #file header
            filehandle.write('#\n#\n#\n#\n')
            filehandle.write('#    Event File: %s\n' %(imagePaths[f]))
            filehandle.write('#    Star Coords: %f %f\n' %(star_coords[0], star_coords[1]))
            filehandle.write('#\n')
            filehandle.write('#    DATE-OBS: %s\n' %(headerTimes[f]))
            filehandle.write('#    Telescope: %s\n' %(telescope))
            filehandle.write('#    Field: %s\n' %(field_name))
            filehandle.write('#    significance: %.3f\n' %(save_sigma[j]))
            filehandle.write('#    Raw lightcurve std: %.4f\n' %(save_lc_std[j]))
            filehandle.write('#    Raw lightcurve mean: %.4f\n' %(save_lc_mean[j]))
            filehandle.write('#    Convolution background std: %.4f\n' %(save_bkg_std[j]))
            filehandle.write('#    Convolution background mean: %.4f\n' %(save_bkg_mean[j]))
            filehandle.write('#    Convolution minimal value: %.4f\n' %(save_minVal[j]))
            filehandle.write('#\n#\n')
            filehandle.write('#filename     time      flux     conv_flux\n')
           
            ''' save data '''
            j=j+1
           #if the portion of the light curve is at the beginning, start at 0th image
            if f - save_chunk <= 0:  
        
                files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= 0 and i < f + save_chunk]  #list of filenames to save
                star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                         #part of light curve to save
                star_save_conv= star_all_conv[np.where(np.in1d(imagePaths, files_to_save))[0]]
                
                #loop through each frame to be saved
                for i in range(0, len(files_to_save)):  
                    filehandle.write('%s %f  %f  %f\n' % (files_to_save[i], float(headerTimes[:f + save_chunk][i].split(':')[2].split('Z')[0]), star_save_flux[i], star_save_conv[i]))
            
            #if portion of light curve to save is not at the beginning
            else:
                
                #if the portion of the light curve to save is at the end of the minute, end at the last image
                if f + save_chunk >= num_images:
        
                    files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= f - save_chunk]    #list of filenames to save (RAB 042222)
                    star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                                                     #part of light curve to save
                    star_save_conv= star_all_conv[np.where(np.in1d(imagePaths, files_to_save))[0]]
                    
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f  %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:][i].split(':')[2].split('Z')[0]), star_save_flux[i], star_save_conv[i]))

                #if the portion of the light curve to save is not at beginning or end of the minute, save the whole portion around the event
                else:  

                    files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= f - save_chunk and i < f + save_chunk]     #list of filenames to save
                    star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                                         #part of light curve to save                    
                    star_save_conv= star_all_conv[np.where(np.in1d(imagePaths, files_to_save))[0]]
                    
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f  %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:f + save_chunk][i].split(':')[2].split('Z')[0]), star_save_flux[i], star_save_conv[i]))



    ''' printout statements'''
    print (datetime.datetime.now(), "Rejected Stars: ", round(((num_stars - len(save_frames)) / num_stars)*100, 2), "%")
    print (datetime.datetime.now(), "Total stars in field:", num_stars)
    print (datetime.datetime.now(), "Candidate events in this minute:", len(save_frames))
    print (datetime.datetime.now(), "Closing:", minuteDir)
    print ("\n")

    #print("Drift calcs",c3)
    
    # return number of stars in field
    gc.collect()
    return minuteDir.name, num_stars
