
"""
Filename:   colibri_main_py3.py
Author(s):  Emily Pass, Rachel Brown, Roman Akhmetshyn, Peter Quigley
Contact:    pquigley@uwo.ca
Created:    2018
Updated:    Mon May  1 14:35:58 2023
    
Description:
Initial Colibri data processing pipeline for flagging candidate KBO occultation
events.
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
import colibri_tools as ct

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

# Timestamp format
OBSDATE_FORMAT = '%Y%m%d'
MINDIR_FORMAT  = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
BARE_FORMAT = '%Y-%m-%d_%H%M%S_%f'

# Star detection parameters
AP_R = 3.  # aperture radius
DETECT_THRESH = 3.3  # sigma threshold for detection
NUM_TO_SKIP  = 1  # number of images to skip for star detection
NUM_TO_STACK = 9  # number of images to stack for star detection
BIASES_TO_STACK = 9       #number of bias images to combine in median bias images
EDGE_BUFFER  = 10  # px from edge of image to ignore
NPY_STARS = 10  # number of stars to save in .npy file
MIN_STARS = 30  # minimum number of stars to analyze minute

# Drift parameters
DRIFT_MULT = 1.0  # multiplier for detection radius after drift
MAX_DRIFT  = 1.0  # px/s
DRIFT_TOL  = 2.5e-2  # px/s

# Analysis parameters
CHUNK_SIZE = 10  # number of images to analyze at once
EVENT_WIDTH = 1.0  # seconds to save on either side of event

# Verbose print statement
verboseprint = lambda *a, **k: None


#--------------------------------functions------------------------------------#

###########################
## Directory Selection
###########################

def getUnprocessedMinutes(obsdate, reprocess=False):
    """
    Return list of unprocessed minute directories for a given date.
    References a list of .npy starlists in the archive directory to determine
    which minutes have already been processed.
    
        Parameters:
            obsdate (str): Date of observation (as marked on the data 
                           directory to be analyzed).

        Returns:
            minute_dirs (list): List of unprocessed minute directories.
            master_bias_list (list): List of master biases and times.

    """

    ## Collect raw data lists

    # Get list of all minute directories for the given date
    minute_dirs = [min_dir for min_dir in DATA_PATH.joinpath(obsdate).iterdir() if min_dir.is_dir()]
    minute_dirs = [min_dir for min_dir in minute_dirs if 'Bias' not in min_dir.name]
    minute_dirs.sort()
    bias_dir = DATA_PATH.joinpath(obsdate, 'Bias')
    
    # If reprocessing, remake the master bias list and return all minute dirs
    if reprocess:
        master_bias_list = cir.makeBiasSet(bias_dir, BASE_PATH, obsdate, BIASES_TO_STACK)
        return minute_dirs, master_bias_list

    ## Generate/collect master bias list

    # Check that the archive directory exists
    obsdate_archive = ARCHIVE_PATH.joinpath(ct.hyphonateDate(obsdate))
    masterbias_dir = obsdate_archive.joinpath('masterBiases')
    if not obsdate_archive.exists():
        obsdate_archive.mkdir()

        # Make master bias set
        master_bias_list = cir.makeBiasSet(bias_dir, BASE_PATH, obsdate, BIASES_TO_STACK)
    
    # Check that there are the same number of items in masterBias directory as in Bias directory
    elif len(list(bias_dir.iterdir())) != len(list(masterbias_dir.iterdir())):
        # Get list of master biases and times
        master_bias_list = cir.getMasterBiasList(obsdate, BASE_PATH)

    # Else, use the masterbias list from the archive
    else:
        # Get list of master biases and times
        master_bias_list = list(masterbias_dir.iterdir())

    
    ## Remove processed minutes from list of all minutes

    # Get list of all minute directories that have already been processed
    # NPY filenames of of the format "YYYYMMDD_HH.MM.SS.fff_3.3sig_pos.npy"
    processed_minutes = list(ARCHIVE_PATH.joinpath(ct.hyphonateDate(obsdate)).glob('*_pos.npy'))
    processed_minutes = [x.name[:21] for x in processed_minutes]

    # Remove processed minutes from list of all minutes
    minute_dirs = [x for x in minute_dirs if x.name not in processed_minutes]

    # Return the list of unprocessed minutes
    return minute_dirs, master_bias_list


###########################
## Main Functions
###########################

def firstOccSearch(minuteDir, MasterBiasList, kernel, exposure_time, sigma_threshold,
                   base_path,obs_date,
                   telescope='TEST',RCDfiles = True, gain_high = True):
    """ 
    Formerly 'main'.
    Detect possible occultation events in selected file and archive results 
    
        Parameters:
            minuteDir (str): Filepath to current directory
            MasterBiasList (list): List of master biases and times
            kernel (arr): Ricker wavelet kernel
            exposure_time (int/float): Camera exposure time (in s)
            sigma_threshold (float): Sensitivity of the detection filter
            base_path (str): Path to root directory containing Archive and Data
            obs_date (str): Date of observation (as marked on the data 
                            directory to be analyzed)
        
        Returns:
            __ (str): Printout of processing tasks
            __ (.npy): .npy format file with star positions (if doesn't exist)
            __ (.txt): .txt format file for each occultation event with names
                       of images to be saved
            __ (int/float): time of saved occultation event images
            __ (float): flux of occulted star in saved images
            star_count (int): Number of detected stars in the given minuteDir
    """


###########################
## Analysis Setup
###########################

    print (f"{datetime.datetime.now()} Opening: {minuteDir}")

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
        return minuteDir.name, 0, 0
    

###########################
## Star Identification
###########################
    
    #print (datetime.datetime.now(), field_name, 'starfinding...',)
    print(f"Starfinding in {field_name}...")
    
    ## Create median combined image for star finding
    try:
        stacked = cir.stackImages(minuteDir, savefolder, NUM_TO_SKIP, NUM_TO_STACK, bias)
    ## Prevents program from crashing if there are corrupt images
    except UnicodeDecodeError:
        print(f"ERROR: UnicodeDecodeError in {minuteDir.name}")
        return minuteDir.name, 0, 0

    ## Make list of star coords and half light radii using a conservative
    ## threshold scaled to the number of images stacked
    star_find_results = tuple(cp.initialFind(stacked, DETECT_THRESH*NUM_TO_STACK**0.5))

    ## Remove stars where centre is too close to edge of frame
    edge_buffer = 10     #number of pixels between edge of star aperture and edge of image
    #star_find_results = tuple(x for x in star_find_results if x[0] + AP_R + edge_buffer < x_length and x[0] - AP_R - edge_buffer > 0)
    #star_find_results = tuple(y for y in star_find_results if y[1] + AP_R + edge_buffer < x_length and y[1] - AP_R - edge_buffer > 0)
    star_find_results = tuple(x for x in star_find_results if x[0] + edge_buffer < x_length and x[0] - edge_buffer > 0)
    star_find_results = tuple(y for y in star_find_results if y[1] + edge_buffer < x_length and y[1] - edge_buffer > 0)


    ## Save the array of star positions as an .npy file
    ## Format: x  |  y  | half light radius
    if len(star_find_results) >= NPY_STARS:
        star_pos_file = base_path.joinpath('ColibriArchive', str(obs_date), minuteDir.name + '_' + str(DETECT_THRESH) + 'sig_pos.npy')
        if star_pos_file.exists():
            star_pos_file.unlink()
        np.save(star_pos_file, star_find_results)


    ## Enforce a minimum number of visible stars in each image
    if len(star_find_results) < MIN_STARS:
        print(f"Insufficient stars in minute: {minuteDir}")
        print (f"{datetime.datetime.now()} Closing: {minuteDir}")
        print ("\n")
        return minuteDir.name, len(star_find_results), 0
        
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
    print(f"Weighting Radius for Starfinding (GaussSigma) = {GaussSigma}")

    ## Import the first frame and last frame for starfinding purposes
    fframe_data,fframe_time = cir.importFramesRCD(imagePaths, NUM_TO_SKIP, 1, bias)   #import first image
    lframe_data,lframe_time = cir.importFramesRCD(imagePaths, len(imagePaths)-1, 1, bias)  #import last image
    headerTimes = [fframe_time] #list of image header times

    ## Refine star positions for first and last image (also returns frame times)
    first_drift = cp.refineCentroid(fframe_data,fframe_time[0], initial_positions, GaussSigma)
    last_drift = cp.refineCentroid(lframe_data,lframe_time[0], first_drift[0], GaussSigma*DRIFT_MULT)

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
    if abs(x_drift) > MAX_DRIFT or abs(y_drift) > MAX_DRIFT:
        print (f"{datetime.datetime.now()} Significant drift. Skipping {minuteDir}...") 
        return minuteDir.name, num_stars, 0
    elif abs(x_drift) > DRIFT_TOL or abs(y_drift) > DRIFT_TOL:
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
                        #sumFlux(first_frame[0], initial_positions[:,0], initial_positions[:,1], AP_R),
                        # (sep.sum_circle(fframe_data, initial_positions[:,0], initial_positions[:,1], AP_R)[0]).tolist(),
                        (sep.sum_circle(fframe_data, initial_positions[:,0], initial_positions[:,1], AP_R)[0]).tolist(),
                        np.ones(np.shape(np.array(initial_positions))[0]) * (Time(fframe_time, precision=9).unix)))


    ## Iteratively process the images and put the results into starData
    if drift:  # If drift compensation is required, enter this loop
    
        print(f"{minuteDir} Drifted - applying drift to photometry {x_drift} {y_drift}")
        
        # Loop through each image in the minute-long dataset in chunks
        residual   = (num_images-1)%CHUNK_SIZE
        for i in range((num_images-1)//CHUNK_SIZE):
            # Read in the images in a given chunk
            imageFile,imageTime = cir.importFramesRCD(imagePaths,CHUNK_SIZE*i+1, CHUNK_SIZE, bias)
            headerTimes = headerTimes + imageTime
            for j in range(CHUNK_SIZE):
                # Process the images in the current chunk
                starData[CHUNK_SIZE*i+j+1] = cp.timeEvolve(imageFile[j],
                                                            deepcopy(starData[CHUNK_SIZE*i+j]),
                                                            imageTime[j],
                                                            AP_R,
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
                                                    AP_R,
                                                    num_stars,
                                                    (x_length, y_length),
                                                    (x_drift, y_drift))        
        gc.collect()
        
            
    else:  # if there is not significant drift, don't account for drift  in photometry
        
        print(f'{minuteDir} No drift')
        
        # Loop through each image in the minute-long dataset in chunks
        residual   = (num_images-1)%CHUNK_SIZE
        for i in range((num_images-1)//CHUNK_SIZE):
            # Read in the images in a given chunk
            imageFile,imageTime = cir.importFramesRCD(imagePaths,CHUNK_SIZE*i+1, CHUNK_SIZE, bias)
            headerTimes = headerTimes + imageTime
            # Process the the current chunk all at once
            starData[CHUNK_SIZE*i+1:CHUNK_SIZE*(i+1)+1] = cp.getStationaryFlux(imageFile,
                                                                               deepcopy(starData[CHUNK_SIZE*i]),
                                                                               imageTime,
                                                                               AP_R,
                                                                               num_stars,
                                                                               (x_length, y_length))
                
            gc.collect()
            
        # Read in the remaining images
        imageFile,imageTime = cir.importFramesRCD(imagePaths,
                                                  num_images-(num_images-1)%CHUNK_SIZE,
                                                  (num_images-1)%CHUNK_SIZE,
                                                  bias)
        headerTimes = headerTimes + imageTime
        # Process the residual chunk
        starData[num_images-residual:] = cp.getStationaryFlux(imageFile,
                                                              deepcopy(starData[num_images-residual-1]),
                                                              imageTime,
                                                              AP_R,
                                                              num_stars,
                                                              (x_length, y_length))
                
        gc.collect()
    

    """
    # Roman's photometry plotting section
    import matplotlib.pyplot as plt
    for starNum in range(0,num_stars):
    flux=starData[:, starNum, 2]
    x_coords_in=starData[0][starNum][0]
    y_coords_in=starData[0][starNum][1]
    x_coords_fi=starData[-1][starNum][0]
    y_coords_fi=starData[-1][starNum][1]
    fig, ax1 = plt.subplots()
    ax1.scatter(range(0,len(starData)), flux,label="initial pos: x="+str(x_coords_in)+" y="+str(y_coords_in)+"\n final pos: x="+str(x_coords_fi)+" y="+str(y_coords_fi))
    plt.legend()
    plot_path=base_path.joinpath('ColibriArchive', str(obs_date), minuteDir.name)
    if not os.path.exists(plot_path):
        os.mkdir(plot_path)
    plt.savefig(plot_path.joinpath('_star_'+str(i) + '.png'))
    
    plt.close()
    for i in range(len(starData[0])):
        flux=[]
        frame=[]
        x_coords_in=starData[0][i][0]
        y_coords_in=starData[0][i][1]
        x_coords_fi=starData[-1][i][0]
        y_coords_fi=starData[-1][i][1]
        for j in range(len(starData)):
            flux.append(starData[j][i][2])
            frame.append(j)
            
        
        fig, ax1 = plt.subplots()
        ax1.scatter(frame, flux,label="initial pos: x="+str(x_coords_in)+" y="+str(y_coords_in)+"\n final pos: x="+str(x_coords_fi)+" y="+str(y_coords_fi))
        plt.legend()
        plot_path=base_path.joinpath('ColibriArchive', str(obs_date), minuteDir.name)
        if not os.path.exists(plot_path):
            os.mkdir(plot_path)
        plt.savefig(plot_path.joinpath('_star_'+str(i) + '.png'))
        
        plt.close()
    """



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

    save_frames = event_frames[np.where(event_frames > 0)]  #frame numbers for each event to be saved
    save_chunk = int(round(EVENT_WIDTH / exposure_time))  #save certain num of frames on both sides of event
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
    return minuteDir.name, num_stars, len(save_frames)
    

#------------------------------------main-------------------------------------#


if __name__ == '__main__':
    
    
###########################
## Argument Parser Setup
###########################

    ## Generate argument parser
    arg_parser = argparse.ArgumentParser(description="First-level data analysis for the Colibri telescopes",
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    ## Add argument functionality
    arg_parser.add_argument('path', help='Path to base directory')
    arg_parser.add_argument('date', help='Observation date (YYYY/MM/DD) of data to be processed')
    arg_parser.add_argument('-s', '--sigma', help='Sigma threshold', default=6.0,type=float)
    arg_parser.add_argument('-R','--RCD', help='Read RCD files directly (otherwise convert to .fits)', action='store_false')
    arg_parser.add_argument('-p', '--noparallel', help='Disable parallelism, run in sequence instead', action='store_false')
    arg_parser.add_argument('-g', '--lowgain', help='Analyze low-gain images', action='store_false')
    arg_parser.add_argument('-t', '--test', help='Test functionality for Peter Quigley', action='store_true')
    arg_parser.add_argument('-r', '--reprocess', help='Reprocess all data', action='store_true')
    
    ## Process argparse list as useful variables
    cml_args = arg_parser.parse_args()
    
    if cml_args.test:
        # Processing parameters
        RCDfiles,gain_high = True,True
        runPar = cml_args.noparallel
        reprocess = True
        sigma_threshold = cml_args.sigma
        
        # Target data
        base_path = pathlib.Path('/', 'home', 'pquigley', 'ColibriRepos')
        obs_date = datetime.date(2022, 8, 12)
        
    else:
        # Processing parameters
        RCDfiles = cml_args.RCD
        runPar = cml_args.noparallel
        gain_high = cml_args.lowgain
        sigma_threshold = cml_args.sigma
        reprocess = cml_args.reprocess

        # Target data
        base_path = pathlib.Path(cml_args.path) # path to data directory
        obsYYYYMMDD = cml_args.date # date to be analyzed formatted as "YYYY/MM/DD"
        obsdatesplit = obsYYYYMMDD.split('/')
        obs_date = datetime.date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
        

    ## Get name of telescope
    try:
        telescope = os.environ['COMPUTERNAME']
    except KeyError:
        telescope = "TEST"
        
    
###########################
## Pre-Analysis Preparation
###########################
    
    '''get filepaths'''
 
    # Get unprocessed minute directories and master bias list
    minute_dirs, MasterBiasList = getUnprocessedMinutes(str(obs_date).replace('-',''), reprocess=reprocess)

    ## Check if there is a valid GPS lock
    imagePaths = sorted(minute_dirs[0].glob('*.rcd'))
    print(imagePaths[0])
    if not cir.testGPSLock(imagePaths[0]):
        print(datetime.datetime.now(), "No GPS Lock established, skipping...")
        
        with open(os.path.join(night_dir,"error.txt"),"w") as f:
            f.write("")
        
        sys.exit()
    

    ''' prepare RickerWavelet/Mexican hat kernel to convolve with light curves'''
    exposure_time = 0.025    # exposure length in seconds
    expected_length = 0.15   # related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing

    kernel_frames = int(round(expected_length / exposure_time))   # width of kernel
    ricker_kernel = RickerWavelet1DKernel(kernel_frames)          # generate kernel


    ''''run pipeline for each folder of data'''
 

###########################
## Run in Parallel
###########################   
 
    #running in parallel (minute directories are split up between cores)
    if runPar == True:
        print('Running in parallel...')
        start_time = timer.time()
        finish_txt=base_path.joinpath('ColibriArchive', str(obs_date), 'primary_summary.txt')
        
        pool_size = multiprocessing.cpu_count() - 2
        pool = Pool(pool_size)
        args = ((minute_dirs[f], MasterBiasList, ricker_kernel, exposure_time, sigma_threshold,
                 base_path,obs_date,telescope,RCDfiles,gain_high) for f in range(len(minute_dirs)))
        
        try:
            star_counts = pool.starmap(firstOccSearch,args)
            end_time = timer.time()
            #starhours = sum(star_minutes)/(2399*60.0)
            #print(f"Calculated {starhours} star-hours\n", file=sys.stderr)
            print(f"Ran for {end_time - start_time} seconds", file=sys.stderr)
            
            with open(finish_txt, 'w') as f:
                f.write(f'# Ran for {end_time - start_time} seconds\n')

                #f.write(f'#Calculated {starhours} star-hours\n')
                f.write("# time, star count\n")
                f.write("#--------------------#\n")
                for results in star_counts:
                    f.write(f"{results[0]}, {results[1]}, {results[2]}\n")
        except:
            logging.exception("failed to parallelize")
            with open(finish_txt, 'w') as f:
                f.write('failed')
            
        pool.close()
        pool.join()


###########################
## Run in Sequence 
###########################  

    else:
        raise NotImplementedError("Not running in parallel needs maitenance.\nSorry for the inconvenience!\n-Peter Q (2022/11/05)")
        
        # Added a check to see if the fits conversion has been done. - MJM 
        #only run this check if we want to process fits files - RAB
        if RCDfiles == False:
                
            #check if conversion indicator file is present
            if not minute_dirs[f].joinpath('converted.txt').is_file():
                        
                print('Converting to .fits')
                
                #make conversion indicator file
                with open(minute_dirs[f].joinpath('converted.txt'), 'a'):
                    os.utime(str(minute_dirs[f].joinpath('converted.txt')))
                            
                #do .rcd -> .fits conversion using the desired gain level
                if gain_high:
                    os.system("python .\\RCDtoFTS.py " + str(minute_dirs[f]) + ' high')
                else:
                    os.system("python .\\RCDtoFTS.py " + str(minute_dirs[f]))
                    
        else:
            print('Already converted raw files to fits format.')
            print('Remove file converted.txt if you want to overwrite.')
        
        for f in range(len(minute_dirs)):
           


            print(f"Running on... {minute_dirs[f]}")

            start_time = timer.time()

            print('Running sequentially...')
            firstOccSearch(minute_dirs[f], MasterBiasList, ricker_kernel, exposure_time, sigma_threshold,
                           base_path,obs_date,telescope,RCDfiles,gain_high)
            
            gc.collect()

            end_time = timer.time()
            print(f"Ran for {end_time - start_time} seconds", file=sys.stderr)

#           with open("logs/timing.log","a") as f:
#               f.write(f"Ran for {end_time - start_time} seconds\n\n")


