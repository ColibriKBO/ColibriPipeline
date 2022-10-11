"""
Created 2018 by Emily Pass

Update: March 1, 2022 - Rachel Brown
Update: Sept. 21, 2022 - Peter Quigley

Update: September 19, 2022 - Roman Akhmetshyn - changed output txt file contents and added a significance calculation


-initial Colibri data processing pipeline for flagging candidate
KBO occultation events
"""

# Module Imports
import sep
import numpy as np
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time
from copy import deepcopy
from multiprocessing import Pool
import pathlib
import multiprocessing
import datetime
import os
import gc
import time as timer
import sys

# Custom Script Imports
import colibri_image_reader as cir


##############################
## Function Definitions
##############################

def averageDrift(positions, times):
    """
    Determines the median x/y drift rates of all stars in a minute (first to
    last image)
    
        Parameters:
            positions (arr): 3D array of star positions [# frames, #stars, #position columns (X & Y)]
            times (arr): Header times of each position
            
        Returns: 
            x_drift_rate (arr): Median x drift rate [px/star]
            y_drift_rate (arr): Median y drift rate [px/star]
    """
    
    '''time difference between frames [s]'''
    times = Time(times, precision=9).unix               #convert position times to unix (float)
    time_interval = np.subtract(times[1], times[0])     #time between first and last frame (s)
    
    '''x and y drift of each star between frames [pixels]'''
    x_drifts = np.subtract(positions[1,:,0], positions[0,:,0])
    y_drifts = np.subtract(positions[1,:,1], positions[0,:,1])
        
    '''get median drift rate across all stars [px/s] '''
    x_drift_rate = np.median(x_drifts/time_interval)   
    y_drift_rate = np.median(y_drifts/time_interval)
     
    return x_drift_rate, y_drift_rate


def clipCutStars(x, y, x_length, y_length):
    """
    When the aperture is near the edge of the field of view sets flux to zero
    to prevent fadeout
    
        Parameters:
            x (arr): x-coordinates of stars
            y (arr): y-coordinates of stars
            x_length (int/float): Length of image in the x-direction
            y_length (int/float): Length of image in the y-direction
            
        Returns:
            ind (arr): Indices of stars deemed to near to the image edge
    """

    edgeThresh = 20.          #number of pixels near edge of image to ignore
    
    
    '''get list of indices where stars too near to edge'''
    
    ind =  np.append(np.where((x < edgeThresh) | \
                              (x > x_length - edgeThresh))[0], \
                     np.where((y < edgeThresh) | \
                              (y > y_length - edgeThresh))[0])
    
    return ind


def dipDetection(fluxProfile, kernel, num, sigma_threshold):
    """
    Checks for geometric dip, and detects dimming using Ricker Wavelet kernel
    
        Parameters:
            fluxProfile (arr): Light curve of star (array of fluxes in each image)
            kernel (arr): Ricker wavelet kernel
            num (int): Current star number
            sigma_threshold (float): sigma_threshold for determining stars
            
        Returns:
            frameNum (int): Frame number of detected event (-1 for no detection
                            or -2 if data unusable)
            lc_arr (arr): Light curve as an array (empty list if no event
                          detected)
            event_type (str): Keyword indicating event type (empty string if
                              no event detected)
    """

    '''' Prunes profiles'''
    light_curve = np.trim_zeros(fluxProfile)
    
    if len(light_curve) == 0:
        print('empty profile: ', num)
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  # reject empty profiles
   
    
    '''perform checks on data before proceeding'''
    
    FramesperMin = 2400      #ideal number of frames in a directory (1 minute)
    minSNR = 5               #median/stddev limit to look for detections
    minLightcurveLen = FramesperMin/4    #minimum length of lightcurve
    
    # reject stars that go out of frame to rapidly
    if len(light_curve) < minLightcurveLen:
        print('Light curve too short: star', num)
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  
    
    #TODO: what should the limits actually be?
    # reject tracking failures
    if abs(np.mean(light_curve[:FramesperMin]) - np.mean(light_curve[-FramesperMin:])) > np.std(light_curve[:FramesperMin]):
        print('Tracking failure: star ', num)
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan 
    
    # reject stars with SNR too low
    if np.median(light_curve)/np.std(light_curve) < minSNR:
        print('Signal to Noise too low: star', num)
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  

    #uncomment to save light curve of each star (doesn't look for dips)
    #return num, light_curve


    '''convolve light curve with ricker wavelet kernel'''
    #will throw error if try to normalize (sum of kernel too close to 0)
    conv = convolve_fft(light_curve, kernel, normalize_kernel=False)    #convolution of light curve with Ricker wavelet
    minLoc = np.argmin(conv)    #index of minimum value of convolution
    minVal = np.min(conv)          #minimum of convolution
    #TODO: The problems with Rocker wavelet smoothing, as currently implemented, are that:
    #1.The wavelet-smoothed light curve has correlated data points, so the original statistics are lost. 
    #In reality, the scatter has been diminished by about the square root of the number  of data points in the width of the wavelet 
    #(which is 6 or 7 data points, if I recall correctly)
    #2.The mean level has been scaled and/or shifted in a way that we (I, at least) don’t currently understand.
    
    #Sep 2022 we don't look for geometric dips any more - Roman A.
    
    # '''geometric dip detection (greater than 40%)'''
    # geoDip = 0.6    #threshold for geometric dip
    # norm_trunc_profile = light_curve/np.median(light_curve)  #normalize light curve
 
    # #if normalized profile at min value of convolution is less than the geometric dip threshold
    # if norm_trunc_profile[minLoc] < geoDip:
        
    #     #get frame number of dip
    #     critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
    #     print (datetime.datetime.now(), "Detected >40% dip: frame", str(critFrame) + ", star", num)
        
    #     return critFrame[0], light_curve, 'geometric'


    '''look for diffraction dip'''
    KernelLength = len(kernel.array)    #number of elements in kernel array
    
    #check if dip is at least one kernel length from edge
    if KernelLength <= minLoc < len(light_curve) - KernelLength:
        
        edgeBuffer = 10     #how many elements from beginning/end of the convolution to exclude
        bkgZone = conv[edgeBuffer: -edgeBuffer]        #background
    
        #dipdetection = 3.75  #dip detection threshold ; Sep 2022 now it's an input parameter - Roman A.
        
    else:
        print('event cutoff star: ', num)
        return -2, [], [], np.nan, np.nan, np.nan, np.nan, -2, np.nan  # reject events that are cut off at the start/end of time series

    #if minimum < background - 3.75*sigma
    # if minVal < np.mean(bkgZone) - dipdetection * np.std(bkgZone):  

    #     #get frame number of dip
    #     critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
    #     print('found significant dip in star: ', num, ' at frame: ', critFrame[0])
        
    #     return critFrame[0], light_curve, 'diffraction'
        
    # else:
    #     return -1, [], ''  # reject events that do not pass dip detection
    
    lightcurve_std=np.std(light_curve)
    
    # event_std=np.std(conv)
    conv_bkg_mean=np.mean(bkgZone)
    #event_mean=np.mean(conv)
    
    significance=(conv_bkg_mean-minVal)/np.std(bkgZone) #significance of the event x*sigma
    
    
    if significance>=sigma_threshold:
        #get frame number of dip
        critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
        print('found significant dip in star: ', num, ' at frame: ', critFrame[0])
        
        return critFrame[0], light_curve, conv, lightcurve_std, np.mean(light_curve), np.std(bkgZone),conv_bkg_mean,minVal,significance
        
    else:
        return -1, light_curve, conv, np.nan, np.nan, np.nan, np.nan, np.nan, significance  # reject events that do not pass dip detection


def getSizeFITS(imagePaths):
    """
    Get number of images in a data directory and image dimensions (.fits only)
    
        Parameters:
            imagePaths (list): List of all image filepaths in data directory
        
        Returns:
            width (int): Width of .fits images
            height (int): Height of .fits images
            frames (int): Number of images in a data directory
    """
    
    '''get number of images in directory'''
    
    frames = len(imagePaths)         #number of images in directory


    '''get width/height of images from first image'''
    
    first_imagePath = imagePaths[0]             #first image in directory
    first_image = fits.open(first_imagePath)    #load first image
    header = first_image[0].header              #image header
    width = header['NAXIS1']                    #X axis dimension
    height = header['NAXIS1']                   #Y axis dimension

    return width, height, frames


def getSizeRCD(imagePaths):
    """
    Get number of images in a data directory and image dimensions (.rcd only).
    Optionally get this from the RCD header.
    
        Parameters:
            imagePaths (list): List of all image filepaths in data directory
        
        Returns:
            width (int): Width of .fits images
            height (int): Height of .fits images
            frames (int): Number of images in a data directory
    """

    frames = len(imagePaths)

    width = 2048
    height = 2048

    # You could also get this from the RCD header by uncommenting the following code
    # with open(filename_first, 'rb') as fid:
    #     fid.seek(81,0)
    #     hpixels = readxbytes(fid, 2) # Number of horizontal pixels
    #     fid.seek(83,0)
    #     vpixels = readxbytes(fid, 2) # Number of vertical pixels

    #     fid.seek(100,0)
    #     binning = readxbytes(fid, 1)

    #     bins = int(binascii.hexlify(binning),16)
    #     hpix = int(binascii.hexlify(hpixels),16)
    #     vpix = int(binascii.hexlify(vpixels),16)
    #     width = int(hpix / bins)
    #     height = int(vpix / bins)

    return width, height, frames


def importFramesFITS(imagePaths, startFrameNum, numFrames, bias):
    """
    Reads in frames from .fits files starting at a specific frame
    
        Parameters:
            imagePaths (list): List of image paths to read in
            startFrameNum (int): Starting frame number
            numFrames (int): How many frames to read in
            bias (arr): 2D array of fluxes from bias image
            
        Returns:
            imagesData (arr): Image data
            imagesTimes (arr): Header times of these images
    """

    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [imagePath for i, imagePath in enumerate(imagePaths) if i >= startFrameNum and i < startFrameNum + numFrames]

    '''get data from each file in list of files to read, subtract bias frame (add 100 first, don't go neg)'''
    for imagePath in files_to_read:

        image = fits.open(imagePath)
        
        header = image[0].header
        
        imageData = image[0].data - bias
        headerTime = header['DATE-OBS']
        
        #change time if time is wrong (29 hours)
        hour = str(headerTime).split('T')[1].split(':')[0]
        imageMinute = str(headerTime).split(':')[1]
        dirMinute = imagePath.parent.name.split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #on red don't need to convert between UTC and local (local is UTC)
            newLocalHour = int(imagePath.parent.name.split('_')[1].split('.')[0])
        
            if int(imageMinute) < int(dirMinute):
               # newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
                newUTCHour = newLocalHour + 1         #add 1 if hour changed over during minute
            else:
                #newUTCHour = newLocalHour + 4
                newUTCHour  = newLocalHour  
        
            #replace bad hour in timestamp string with correct hour
            newUTCHour = str(newUTCHour)
            newUTCHour = newUTCHour.zfill(2)
        
            replaced = str(headerTime).replace('T' + hour, 'T' + newUTCHour).strip('b').strip(' \' ')
        
            #encode into bytes
            #newTimestamp = replaced.encode('utf-8')
            headerTime = replaced
            
        image.close()

        imagesData.append(imageData)
        imagesTimes.append(headerTime)
         
    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData, imagesTimes


def initialFind(imageData, detect_thresh):
    """
    Locates the stars in an image
    
        Parameters:
            imageData (arr): 2D array of image flux data
            detect_thresh (float): Detection threshold for star finding
            
        Returns:
            positions (arr): [x, y, half light radius] of all stars in pixels
    """

    ''' Background extraction for initial time slice'''
    imageData_new = deepcopy(imageData)
    bkg = sep.Background(imageData_new)
    bkg.subfrom(imageData_new)
    thresh = detect_thresh * bkg.globalrms  # set detection threshold to mean + 3 sigma

    
    ''' Identify stars in initial time slice '''
    objects = sep.extract(imageData_new, thresh)


    ''' Characterize light profile of each star '''
    halfLightRad = np.sqrt(objects['npix'] / np.pi) / 2.  # approximate half light radius as half of radius

    
    ''' Generate tuple of (x,y,r) positions for each star'''
    positions = zip(objects['x'], objects['y'], halfLightRad)

    return positions



def refineCentroid(imageData, time, coords, sigma):
    """
    Refines the centroid for each star for an image based on previous coords
    
        Parameters:
            imageData (arr): 2D array of flux data for a single image
            time (str): Header time of image
            coords (list): Coordinates of stars in previous image
            sigma (float): Guassian sigma weighting
            
        Return:
            Coords (tuple): two zipped lists of star coordinates
            time (str): Header time of image
    """

    '''initial x, y positions'''
    x_initial = [pos[0] for pos in coords]
    y_initial = [pos[1] for pos in coords]

    '''Flatten imageData to 2D array'''
    if imageData.ndim == 3:
        imageData = imageData[0]

    '''use an iterative 'windowed' method from sep to get new position'''
    new_pos = np.array(sep.winpos(imageData, x_initial, y_initial, sigma, subpix=5))[0:2, :]
    x = new_pos[:][0].tolist()
    y = new_pos[:][1].tolist()
    
    '''returns tuple x, y (python 3: zip(x, y) -> tuple(zip(x,y))) and time'''
    return tuple(zip(x, y)), time


def runParallel(minuteDir, MasterBiasList, ricker_kernel, exposure_time, sigma_threshold):
    firstOccSearch(minuteDir, MasterBiasList, ricker_kernel, exposure_time, sigma_threshold)
    gc.collect()
    #runParallel should be scrapped
    

def sumFlux(data, x_coords, y_coords, l):
    '''
    Function to sum up flux in square aperture of size
    
        Parameters:
            data (arr): 2D image flux array
            x_coords (arr/list): x-coordinates of the stars
            y_coords (arr/list): y-coordinates of the stars
            l (int): "Radius" of the square aperture
        
        Returns:
            star_fluxes (list): Fluxes of each star
    '''
    
    '''loop through each x and y coordinate, adding up flux in square (2l+1)^2'''
 
    star_flux_lists = [[data[y][x]
                        for x in range(int(x_coords[star] - l), int(x_coords[star] + l) + 1) 
                        for y in range(int(y_coords[star] - l), int(y_coords[star] + l) + 1)]
                           for star in range(0, len(x_coords))]
    
    star_fluxes = [sum(fluxlist) for fluxlist in star_flux_lists]
    
    return star_fluxes



def timeEvolve(imageData, imageTime, prevStarData, r, numStars, 
               x_length, y_length, 
               x_drift=0, y_drift=0):
    """
    Adjusts aperture based on star drift and calculates flux in aperture
    
        Parameters:
            imageData (arr): 2D image flux array
            imageTime (int/float): Header image time
            prevStarData (arr): Star data (coordinates, flux, time) from previous image
            r (int): Aperture radius to sum flux (in pixels)
            numStars (int): Number of stars in image
            x_length (int/float): Image length in x-direction
            y_length (int/float): Image length in y-direction
            x_drift (float): Drift rate in the x-direction (in px/s)
            y_drift (float): Drift rate in the y-direction (in px/s)
            
        Returns:
            star_data (tuple): New star coordinates, image flux, time as tuple
     """

    '''get proper frame times to apply drift'''
    frame_time = Time(imageTime, precision=9).unix   #current frame time from file header (unix)
    drift_time = frame_time - prevStarData[1,3]    #time since previous frame [s]
    
    '''add drift to each star's coordinates based on time since last frame'''
    x = np.array([prevStarData[ind, 0] + x_drift*drift_time for ind in range(0, numStars)])
    y = np.array([prevStarData[ind, 1] + y_drift*drift_time for ind in range(0, numStars)])
    
    '''get list of indices near edge of frame'''
    EdgeInds = clipCutStars(x, y, x_length, y_length)
    EdgeInds = np.sort(np.unique(EdgeInds))
    
    '''remove stars near edge of frame'''
    xClip = np.delete(np.array(x), EdgeInds)
    yClip = np.delete(np.array(y), EdgeInds)
    
    '''add up all flux within aperture'''
    sepfluxes = (sep.sum_circle(imageData, xClip, yClip, r, bkgann = (r + 6., r + 11.))[0]).tolist()
    #fluxes = sumFlux(data, xClip, yClip, l)
    
    '''set fluxes at edge to 0'''
    for i in EdgeInds:
     #   fluxes.insert(i, 0)
        sepfluxes.insert(i,0)
        
    '''returns x, y star positions, fluxes at those positions, times'''
  #  star_data = tuple(zip(x, y, fluxes, np.full(len(fluxes), frame_time)))
    star_data = tuple(zip(x, y, sepfluxes, np.full(len(sepfluxes), frame_time)))
    return star_data



#############
# Former main
#############

def firstOccSearch(minuteDir, MasterBiasList, kernel, exposure_time, sigma_threshold):
    """ 
    Formerly 'main'.
    Detect possible occultation events in selected file and archive results 
    
        Parameters:
            minuteDir (str): Filepath to current directory
            MasterBiasList (list): List of master biases and times
            kernel (arr): Ricker wavelet kernel
            exposure_time (int/float): Camera exposure time (in __)
            sigma_threshold (float): Sensitivity of the detection filter
        
        Returns:
            __ (str): Printout of processing tasks
            __ (.npy): .npy format file with star positions (if doesn't exist)
            __ (.txt): .txt format file for each occultation event with names
                       of images to be saved
            __ (int/float): time of saved occultation event images
            __ (float): flux of occulted star in saved images
    """

    global telescope
    
    print (datetime.datetime.now(), "Opening:", minuteDir)
    

    ''' create folder for results '''
    
    day_stamp = obs_date
    savefolder = base_path.joinpath('ColibriArchive', str(day_stamp))
    if not savefolder.exists():
        savefolder.mkdir()      

    '''load in appropriate master bias image from pre-made set'''
    bias = cir.chooseBias(minuteDir, MasterBiasList, obs_date)


    ''' adjustable parameters '''
    
    ap_r = 3.   #radius of aperture for flux measuremnets
    detect_thresh = 4.   #threshold for object detection


    ''' get list of image names to process'''       
    
    if RCDfiles == True: # Option for RCD or fits import - MJM 20210901
        imagePaths = sorted(minuteDir.glob('*.rcd'))
    else:
        imagePaths = sorted(minuteDir.glob('*.fits'))  
        
    del imagePaths[0]
    
    field_name = imagePaths[0].name.split('_')[0]  #which of 11 fields are observed
    
    
    ''' get 2d shape of images, number of image in directory'''
    
    if RCDfiles == True:
        x_length, y_length, num_images = cir.getSizeRCD(imagePaths) 
    else:
        x_length, y_length, num_images = getSizeFITS(imagePaths)

    print (datetime.datetime.now(), "Imported", num_images, "frames")
    
    #check if there are enough images in the current directory 
    minNumImages = len(kernel.array)*3         #3x kernel length
    if num_images < minNumImages:
        print (datetime.datetime.now(), "Insufficient number of images, skipping...")
        return
    

    ''' create star positional data from median combined stack'''
    
    print (datetime.datetime.now(), field_name, 'starfinding...',)
    
    #name of .npy file to save star positions in 
    star_pos_file = base_path.joinpath('ColibriArchive', str(day_stamp), minuteDir.name + '_' + str(detect_thresh) + 'sig_pos.npy')

    # Remove position file if it exists - MJM (modified RAB Feb 2022)
    if star_pos_file.exists():
        star_pos_file.unlink()

    numtoStack = 9          #number of images to include in stack
    startIndex = 1          #which image to start stack at (vignetting in 0th image)

    i = 0                   #counter used if several images are poor
    min_stars = 30          #minimum stars in an image
    star_find_results = []  #to store results of star finding
        
    #run until the minimum number of stars is found
    while len(star_find_results) < min_stars:
        
        startIndex += i     #adjust number to start stack if there weren't enough stars
        
        #print('stacking images %i to %i\n' %(startIndex, numtoStack))
        
        #create median combined image for star finding
        stacked = cir.stackImages(minuteDir, savefolder, startIndex, numtoStack, bias)

        #make list of star coords and half light radii
        star_find_results = tuple(initialFind(stacked, detect_thresh))

        #remove stars where centre is too close to edge of frame
        edge_buffer = 1     #number of pixels between edge of star aperture and edge of image
        star_find_results = tuple(x for x in star_find_results if x[0] + ap_r + edge_buffer < x_length and x[0] - ap_r - edge_buffer > 0)
        star_find_results = tuple(y for y in star_find_results if y[1] + ap_r + edge_buffer < x_length and y[1] - ap_r - edge_buffer > 0)

        #increase start image counter
        i += 1
            
        #check if we've reached the end of the minute, return error if so
        if (1 + i + 9) >= num_images:
            print('no good images in minute: ', minuteDir)
            print (datetime.datetime.now(), "Closing:", minuteDir)
            print ("\n")
            return -1

    #print('star finding file index: ', i)
    
    #save to .npy file
    #star position file format: x  |  y  | half light radius
    np.save(star_pos_file, star_find_results)
        
    #save radii and positions in separate arrays
    star_find_results = np.array(star_find_results)
    radii = star_find_results[:,-1]
    initial_positions = star_find_results[:,:-1]

    num_stars = len(initial_positions)      #number of stars in image
    
    #print(datetime.datetime.now(), 'Done star finding. Number of stars found: ', num_stars) 
    
    
    ''' Drift calculations '''

    if RCDfiles == True: # Choose to open rcd or fits - MJM
        first_frame = cir.importFramesRCD(imagePaths, 0, 1, bias)   #import first image
        headerTimes = [first_frame[1]] #list of image header times
        last_frame = cir.importFramesRCD(imagePaths, len(imagePaths)-1, 1, bias)  #import last image

    else:
        first_frame = importFramesFITS(imagePaths, 0, 1, bias)      #data and time from 1st image
        headerTimes = [first_frame[1]]  #list of image header times
        last_frame = importFramesFITS(imagePaths, len(imagePaths)-1, 1, bias) #data and time from last image

    drift = False     # variable to check whether stars have drifted since last frame

    drift_pos = np.empty([2, num_stars], dtype = (np.float64, 2))  #array to hold first and last positions
    drift_times = []   #list to hold times for each set of drifted coords
    GaussSigma = np.mean(radii * 2. / 2.35)  # calculate gaussian sigma for each star's light profile

    #refined star positions and times for first image
    first_drift = refineCentroid(*first_frame, initial_positions, GaussSigma)
    drift_pos[0] = first_drift[0]
    drift_times.append(first_drift[1])

    #refined star positions and times for last image
    last_drift = refineCentroid(*last_frame, drift_pos[0], GaussSigma)
    drift_pos[1] = last_drift[0]
    drift_times.append(last_drift[1])

    #get median drift rate [px/s] in x and y over the minute
    x_drift, y_drift = averageDrift(drift_pos, drift_times)

    #check drift rates
    driftTolerance = 2.5e-2   #px per s
    
    if abs(x_drift) > driftTolerance or abs(y_drift) > driftTolerance:
        drift = True

    driftErrorThresh = 1        #threshold for drift that is manageable
    
    #check if stars have drifted too much, return error if so
    if abs(np.median(x_drift)) > driftErrorThresh or abs(np.median(y_drift)) > driftErrorThresh:
        print (datetime.datetime.now(), "Significant drift, skipping ", minuteDir) 
        return -1

    ''' flux and time calculations with optional time evolution '''
    
    #image data (2d array with dimensions: # of images x # of stars)
    starData = np.empty([num_images, num_stars], dtype=(np.float64, 4))
    
    #get first image data from initial star positions
    print(first_frame[0].shape,initial_positions.shape)
    starData[0] = tuple(zip(initial_positions[:,0], 
                        initial_positions[:,1], 
                        #sumFlux(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r),
                        (sep.sum_circle(first_frame[0], initial_positions[:,0], initial_positions[:,1], ap_r)[0]).tolist(), 
                        np.ones(np.shape(np.array(initial_positions))[0]) * (Time(first_frame[1], precision=9).unix)))

    if drift:  # time evolve moving stars
    
        print('drifted - applying drift to photometry', x_drift, y_drift)
        
        #loop through each image in the minute-long dataset
        for i in range(1, num_images):
            
            #import .rcd image data
            if RCDfiles == True:
                #image file contains both image array and header time
                imageFile = cir.importFramesRCD(imagePaths, i, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list
            
            #import .fits image data
            else:
                #image file contains both image array and header time
                imageFile = importFramesFITS(imagePaths, i, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list

            #calculate star fluxes from image
            starData[i] = timeEvolve(*imageFile, deepcopy(starData[i - 1]), 
                                      ap_r, num_stars, x_length, y_length, x_drift, y_drift)
    
            
    else:  # if there is not significant drift, don't account for drift  in photometry
        
        print('no drift')
        
        #loop through each image in the minute-long dataset
        for i in range(1, num_images):
            
            #import .rcd image data
            if RCDfiles == True:
                #image file contains both image array and header time
                imageFile = cir.importFramesRCD(imagePaths, i, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list
            
            #import .fits image data
            else:
                #image file contains both image array and header time
                imageFile = importFramesFITS(imagePaths, i, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list
            
            #calculate star fluxes from image
            starData[i] = timeEvolve(*imageFile, deepcopy(starData[i - 1]), 
                                     ap_r, num_stars, x_length, y_length)

    # data is an array of shape: [frames, star_num, {0:star x, 1:star y, 2:star flux, 3:unix_time}]

    #print (datetime.datetime.now(), 'Photometry done.')

    ''' Dip detection '''
   
    #perform dip detection for all stars
    
    dipResults = []     #array to hold results of dip detection
    
    #loop through each detected object
    for starNum in range(0, num_stars):
        dipResults.append(dipDetection(starData[:, starNum, 2], kernel, starNum, sigma_threshold))

    #transform into a multidimensional array
    dipResults = np.array(dipResults)
   
    # event_frames = dipResults[:,0]         #array of event frames (-1 if no event detected, -2 if incomplete data)
    # light_curves = dipResults[:,1]         #array of light curves (empty if no event detected)
    # dip_types = dipResults[:,2]             #array of keywords describing type of dip detected
    
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
        
        date = headerTimes[f][0].split('T')[0]                                 #date of event
        time = headerTimes[f][0].split('T')[1].split('.')[0].replace(':','')   #time of event
        star_coords = initial_positions[np.where(event_frames == f)[0][0]]     #coords of occulted star
        mstime = headerTimes[f][0].split('T')[1].split('.')[1]                 # micros time of event
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
        #     filehandle.write('#    DATE-OBS: %s\n' %(headerTimes[f][0]))
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
            filehandle.write('#    DATE-OBS: %s\n' %(headerTimes[f][0][:26]))
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
                    filehandle.write('%s %f  %f  %f\n' % (files_to_save[i], float(headerTimes[:f + save_chunk][i][0].split(':')[2].split('Z')[0]), star_save_flux[i], star_save_conv[i]))
            
            #if portion of light curve to save is not at the beginning
            else:
                
                #if the portion of the light curve to save is at the end of the minute, end at the last image
                if f + save_chunk >= num_images:
        
                    files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= f - save_chunk]    #list of filenames to save (RAB 042222)
                    star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                                                     #part of light curve to save
                    star_save_conv= star_all_conv[np.where(np.in1d(imagePaths, files_to_save))[0]]
                    
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f  %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:][i][0].split(':')[2].split('Z')[0]), star_save_flux[i], star_save_conv[i]))

                #if the portion of the light curve to save is not at beginning or end of the minute, save the whole portion around the event
                else:  

                    files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= f - save_chunk and i < f + save_chunk]     #list of filenames to save
                    star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                                         #part of light curve to save                    
                    star_save_conv= star_all_conv[np.where(np.in1d(imagePaths, files_to_save))[0]]
                    
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f  %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:f + save_chunk][i][0].split(':')[2].split('Z')[0]), star_save_flux[i], star_save_conv[i]))
                    

    ''' printout statements'''
    print (datetime.datetime.now(), "Rejected Stars: ", round(((num_stars - len(save_frames)) / num_stars)*100, 2), "%")
    print (datetime.datetime.now(), "Total stars in field:", num_stars)
    print (datetime.datetime.now(), "Candidate events in this minute:", len(save_frames))
    print (datetime.datetime.now(), "Closing:", minuteDir)
    print ("\n")


"""---------------------------------SCRIPT STARTS HERE-------------------------------------------"""
'''set parameters for running code'''

RCDfiles = True         #True for reading .rcd files directly. Otherwise, fits conversion will take place.
runPar = True          #True if you want to run directories in parallel
gain_high = True           #gain level for .rcd files ('low' == False or 'high' == True)
try: # get name of telescope
    telescope = os.environ['COMPUTERNAME']
except KeyError:
    telescope = "TEST"

'''get arguments'''
if len(sys.argv) == 4:
    base_path = pathlib.Path(sys.argv[1]) # path to data directory
    obsYYYYMMDD = sys.argv[2] # date to be analyzed formatted as "YYYY/MM/DD"
    obsdatesplit = obsYYYYMMDD.split('/')
    obs_date = datetime.date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
    sigma_threshold=float(sys.argv[3]) #usually 3.75

elif len(sys.argv) == 2 and sys.argv[1] == "Test": # default 
   base_path = pathlib.Path('/', 'home', 'pquigley', 'ColibriRepos')  #path to main directory
   obs_date = datetime.date(2022, 8, 12)    #date observations
   sigma_threshold=3.75

else:
    base_path = pathlib.Path('/', 'E:','/Colibri', 'Green')  #path to main directory
    obs_date = datetime.date(2022, 9, 8)    #date observations
    sigma_threshold=3.75





if __name__ == '__main__':
    
    
    '''get filepaths'''
 
    #directory for night-long dataset
    night_dir = base_path.joinpath('ColibriData', str(obs_date).replace('-', ''))      #path to data
 
    #subdirectories of minute-long datasets (~2400 images per directory)
    minute_dirs = [f for f in night_dir.iterdir() if f.is_dir()]  
    
    #parent directory of bias images
    bias_dir = [f for f in minute_dirs if 'Bias' in f.name][0]
    
    #remove bias directory from list of image directories and sort
    minute_dirs = [f for f in minute_dirs if 'Bias' not in f.name]
    minute_dirs.sort()
    
    print ('folders', [f.name for f in minute_dirs])
         
    
    '''get median bias image to subtract from all frames'''
    
    NumBiasImages = 9       #number of bias images to combine in median bias images

    #get 2d numpy array with bias datetimes and master bias filepaths
    MasterBiasList = cir.makeBiasSet(bias_dir, base_path, obs_date, NumBiasImages)
    

    ''' prepare RickerWavelet/Mexican hat kernel to convolve with light curves'''
    exposure_time = 0.025    # exposure length in seconds
    expected_length = 0.15   # related to the characteristic scale length, length of signal to boost in convolution, may need tweaking/optimizing

    kernel_frames = int(round(expected_length / exposure_time))   # width of kernel
    ricker_kernel = RickerWavelet1DKernel(kernel_frames)          # generate kernel


    ''''run pipeline for each folder of data'''
    
    #running in parallel (minute directories are split up between cores)
    if runPar == True:
        print('Running in parallel...')
        start_time = timer.time()
        pool_size = multiprocessing.cpu_count() - 2
        pool = Pool(pool_size)
        args = ((minute_dirs[f], MasterBiasList, ricker_kernel, exposure_time, sigma_threshold) for f in range(0,len(minute_dirs)))
        pool.starmap(runParallel,args)
        pool.close()
        pool.join()

        end_time = timer.time()
        print('Ran for %s seconds' % (end_time - start_time))
        
    #running in sequence
    else:
        
        for f in range(0, len(minute_dirs)):
           
            # Added a check to see if the fits conversion has been done. - MJM 
            #only run this check if we want to process fits files - RAB
            if RCDfiles == False:
                
                    #check if conversion indicator file is present
                    if not minute_dirs[f].joinpath('converted.txt').is_file():
                        
                        print('converting to .fits')
                        
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

            print('running on... ', minute_dirs[f])

            start_time = timer.time()

            print('Running sequentially...')
            firstOccSearch(minute_dirs[f], MasterBiasList, ricker_kernel, exposure_time, sigma_threshold)
            firstOccSearch(minute_dirs[f], MasterBiasList, ricker_kernel, exposure_time, sigma_threshold)
            
            gc.collect()

            end_time = timer.time()
            print('Ran for %s seconds' % (end_time - start_time))

      
