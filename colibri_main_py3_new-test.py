"""
Created 2018 by Emily Pass

Update: March 1, 2022 - Rachel Brown

-initial Colibri data processing pipeline for flagging candidate
KBO occultation events
"""

import sep
import numpy as np
import numba as nb
import sys
from astropy.io import fits
from astropy.convolution import convolve_fft, RickerWavelet1DKernel
from astropy.time import Time
from copy import deepcopy
from multiprocessing import Pool
#import matplotlib.pyplot as plt
import pathlib
import multiprocessing
import datetime
import os
import gc
import time as timer


def averageDrift(positions, times):
    """ Determines the median x/y drift rates of all stars in a minute (first to last image)
    input: 3D array of star positions [# frames, #stars, #position columns (X & Y)], 
           header times of each position
    returns: median x, y drift rates [px/s] taken over all stars"""
    
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


def chooseBias(obs_folder, MasterBiasList):
    """ choose correct master bias by comparing time to the observation time
    input: filepath to current minute directory, 2D numpy array of [bias datetimes, bias filepaths]
    returns: bias image that is closest in time to observation"""
    
    #current hour of observations
    current_dt = getDateTime(obs_folder)
    print('current date: ', current_dt)
    
    '''make array of time differences between current and biases'''
    bias_diffs = np.array(abs(MasterBiasList[:,0] - current_dt))
    bias_i = np.argmin(bias_diffs)    #index of best match
    
    '''select best master bias using above index'''
    bias_dt = MasterBiasList[bias_i][0]
    bias_image = MasterBiasList[bias_i][1]
                
    #load in new master bias image
    print('current bias time: ', bias_dt)
    bias = fits.getdata(bias_image)
        
    return bias


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


def dipDetection(fluxProfile, kernel, num, sigma_threshold):
    """ Checks for geometric dip, and detects dimming using Ricker Wavelet (Mexican Hat) kernel
    input: light curve of star (array of fluxes in each image), Ricker wavelet kernel, 
    current star number
    returns: Frame number of detected event (-1 for no detection or -2 if data unusable),
    light curve as an array (empty list if no event detected), 
    keyword indicating event type (empty string if no event detected)"""

    '''' Prunes profiles'''
    light_curve = np.trim_zeros(fluxProfile)
    
    if len(light_curve) == 0:
        print('empty profile: ', num)
        return -2, [], ''  # reject empty profiles
   
    
    '''perform checks on data before proceeding'''
    
    FramesperMin = 2400      #ideal number of frames in a directory (1 minute)
    minSNR = 5               #median/stddev limit to look for detections
    minLightcurveLen = FramesperMin/4    #minimum length of lightcurve
    
    # reject stars that go out of frame to rapidly
    if len(light_curve) < minLightcurveLen:
        print('Light curve too short: star', num)
        return -2, [], ''  
    
    #TODO: what should the limits actually be?
    # reject tracking failures
    if abs(np.mean(light_curve[:FramesperMin]) - np.mean(light_curve[-FramesperMin:])) > np.std(light_curve[:FramesperMin]):
        print('Tracking failure: star ', num)
        return -2, [], '' 
    
    # reject stars with SNR too low
    if np.median(light_curve)/np.std(light_curve) < minSNR:
        print('Signal to Noise too low: star', num)
        return -2, [], ''  

    #uncomment to save light curve of each star (doesn't look for dips)
    #return num, light_curve


    '''convolve light curve with ricker wavelet kernel'''
    #will throw error if try to normalize (sum of kernel too close to 0)
    conv = convolve_fft(light_curve, kernel, normalize_kernel=False)    #convolution of light curve with Ricker wavelet
    minLoc = np.argmin(conv)    #index of minimum value of convolution
    minVal = min(conv)          #minimum of convolution
    
   
    #geoDip = 0.6    #threshold for geometric dip
    norm_trunc_profile = light_curve/np.median(light_curve)  #normalize light curve
 
    #if normalized profile at min value of convolution is less than the geometric dip threshold
    
#    if norm_trunc_profile[minLoc] < geoDip:
#        
#        #get frame number of dip
#        critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
#        print (datetime.datetime.now(), "Detected >40% dip: frame", str(critFrame) + ", star", num)
#        
#        return critFrame[0], light_curve, 'geometric'


    '''if no geometric dip, look for smaller diffraction dip'''
    KernelLength = len(kernel.array)    #number of elements in kernel array
    
    #check if dip is at least one kernel length from edge
    if KernelLength <= minLoc < len(light_curve) - KernelLength:
        
        edgeBuffer = 10     #how many elements from beginning/end of the convolution to exclude
        bkgZone = conv[edgeBuffer: -edgeBuffer]        #background
    
       # dipdetection = 3.75  #dip detection threshold 
        
    else:
        print('event cutoff star: ', num)
        return -2, [], ''  # reject events that are cut off at the start/end of time series
    
    event_std=np.std(conv)
    event_mean=np.mean(conv)
    
    significance=(event_mean-minVal)/event_std/np.std(bkgZone)
    
    if significance>=sigma_threshold:

    #if minimum < background - 3.75*sigma
    #if minVal < np.mean(bkgZone) - dipdetection * np.std(bkgZone):  

        #get frame number of dip
        critFrame = np.where(fluxProfile == light_curve[minLoc])[0]
        print('found significant dip in star: ', num, ' at frame: ', critFrame[0])
        
        return critFrame[0], light_curve, conv, event_std, event_mean, minVal, np.min(norm_trunc_profile)
        
    else:
        return -1, [], ''  # reject events that do not pass dip detection
    
    
def getBias(filepath, numOfBiases, gain):
    """ get median bias image from a set of biases 
    input: filepath to bias image directory, 
    number of bias  images to take median of, image gain ('high' or 'low')
    return: median bias image"""
    
    print('Calculating median bias...')
    
    #for .fits files
    if RCDfiles == False:
        # Added a check to see if the fits conversion has been done.
        # Comment out if you only want to check for presence of fits files.
        # If commented out, be sure to uncomment the 'if not glob(...)' below

        checkfile = filepath.joinpath('converted.txt')

        if checkfile.is_file() == False:

            with open(filepath.joinpath('converted.txt'), 'a'):
                os.utime(filepath.joinpath('converted.txt'))
            
                if gain == 'high':
                    os.system("python ../RCDtoFTS.py " + str(filepath) + '/ ' + gain)
            
                else:
                    os.system("python ../RCDtoFTS.py " + str(filepath) + '/')

        else:
            print('Already converted raw files to fits format.')
            print('Remove file converted.txt if you want to overwrite.')
    

        '''get list of bias images to combine'''
        biasFileList = sorted(filepath.glob('*.fits'))
        biases = []   #list to hold bias data
    

        '''append data from each bias image to list of biases'''
        for i in range(0, numOfBiases):
            biases.append(fits.getdata(biasFileList[i]))
    
        '''take median of bias images'''
        biasMed = np.median(biases, axis=0)
    
    #for .rcd files
    else:

        '''get list of images to combine'''
        rcdbiasFileList = sorted(filepath.glob('*.rcd'))
    
        #import images, using array of zeroes as bias
        rcdbiases = importFramesRCD(rcdbiasFileList, 0, numOfBiases, np.zeros((2048,2048)), gain)[0]
    
        '''take median of bias images'''
        biasMed = np.median(rcdbiases, axis=0)
    
    return biasMed


def getDateTime(folder):
    """function to get date and time of folder, then make into python datetime object
    input: filepath to the folder
    returns: datetime object"""

    #time is in format ['hour', 'minute', 'second', 'msec'] 
    folderTime = folder.name.split('_')[-1].strip('/').split('.')  #get time folder was created from folder name

    folderDate = obs_date
    
    #date is in format ['year', 'month', 'day']
    folderTime = datetime.time(int(folderTime[0]), int(folderTime[1]), int(folderTime[2]))       #convert to time object
    
    #combine date and time
    folderDatetime = datetime.datetime.combine(folderDate, folderTime)                     #combine into datetime object
    
    return folderDatetime


def getSizeFITS(imagePaths):
    """Get number of images in a data directory and image dimensions (.fits only)
    input: list of all image filepaths in data directory
    returns: width, height of fits image, number of images in directory"""
    
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
    """ MJM - Get the size of the images and number of frames """
    '''input: list of paths to .rcd images
    returns: width of images, height of images, number of frames'''

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
    """ reads in frames from fits files starting at frame_num
    input: list of image paths to read in, starting frame number, how many frames to read in, 
    bias image (2D array of fluxes)
    returns: array of image data arrays, array of header times of these images"""

    imagesData = []    #list to hold image data
    imagesTimes = []   #list to hold image times
    
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [imagePath for i, imagePath in enumerate(imagePaths) if i >= startFrameNum and i < startFrameNum + numFrames]


    '''get data from each file in list of files to read, subtract bias frame '''
    for imagePath in files_to_read:

        image = fits.open(imagePath)        #open image
        
        header = image[0].header            #read in header data
        
        imageData = image[0].data - bias    #get image counts and subtract off bias
        
        headerTime = header['DATE-OBS']     #get timestamp
        
        
        '''change time if time is wrong (29 hours)'''
        hour = str(headerTime).split('T')[1].split(':')[0]
        imageMinute = str(headerTime).split(':')[1]
        dirMinute = imagePath.parent.name.split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #on red /greendon't need to convert between UTC and local (local is UTC)
            newLocalHour = int(imagePath.parent.name.split('_')[1].split('.')[0])
        
            #if hour changed over during the minute of observations    
            if int(imageMinute) < int(dirMinute):
               # newUTCHour = newLocalHour + 4 + 1     #add 1 plus 4 to convert to UTC
                newUTCHour = newLocalHour + 1          #add 1 
            
            #if hour has not changed over, can take new hour directly from directory name
            else:
                #newUTCHour = newLocalHour + 4          #add 4 to convert to UTC
                newUTCHour  = newLocalHour              #no addition necessary
        
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


def importFramesRCD(imagePaths, startFrameNum, numFrames, bias, gain):
    """ reads in frames from .rcd files starting at frame_num
    input: list of filenames to read in, starting frame number, how many frames to read in, 
    bias image (2D array of fluxes), gain
    returns: array of image data arrays, array of header times of these images"""
    
    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    hnumpix = 2048
    vnumpix = 2048

    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [imagePath for i, imagePath in enumerate(imagePaths) if i >= startFrameNum and i < startFrameNum + numFrames]
    
    for imagePath in files_to_read:

        data, header = readRCD(imagePath)
        headerTime = header['timestamp']

        images = nb_read_data(data)
        image = split_images(images, hnumpix, vnumpix, gain)
        image = np.subtract(image,bias)

        #change time if time is wrong (29 hours)
        hour = str(headerTime).split('T')[1].split(':')[0]
        imageMinute = str(headerTime).split(':')[1]
        dirMinute = imagePath.parent.name.split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #for red: local time is UTC time (don't need +4)
            newLocalHour = int(imagePath.parent.name.split('_')[1].split('.')[0])
        
            if int(imageMinute) < int(dirMinute):
                #newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
                newUTCHour = newLocalHour + 1
            else:
                #newUTCHour = newLocalHour + 4
                newUTCHour = newLocalHour
        
            #replace bad hour in timestamp string with correct hour
            newUTCHour = str(newUTCHour)
            newUTCHour = newUTCHour.zfill(2)
        
            replaced = str(headerTime).replace('T' + hour, 'T' + newUTCHour).strip('b').strip(' \' ')
        
            #encode into bytes
            headerTime = replaced


        imagesData.append(image)
        imagesTimes.append(headerTime)

    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData, imagesTimes


def initialFind(imageData, detect_thresh_level):
    """ Locates the stars in an image
    input: flux data in 2D array for an image, detection threshold for star finding
    returns: [x, y, half light radius] of all stars in pixels"""

    ''' Background extraction'''
    imageData_new = deepcopy(imageData)
    bkg = sep.Background(imageData_new)
    bkg.subfrom(imageData_new)
    detect_thresh = detect_thresh_level * bkg.globalrms  # set detection threshold to mean + 3 sigma

    
    ''' Identify stars in background subtracted data'''
    objects = sep.extract(imageData_new, detect_thresh)


    ''' Approximate star's half light radius as half the radius'''
    halfLightRad = np.sqrt(objects['npix'] / np.pi) / 2.  

    
    ''' Generate tuple of (x,y,r) positions for each star'''
    positions = zip(objects['x'], objects['y'], halfLightRad)

    return positions


def makeBiasSet(filepath, numOfBiases, gain):
    """ get set of median-combined biases for entire night that are sorted and indexed by time,
    these are saved to disk and loaded in when needed
    input: filepath (Path object) to bias image directories, 
    number of biases images to combine for master, gain keyword 
    return: array with bias image times and filepaths to saved biases on disk"""

    biasFolderList = [f for f in filepath.iterdir() if f.is_dir()]
    
    ''' create folders for results '''
    
    day_stamp = obs_date
    save_path = base_path.joinpath('ColibriArchive', str(day_stamp))
    bias_savepath = save_path.joinpath('masterBiases')
    
    if not save_path.exists():
        save_path.mkdir()
    print(save_path)
        
    if not bias_savepath.exists():
        bias_savepath.mkdir()
        
        
    ''' make median combined image for each minute where biases were taken '''
    
    #make list of times and corresponding master bias images
    biasList = []
    
    #loop through each folder of biases
    for folder in biasFolderList:
        masterBiasImage = getBias(folder, numOfBiases, gain)      #get median combined image from this folder
        
        #save as .fits file if doesn't already exist
        hdu = fits.PrimaryHDU(masterBiasImage)
        biasFilepath = bias_savepath.joinpath(folder.name + '.fits') 
        
        if not biasFilepath.exists():
            hdu.writeto(biasFilepath)
        
        folderDatetime = getDateTime(folder)
        
        biasList.append((folderDatetime, biasFilepath))
    
    #package times and filepaths into array, sort by time
    biasList = np.array(biasList)
    ind = np.argsort(biasList, axis=0)
    biasList = biasList[ind[:,0]]
    print('bias times: ', biasList[:,0])
    
    return biasList


def refineCentroid(imageData, time, coords, sigma):
    """ Refines the centroid for each star for an image based on previous coords 
    input: flux data in 2D array for single image, header time of image, 
    coord of stars in previous image, weighting (Gauss sigma)
    returns: new [x, y] positions, header time of image """

    '''initial x, y positions'''
    x_initial = [pos[0] for pos in coords]
    y_initial = [pos[1] for pos in coords]
    
    '''use an iterative 'windowed' method from sep to get new position'''
    new_pos = np.array(sep.winpos(imageData, x_initial, y_initial, sigma, subpix=5))[0:2, :]
    x = new_pos[:][0].tolist()
    y = new_pos[:][1].tolist()
    
    '''returns tuple x, y (python 3: zip(x, y) -> tuple(zip(x,y))) and time'''
    return tuple(zip(x, y)), time


def runParallel(minuteDir, MasterBiasList, ricker_kernel, exposure_time, gain):
    firstOccSearch(minuteDir, MasterBiasList, ricker_kernel, exposure_time, gain)
    gc.collect()
    

def stackImages(folder, save_path, startIndex, numImages, bias, gain):
    """make median combined image of first numImages in a directory
    input: current directory of images (path object), directory to save stacked image in (path object), starting index (int), 
    number of images to combine (int), bias image (2d numpy array), gain level ('low' or 'high')
    return: median combined bias subtracted image for star detection"""
    
    #for .fits files:
    if RCDfiles == False: 
        fitsimageFileList = sorted(folder.glob('*.fits'))
        fitsimageFileList.sort(key=lambda f: int(f.name.split('_')[2].split('.')[0]))
        fitsimages = []   #list to hold bias data
    
        '''append data from each image to list of images'''
    #    for i in range(startIndex, numImages):
    #        fitsimages.append(fits.getdata(fitsimageFileList[i]))
        
        fitsimages = importFramesFITS(fitsimageFileList, startIndex, numImages, bias)[0]
    
        '''take median of images and subtract bias'''
        fitsimageMed = np.median(fitsimages, axis=0)
        hdu = fits.PrimaryHDU(fitsimageMed)
    
    else:
        #for rcd files:
        '''get list of images to combine'''
        rcdimageFileList = sorted(folder.glob('*.rcd'))         #list of .rcd images
        rcdimages = importFramesRCD(rcdimageFileList, startIndex, numImages, bias, gain)[0]     #import images & subtract bias
    
        imageMed = np.median(rcdimages, axis=0)          #get median value
    
        '''save median combined bias subtracted image as .fits'''
        hdu = fits.PrimaryHDU(imageMed) 

    medFilepath = save_path.joinpath(gain + folder.name + '_medstacked.fits')     #save stacked image

    #if image doesn't already exist, save to path
    if not os.path.exists(medFilepath):
        hdu.writeto(medFilepath)
   
    return imageMed


def sumFlux(data, x_coords, y_coords, l):
    '''function to sum up flux in square aperture of size: centre +/- l pixels
    input: image data [2D array], stars x coords, stars y coords, 'radius' of square side [px]
    returns: list of fluxes for each star'''
    
    '''loop through each x and y coordinate, adding up flux in square (2l+1)^2'''
 
    star_flux_lists = [[data[y][x]
                        for x in range(int(x_coords[star] - l), int(x_coords[star] + l) + 1) 
                        for y in range(int(y_coords[star] - l), int(y_coords[star] + l) + 1)]
                           for star in range(0, len(x_coords))]
    
    star_fluxes = [sum(fluxlist) for fluxlist in star_flux_lists]
    
    return star_fluxes


def timeEvolve(imageData, imageTime, prevStarData, x_drift, y_drift, r, numStars, x_length, y_length):
    """ Adjusts aperture based on star drift and calculates flux in aperture 
    input: image data (flux in 2d array), image header time, star data (coords, flux, time) from previous image, 
    x px/s drift rate, y px/s drift rate, aperture radius to sum flux in [px], 
    number of stars, x image length, y image length
    returns: new star coords [x,y], image flux, times as tuple"""

    '''get proper frame times to apply drift'''
    frame_time = Time(imageTime, precision=9).unix   #current frame time from file header (unix)
    drift_time = frame_time - prevStarData[1,3]      #time since previous frame [s]
    
    '''add drift to each star's coordinates based on time since last frame'''
    x = [prevStarData[ind, 0] + x_drift*drift_time for ind in range(0, numStars)]
    y = [prevStarData[ind, 1] + y_drift*drift_time for ind in range(0, numStars)]
    
    '''get list of indices near edge of frame'''
    EdgeInds = clipCutStars(x, y, x_length, y_length)
    EdgeInds = list(set(EdgeInds))
    EdgeInds.sort()
    
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


def timeEvolveNoDrift(imageData, imageTime, prevStarData, r, numStars, x_length, y_length):
    """ Adjusts aperture based on star drift and calculates flux in aperture 
    input: image data (flux in 2d array), image header times, star data from previous image (coords, flux, time), 
    aperture length to sum flux in, number of stars, x image length, y image length
    returns: new star coords [x,y], image flux, times as tuple"""
    
    frame_time = Time(imageTime, precision=9).unix  #current frame time from file header (unix)
     
    ''''get each star's coordinates (not accounting for drift)'''
    x = [prevStarData[ind, 0] for ind in range(0, numStars)]
    y = [prevStarData[ind, 1] for ind in range(0, numStars)]
    
    '''get list of indices near edge of frame'''
    EdgeInds = clipCutStars(x, y, x_length, y_length)
    EdgeInds = list(set(EdgeInds))
    EdgeInds.sort()
    
    '''remove stars near edge of frame'''
    xClip = np.delete(np.array(x), EdgeInds)
    yClip = np.delete(np.array(y), EdgeInds)
    
    '''add up all flux within aperture'''
    sepfluxes = (sep.sum_circle(imageData, xClip, yClip, r, bkgann = (r + 6., r + 11.))[0]).tolist()
   # fluxes = (sumFlux(data, xClip, yClip, l))

    '''set fluxes at edge to 0'''
    for i in EdgeInds:
       # fluxes.insert(i, 0)
        sepfluxes.insert(i,0)
    
    '''returns x, y star positions, fluxes at those positions, times'''
    #star_data = tuple(zip(x, y, fluxes, np.full(len(fluxes), frame_time)))
    star_data = tuple(zip(x, y, sepfluxes, np.full(len(sepfluxes), frame_time)))
    return star_data


#############
# RCD reading section - MJM 20210827
#############

# Function for reading specified number of bytes
def readxbytes(fid, numbytes):
    for i in range(1):
        data = fid.read(numbytes)
        if not data:
            break
    return data

# Function to read 12-bit data with Numba to speed things up
@nb.njit(nb.uint16[::1](nb.uint8[::1]),fastmath=True,parallel=True)
def nb_read_data(data_chunk):
    """data_chunk is a contigous 1D array of uint8 data)
    eg.data_chunk = np.frombuffer(data_chunk, dtype=np.uint8)"""
    #ensure that the data_chunk has the right length

    assert np.mod(data_chunk.shape[0],3)==0

    out=np.empty(data_chunk.shape[0]//3*2,dtype=np.uint16)
 #   image1 = np.empty((2048,2048),dtype=np.uint16)
 #   image2 = np.empty((2048,2048),dtype=np.uint16)

    for i in nb.prange(data_chunk.shape[0]//3):
        fst_uint8=np.uint16(data_chunk[i*3])
        mid_uint8=np.uint16(data_chunk[i*3+1])
        lst_uint8=np.uint16(data_chunk[i*3+2])

        out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
        out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

    return out


# Function to split high and low gain images
def split_images(data,pix_h,pix_v,gain):
    interimg = np.reshape(data, [2*pix_v,pix_h])

    if gain == 'low':
        image = interimg[::2]
    else:
        image = interimg[1::2]

    return image

# Function to read RCD file data
def readRCD(filename):
    '''reads .rcd file
    input: path to file [string or pathlib path]
    returns: table with pixel data, header dictionary'''

    hdict = {}  #dictionary to hold data

    with open(filename, 'rb') as fid:

        # Go to start of file
        fid.seek(0,0)

        # Serial number of camera
        fid.seek(63,0)
        hdict['serialnum'] = readxbytes(fid, 9)

        # Timestamp
        fid.seek(152,0)
        hdict['timestamp'] = readxbytes(fid, 29).decode('utf-8')

        # Load data portion of file
        fid.seek(384,0)

        table = np.fromfile(fid, dtype=np.uint8, count=12582912)

    return table, hdict


#############
# End RCD section
#############


def firstOccSearch(minuteDir, MasterBiasList, kernel, exposure_time, gain, sigma_threshold):
    """ formerly 'main'
    Detect possible occultation events in selected file and archive results 
    
    input: filepath to current folder, list of master biases and times, Rickerwavelet kernel, camera exposure time,
    gain level of images
    
    output: printout of processing tasks, .npy file with star positions (if doesn't exist), 
    .txt file for each occultation event with names of images to be saved, the time 
    of that image, flux of occulted star in image
    """

    global telescope
    
    print (datetime.datetime.now(), "Opening:", minuteDir)
    print('Minute Dir: %s' % minuteDir)

    

    ''' create folder for results '''
    day_stamp = obs_date # Changed to observation date - MJM
    savefolder = base_path.joinpath('ColibriArchive', str(day_stamp))
    if not savefolder.exists():
        savefolder.mkdir()      

        
    '''load in appropriate master bias image from pre-made set'''
    
    bias = chooseBias(minuteDir, MasterBiasList)


    ''' adjustable parameters '''
    
    ap_r = 3.   #radius of aperture for flux measuremnets
    detect_thresh_level = 4.   #threshold for object detection


    ''' get list of image names to process'''       
    
    if RCDfiles == True: # Option for RCD or fits import - MJM 20210901
        imagePaths = sorted(minuteDir.glob('*.rcd'))
    else:
        imagePaths = sorted(minuteDir.glob('*.fits'))  
        
    del imagePaths[0]
    
    field_name = imagePaths[0].name.split('_')[0]  #which of 11 fields are observed
    
    
    ''' get 2d shape of images, number of image in directory'''
    
    if RCDfiles == True:
        x_length, y_length, num_images = getSizeRCD(imagePaths) 
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
    star_pos_file = base_path.joinpath('ColibriArchive', str(obs_date), minuteDir.name + '_' + str(detect_thresh_level) + 'sig_pos.npy')

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
        stacked = stackImages(minuteDir, savefolder, startIndex, numtoStack, bias, gain)
        
        #make list of star coords and half light radii
        star_find_results = tuple(initialFind(stacked, detect_thresh_level))
        
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
        first_frame = importFramesRCD(imagePaths, 0, 1, bias, gain)   #import first image
        headerTimes = [first_frame[1]] #list of image header times
        last_frame = importFramesRCD(imagePaths, len(imagePaths)-1, 1, bias, gain)  #import last image
    
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
                imageFile = importFramesRCD(imagePaths, i, 1, bias, gain)
                headerTimes.append(imageFile[1])  #add header time to list
            
            #import .fits image data
            else:
                #image file contains both image array and header time
                imageFile = importFramesFITS(imagePaths, i, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list

            #calculate star fluxes from image
            starData[i] = timeEvolve(*imageFile, deepcopy(starData[i - 1]), 
                                     x_drift, y_drift, ap_r, num_stars, x_length, y_length)
    
            
    else:  # if there is not significant drift, don't account for drift  in photometry
        
        print('no drift')
        
        #loop through each image in the minute-long dataset
        for i in range(1, num_images):
            
            #import .rcd image data
            if RCDfiles == True:
                #image file contains both image array and header time
                imageFile = importFramesRCD(imagePaths, i, 1, bias, gain)
                headerTimes.append(imageFile[1])  #add header time to list
            
            #import .fits image data
            else:
                #image file contains both image array and header time
                imageFile = importFramesFITS(imagePaths, i, 1, bias)
                headerTimes.append(imageFile[1])  #add header time to list
            
            #calculate star fluxes from image
            starData[i] = timeEvolveNoDrift(*imageFile, deepcopy(starData[i - 1]), 
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
    
    event_frames = dipResults[:,0]         #array of event frames (-1 if no event detected, -2 if incomplete data)
    light_curves = dipResults[:,1]         #array of light curves (empty if no event detected)
    conv_flux=dipResults[:,2]
    event_std=dipResults[:,3]
    significanse= dipResults[:,1] 
    minVal= dipResults[:,5] 
    minVal_unc= dipResults[:,6] 
    meanFlux= dipResults[:,4] 
    #dip_types = dipResults[:,2]             #array of keywords describing type of dip detected

    ''' data archival '''
    
    secondsToSave =  1.0    #number of seconds on either side of event to save 
    save_frames = event_frames[np.where(event_frames > 0)]  # frame numbers for each event to be saved
    save_chunk = int(round(secondsToSave / exposure_time))  # save certain num of frames on both sides of event
    save_curves = light_curves[np.where(event_frames > 0)]  # light curves for each star to be saved
    #save_types = dip_types[np.where(event_frames > 0)]
    #loop through each detected event
    for f in save_frames:
        
        date = headerTimes[f][0].split('T')[0]                                 # date of event
        time = headerTimes[f][0].split('T')[1].split('.')[0].replace(':','')   # time of event
        mstime = headerTimes[f][0].split('T')[1].split('.')[1]                 # micros time of event
        star_coords = initial_positions[np.where(event_frames == f)[0][0]]     # coords of occulted star
   
       # print(datetime.datetime.now(), ' saving event in frame', f)
        
        star_all_flux = save_curves[np.where(save_frames == f)][0]  #total light curve for current occulted star
        
        #text file to save results in
        #saved file format: 'det_date_time_star#_telescope.txt'

        savefile = base_path.joinpath('ColibriArchive', str(obs_date), 'det_' + date + '_' + time + '_' + mstime + '_star' + str(np.where(event_frames == f)[0][0]) + '_' + telescope + '.txt')
        #columns: fits filename and path | header time (seconds) |  star flux
        
        #open file to save results
        with open(savefile, 'w') as filehandle:
            
            #file header
            filehandle.write('#\n#\n#\n#\n')
            filehandle.write('#    Event File: %s\n' %(imagePaths[f]))
            filehandle.write('#    Star Coords: %f %f\n' %(star_coords[0], star_coords[1]))
            filehandle.write('#\n')
            filehandle.write('#    DATE-OBS: %s\n' %(headerTimes[f][0][:26]))
            filehandle.write('#    Telescope: %s\n' %(telescope))
            filehandle.write('#    Field: %s\n' %(field_name))
            #filehandle.write('#    Dip Type: %s\n' %(save_types[np.where(save_frames == f)][0]))
            filehandle.write('#    significanse: %.2f\n' %(significanse))
            filehandle.write('#    Min val (cunv): %.2f\n' %(minVal))
            filehandle.write('#    Min val (uncunv): %.2f\n' %(minVal_unc))
            filehandle.write('#    Mean Flux: %.2f\n' %(meanFlux))
            filehandle.write('#    Median Flux: %.2f\n' %(np.median(star_all_flux)))
            filehandle.write('#    Stddev: %.3f\n' %(event_std))
            filehandle.write('#    Stddev Flux: %.3f\n' %(np.std(star_all_flux)))
            filehandle.write('#\n#\n')
            filehandle.write('#filename     time      flux     conv_flux\n')
           
            ''' save data '''
            
           #if the portion of the light curve is at the beginning, start at 0th image
            if f - save_chunk <= 0:  
        
                files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= 0 and i < f + save_chunk]  #list of filenames to save
                star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                         #part of light curve to save
      
                #loop through each frame to be saved
                for i in range(0, len(files_to_save)):  
                    filehandle.write('%s %f  %f  %f\n' % (files_to_save[i], float(headerTimes[:f + save_chunk][i][0].split(':')[2].split('Z')[0]), star_save_flux[i], conv_flux))
            
            #if portion of light curve to save is not at the beginning
            else:
                
                #if the portion of the light curve to save is at the end of the minute, end at the last image
                if f + save_chunk >= num_images:
        
                    files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= f - save_chunk]    #list of filenames to save (RAB 042222)
                    star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                                                     #part of light curve to save
       
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f  %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:][i][0].split(':')[2].split('Z')[0]), star_save_flux[i], conv_flux))

                #if the portion of the light curve to save is not at beginning or end of the minute, save the whole portion around the event
                else:  

                    files_to_save = [imagePath for i, imagePath in enumerate(imagePaths) if i >= f - save_chunk and i < f + save_chunk]     #list of filenames to save
                    star_save_flux = star_all_flux[np.where(np.in1d(imagePaths, files_to_save))[0]]                                         #part of light curve to save                    
                   
                    #loop through each frame to save
                    for i in range(0, len(files_to_save)): 
                        filehandle.write('%s %f %f  %f\n' % (files_to_save[i], float(headerTimes[f - save_chunk:f + save_chunk][i][0].split(':')[2].split('Z')[0]), star_save_flux[i], conv_flux))
                    

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
telescope = os.environ['COMPUTERNAME']   #name of telescope
gain = 'high'           #gain level for .rcd files ('low' or 'high')

'''get arguments'''
if len(sys.argv) > 1:
    base_path = pathlib.Path(sys.argv[1])
    obsYYYYMMDD = sys.argv[2]
    obsdatesplit = obsYYYYMMDD.split('/')
    obs_date = datetime.date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))

else:
    base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri', telescope)  #path to main directory
    obs_date = datetime.date(2021, 8, 4)    #date observations

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
    MasterBiasList = makeBiasSet(bias_dir, NumBiasImages, gain)
    

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
        args = ((minute_dirs[f], MasterBiasList, ricker_kernel, exposure_time, gain) for f in range(0,len(minute_dirs)))
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
                            if gain == 'high':
                                os.system("python .\\RCDtoFTS.py " + str(minute_dirs[f]) + ' ' + gain)
                            else:
                                os.system("python .\\RCDtoFTS.py " + str(minute_dirs[f]))
                    
                    else:
                        print('Already converted raw files to fits format.')
                        print('Remove file converted.txt if you want to overwrite.')

            print('running on... ', minute_dirs[f])

            start_time = timer.time()

            sigma_threshold=3.75

            print('Running sequentially...')
            firstOccSearch(minute_dirs[f], MasterBiasList, ricker_kernel, exposure_time, gain, sigma_threshold)
            
            gc.collect()

            end_time = timer.time()
            print('Ran for %s seconds' % (end_time - start_time))

      
