"""
Filename:   fits_funcs.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Tue Jan 24 14:19:38 2023
Updated:    Tue Jan 24 14:19:38 2023
    
Usage:      import fits_funcs as ff
A repository for the old methods to read/analyze FITS files. Not in use as of
2023/01/24.
"""

# Module Imports
import os,sys
import numpy as np
from astropy.io import fits


#--------------------------------functions------------------------------------#

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