"""
Filename:   colibri_image_reader.pyx
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Tue Sep 27 10:25:21 2022
Updated:    Tue Sep 27 10:25:21 2022
    
Usage: import colibri_image_reader as cir
"""

# Module Imports
import pathlib
import os
import datetime
import cython
import numpy as np
from astropy.io import fits
from time import time

# Custom Script Imports
from bitconverter import conv_12to16

# Cython-Numpy Interface
cimport numpy as np
np.import_array()

# Compile Typing Definitions 
ctypedef np.uint8_t UI8
ctypedef np.uint16_t UI16
ctypedef np.float64_t F64


##############################
## DateTime Information
##############################

def getDateTime(folder, obs_date):
    """
    Function to get date and time of folder, then make into python datetime object
    
        Parameters:
            folder (Path): Filepath object to the folder
            obs_date (datetime): datetime object of the observing date
            
        Returns:
            folderDatetime (datetime): datetime object of folder date and time
    """

    # Extract time information from Path object ['hour', 'min', 'sec', 'msec']
    folder_time = folder.name.split('_')[-1].strip('/').split('.')
    folder_time = datetime.time(int(folder_time[0]),
                               int(folder_time[1]),
                               int(folder_time[2]))
    
    # Combine the date and time objects as a datetime object
    folder_datetime = datetime.datetime.combine(obs_date, folder_time)
    
    return folder_datetime


##############################
## Retreive Data
##############################

@cython.boundscheck(False)
@cython.wraparound(False)
def readRCD(filename):
    """
    Reads .rcd file and Windows file modification time.
    
        Parameters:
            filename (Path): Path to .rcd file
            
        Returns:
            table (arr): Table with image pixel data
            timestamp (str): Timestamp of observation
            file_mod_time (str): File modification time from the system in ISO format
    """

    ## Type definitions
    cdef np.ndarray table

    ## Open .rcd file and extract the observation timestamp and data
    with open(filename, 'rb') as fid:
        # Timestamp from file
        fid.seek(152, 0)
        timestamp = fid.read(29).decode('utf-8')

        # Load data portion of file
        fid.seek(384, 0)
        table = np.fromfile(fid, dtype=np.uint8, count=12582912)

    # Get the file modification time from the Windows system
    file_mod_time_epoch = os.path.getmtime(filename)
    file_mod_time = datetime.utcfromtimestamp(file_mod_time_epoch)

    # Format the modification time without the 'Z'
    formatted_mod_time = file_mod_time.strftime('%Y-%m-%dT%H:%M:%S.') + f'{file_mod_time.microsecond * 1000:09d}'

    return table, formatted_mod_time


def readFITS(filename):
    """
    Reads .fits file
    
        Parameters:
            filename (Path): Path to .fits file
            
        Returns:
            table (arr): Table with image pixel data
            timestamp (str): Timestamp of observation
    """
    
    ## Type definitions
    cdef np.ndarray table
    
    ## Open .fits file and extract the observation timestamp and data
    with fits.open(filename) as hdul:
        # Timestamp
        timestamp = hdul[0].header['DATE-OBS']
        
        # Load data portion of file
        table = hdul[0].data
    
    return table, timestamp


def importFramesFITS(list image_paths,
                    int  start_frame=0, 
                    int  num_frames=1,):
    """
    Reads in frames from .fits files starting at a specific frame.
    Assumes that this has already been converted from .rcd files and dark subtracted!
    
        Parameters:
            imagePaths (list): List of image paths to read in
            startFrameNum (int): Starting frame number
            numFrames (int): How many frames to read in
            
        Returns:
            image_array (arr): Image data
            image_times (list): Header times of these images
    """

    ## Type definitions
    cdef int IMG_DIM,frame,end_frame
    cdef np.ndarray[UI8, ndim=1] data
    cdef np.ndarray[F64, ndim=3] img_array
    cdef list img_times = []
    cdef list images_to_read

    
    ## Define end frame. Then evaluate if this is larger than the array
    frame = 0
    end_frame = start_frame + num_frames
    #print(f"Start = {start_frame}; End = {end_frame}; Len = {len(image_paths)}")

    ## Set images to be read
    if end_frame > len(image_paths):
        print("Not enough frames in given list to import")
        images_to_read = image_paths[start_frame:]
    elif end_frame == len(image_paths):
        print("At end of given list of frames")
        images_to_read = image_paths[start_frame:]
    else:
        images_to_read = image_paths[start_frame:end_frame]

    ## Define pixel dimension of the square image and the memory array and list
    IMG_DIM = 2048
    img_array = np.zeros((len(images_to_read), IMG_DIM, IMG_DIM), dtype=np.float64)

    for fname in images_to_read:
        #print(frame)
        
        # Load in the dark-subtracted image data and header timestamp
        data,timestamp = readFITS(fname)
        
        # Timestamp formatted as YYYY-MM-DDThh:mm:ss.dddddddddZ
        # Roll over the time if it exceeded 24h
        hour = timestamp.split('T')[1].split(':')[0]
        if int(hour) > 23:
            timestamp = timestamp.replace('T' + hour, 'T' + str(int(hour) % 24))
        
        # Add corrected image and time data to appropriate array/list
        img_array[frame] = data
        img_times.append(timestamp)
        frame += 1

    ## Check if only one frame was called: if so, ndim=3 -> ndim=2
    if num_frames == 1:
        return img_array[0],img_times
    else:
        return img_array,img_times           



@cython.boundscheck(False)
@cython.wraparound(False)
def importFramesRCD(image_paths,
                    int  start_frame=0, 
                    int  num_frames=1,
                    np.ndarray[F64, ndim=2] dark=np.zeros((1,1),dtype=np.float64)):
    """
    Reads in frames from .rcd files starting at a specific frame.
    
        Parameters:
            image_paths (Path): pathlib object of image paths to read in
            start_frame (int): Starting frame number
            num_frames (int): How many frames to read in
            dark (arr): 2D array of fluxes from dark image
            
        Returns:
            img_array (arr): Image data
            img_times (list): Header times of these images
    """
    
    ## Type definitions
    cdef int IMG_DIM,frame,end_frame
    cdef np.ndarray[UI8, ndim=1] data
    cdef np.ndarray[UI16, ndim=2] image
    cdef np.ndarray[F64, ndim=3] img_array
    cdef list img_times
    cdef str timestamp,hour
    
    
    ## Define pixel dimension of the square image and the memory array and list
    IMG_DIM = 2048
    img_array = np.zeros((num_frames, IMG_DIM, IMG_DIM), dtype=np.float64)
    img_times = []
    
    ## Define end frame. Then evaluate if this is larger than the array
    ## Loop which iteratively reads in the files and processes them
    frame = 0
    end_frame = start_frame + num_frames
    #print(f"Start = {start_frame}; End = {end_frame}; Len = {len(image_paths)}")
    if end_frame > len(image_paths):
        for fname in image_paths[start_frame:]:
            #print(frame)
            
            # Load in the image data and header timestamp and subtract the dark
            data,timestamp = readRCD(fname)
            image = split_images(conv_12to16(data), IMG_DIM, IMG_DIM)
            
            # Timestamp formatted as YYYY-MM-DDThh:mm:ss.dddddddddZ
            # Roll over the time if it exceeded 24h
            hour = timestamp.split('T')[1].split(':')[0]
            if int(hour) > 23:
                timestamp = timestamp.replace('T' + hour, 'T' + str(int(hour) % 24))
            
            # Add corrected image and time data to appropriate array/list
            img_array[frame] = image
            img_times.append(timestamp)
            frame += 1
        UserWarning("Not enough frames in given list to import")
            
    elif end_frame == len(image_paths):
        for fname in image_paths[start_frame:]:
            #print(frame)
            
            # Load in the image data and header timestamp and subtract the dark
            data,timestamp = readRCD(fname)
            image = split_images(conv_12to16(data), IMG_DIM, IMG_DIM)
            
            # Timestamp formatted as YYYY-MM-DDThh:mm:ss.dddddddddZ
            # Roll over the time if it exceeded 24h
            hour = timestamp.split('T')[1].split(':')[0]
            if int(hour) > 23:
                timestamp = timestamp.replace('T' + hour, 'T' + str(int(hour) % 24))
            
            # Add corrected image and time data to appropriate array/list
            img_array[frame] = image
            img_times.append(timestamp)
            frame += 1
        UserWarning("At end of given list of frames")
    
    else:
        for fname in image_paths[start_frame:end_frame]:
            #print(frame)
            
            # Load in the image data and header timestamp and subtract the dark
            data,timestamp = readRCD(fname)
            image = split_images(conv_12to16(data), IMG_DIM, IMG_DIM)
            
            # Timestamp formatted as YYYY-MM-DDThh:mm:ss.dddddddddZ
            # Roll over the time if it exceeded 24h
            hour = timestamp.split('T')[1].split(':')[0]
            if int(hour) > 23:
                timestamp = timestamp.replace('T' + hour, 'T' + str(int(hour) % 24))
            
            # Add corrected image and time data to appropriate array/list
            img_array[frame] = image
            img_times.append(timestamp)
            frame += 1
            
    
    img_array = np.subtract(img_array, dark, dtype=np.float64)
    
    ## Check if only one frame was called: if so, ndim=3 -> ndim=2
    if num_frames == 1:
        return img_array[0],img_times
    else:
        return img_array,img_times           


##############################
## Image Analysis
##############################

def testGPSLock(filepath):
    """
    Check to see if there was a valid GPS lock established for the given image.

    Args:
        filepath (str/Path): Full filepath to the RCD image to be checked.

    Returns:
        gpsLock (bool): Returns true/false on whether a GPS lock was
                        established for the given image

    """
    cdef int ControlBlock2
    cdef bint gpsLock
    
    ## Open .rcd file and extract the ControlBlock2 variable which represents
    ## the 8 boolean values following in the metadata (specifically the GPS
    ## lock and GPS error variables) as an int
    with open(filepath, 'rb') as fid:
        fid.seek(140,0)
        ControlBlock2 = ord(fid.read(1))
        
    ## Compare ControlBlock2 variable with expected 96 == 0b01100000 sequence
    ## which represents a GPS locked, upper-left quadrant image with no GPS
    ## error.
    gpsLock = (ControlBlock2 == 96)
    print("GPS Control Block: {}".format(ControlBlock2))
    
    return gpsLock
    

@cython.wraparound(False)
def split_images(np.ndarray[UI16, ndim=1] data,
                 int X_PIX,
                 int Y_PIX,
                 *, 
                 bint gain_high=True):
    """
    Function to extract either the low- or high-gain image from the data
    
        Parameters:
            data (arr): 1D array of combined image data
            X_PIX (int): Width of the image in pixels
            Y_PIX (int): Depth of the image in pixels
            
            Keyword Only:
            gain_high (bool): Gain keyword (where high gain = True)
                
        Returns:
            highgain_image (arr): Extracted high-gain image
            lowgain_image (arr): Extracted low-gain image
    """
    
    ## Type definitions
    cdef np.ndarray data_2D,extracted_image
    
    data_2D = np.reshape(data, [2*Y_PIX,X_PIX])

    if gain_high:
        extracted_image = data_2D[1::2]
    else:
        extracted_image  = data_2D[::2]
        
    return extracted_image


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
    
    #TODO: complete this
    pass


def getSizeRCD(image_paths): #TODO: consider getting rid of this function
    """
    Get number of images in a data directory and image dimensions (.rcd only).
    Optionally get this from the RCD header.
    
        Parameters:
            image_paths (list/Path): List/pathlib Path object of all image
                                     filepaths in data directory
        
        Returns:
            width (int): Width of .fits images
            height (int): Height of .fits images
            frames (int): Number of images in a data directory
    """

    cdef int width,height,frames
    width,height = (2048,2048)
    frames = len(image_paths)
    
    return width, height, frames


##############################
## Dark Analysis
##############################

#TODO: modify usage in FirstOcc fxn to only pass the master dark directory
def makeMasterDark(dirpath,
                   int  num_darks=9,
                   *, 
                   bint RCD_files=True, 
                   bint gain_high=True): 
    """
    Get median dark image from a set of darks
    
        Parameters:
            dirpath (Path): pathlib Path object for individual dark image directories
            num_darks (int): Number of dark images to take median of
            
            Keyword Only:
            RCD_files (bool): Using RCD files (True) or FITS files (False)
            gain_high (bool): Gain keyword (where high gain = True)
            
        Return:
            darkMed (arr): Median dark image
    """
    
    ## Type definitions
    cdef np.ndarray RCD_darks,dark_median
    
    ## Read in num_darks frames from the directory and median combine them
    if RCD_files:
        RCD_dark_list = sorted(dirpath.glob("*.rcd"))
        RCD_darks = importFramesRCD(RCD_dark_list,num_frames=num_darks)[0]
        dark_median = np.median(RCD_darks, axis=0)
    
    ## For use with .fits files
    else:
        #TODO: impliment .fits alternative -> this is the old implimentation
        pass
    
    return dark_median


def makeDarkSet(dirpath,
                base_path, 
                obs_date,
                int num_darks=9):
    """
    Get set of median-combined darks for entire night that are sorted and
    indexed by time, these are saved to disk and loaded in when needed
    
    !Note: Not currently sorting by time
    
        Parameters:
            dirpath (Path): Filepath to dark image directory
            base_path (Path): pathlib.Path object to directory containing data
                              and archive directories
            obs_date (datetime/str): The observing date
            num_darks (int): Number of dark images to combine for master
            
        Return:
            dark_arr (arr): Dark image times and filepaths to saved darks
    """

    ## Type definitions
    cdef list dark_dirs,dark_list
    cdef np.ndarray masterdark_img,dark_arr

    ## Make list of dark folders
    dark_dirs = [d for d in dirpath.iterdir() if d.is_dir()]
    
    ## Generate master dark folder
    archive_path    = base_path.joinpath("ColibriArchive",str(obs_date))
    masterdark_path = archive_path.joinpath("masterDarks")
    
    # Create directories if they do not exist yet
    if not archive_path.exists():
        archive_path.mkdir()
    if not masterdark_path.exists():
        masterdark_path.mkdir()
        
    ## Make list of times and corresponding master dark images
    dark_list = []
    for i,folder in enumerate(dark_dirs):
        # Get median combined master image for this time
        masterdark_img = makeMasterDark(folder,num_darks)
        
        # Save master dark as a .fits file if it doesn't already exist
        dark_filepath = masterdark_path.joinpath(folder.name + ".fits")
        if not dark_filepath.exists():
            dark_fits = fits.PrimaryHDU(masterdark_img)
            dark_fits.writeto(dark_filepath)
        
        # Get datetime object of the current dark directory
        folder_datetime = getDateTime(folder, obs_date)
        
        # Package time and filepath into array
        #! Note: not currently sorting by time
        
        dark_list.append((folder_datetime,dark_filepath))
        
    ## Convert list of tuples to numpy array. Uncomment to sort by time
    dark_arr = np.array(dark_list)
    #ind = np.argsort(dark_arr, axis=0)
    #dark_list = dark_arr[ind[:,0]]
    print(f'dark times: {dark_arr[:,0]}')
    
    return dark_arr


def chooseDark(obs_folder,masterdark_list, obs_date):
    """
    Choose correct master dark by comparing time to the observation time
    
        Parameters:
            obs_folder (Path): Filepath to current minute directory
            masterdark_list (arr): 2D array of [dark datetimes, dark filepaths]
            obs_date (datetime): datetime object of current minute directory
            
        Returns:
            best_dark (bit): Bitmap of best master dark image
    """
    
    ## Type definitions
    cdef np.ndarray diff_time
    cdef int best_dark_indx
    
    ## Get current datetime of observations
    dir_datetime = getDateTime(obs_folder, obs_date)
    print(f"current date: {dir_datetime}")
    
    ## Make array of time differences between current obs_folder and darks
    diff_time     = np.array(abs(masterdark_list[:,0] - dir_datetime))

    ## Select best master dark to use
    best_dark_indx = np.argmin(diff_time)
    best_dark_time = masterdark_list[best_dark_indx][0] 
    best_dark_file = masterdark_list[best_dark_indx][1]
    
    ## Load in new master dark image
    best_dark = fits.getdata(best_dark_file)
    best_dark = best_dark.astype(np.float64)
    print(f"using dark from: {best_dark_time}")
    
    return best_dark


##############################
## Stack Images
##############################

#TODO: combine this function with makeMasterDark
def stackImages(folder,
                save_path, 
                int  start_frame, 
                int  num_frames, 
                np.ndarray[F64, ndim=2] dark,
                *, 
                bint RCDfiles=True,
                bint gain_high=True):
    """
    Make median combined image of first numImages in a directory
    
        Parameters:
            folder (str): Directory of images to be stacked
            save_path (str): Directory to save stacked image in
            start_frame (int): Image starting index
            num_frames (int): Number of images to combine
            dark (arr): 2D flux array from the dark image
            
            Keyword Only:
            RCDfiles (bool): Reading .rcd files directly, rather than converting
            gain_high (bool): gain level for .rcd files
            
        Returns:
            imageMed (arr): Median combined, dark-subtracted image for star
                            detection
    """

    ## Type definitions
    cdef list RCD_file_list
    cdef np.ndarray RCD_imgs,median_img
    
    ## Read in stack of .rcd images and median combine them
    if RCDfiles:
        RCD_file_list = sorted(folder.glob("*.rcd"))
        RCD_imgs = importFramesRCD(RCD_file_list,start_frame,num_frames,dark)[0]
        median_img = np.median(RCD_imgs, axis=0)
    
    ## Read in stack of .fits images and median combine them
    else:
        fitsimageFileList = sorted(folder.glob('*.fits'))
        fitsimageFileList.sort(key=lambda f: int(f.name.split('_')[2].split('.')[0]))
        fitsimages = []   #list to hold dark data
    
        '''append data from each image to list of images'''
        for i in range(start_frame, num_frames):
            fitsimages.append(fits.getdata(fitsimageFileList[i]))
        
        fitsimages = importFramesFITS(fitsimageFileList, start_frame, num_frames, dark)[0]
    
        '''take median of images and subtract dark'''
        fitsimageMed = np.median(fitsimages, axis=0)
        median_img = fitsimageMed
    
    ## Define new image path
    if gain_high:
        median_filepath = save_path.joinpath('high' + folder.name + '_medstacked.fits')
    else:
        median_filepath = save_path.joinpath('low' + folder.name + '_medstacked.fits')
        
    ## Check to see if this set was already median combined. Else save the image
    if not median_filepath.exists():
        stack_fits = fits.PrimaryHDU(median_img)
        stack_fits.writeto(median_filepath)
        
    
    return median_img
