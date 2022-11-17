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
## Bitreading Functions
##############################

@cython.wraparound(False)
def noDriftMask(np.ndarray[UI16, ndim=2] star_ind,
                int box_dim=7,
                bint gain_high=True):
    """
    Create an array for reading in relevant bits (containing stars) for this
    minute directory. Must pre-eliminate stars too close to the edge

    Args:
        star_ind (list): Pixel coordinates of the star centrodis to be analyzed.
        box_dim (int, optional): Width of integration box (in px). Must be odd
        gain_high (bool, optional): Analyze high gain image over the low gain
                                    image. Defaults to True.

    Returns:
        seek_ind (arr): Array of bits to seek from the RCD file.
        
    """
    
    ## Specific container definitions
    cdef int half_box = box_dim//2
    cdef np.ndarray seek_ind = np.empty((len(clipped_ind)*box_dim,2))

    ## Index definitions
    cdef int i,j
    cdef np.ndarray star


    ## Loop to read in and sum the pixel box.
    ## Uses two cases for half-box being even and odd
    if half_box%2 == 0: # even half-box case
        for i,star in enumerate(star_ind):
            if star[1]%2 == 0: # even pixel case
                for j in range(box_dim):
                    seek_ind[i*box_dim + j,0] = 384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box)
                    seek_ind[i*box_dim + j,1] = i
                    
            else: # odd pixel case
                for j in range(box_dim):
                    seek_ind[i*box_dim + j,0] = 384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box - 1)
                    seek_ind[i*box_dim + j,1] = i
                
            
    else: # odd half-box case
        for i,star in enumerate(star_ind):
            if star[1]%2 == 1: # even pixel case
                for j in range(box_dim):
                    seek_ind[i*box_dim + j,0] = 384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box)
                    seek_ind[i*box_dim + j,1] = i
                    
            else: # odd pixel case
                for j in range(box_dim):
                    seek_ind[i*box_dim + j,0] = 384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box - 1)
                    seek_ind[i*box_dim + j,1] = i
                

    ## Sort seek_inds to eliminate backtracking and return the inds
    seek_ind = seek_ind[seek_ind[:,0]]
    return seek_ind[0],seek_ind[1]



@cython.wraparound(False)
def fluxBitString(list mindir,
                  np.ndarray[UI16, ndim=2] star_coords,
                  int box_dim=7,
                  int l=2048,
                  int pixel_buffer=20,
                  bint gain_high=True):
    """
    

    Args:
        mindir (list): DESCRIPTION.
        star_coords (list): Pixel coordinates of the star centrodis to be analyzed.
        box_dim (int, optional): Width of integration box (in px). Must be odd
                                 to be symmetric.
        l (int, optional): Dimension of the square image.
        pixel_buffer (int, optional): Buffer width from the image edge that
                                      a star must be to be analyzed.
        timestamp (bool, optional): Return the image timestamp. Defaults to True.
        gain_high (bool, optional): Analyze high gain image over the low gain
                                    image. Defaults to True.

    Returns:
        None.

    """

    ## Type definitions
    cdef int i,ind
    cdef str path
    cdef np.ndarray clipped_ind,seek_ind,,identifier,bit_buffer,flux

    ## Integration variables
    half_box = box_dim//2
    ints_to_read = (box_dim + 1)*(3/2)
    bits_to_read = ints_to_read*8
    
    ## Eliminate stars too close to the border
    clipped_ind = star_coords[np.all(star_ind > pixel_buffer, axis=1) & \
                           np.all(star_ind < l - pixel_buffer, axis=1)]
    
    ## Get indexes of the stars to sum and create the bit buffer for tmp storage
    seek_ind,identifier = noDriftMask(clipped_ind,box_dim,gain_high)
    bit_buffer = np.empty(len(seek_ind)*bits_to_read,dtype=np.uint8)
    
    ## For each image in the directory, read the timestamp and then seek
    ## indices and read in the relevant bits for all stars. Then convert
    ## to uint16 type. Group the relevant integers and sum the fluxes.
    cdef list timestamps = []
    for path in mindir:
        with open(path,'rb') as fid:
            # Get frame timestamp
            fid.seek(152,0)
            timestamps.append(fid.read(29).decode('utf-8'))
            
            # Get bitstring
            for i,ind in enumerate(seek_ind):
                fid.seek(ind,0)
                bit_buffer[i*ints_to_read:(i+1)*ints_to_read] = np.fromfile(fid, dtype=np.uint8, count=bits_to_read)
            
            # Convert 8-bit imposter ints to 16-bit proper ints, reshape, and sum
            partial_flux = (conv_12to16(bit_buffer)).reshape((len(seek_ind),box_dim+1))
            
                
                


@cython.wraparound(False)
def fluxFromBits(filename,
                 list star_ind,
                 int box_dim=7,
                 bint timestamp=True,
                 bint gain_high=True):
    """
    Read specific stars from their pixel coordinates and returns an integrated
    flux calculated using a box method. Only works with 2048x2048 RCD images.

    Args:
        filename (str/Path): Path or pathlib object to the frame to be analyzed.
        star_ind (list): Pixel coordinates of the stars to be analyzed.
        box_dim (int, optional): Width of integration box (in px). Must be odd
                                 to be symmetric.
        timestamp (bool, optional): Return the image timestamp. Defaults to True.
        gain_high (bool, optional): Analyze high gain image over the low gain
                                    image. Defaults to True.

    Returns:
        flux (arr): Flux corresponding to the star_ind stars integrated over a
                    square area.
        img_time (str, optional): Image timestamp. Returned if timestamp=True.
        
    """
    
    raise DeprecationWarning
    
    ## Type definitions
    cdef int half_box,ints_to_read,bits_to_read
    cdef str img_time
    cdef int i,j
    cdef np.ndarray bit_buffer, flux
    
    ## Assert that the integration box be symmetric
    #assert box_dim%2 == 1
    
    ## Integration variables
    half_box = box_dim//2
    ints_to_read = (box_dim + 1)*(3/2)
    bits_to_read = ints_to_read*8
    
    bit_buffer = np.empty((box_dim, ints_to_read),dtype=np.uint8)
    flux = np.zeros(len(star_ind))
    
    
    ## Loop to read in and sum the pixel box.
    ## Uses two cases for half-box being even and odd
    if half_box%2 == 0: # even half-box case
        with open(filename, 'rb') as fid:
            for i,star in enumerate(star_ind):
                if star[1]%2 == 0: # even pixel case
                    for j in range(box_dim):
                        fid.seek(384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box), 0)
                        bit_buffer[j] = np.fromfile(fid, dtype=np.uint8, count=bits_to_read)
                        
                    flux[i] = np.sum(conv_12to16(bit_buffer.flatten())[:box_dim-1])
                
                else: # odd pixel case
                    for j in range(box_dim):
                        fid.seek(384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box - 1), 0)
                        bit_buffer[j] = np.fromfile(fid, dtype=np.uint8, count=bits_to_read)
                        
                    flux[i] = np.sum(conv_12to16(bit_buffer.flatten())[1:])
                    
            if timestamp:
                fid.seek(152,0)
                img_time = fid.read(29).decode('utf-8')
                return flux,img_time
            else:
                return flux
                    
                    
    else: # odd half-box case
        with open(filename, 'rb') as fid:
            for i,star in enumerate(star_ind):
                if star[1]%2 == 1: # odd pixel case
                    for j in range(box_dim):
                        fid.seek(384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box), 0)
                        bit_buffer[j] = np.fromfile(fid, dtype=np.uint8, count=bits_to_read)
                        
                    flux[i] = np.sum(conv_12to16(bit_buffer.flatten())[:box_dim-1])
                
                else: # even pixel case
                    for j in range(box_dim):
                        fid.seek(384 + 24576*(gain_high + 2*(star[0] + j - half_box)) + 12*(star[1] - half_box - 1), 0)
                        bit_buffer[j] = np.fromfile(fid, dtype=np.uint8, count=bits_to_read)
                        
                    flux[i] = np.sum(conv_12to16(bit_buffer.flatten())[1:])
                    
            if timestamp:
                fid.seek(152,0)
                img_time = fid.read(29).decode('utf-8')
                return flux,img_time
            else:
                return flux


##############################
## Retreive Data
##############################

@cython.boundscheck(False)
@cython.wraparound(False)
def readRCD(filename):
    """
    Reads .rcd file
    
        Parameters:
            filename (Path): Path to .rcd file
            
        Returns:
            table (arr): Table with image pixel data
            timestamp (str): Timestamp of observation
    """

    ## Type definitions
    cdef np.ndarray table

    ## Open .rcd file and extract the observation timestamp and data
    with open(filename, 'rb') as fid:
        # Timestamp
        fid.seek(152,0)
        timestamp = fid.read(29).decode('utf-8')

        # Load data portion of file
        fid.seek(384,0)
        table = np.fromfile(fid, dtype=np.uint8, count=12582912)

    return table, timestamp
            


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

    pass


#@cython.boundscheck(False)
@cython.wraparound(False)
def importFramesRCD(image_paths,
                    int  start_frame=0, 
                    int  num_frames=1,
                    np.ndarray[F64, ndim=2] bias=np.zeros((1,1),dtype=np.float64)):
    """
    Reads in frames from .rcd files starting at a specific frame.
    
        Parameters:
            image_paths (Path): pathlib object of image paths to read in
            start_frame (int): Starting frame number
            num_frames (int): How many frames to read in
            bias (arr): 2D array of fluxes from bias image
            
        Returns:
            img_array (arr): Image data
            img_times (arr): Header times of these images
    """
    
    ## Type definitions
    cdef int IMG_DIM,frame
    cdef np.ndarray img_array,data,image
    cdef list img_times
    
    
    ## Define pixel dimension of the square image and the memory array and list
    IMG_DIM = 2048
    img_array = np.zeros((num_frames, IMG_DIM, IMG_DIM), dtype=np.float64)
    img_times = []
    
    ## Loop which iteratively reads in the files and processes them
    for i,fname in enumerate(image_paths):
        # Check if we are below start_frame
        if i < start_frame:
            continue
        
        # Check if we exceed the requested frames
        frame = i - start_frame
        #print(i,frame)
        if frame >= num_frames:
            break
        
        # Load in the image data and header timestamp and subtract the bias
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
        
        
    else:
        frame += 1
    
    ## Check if only one frame was called: if so, ndim=3 -> ndim=2
    if num_frames == 1:
        img_array = img_array[0]
    ## Check if we ran out of frames
    elif num_frames != frame:
        img_array = img_array[:frame]
        print(f"We ran out of frames! Only {frame} of {num_frames}.")
        print(f"Contracting array...")
        
    
    img_array = np.subtract(img_array, bias, dtype=np.float64)
    
    return img_array,img_times        


##############################
## Image Analysis
##############################

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
## Bias Analysis
##############################

#TODO: modify usage in FirstOcc fxn to only pass the master bias directory
def makeMasterBias(dirpath,
                   int  num_biases=9,
                   *, 
                   bint RCD_files=True, 
                   bint gain_high=True): 
    """
    Get median bias image from a set of biases
    
        Parameters:
            dirpath (Path): pathlib Path object for individual bias image directories
            num_biases (int): Number of bias images to take median of
            
            Keyword Only:
            RCD_files (bool): Using RCD files (True) or FITS files (False)
            gain_high (bool): Gain keyword (where high gain = True)
            
        Return:
            biasMed (arr): Median bias image
    """
    
    ## Type definitions
    cdef np.ndarray RCD_biases,bias_median
    
    ## Read in num_biases frames from the directory and median combine them
    if RCD_files:
        RCD_bias_list = sorted(dirpath.glob("*.rcd"))
        RCD_biases = importFramesRCD(RCD_bias_list,num_frames=num_biases)[0]
        bias_median = np.median(RCD_biases, axis=0)
    
    ## For use with .fits files
    else:
        #TODO: impliment .fits alternative -> this is the old implimentation
        pass
    
    return bias_median


def makeBiasSet(dirpath,
                base_path, 
                obs_date,
                int num_biases=9):
    """
    Get set of median-combined biases for entire night that are sorted and
    indexed by time, these are saved to disk and loaded in when needed
    
    !Note: Not currently sorting by time
    
        Parameters:
            dirpath (Path): Filepath to bias image directory
            base_path (Path): pathlib.Path object to directory containing data
                              and archive directories
            obs_date (datetime/str): The observing date
            num_biases (int): Number of bias images to combine for master
            
        Return:
            bias_arr (arr): Bias image times and filepaths to saved biases
    """

    ## Type definitions
    cdef list bias_dirs,bias_list
    cdef np.ndarray masterbias_img,bias_arr

    ## Make list of bias folders
    bias_dirs = [d for d in dirpath.iterdir() if d.is_dir()]
    
    ## Generate master bias folder
    archive_path    = base_path.joinpath("ColibriArchive",str(obs_date))
    masterbias_path = archive_path.joinpath("masterBiases")
    
    # Create directories if they do not exist yet
    if not archive_path.exists():
        archive_path.mkdir()
    if not masterbias_path.exists():
        masterbias_path.mkdir()
        
    ## Make list of times and corresponding master bias images
    bias_list = []
    for i,folder in enumerate(bias_dirs):
        # Get median combined master image for this time
        masterbias_img = makeMasterBias(folder,num_biases)
        
        # Save master bias as a .fits file if it doesn't already exist
        bias_filepath = masterbias_path.joinpath(folder.name + ".fits")
        if not bias_filepath.exists():
            bias_fits = fits.PrimaryHDU(masterbias_img)
            bias_fits.writeto(bias_filepath)
        
        # Get datetime object of the current bias directory
        folder_datetime = getDateTime(folder, obs_date)
        
        # Package time and filepath into array
        #! Note: not currently sorting by time
        
        bias_list.append((folder_datetime,bias_filepath))
        
    ## Convert list of tuples to numpy array. Uncomment to sort by time
    bias_arr = np.array(bias_list)
    #ind = np.argsort(bias_arr, axis=0)
    #bias_list = bias_arr[ind[:,0]]
    print(f'bias times: {bias_arr[:,0]}')
    
    return bias_arr


def chooseBias(obs_folder,masterbias_list, obs_date):
    """
    Choose correct master bias by comparing time to the observation time
    
        Parameters:
            obs_folder (Path): Filepath to current minute directory
            masterbias_list (arr): 2D array of [bias datetimes, bias filepaths]
            obs_date (datetime): datetime object of current minute directory
            
        Returns:
            best_bias (bit): Bitmap of best master bias image
    """
    
    ## Type definitions
    cdef np.ndarray diff_time
    cdef int best_bias_indx
    
    ## Get current datetime of observations
    dir_datetime = getDateTime(obs_folder, obs_date)
    print(f"current date: {dir_datetime}")
    
    ## Make array of time differences between current obs_folder and biases
    diff_time     = np.array(abs(masterbias_list[:,0] - dir_datetime))

    ## Select best master bias to use
    best_bias_indx = np.argmin(diff_time)
    best_bias_time = masterbias_list[best_bias_indx][0] 
    best_bias_file = masterbias_list[best_bias_indx][1]
    
    ## Load in new master bias image
    best_bias = fits.getdata(best_bias_file)
    best_bias = best_bias.astype(np.float64)
    print(f"using bias from: {best_bias_time}")
    
    return best_bias


##############################
## Stack Images
##############################

#TODO: combine this function with makeMasterBias
def stackImages(folder,
                save_path, 
                int  start_frame, 
                int  num_frames, 
                np.ndarray[F64, ndim=2] bias,
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
            bias (arr): 2D flux array from the bias image
            
            Keyword Only:
            RCDfiles (bool): Reading .rcd files directly, rather than converting
            gain_high (bool): gain level for .rcd files
            
        Returns:
            imageMed (arr): Median combined, bias-subtracted image for star
                            detection
    """

    ## Type definitions
    cdef list RCD_file_list
    cdef np.ndarray RCD_imgs,median_img
    
    ## Read in stack of .rcd images and median combine them
    if RCDfiles:
        RCD_file_list = sorted(folder.glob("*.rcd"))
        RCD_imgs = importFramesRCD(RCD_file_list,start_frame,num_frames,bias)[0]
        median_img = np.median(RCD_imgs, axis=0)
    
    ## Read in stack of .fits images and median combine them
    else:
        fitsimageFileList = sorted(folder.glob('*.fits'))
        fitsimageFileList.sort(key=lambda f: int(f.name.split('_')[2].split('.')[0]))
        fitsimages = []   #list to hold bias data
    
        '''append data from each image to list of images'''
        for i in range(start_frame, num_frames):
            fitsimages.append(fits.getdata(fitsimageFileList[i]))
        
        fitsimages = importFramesFITS(fitsimageFileList, start_frame, num_frames, bias)[0]
    
        '''take median of images and subtract bias'''
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
