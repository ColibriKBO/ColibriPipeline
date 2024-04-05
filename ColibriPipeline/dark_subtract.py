# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 11:37:02 2021

@author: Rachel A. Brown
"""

import astropy
from astropy.io import fits
import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import datetime
import pathlib
import numba as nb
import binascii
import os

'''------------rcd reading section (Mike Mazur) -----------'''
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
    image1 = np.empty((2048,2048),dtype=np.uint16)
    image2 = np.empty((2048,2048),dtype=np.uint16)

    for i in nb.prange(data_chunk.shape[0]//3):
        fst_uint8=np.uint16(data_chunk[i*3])
        mid_uint8=np.uint16(data_chunk[i*3+1])
        lst_uint8=np.uint16(data_chunk[i*3+2])

        out[i*2] =   (fst_uint8 << 4) + (mid_uint8 >> 4)
        out[i*2+1] = ((mid_uint8 % 16) << 8) + lst_uint8

    return out

def getSizeRCD(filenames):
    """ MJM - Get the size of the images and number of frames """
    filename_first = filenames[0]
    frames = len(filenames)

    width = 2048
    height = 2048

    # You could also get this from the RCD header by uncommenting the following code
    with open(filename_first, 'rb') as fid:
         fid.seek(81,0)
         hpixels = readxbytes(fid, 2) # Number of horizontal pixels
         fid.seek(83,0)
         vpixels = readxbytes(fid, 2) # Number of vertical pixels

         fid.seek(100,0)
         binning = readxbytes(fid, 1)

         bins = int(binascii.hexlify(binning),16)
         hpix = int(binascii.hexlify(hpixels),16)
         vpix = int(binascii.hexlify(vpixels),16)
         width = int(hpix / bins)
         height = int(vpix / bins)

    return width, height, frames

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

    hdict = {}

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

def importFramesRCD(parentdir, filenames, start_frame, num_frames, dark, gain):
    """ reads in frames from .rcd files starting at frame_num
    input: parent directory (minute), list of filenames to read in, starting frame number, how many frames to read in, 
    dark image (2D array of fluxes)
    returns: array of image data arrays, array of header times of these images"""
    
    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    hnumpix = 2048
    vnumpix = 2048
    
    imgain = gain
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]
    
    for filename in files_to_read:


        data, header = readRCD(filename)
        headerTime = header['timestamp']

        images = nb_read_data(data)
        image = split_images(images, hnumpix, vnumpix, imgain)
        image = np.subtract(image,dark)

        #change time if time is wrong (29 hours)
        hour = str(headerTime).split('T')[1].split(':')[0]
        fileMinute = str(headerTime).split(':')[1]
        dirMinute = str(parentdir).split('_')[1].split('.')[1]
      #  dirMinute = '30'
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #for red: local time is UTC time (don't need +4)
            newLocalHour = int(parentdir.split('_')[1].split('.')[0])
        
            if int(fileMinute) < int(dirMinute):
                newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
               # newUTCHour = newLocalHour + 1         #FOR RED
            else:
                newUTCHour = newLocalHour + 4
               # newUTCHour = newLocalHour              #FOR RED
        
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

#############
# End RCD section
#############

def getDateTime(folder):
    """function to get date and time of folder, then make into python datetime object
    input: filepath 
    returns: datetime object"""
    
    #time is in format ['hour', 'minute', 'second', 'msec']
    folderDate = str(folder.name).split('_')[0]                 #get date folder was created from its name
    #folderDate = '20220121'
    folderTime = str(folder.name).split('_')[1].split('.')
  # folderTime = '19.30.00.000'
   # folderTime = folderTime.split('.')
    folderDate = datetime.date(int(folderDate[:4]), int(folderDate[4:6]), int(folderDate[-2:]))  #convert to date object
    folderTime = datetime.time(int(folderTime[0]), int(folderTime[1]), int(folderTime[2]))       #convert to time object
    folderDatetime = datetime.datetime.combine(folderDate, folderTime)                     #combine into datetime object
    
    return folderDatetime

def chooseDark(obs_folder, MasterDarkList):
    """ choose correct master dark by comparing time to the observation time
    input: filepath to current minute directory, 2D numpy array of [dark datetimes, dark filepaths]
    returns: dark image that is closest in time to observation"""
    
    #current hour of observations
    current_dt = getDateTime(obs_folder)
    
    '''make array of time differences between current and darks'''
    dark_diffs = np.array(abs(MasterDarkList[:,0] - current_dt))
    dark_i = np.argmin(dark_diffs)    #index of best match
    
    '''select best master dark using above index'''
    dark_image = MasterDarkList[dark_i][1]
                
    #load in new master dark image
    dark = fits.getdata(dark_image)
        
    return dark

def getDark(filepath, numOfDarks, gain):
    """ get median dark image from a set of darks (length =  numOfDarks) from filepath
    input: dark image directory (path object), number of dark images to take median from (int), gain level ('low' or 'high')
    return: median dark image"""
    
    #FOR FITS:
    # Added a check to see if the fits conversion has been done.
    # Comment out if you only want to check for presence of fits files.
    # If commented out, be sure to uncomment the 'if not glob(...)' below
  #  if filepath.joinpath('converted.txt').is_file == False:
  #      with open(filepath + 'converted.txt', 'a'):
 #            os.utime(filepath + 'converted.txt')
            
 #            if gain == 'high':
 #                os.system("python .\\RCDtoFTS.py " + str(filepath) + ' ' + gain)
 #            else:
 #                os.system("python .\\RCDtoFTS.py " + str(filepath))
 # #   else:
 # #       print('Already converted raw files to fits format.')
 # #       print('Remove file converted.txt if you want to overwrite.')

 #    #for .fits files
 #    '''get list of dark images to combine'''
 #   darkFileList = sorted(filepath.glob('*.fits'))
 #   darks = []   #list to hold dark data
    
 #    '''append data from each dark image to list of darks'''
 #   for i in range(0, numOfDarks):
 #        darks.append(fits.getdata(darkFileList[i]))
 
     
        
    #for rcd files:
    '''get list of images to combine'''
    rcddarkFileList = sorted(filepath.glob('*.rcd'))
    
    #import images, using array of zeroes as dark
    rcddarks = importFramesRCD(filepath, rcddarkFileList, 0, numOfDarks, np.zeros((2048,2048)), gain)[0]
    
    '''take median of dark images'''
    darkMed = np.median(rcddarks, axis=0)
    
    return darkMed

def makeDarkSet(filepath, numOfDarks, savefolder, gain):
    """ get set of median-combined darks for entire night that are sorted and indexed by time,
    these are saved to disk and loaded in when needed
    input: filepath (string) to dark image directories, number of darks images to combine for master
    return: array with dark image times and filepaths to saved darks on disk"""
    
    darkFolderList = [f for f in filepath.iterdir() if f.is_dir()]   #list of dark folders
    
    ''' create folder for results, save dark images '''
    dark_savepath = savefolder.joinpath(gain + '_masterDarks')

    if not dark_savepath.exists():
        dark_savepath.mkdir()      #make folder to hold master dark images in
        
    #make list of times and corresponding master dark images
    darkList = []
    
    #loop through each folder of darks
    for folder in darkFolderList:
        masterDarkImage = getDark(folder, numOfDarks, gain)      #get median combined image from this folder
        
        #save as .fits file if doesn't already exist
        hdu = fits.PrimaryHDU(masterDarkImage)
        darkFilepath = dark_savepath.joinpath(folder.name + '_' + gain + '_meddark.fits')

        
        if not os.path.exists(darkFilepath):
            hdu.writeto(darkFilepath)
        
        folderDatetime = getDateTime(folder)
        
        darkList.append((folderDatetime, darkFilepath))
    
    #package times and filepaths into array, sort by time
    darkList = np.array(darkList)
    ind = np.argsort(darkList, axis=0)
    darkList = darkList[ind[:,0]]
    
    return darkList


#----------------------------------main---------------------------------------#

if __name__ == '__main__':

    '''------------set up--------------------'''
    RCDFiles = True                             #set True to read .rcd files, False to read .fits files

    #observation info
    obs_date = datetime.date(2021, 11, 9)            #date of observation
    obs_time = datetime.time(2, 59, 16)              #time of observation (to the second)
    telescope = 'Red'                                #telescope identifier
    field_name = 'field2'                            #name of field observed

    #image info
    image_index = '0000036'                          #index of image to use
    gain = 'high'                                    #gain of image to use


    '''-----------------set up filepaths-----------------'''
    base_path = pathlib.Path('/', 'home', 'rbrown', 'Documents', 'Colibri', telescope)  #base path for telescope data

    data_path = base_path.joinpath('ColibriData', str(obs_date).replace('-', ''))    #path that holds data

    #get exact name of desired minute directory
    subdirs = [f.name for f in data_path.iterdir() if f.is_dir()]                   #all minutes in night directory
    minute_dir = [f for f in subdirs if str(obs_time).replace(':', '.') in f][0]    #minute we're interested in

    minute_path = data_path.joinpath(minute_dir)            #path to minute images we're interested in 

    saveDirpath = base_path.joinpath('ColibriArchive', str(obs_date).replace('-','') + '_diagnostics')  #path to save images in

    subSavepath = saveDirpath.joinpath('subtract_images')               #path to save subtracted images in
    masterDarkpath = saveDirpath.joinpath(gain + '_masterDarks')       #path that holds master dark images

    #make directories to hold results in if doesn't already exist
    saveDirpath.mkdir(parents=True, exist_ok=True)
    subSavepath.mkdir(parents=True, exist_ok=True)
    masterDarkpath.mkdir(parents=True, exist_ok=True)

    '''---------------make/load median combined dark---------------------'''

    #get dark median dark image to subtract
    darkList = sorted(masterDarkpath.glob('*.fits'))

    #make median combined dark images if they don't already exist
    if len(darkList) == 0:
        NumDarkImages = 9           #number of darks to include in median combine
        darkList = makeDarkSet(base_path.joinpath('ColibriData', str(obs_date).replace('-',''), 'Dark'), NumDarkImages, saveDirpath, gain)  #list of dark combines and times

    #need datetimes of dark images if already created
    else:
        darkList2D = []     #2D list to hold dark dates and images
        
        #get time for each dark image in list
        for darkImPath in darkList:
            darkTime = getDateTime(darkImPath)
            darkList2D.append((darkTime, darkImPath))
        
        #package into 2D array
        darkList = np.array(darkList2D)
        ind = np.argsort(darkList, axis=0)
        darkList = darkList[ind[:,0]]
        
    #median combined dark image for correct time of night
    darkMed = chooseDark(minute_path, darkList)  

    '''------------------read in images and subtract dark-----------------'''  

    #get list of images in minute folder
    if RCDFiles == True:
        imagepath = sorted(minute_path.glob('*' + image_index + '.rcd'))

    #for .fits images
    else:
        imagepath = sorted(minute_path.glob('*' + image_index + '.fits'))[0]

    imagename = imagepath[0].name       #string containing image name

    imageSavepath = subSavepath.joinpath(minute_dir + '_' + gain + '_sub_' + imagename.replace('.rcd','.fits'))    #path (incl filename) to save dark subtracted image to


    #import desired image and subtract the correct dark
    if RCDFiles == True:
        #import image and do dark subtraction
        image = importFramesRCD(minute_dir, imagepath, 0, 1, darkMed, gain)
        imageData = image[0]

        #save subtracted image as .fits file (does not preserve header data)
        hdu = fits.PrimaryHDU(imageData)
            
        if not os.path.exists(imageSavepath):
            hdu.writeto(imageSavepath)
            
    else:
        #load in image to do subtraction on
        image = fits.open(imagepath)
        
        #do subtraction
        image[0].data = image[0].data - darkMed
        imageData = image[0].data


        #save new image file
        image.writeto(imageSavepath)

    '''-----calculate and save subtracted image stats'''
    #calculate stats
    med = np.median(imageData)
    mean = np.mean(imageData)
    RMS = np.sqrt(np.mean(np.square(imageData)))

    #print stats to console
    print('Image: ', imageSavepath.name)
    print('Median value: ', med)
    print('Mean value: ', mean)
    print('RMS: ', RMS)    

    #append statistics to file
    imageStatFile = imageSavepath.parent.joinpath('subtractedImageStats.txt')

    #make new file with header if it doesn't exist
    if not imageStatFile.exists():
        with open(imageStatFile, 'w') as file:
            
            file.write('#imageName    Median    Mean    RMS\n')
            
    #write to existing file
    with open(imageStatFile, 'a') as file:
        file.write('%s %f %f %f\n' %(imageSavepath.name, med, mean, RMS))
                    