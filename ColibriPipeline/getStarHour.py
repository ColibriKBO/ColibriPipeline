# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 11:06:40 2022

@author: Roman A.

Return star-hours for each field observed in the night directory based on number of frames of certain fields and
number of stars in the mid-frame

This script is used by timeline.py
"""
from pathlib import Path
import sep
from datetime import datetime, date, time
import os
import numpy as np
import numba as nb
from copy import deepcopy
from astropy.io import fits


#----------------------------------------M. Mazur's stuff to read .rcd files------------------------------------------

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

def importFramesRCD(filenames, start_frame, num_frames, dark, gain):
    """ reads in frames from .rcd files starting at frame_num
    input: parent directory (minute), list of filenames to read in, starting frame number, how many frames to read in, 
    dark image (2D array of fluxes)
    returns: array of image data arrays, array of header times of these images"""
    
    imagesData = []    #array to hold image data
    
    hnumpix = 2048
    vnumpix = 2048
    
    #imgain = 'low'
    imgain = gain
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]
    
    for filename in files_to_read:


        data, header = readRCD(filename)

        images = nb_read_data(data)
        image = split_images(images, hnumpix, vnumpix, imgain)
        image = np.subtract(image,dark)

        imagesData.append(image)
        

    '''make into array'''
    imagesData = np.array(imagesData, dtype='float64')
    
    '''reshape, make data type into floats'''
    if imagesData.shape[0] == 1:
        imagesData = imagesData[0]
        imagesData = imagesData.astype('float64')
        
    return imagesData

#---------------------------------------end of RCD stuff-----------------------------------------------

def initialFindFITS(data, detect_thresh):
    """ Locates the stars in the initial time slice 
    input: flux data in 2D array for a fits image, star detection threshold (float)
    returns: [x, y, half light radius] of all stars in pixels"""

    ''' Background extraction for initial time slice'''
    data_new = deepcopy(data)           #make copy of data
    bkg = sep.Background(data_new)      #get background array
    bkg.subfrom(data_new)               #subtract background from data
    thresh = detect_thresh * bkg.globalrms         # set detection threshold to mean + 3 sigma

    #sep.set_extract_pixstack(600000)
    ''' Identify stars in initial time slice '''
    objects = sep.extract(data_new, thresh)#, deblend_nthresh = 1)


    ''' Characterize light profile of each star '''
    halfLightRad = np.sqrt(objects['npix'] / np.pi) / 2.  # approximate half light radius as half of radius

    
    ''' Generate tuple of (x,y,r) positions for each star'''
    positions = zip(objects['x'], objects['y'], halfLightRad)
    

    return positions

def getDark(filepath, numOfDarks, gain):
    """ get median dark image from a set of darks (length =  numOfDarks) from filepath
    input: dark image directory (path object), number of dark images to take median from (int), gain level ('low' or 'high')
    return: median dark image"""    
        
    #for rcd files:
    '''get list of images to combine'''
    rcddarkFileList = sorted(filepath.glob('*.rcd'))
    
    #import images, using array of zeroes as dark
    rcddarks = importFramesRCD( rcddarkFileList, 0, numOfDarks, np.zeros((2048,2048)), gain)[0]
    
    '''take median of dark images'''
    darkMed = np.median(rcddarks, axis=0)
    
    return darkMed

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
    folderDate = date(int(folderDate[:4]), int(folderDate[4:6]), int(folderDate[-2:]))  #convert to date object
    folderTime = time(int(folderTime[0]), int(folderTime[1]), int(folderTime[2]))       #convert to time object
    folderDatetime = datetime.combine(folderDate, folderTime)                     #combine into datetime object
    
    return folderDatetime

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

def fieldCoords(fieldname):
    """
    Get field coordinates based on field index

    Parameters
    ----------
    fieldname : str
        Name of the field.

    Returns
    -------
    coords : array
        A 1D array of field coordinates [Ra, dec].

    """

    if fieldname == 'field1':
        coords=[273.736, -18.640]
    if fieldname == 'field2':
        coords=[92.419,  23.902]
    if fieldname == 'field3':
        coords=[287.740, -17.914]
    if fieldname == 'field4':
        coords=[105.436, 22.379]
    if fieldname == 'field5':
        coords=[254.789, -27.225]
    if fieldname == 'field6':
        coords=[129.972, 19.312]
    if fieldname == 'field7':
        coords=[75.678,  23.580]
    if fieldname == 'field8':
        coords=[306.006, -14.551]
    if fieldname == 'field9':
        coords=[239.923, -25.287]
    if fieldname == 'field10':
        coords=[56.973,  23.942]
    if fieldname == 'field11':
        coords=[318.700, -11.365]
    if fieldname == 'field12':
        coords=[226.499, -22.274]
    if fieldname == 'field13':
        coords=[334.365, -10.910]
    if fieldname == 'field14':
        coords=[212.040, -17.675]
    if fieldname == 'field15':
        coords=[39.313,  17.413]
    if fieldname == 'field16':
        coords=[143.292, 10.261]
    if fieldname == 'field17':
        coords=[348.814, -0.699]
    if fieldname == 'field18':
        coords=[155.530, 5.914]
    if fieldname == 'field19':
        coords=[1.693,   3.707]
    if fieldname == 'field20':
        coords=[15.529,  2.557]
    if fieldname == 'field21':
        coords=[25.171,  14.130]
    if fieldname == 'field22':
        coords=[198.755, -11.953]
    if fieldname == 'field23':
        coords=[184.631, -3.816]
    if fieldname == 'field24':
        coords=[172.488, 0.500]

        
    return coords

def getStarHour(main_path, obs_date, threshold=4, gain='high'):
    """
    Get string that will tell you field coordinates and star-hours

    Parameters
    ----------
    main_path : str
        base_path for observations, usually it's D:.
    obs_date : str
        Observation date eg 2022-11-27.
    threshold : int, optional
        Star-finding threshold for SEP. The default is 4.
    gain : str, optional
        Image reading gain. The default is 'high'.

    Returns
    -------
    summary : str
        A list of strings with fields, coordinates and starhours.

    """

    main_path=Path(main_path)
    data_path=main_path.joinpath('/ColibriData',str(obs_date).replace('-', '')) #path of observed night

    
    minute_dirs=[d for d in data_path.iterdir() if (os.path.isdir(d) and d.name != 'Dark')] #list of minute dirs
    
    total_frames=[] #total list of all frames observed
    fields=[] #list of all fields observed
    for dirs in minute_dirs: #loop through each minute
        frames=[f for f in dirs.iterdir()]
        total_frames.append(frames)
        fields.append(str(frames[0].name).split('_')[0])
        
    all_frames = [] #flatten the list of frames
    for sublist in total_frames:
        all_frames.extend(sublist)
        
    fields=list(dict.fromkeys(fields))
    print(fields)
    
    summary=[]
    for field in fields:
        stars=[f for f in all_frames if field in f.name] #list of frames for scpecific filed
        #get Darks in order to read the image
        NumDarkImages = 9
        MasterDarkList = makeDarkSet(data_path.joinpath('Dark'), NumDarkImages, main_path.joinpath('tmp'), gain)
        folder=stars[int(len(stars) / 2)].parent
        
        dark = chooseDark(folder, MasterDarkList)
        #read mid frame in the list
        try:
            img=importFramesRCD([stars[int(len(stars) / 2)]], 0, 1, dark, gain) #import dark reduced frame as 2d array
        
            star_pos=list(initialFindFITS(img,threshold)) #find stars on 2D array 
        except: #in case of corrupt images
            star_pos=[]
        i=0
        while len(star_pos)<20: #in case of bad frames with no stars
            print("not enough stars!")
            try:
                folder=stars[i].parent #iterate every 1000th frame of the list to find enough stars
                dark = chooseDark(folder, MasterDarkList)
                print(stars[i])
                img=importFramesRCD([stars[i]], 0, 1, dark, gain)
            
                star_pos=list(initialFindFITS(img,threshold))
                print(len(star_pos))
                i+=1000
            except IndexError:
                
                break
        

        coords=fieldCoords(field)
        
        #assuming exposure time is 25ms
        output=f'{field} observed  {len(stars)*0.025/60/60*len(star_pos):.1f}  star-hours, Ra: {coords[0]} dec: {coords[1]}\n'
        print(output)
        summary.append(output)
        
    return summary
        

if __name__ == '__main__':
    print(getStarHour('D:','2023-03-29'))
