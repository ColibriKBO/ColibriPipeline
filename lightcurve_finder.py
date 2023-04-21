# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 12:22:17 2022

@author: Roman A.

Reads time and coordinates from existing occultation detection file, performs photometry, saves data file and plots
lightcurve of 3 telescopes at specific event
"""

from pathlib import Path
import sep
import re
import os
import shutil
from datetime import datetime, date, time
from astropy import wcs
import numpy as np
import numba as nb
from astropy.io import fits
from astroquery.astrometry_net import AstrometryNet
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def findLightCurve(event_dir, mindir_list,telescope):
    '''
    Reads time and coordinates from existing occultation detection file, performs photometry, saves data file and plots
    lightcurve of 3 telescopes at specific event
    
    Parameters
    ----------
    event_dir : path type
        Folder of matched event.
    mindir_list : list of path type objects
        List of observed minutes.
    telescope : str
        Telescope name for detection.

    Returns
    -------
    None.

    '''
    
    d=event_dir #folder with event txts 

    print(f'No {telescope} in {d}') #???
    detection = [f for f in d.iterdir() if '.txt' in f.name ][0] #random detection txt files
    det_time=datetime.strptime(readTime(detection), Tformat) #get event detection time
    print(det_time)
    
    #loop through minute dirs and find a minute that will have closest time to event
    for i in range(len(mindir_list)):
        # if (minute_names[i]>det_time and i==0):
        #     print('search lightcurve in ',minute_names[i])
        minute_name=datetime.strptime(mindir_list[i].name.split('_')[1],format_data)#time of minute dir
        
        #event time in higher than minute time it was detected in
        if minute_name>det_time:
            
            print('search lightcurve in ',mindir_list[i-1])
            lc_path=mindir_list[i-1]
        elif (minute_name<det_time and i==len(mindir_list)-1):
            print('search lightcurve in ',mindir_list[i])
            lc_path=mindir_list[i]
            
    imagePaths = sorted(lc_path.glob('*.rcd')) #path for frames
    headerTimes=[] #store image header times
    try: 
        minuteDir=lc_path
    except:
        print("no times matched")
        
    #make bias set and find bias for this specific event and specific telescope
    bias_dir = [f for f in green_minutesdir.iterdir() if 'Bias' in f.name][0]
    MasterBiasList = makeBiasSet(bias_dir, 10)
    bias = chooseBias(minuteDir, MasterBiasList)
    print("reading image headers...")
    for i in range(len(imagePaths)): #loop to get header times
        
        
        #image file contains both image array and header time
        imageTime = importTimesRCD(imagePaths, i, 1)
        headerTimes.append(imageTime)  #add header time to list
        
    print("selecting frames...")
    for i in range(len(headerTimes)):
        #get time variable of headers
        header_time=datetime.strptime(headerTimes[i][0].split('T')[1][:-3], Tformat) #convert to time variable
        
        if header_time>det_time: #find frame header which is close to event time
        
            
            print('time found:  ',headerTimes[i])
            mid_frame=imagePaths[i]
            mid_index=i #this is where dip happened
            break
            
    try:
        num_of_frames=40
        frames_for_lc=imagePaths[i-num_of_frames:i+num_of_frames] #get 40 frames before and after dip
    except:
        try: #if dip happened in the biginning or end of minute
            num_of_frames=20
            frames_for_lc=imagePaths[i-num_of_frames:i+num_of_frames]
        except:
            num_of_frames=10
            frames_for_lc=imagePaths[i-num_of_frames:i+num_of_frames]
    
    print("creating median combined image for astrometry")
    med_stack,med_stack_path=stackImages(lc_path, matched_dir, mid_index-num_of_frames, 9, bias) #med stack 9 frames of sequence
    
    transform, transform_file=getTransform(med_stack_path) #get wcs tranformation from median frame
    
    star_RaDec=readRAdec(detection) #read star ra dec coords from detection file
    
    star_XY=getXY(transform_file, star_RaDec) #use wcs to get X Y of the star for this med frame
    print(star_XY[0],star_XY[1])
    
    starRaDec=getRAdec(transform_file, star_XY) #tranform again just to be sure
    
    flux=[] #list to store flux
    
    imageFile, headerTimes = importFramesRCD(frames_for_lc, 0, len(frames_for_lc), bias) #import selected frames
    r=3 #aperture rad
    #calculate star fluxes from image
    for frame in imageFile: #do photometry on each image
        flux.append(sep.sum_circle(frame, [star_XY[0]], [star_XY[1]], r, bkgann = (r + 5., r + 8.))[0])
        
    #make directory to save lightcurves in
    
    savefile = d.joinpath('star'+'_' + telescope + ".txt")
    field_name = str(frames_for_lc[0].name).split('_')[0]
    #open file to save results
    with open(savefile, 'w') as filehandle:
        
        #file will have the same format as detection txt
        filehandle.write('#\n#\n#\n#\n')
        filehandle.write('#    First Image File: %s\n' %(frames_for_lc[0]))
        filehandle.write('#    Star Coords: %f %f\n' %(star_XY[0], star_XY[1]))
        filehandle.write('#    RA Dec Coords: %f %f\n'%(starRaDec[0],starRaDec[1]))
        filehandle.write('#    DATE-OBS: %s - %s\n' %(headerTimes[0],headerTimes[-1]))
        filehandle.write('#    Telescope: %s\n' %(telescope))
        filehandle.write('#    Field: %s\n' %(field_name))
        filehandle.write('#\n#\n#\n')
        filehandle.write('#filename     time      flux      conv_flux\n')
      
        
        #loop through each frame to be saved
        for i in range(len(frames_for_lc)):  
           # filehandle.write('%s %f  %f\n' % (files_to_save[i], float(headerTimes[i][0].split(':')[2].split('Z')[0]), star_save_flux[i]))
            filehandle.write('%s  %s  %f\n' % (frames_for_lc[i], headerTimes[i].split('T')[1].split(':')[2][:-3], flux[i]))

    print ("\n")

def plotEvent(detection_dir):
    '''
    Plots occultation lightcurve for 3 telescopes based on txts provided earlier

    Parameters
    ----------
    detection_dir : path
        Path of the matched detection event.

    Returns
    -------
    None.

    '''

    savefiles = [f for f in d.iterdir() if '.txt' in f.name ] #txts that have flux and time data
  
    
    fig, ax1 = plt.subplots()
    for savefile in savefiles:
    
        flux = pd.read_csv(savefile, delim_whitespace = True, 
               names = ['filename', 'time', 'flux', 'conv_flux'], comment = '#')
    

        telescope,sigma=readFile(savefile)   #read telescope name of file and event significance
        coords=readRAdec(savefile)

  
        seconds=[]
        for times in flux['time']:
            
            seconds.append(float(times))
            
        if telescope=="BLUEBIRD":
            color='blue'
        elif telescope=="REDBIRD":
            color='red'
        elif telescope=="GREENBIRD":
            color='green'
        ax1.plot(seconds, flux['flux'],color=color,label=telescope+' RaDec:'+str(coords[0])+' '+str(coords[1])+' Sigma: '+str(sigma))
    # ax1.hlines(med, min(seconds), max(seconds), color = 'black', label = 'median: %i' % med)                                                                                                 
    # ax1.hlines(med + std, min(seconds), max(seconds), linestyle = '--', color = 'black', label = 'stddev: %.3f' % std)
    # ax1.hlines(med - std, min(seconds), max(seconds), linestyle = '--', color = 'black')
   
    ax1.set_xlabel('time (seconds)')
    ax1.set_ylabel('Counts/circular aperture')
    # ax1.set_title('Star [%.1f, %.1f], SNR = %.2f' %( float(coords[0]), float(coords[1]), SNR))
    #ax1.set_xticks(np.arange(min(seconds), max(seconds)+1, 1.0))
    ax1.tick_params(axis="x", labelsize=7)
    ax1.legend()
    plt.title(label=str(datetime.strptime(readDateTime(savefiles[0]), '%Y-%m-%dT%H:%M:%S')))
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(d.joinpath(d.name+'_star_'+telescope + '.png'), bbox_inches = 'tight',dpi=500)
    plt.savefig(d.joinpath(d.name+'_star_'+telescope + '.svg'), bbox_inches = 'tight',dpi=800)
    # plt.show()
    plt.close()
    

def readFile(filepath):                                      #redifinition from Colibri Pipeline's function
   
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            

           

            # #get star coords
            # if i == 5:
            #     star_coords = line.split(':')[1].split(' ')[1:3]
            #     star_x = float(star_coords[0])
            #     star_y = float(star_coords[1])
            
            # #get event time
            # elif i == 7:
            #     event_time = line.split('T')[2].split('\n')[0]
                
            if i == 8:
                telescope = line.split(': ')[1].split('\n')[0]
                
            elif i == 10:
                try:
                    sigma = line.split(':')[1].split('\n')[0].strip(" ")
                except:
                    sigma='-'
                
            # elif i == 11:
            #     star_med = line.split(':')[1].split('\n')[0].strip(" ")
                
            # elif i == 12:
            #     star_std = line.split(':')[1].split('\n')[0].strip(' ')
                
        #reset event frame to match index of the file
        

    #return all event data from file as a tuple    
    return telescope, sigma


def importTimesRCD(imagePaths, startFrameNum, numFrames):
    """
    Reads in frames from .rcd files starting at a specific frame
    
        Parameters:
            imagePaths (list): List of image paths to read in
            startFrameNum (int): Starting frame number
            numFrames (int): How many frames to read in
            bias (arr): 2D array of fluxes from bias image
            
        Returns:
            imagesData (arr): Image data
            imagesTimes (arr): Header times of these images
    """
    
    imagesTimes = []   #array to hold image times
    
    

    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [imagePath for i, imagePath in enumerate(imagePaths) if i >= startFrameNum and i < startFrameNum + numFrames]
    
    for imagePath in files_to_read:
        
        

        header = readRCDheader(imagePath)
        headerTime = header['timestamp']

       

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


        
        imagesTimes.append(headerTime)

    
        
    return imagesTimes

    
    
def readRAdec(filepath):                                    #modified readFile from Colibri Pipeline
    '''read in a .txt detection file and get information from it'''
    
    #make dataframe containing image name, time, and star flux value for the star
    # starData = pd.read_csv(filepath, delim_whitespace = True, 
    #        names = ['filename', 'time', 'flux'], comment = '#')

    
    
    #get header info from file
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
          
            #get star coords
            
                
            if i==6:
                star_coords = line.split(':')[1].split(' ')[1:3]
                star_ra = float(star_coords[0])
                star_dec = float(star_coords[1])
                
        

    #return    
    return (star_ra,star_dec)

def getXY(transform_file, star_RaDec):
    '''get WCS transform ([RA,dec] -> [X,Y]) from astrometry.net header
    input: astrometry.net output file, star position array (x,y)
    returns: coordinate transform'''
    
    #load in transformation information
    transform_im = fits.open(transform_file)
    transform = wcs.WCS(transform_im[0].header)
    
    #get star coordinates from observation image (.npy file)
    
    star_RaDec=np.array(star_RaDec)
    
    #get transformation
  #  world = transform.wcs_pix2world(star_pos, 0)
   # print(world)
    px = transform.all_world2pix(star_RaDec[0],star_RaDec[1],0)
   # print(px)
    
    #optional: save text file with transformation
    # with open(savefile, 'w') as filehandle:
    #     filehandle.write('#\n#\n#\n#\n#X  Y  RA  Dec\n')
    
    #     for i in range(0, len(star_pos)):
    #         #output table: | x | y | RA | Dec | 
    #         filehandle.write('%f %f %f %f\n' %(star_pos[i][0], star_pos[i][1], px[i][0], px[i][1]))
      
    # coords = np.array([star_pos[:,0], star_pos[:,1], px[:,0], px[:,1]]).transpose()
    return px

def getRAdec(transform_file, star_XY):
    '''get WCS transform from astrometry.net header
    input: astrometry.net output file (path object), star position file (.npy path object), filename to save to (path object)
    returns: coordinate transform'''
    
    transform_im = fits.open(transform_file)
    transform = wcs.WCS(transform_im[0].header)
    
    #get star coordinates from observation image (.npy file)
    star_pos = np.array(star_XY)
    
    #get transformation
    world = transform.all_pix2world(star_pos[0],star_pos[1], 0,ra_dec_order=True) #2022-07-21 Roman A. changed solution function to fit SIP distortion
   # print(world)
   # px = transform.wcs_world2pix(world, 0)
   # print(px)
      
    # coords = np.array([star_pos[:,0], star_pos[:,1], world[:,0], world[:,1]]).transpose()
    return world

def getSolution(image_file, save_file, order):
    '''send request to solve image from astrometry.net'''
    '''send request to solve image from astrometry.net
    input: path to the image file to submit, filepath to save the WCS solution header to, order of soln
    returns: WCS solution header'''

    #astrometry.net API
    ast = AstrometryNet()

    #key for astrometry.net account
    ast.api_key = 'vbeenheneoixdbpb'    #key for Rachel Brown's account (040822)
 #   ast.show_allowed_settings()
    wcs_header = ast.solve_from_image(image_file, crpix_center = True, tweak_order = order, force_image_upload=True)
  #  wcs_header = ast.solve_from_image(image_file, tweak_order = order, force_image_upload=True)


    #save solution to file
    if not save_file.exists():
            wcs_header.tofile(save_file)

    return wcs_header

def getTransform(median_image):
    '''get astrometry.net transform for a given minute'''
   
    #get median combined image
    
    
    #get name of transformation header file to save
    transform_file = median_image.with_name(median_image.name.strip('_medstacked.fits') + '_wcs.fits')
    
    #check if the tranformation has already been calculated and saved
    if transform_file.exists():
        
        #open file and get transformation
        wcs_header = fits.open(transform_file)
        transform = wcs.WCS(wcs_header[0])
    #calculate new transformation
    else:
        #get WCS header from astrometry.net plate solution
        wcs_header = getSolution(median_image, transform_file, 4)
    
        #calculate coordinate transformation
        transform = wcs.WCS(wcs_header)
    
    #add to dictionary
    
    
    return transform, transform_file

def stackImages(folder, save_path, startIndex, numImages, bias):
    """
    Make median combined image of first numImages in a directory
    
        Parameters:
            folder (str): Directory of images to be stacked
            save_path (str): Directory to save stacked image in
            startIndex (int): Image starting index
            numImages (int): Number of images to combine
            bias (arr): 2D flux array from the bias image
            
        Returns:
            imageMed (arr): Median combined, bias-subtracted image for star
                            detection
    """
    
   
    #for rcd files:
    '''get list of images to combine'''
    rcdimageFileList = sorted(folder.glob('*.rcd'))         #list of .rcd images
    rcdimages = importFramesRCD(rcdimageFileList, startIndex, numImages, bias)[0]     #import images & subtract bias

    imageMed = np.median(rcdimages, axis=0)          #get median value

    '''save median combined bias subtracted image as .fits'''
    hdu = fits.PrimaryHDU(imageMed) 

    if gain_high:
        medFilepath = save_path.joinpath('high' + folder.name + '_medstacked.fits')     #save stacked image
    else:
        medFilepath = save_path.joinpath('low' + folder.name + '_medstacked.fits')     #save stacked image

    #if image doesn't already exist, save to path
    if not os.path.exists(medFilepath):
        hdu.writeto(medFilepath)
   
    return imageMed,medFilepath

    
def getBias(filepath, numOfBiases):
    """
    Get median bias image from a set of biases
    
        Parameters:
            filepath (str): Filepath to bias image directory
            numOfBiases (int): Number of bias  images to take median of
            
        Return:
            biasMed (arr): Median bias image
    """
    
    print('Calculating median bias...')
    
   

    '''get list of images to combine'''
    rcdbiasFileList = sorted(filepath.glob('*.rcd'))

    #import images, using array of zeroes as bias
    rcdbiases = importFramesRCD(rcdbiasFileList, 0, numOfBiases, np.zeros((2048,2048)))[0]

    '''take median of bias images'''
    biasMed = np.median(rcdbiases, axis=0)
    
    return biasMed

def getDateTime(folder):
    """
    Function to get date and time of folder, then make into python datetime object
    
        Parameters:
            folder (str): Filepath to the folder
            
        Returns:
            folderDatetime (datetime): datetime object of folder date and time
    """

    #time is in format ['hour', 'minute', 'second', 'msec'] 
    folderTime = folder.name.split('_')[-1].strip('/').split('.')  #get time folder was created from folder name

    folderDate = date(int(obs_date.split('-')[0]), int(obs_date.split('-')[1]), int(obs_date.split('-')[2]))
    
    #date is in format ['year', 'month', 'day']
    folderTime = time(int(folderTime[0]), int(folderTime[1]), int(folderTime[2]))       #convert to time object
    
    #combine date and time
    folderDatetime = datetime.combine(folderDate, folderTime)                     #combine into datetime object
    
    return folderDatetime

def chooseBias(obs_folder, MasterBiasList):
    """
    Choose correct master bias by comparing time to the observation time
    
        Parameters:
            obs_folder (str): Filepath to current minute directory
            MaserBiasList (arr): 2D array of [bias datetimes, bias filepaths]
            
        Returns:
            bias (bin): Bitmap of best master bias image
    """
    
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

def makeBiasSet(filepath, numOfBiases):
    """
    Get set of median-combined biases for entire night that are sorted and
    indexed by time, these are saved to disk and loaded in when needed
    
        Parameters:
            filepath (str): Filepath to bias image directory
            numOfBiases (int): Number of bias images to comvine for master
            
        Return:
            biasList (arr): Bias image times and filepaths to saved biases
    """

    biasFolderList = [f for f in filepath.iterdir() if f.is_dir()]
    
    ''' create folders for results '''
    
    day_stamp = obs_date
    save_path = base_path.joinpath('ColibriArchive', str(day_stamp))
    bias_savepath = save_path.joinpath('masterBiases')
    
    if not save_path.exists():
        save_path.mkdir()
        
    if not bias_savepath.exists():
        bias_savepath.mkdir()
        
        
    #make list of times and corresponding master bias images
    biasList = []
    
    #loop through each folder of biases
    for folder in biasFolderList:
        masterBiasImage = getBias(folder, numOfBiases)      #get median combined image from this folder
        
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

def readxbytes(fid, numbytes):
    for i in range(1):
        data = fid.read(numbytes)
        if not data:
            break
    return data
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
def split_images(data,pix_h,pix_v):
    interimg = np.reshape(data, [2*pix_v,pix_h])

    if gain_high:
        image = interimg[1::2]
    else:
        image = interimg[::2]

    return image

def readRCDheader(filename):
    """
    Reads .rcd file
    
        Parameters:
            filename (str): Path to .rcd file
            
        Returns:
            table (arr): Table with image pixel data
            hdict (dic?): Header dictionary
    """

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

    return  hdict

def readRCD(filename):
    """
    Reads .rcd file
    
        Parameters:
            filename (str): Path to .rcd file
            
        Returns:
            table (arr): Table with image pixel data
            hdict (dic?): Header dictionary
    """

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

def importFramesRCD(imagePaths, startFrameNum, numFrames, bias):
    """
    Reads in frames from .rcd files starting at a specific frame
    
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
    
    hnumpix = 2048
    vnumpix = 2048

    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [imagePath for i, imagePath in enumerate(imagePaths) if i >= startFrameNum and i < startFrameNum + numFrames]
    
    for imagePath in files_to_read:
        
        

        data, header = readRCD(imagePath)
        headerTime = header['timestamp']

        images = nb_read_data(data)
        image = split_images(images, hnumpix, vnumpix)
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
    
   

def readTime(filepath):                                    #modified readFile from Colibri Pipeline
    '''read in a .txt detection file and get information from it'''
    
    #make dataframe containing image name, time, and star flux value for the star
    # starData = pd.read_csv(filepath, delim_whitespace = True, 
    #        names = ['filename', 'time', 'flux'], comment = '#')

    # first_frame = int(starData['filename'][0].split('_')[-1].split('.')[0])
    
    #get header info from file
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            

            #get event frame number
            # if i == 4:
            #     event_frame = int(line.split('_')[-1].split('.')[0])

            #get star coords
            
                
            if i==7:
                event_time=line.split(': ')[1].split('T')[1][:-4]
                
        #reset event frame to match index of the file
        # event_frame = event_frame - first_frame

    #return    
    return event_time

def readDateTime(filepath):                                    #modified readFile from Colibri Pipeline
    '''read in a .txt detection file and get information from it'''
    
    #make dataframe containing image name, time, and star flux value for the star
    # starData = pd.read_csv(filepath, delim_whitespace = True, 
    #        names = ['filename', 'time', 'flux'], comment = '#')

    # first_frame = int(starData['filename'][0].split('_')[-1].split('.')[0])
    
    #get header info from file
    with filepath.open() as f:
        
        #loop through each line of the file
        for i, line in enumerate(f):
            

            #get event frame number
            # if i == 4:
            #     event_frame = int(line.split('_')[-1].split('.')[0])

            #get star coords
            
                
            if i==7:
                event_time=line.split(': ')[1].split('.')[0]
                
        #reset event frame to match index of the file
        # event_frame = event_frame - first_frame

    #return    
    return event_time

'''-----------------MAIN-------------------'''

arg_parser = argparse.ArgumentParser(description=""" Run lightcurve finding processing
    Usage:

    """,
    formatter_class=argparse.RawTextHelpFormatter)

arg_parser.add_argument('-b', '--basedir', help='Base directory for data (typically d:)', default='d:')
arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')


cml_args = arg_parser.parse_args()
obsYYYYMMDD = cml_args.date
obs_date = datetime.date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))

cml_args = arg_parser.parse_args()
base_path = Path(cml_args.basedir)

gain_high = True #read only images in high gain

# base_path=Path('/','D:')
# matched_dir=base_path.joinpath('/ColibriArchive',str(obs_date),'matched')
matched_dir=base_path.joinpath('ColibriArchive',str(obs_date),'matched') #directory with matched events


# get list of observed minute dirs
green_minutesdir=base_path.joinpath('ColibriData',str(obs_date).replace('-', ''))
green_minutes=[f for f in green_minutesdir.iterdir() if str(obs_date).replace('-', '') in f.name]
blue_minutesdir=Path('/','B:',str(obs_date).replace('-', ''))
blue_minutes=[f for f in blue_minutesdir.iterdir() if str(obs_date).replace('-', '') in f.name]
red_minutesdir=Path('/','R:',str(obs_date).replace('-', ''))
red_minutes=[f for f in red_minutesdir.iterdir() if str(obs_date).replace('-', '') in f.name]

#get hhmmmss of detections
time_patterns=[times for file in sorted(os.listdir(matched_dir)) if '.txt' in file for times in re.findall("_(\d{6})_", file)]

#create separate folder for each matched star
for pattern in time_patterns:
    if not os.path.exists(matched_dir.joinpath(str(pattern))):
        os.mkdir(matched_dir.joinpath(str(pattern)))

files=[f for f in matched_dir.iterdir() if 'det' in f.name]

for pattern in time_patterns: #copy each event to designated event folder
    for f in files:
        if pattern in f.name:
            
            shutil.copy2(str(f), str(matched_dir.joinpath(str(pattern))))
        
dirs=[d for d in matched_dir.iterdir() if d.is_dir()] #list of event folders

Tformat="%H:%M:%S.%f"
format_data="%H.%M.%S.%f"

for d in dirs:
    event_telescopes=[file.split('_')[-1].split('.')[0] for file in sorted(os.listdir(d))]
    if 'GREENBIRD' not in event_telescopes:
        findLightCurve(d, green_minutes,'GREENBIRD')
        
    elif 'BLUEBIRD' not in event_telescopes:
        findLightCurve(d, blue_minutes,'BLUEBIRD')
        
    elif 'REDBIRD' not in event_telescopes:
        findLightCurve(d, red_minutes,'REDBIRD')
            
    plotEvent(d)
