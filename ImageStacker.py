# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 22:50:14 2023

@author: Roman A.

Stacks 1-minute long images using clipped mean
"""

from pathlib import Path
import numpy as np
from datetime import datetime, date, time
import argparse
from astropy import stats
from astropy.io import fits
import  os

import time as T
import numba as nb

import multiprocessing
from multiprocessing import Pool
import logging

import astrometrynet_funcs


    

def getWCS(image):
    """
    Finds median image that best fits for the time of the detection and uses it to get Astrometry solution.
    Required to have a list of median-combined images (median_combos)

    Parameters
    ----------
    date : str
        Time of the detection file HH.MM.SS.ms

    Returns
    -------
    transform : file headers
        Headers of the WCS file that contain transformation info.

    """
    
    median_str="/mnt/"+str(image).replace(':', '/').replace('\\', '/') #10-12 Roman A.
    median_str=median_str.lower()
    
    #get name of transformation header file to save
    transform_file = image.with_name(image.name.strip('_medstacked.fits') + '_wcs.fits')
    
    transform_str=str(transform_file).split('\\')[-1] #10-12 Roman A.
    

    #get WCS header from astrometry.net plate solution
    soln_order = 4
    
    try:
        wcs_header = astrometrynet_funcs.getLocalSolution(median_str, transform_str, soln_order) #10-12 Roman A.
    except:
        print('could not run localy')
        wcs_header = astrometrynet_funcs.getSolution(image, transform_file, soln_order)

    
    return wcs_header

def chooseBias(obs_folder, MasterBiasList):
    """ choose correct master bias by comparing time to the observation time
    input: filepath to current minute directory, 2D numpy array of [bias datetimes, bias filepaths]
    returns: bias image that is closest in time to observation"""
    
    #current hour of observations
    current_dt = getDateTime(obs_folder)
    
    '''make array of time differences between current and biases'''
    bias_diffs = np.array(abs(MasterBiasList[:,0] - current_dt))
    bias_i = np.argmin(bias_diffs)    #index of best match
    
    '''select best master bias using above index'''
    bias_image = MasterBiasList[bias_i][1]
                
    #load in new master bias image
    bias = fits.getdata(bias_image)
        
    return bias

def chooseBias_Med(img_median, MasterBiasList):
    """ choose correct master bias by comparing time to the observation time
    input: filepath to current minute directory, 2D numpy array of [bias datetimes, bias filepaths]
    returns: bias image that is closest in time to observation"""
    
    
    
    '''make array of time differences between current and biases'''

    bias_diffs = np.array(abs(MasterBiasList[:,2] - img_median))
    bias_i = np.argmin(bias_diffs)    #index of best match
    
    '''select best master bias using above index'''
    bias_image = MasterBiasList[bias_i][1]
                
    #load in new master bias image
    bias = fits.getdata(bias_image)

    return bias

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

def importFramesRCD(parentdir, filenames, start_frame, num_frames, bias, gain):
    """ reads in frames from .rcd files starting at frame_num
    input: parent directory (minute), list of filenames to read in, starting frame number, how many frames to read in, 
    bias image (2D array of fluxes)
    returns: array of image data arrays, array of header times of these images"""
    
    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    hnumpix = 2048
    vnumpix = 2048
    
    #imgain = 'low'
    imgain = gain
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]
    
    for filename in files_to_read:


        data, header = readRCD(filename)
        headerTime = header['timestamp']

        images = nb_read_data(data)
        image = split_images(images, hnumpix, vnumpix, imgain)
        image = np.subtract(image,bias)

        #change time if time is wrong (29 hours)
        hour = str(headerTime).split('T')[1].split(':')[0]
        fileMinute = str(headerTime).split(':')[1]
        dirMinute = str(parentdir).split('_')[1].split('.')[1]
      #  dirMinute = '30'
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #for red: local time is UTC time (don't need +4)
            newLocalHour = int(parentdir.name.split('_')[1].split('.')[0])
        
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

def getBias(filepath, numOfBiases, gain):
    """ get median bias image from a set of biases (length =  numOfBiases) from filepath
    input: bias image directory (path object), number of bias images to take median from (int), gain level ('low' or 'high')
    return: median bias image"""
    
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
 #    '''get list of bias images to combine'''
 #   biasFileList = sorted(filepath.glob('*.fits'))
 #   biases = []   #list to hold bias data
    
 #    '''append data from each bias image to list of biases'''
 #   for i in range(0, numOfBiases):
 #        biases.append(fits.getdata(biasFileList[i]))
 
     
        
    #for rcd files:
    '''get list of images to combine'''
    rcdbiasFileList = sorted(filepath.glob('*.rcd'))
    
    #import images, using array of zeroes as bias
    rcdbiases = importFramesRCD(filepath, rcdbiasFileList, 0, numOfBiases, np.zeros((2048,2048)), gain)[0]
    
    '''take median of bias images'''
    biasMed = np.median(rcdbiases, axis=0)
    
    return biasMed

def makeBiasSet(filepath, numOfBiases, savefolder, gain):
    """ get set of median-combined biases for entire night that are sorted and indexed by time,
    these are saved to disk and loaded in when needed
    input: filepath (string) to bias image directories, number of biases images to combine for master
    return: array with bias image times and filepaths to saved biases on disk"""
    
    biasFolderList = [f for f in filepath.iterdir() if f.is_dir()]   #list of bias folders
    
    ''' create folder for results, save bias images '''
    bias_savepath = savefolder.joinpath(gain + '_masterBiases')

    if not bias_savepath.exists():
        bias_savepath.mkdir()      #make folder to hold master bias images in
        
    #make list of times and corresponding master bias images
    biasList = []
    
    #loop through each folder of biases
    for folder in biasFolderList:
        masterBiasImage = getBias(folder, numOfBiases, gain)      #get median combined image from this folder
        
        #save as .fits file if doesn't already exist
        hdu = fits.PrimaryHDU(masterBiasImage)
        biasFilepath = bias_savepath.joinpath(folder.name + '_' + gain + '_medbias.fits')

        
        if not os.path.exists(biasFilepath):
            hdu.writeto(biasFilepath)
        
        folderDatetime = getDateTime(folder)
        
        biasMed=np.median(masterBiasImage)
        
        biasList.append((folderDatetime, biasFilepath, biasMed))
    
    #package times and filepaths into array, sort by time
    biasList = np.array(biasList)
    ind = np.argsort(biasList, axis=0)
    biasList = biasList[ind[:,0]]
    
    return biasList

def readxbytes(fid,numbytes):
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

def NumpyMean(filelist, gain):
    chunk_stack=[]
    hnumpix = 2048
    vnumpix = 2048
    imgain=gain
    for f in filelist:
        
        fid = open(f, 'rb')
        
        fid.seek(384,0)
        
        table = np.fromfile(fid, dtype=np.uint8, count=12582912)
        testimages=nb_read_data(table)

        image = split_images(testimages, hnumpix, vnumpix, imgain)
        chunk_stack.append(image)
    
    return np.array(np.average(chunk_stack, axis=0))

def clippedMean(filelist, hiclips, loclips,gain):
    '''
    

    Parameters
    ----------
    filelist : list
        List of image file paths.
    hiclips : int
        Number of images with high values to cu.
    loclips : int
        Number of images with low values to cut.
    gain : str
        high or low.

    Returns
    -------
    stack : numpy array
        Array representing stacked image.

    '''
    stackArray = np.zeros([2048,2048])
    hiArray = np.zeros([2048,2048,hiclips+1])
    loArray = np.zeros([2048,2048,loclips+1])
    hiTempArray = np.zeros([2048,2048])
    loTempArray = np.zeros([2048,2048])
    hiLoTempArray = np.zeros([2048,2048])
        
    stackcount = 0
    imgain=gain
    hnumpix = 2048
    vnumpix = 2048
    
    for f in filelist:
        # print(f.name)
        fid = open(f, 'rb')
        
        fid.seek(384,0)
        
        table = np.fromfile(fid, dtype=np.uint8, count=12582912)
        testimages=nb_read_data(table)

        image = split_images(testimages, hnumpix, vnumpix, imgain)
        
        
        if (hiclips > 0) and (loclips == 0):
        
            np.copyto(hiArray[:,:,-1],image)
            hiArray = -np.sort(-hiArray,axis=2)
            np.copyto(hiTempArray,hiArray[:,:,-1])
            stackArray = np.add(stackArray,hiTempArray)
            
        
        if (hiclips == 0) and (loclips > 0):
            np.copyto(loArray[:,:,-1],image)
            loArray = -np.sort(-loArray,axis=2)
            np.copyto(loTempArray,loArray[:,:,0])
            
            if stackcount > loclips:
                stackArray = np.add(stackArray,loTempArray)
        
        if (hiclips == 0) and (loclips == 0):
            stackArray = np.add(stackArray,image)   
        
        stackcount += 1
        
    stack = stackArray/(stackcount-hiclips-loclips)
        
    return stack        

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=""" Run lightcurve finding processing
        Usage:

        """,
        formatter_class=argparse.RawTextHelpFormatter)

    arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')


    cml_args = arg_parser.parse_args()
    obsYYYYMMDD = cml_args.date
    obs_date = date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
    
    gain='high' #gain of images to read
    chunk=40 #number of images to stack for each core
    hiclip_value=1 #number of images with high values to cut (less than 5 is okay)
    loclip_value=0 #number of images with low values to cut (usually don't need)
    NumBiasImages=50 #median stack all biases
    data_path=Path('/','D:','/ColibriData',str(obs_date).replace('-','')) #path for night
    base_path=Path('/','D:') 
    
    bias_save_folder=base_path.joinpath('/StackedData','Biases') #folder to save median stacked biases
    
    if not os.path.exists(bias_save_folder):
        os.makedirs(bias_save_folder)
    
    print("Getting list of Biases...")
    MasterBiasList = makeBiasSet(data_path.joinpath('Bias'), NumBiasImages, bias_save_folder, 
                                 gain) #make list of available biases
    
    minutes=[f for f in data_path.iterdir() if (os.path.isdir(f) and "Bias" not in f.name)]
    #list of minutes in the night
    
    t1=T.time()#time when it all starts
    
    for minute in minutes:
        print(minute)
        start_time = T.time()#time when i minute stacking starts
        files=[f for f in minute.iterdir() if ".rcd" in f.name] #list of files in a minute dir
        field=files[0].name.split('_')[0] #read field number from first image name
    
        try:
            header = readRCD(files[-1])[1] #read header of the last image
        except:
            print("corrupted last frame!")
            continue
            
        stack_time=header['timestamp'] #get time for the stack
        
        # stacked=clippedMean(files,1,0,'high')
        
        # print('Running in parallel...')
        
        # pool_size = multiprocessing.cpu_count() -2
        # pool = Pool(pool_size)
        # # args = ((files[f:f+chunk],hiclip_value,loclip_value,gain)for f in range(0,len(files),chunk))
        # args = ((files[f:f+chunk],gain) for f in range(0,len(files),chunk))
        # # stacked=[]
        # try:
        #     # stacked= pool.starmap(clippedMean,args)
        #     stacked= pool.starmap(NumpyMean,args)
        #     # pool.starmap(clippedMean,args)
        # except:
        #     logging.exception("failed to parallelize")
        
        # pool.close()
        # pool.join()
        arr = np.zeros((2048, 2048))
        for file in files:
            f_time=T.time()
            fid = open(file, 'rb')
            
            fid.seek(384,0)
            
            table = np.fromfile(fid, dtype=np.uint8, count=12582912)
            testimages=nb_read_data(table)

            image = split_images(testimages, 2048, 2048, 'high')
            
            arr=np.add(arr,image)
            print(T.time()-f_time)
        stacked_img=arr/len(files)
        
        # print(len(stacked), stacked[0].shape)
    
        # stacked_img=np.mean(stacked,axis=0)


        flat_img=stacked_img.flatten()
        
        stack_med=stats.sigma_clipped_stats(flat_img, sigma=3, maxiters=100,axis=0)[1]
        print("Stack image median: ",stack_med)

        bias = chooseBias_Med(stack_med, MasterBiasList) #choose best bias according to time
        
        print("Bias median: ", np.median(bias))

        reduced_image = np.subtract(stacked_img,bias) #Image - bias
        
        
        hdu = fits.PrimaryHDU(reduced_image) #save stacked image as .fits
        
        save_path=base_path.joinpath('/StackedData',field,str(obs_date))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Filepath = save_path.joinpath(minute.name+'_clippedmean.fits')
        
        hdu.writeto(Filepath, overwrite=True)
        print("Stack time: ", stack_time)
        fits.setval(Filepath, 'DATE-OBS', value=stack_time)
        
        hdu = fits.PrimaryHDU(bias) #save used bias as .fits
        
        save_path=base_path.joinpath('/StackedData',field,str(obs_date),'Bias')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        Bias_Filepath = save_path.joinpath(minute.name+'_medianbias.fits')
        
        hdu.writeto(Bias_Filepath, overwrite=True)
        
        print("Finished stacking "+minute.name+" in %.2f seconds" % (T.time() - start_time))
        
        print("Calculating WCS for the image...")
        try:
            wcs_headers=getWCS(Filepath)

            hdu = fits.open(Filepath, 'update')
            hdu[0].header.update(wcs_headers)
            hdu.close()
        except:
            print('wcs fail')
            continue
        
    print("Finished stacking night in %.2f seconds" % (T.time() - t1))