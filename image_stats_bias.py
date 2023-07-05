#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 11:53:28 2021
Updated on Tues. June 21, 2022
Read in all bias images from a single night, get the median value, save a text file with 
filename | time | median | mean | mode | sensor temp | base temp | FPGA temp

@author: Rachel A. Brown

17-07-2022 modified by Roman A. : added readnoise.py into this script, enabling to see read noise of each directory in
            the output txt. Modified date folder input method so that the script can be run easier from powershell
            
14-08-2022 modified by Roman A. : read noise is now written to separate txt file, the script produces 2 graphs
       that show how bias mean and median varies.
"""

import numpy as np
import numba as nb
import pandas as pd
import scipy.stats 
import binascii
import datetime
import pathlib
import lightcurve_maker
import math

import matplotlib.pyplot as plt
from pathlib import Path
import os
import argparse

#Mike's rcd section ---------------------------------------------------------
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

   # width = 2048
   # height = 2048

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
        
        #sensor temperature of camera
        fid.seek(91,0)
        hdict['sensortemp'] = readxbytes(fid, 2)
        
        #sensor temperature of camera
        fid.seek(141,0)
        hdict['basetemp'] = readxbytes(fid, 2)
        
        #sensor temperature of camera
        fid.seek(143,0)
        hdict['FPGAtemp'] = readxbytes(fid, 2)

        # Timestamp
        fid.seek(152,0)
        hdict['timestamp'] = readxbytes(fid, 29).decode('utf-8')

        # Load data portion of file
        fid.seek(384,0)

        table = np.fromfile(fid, dtype=np.uint8, count=12582912)

    return table, hdict
#--------------------------------------------------------------------------------------------



def importFramesRCD(parentdir, filenames, start_frame, num_frames, bias, gain):
    """ reads in frames from .rcd files starting at frame_num
    input: parent directory (minute), list of filenames to read in, starting frame number, how many frames to read in, 
    bias image (2D array of fluxes)
    returns: array of image data array, array of header time, tuple of sensor temperatures"""
    
    imagesData = []    #array to hold image data
    imagesTimes = []   #array to hold image times
    
    hnumpix = 2048      #height of image [px]
    vnumpix = 2048      #width of image [px]
    
    imgain = gain       #gain level to use
    
    '''list of filenames to read between starting and ending points'''
    files_to_read = [filename for i, filename in enumerate(filenames) if i >= start_frame and i < start_frame + num_frames]
    
    for filename in files_to_read:

        #read in pixel data and header
        data, header = readRCD(filename)
        headerTime = header['timestamp']
        sensorTemp = float(int(binascii.hexlify(header['sensortemp']), 16))/10.
        baseTemp = float(int(binascii.hexlify(header['basetemp']), 16))/10.
        FPGAtemp = float(int(binascii.hexlify(header['FPGAtemp']), 16))/10.

        #get image with correct gain from data
        images = nb_read_data(data)
        image = split_images(images, hnumpix, vnumpix, imgain)      
        #image = np.subtract(image,bias)

        #change time if time is wrong (29 hours)
        hour = str(headerTime).split('T')[1].split(':')[0]
        fileMinute = str(headerTime).split(':')[1]
        dirMinute = str(parentdir).split('_')[1].split('.')[1]
        
        #check if hour is bad, if so take hour from directory name and change header
        if int(hour) > 23:
            
            #directory name has local hour, header has UTC hour, need to convert (+4)
            #for red: local time is UTC time (don't need +4)
            newLocalHour = int(parentdir.name.split('_')[1].split('.')[0])
        
            if int(fileMinute) < int(dirMinute):
               # newUTCHour = newLocalHour + 4 + 1     #add 1 if hour changed over during minute
                newUTCHour = newLocalHour + 1          #if telescope in UTC
            else:
               # newUTCHour = newLocalHour + 4
                newUTCHour = newLocalHour              #if telescope in UTC
        
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
        
    return imagesData, imagesTimes, (sensorTemp, baseTemp, FPGAtemp)

#-------------------------Read noise section copied from readnoise.py----------------------#17-07-2022 Roman A.
def get_ReadNoise(FirstBias,SecondBias,gain): 
    
    if gain=='high': #gain string to int
        gain=0.82
        
    else:
        gain=18.98
    
    
    diffImage=FirstBias-SecondBias #subtract one image from another. This results in a differential image of the biases.
    dev = np.std(diffImage)         #standard deviation of the differential image on a pixel per pixel basis
    ReadNoise=dev*gain/math.sqrt(2) #read noise by https://www.photometrics.com/wp-content/uploads/2019/10/read-noise-calculator.pdf
    return ReadNoise

#--------------------------------------------------------------------------------
telescope = os.environ['COMPUTERNAME']       #identifier for telescope
gain = 'high'           #gain level for .rcd files ('low' or 'high')

if __name__ == '__main__':
   # if len(sys.argv) > 1:
   #     date = sys.argv[1]   #night directory
   
    '''--------------observation & solution info----------------'''
    #obs_date = datetime.date(2022, 7, 6)           #date of observation
#    #print('Telescope: Green, gain: HIGH\n')
#    obs_date=input("Input obesrvation date (ex. 2022-07-30): ")  #17-07-2022 Roman A.
#    telescope = 'Blue'                             #telescope identifier
#    gain = 'high'           #keyword for gain - 'high' or 'low'
    
    try:
        #2022.09.12 Roman A. --- modifications for automation
        arg_parser = argparse.ArgumentParser(description=""" Run image stats script.
    		Usage:

    		""",formatter_class=argparse.RawTextHelpFormatter)

        arg_parser.add_argument('-b', '--basedir', help='Base directory for data (typically d:)', default='d:')
        arg_parser.add_argument('-d', '--date', help='Observation date (YYYY/MM/DD) of data to be processed.')

        cml_args = arg_parser.parse_args()

        base_path = pathlib.Path(cml_args.basedir)
        obsYYYYMMDD = cml_args.date
        obs_date = datetime.date(int(obsYYYYMMDD.split('/')[0]), int(obsYYYYMMDD.split('/')[1]), int(obsYYYYMMDD.split('/')[2]))
        
        '''------------set up paths to directories------------------'''

        #base_path = pathlib.Path('/', 'D:')

        data_path = base_path.joinpath('/ColibriData', str(obs_date).replace('-', ''), 'Bias')    #path to bias directories
        save_path = base_path.joinpath('/ColibriArchive', str(obs_date).replace('-','') + '_diagnostics', 'Bias_Stats')  #path to save results to
        
        save_path.mkdir(parents=True, exist_ok=True)        #make save directory if it doesn't already exist

        save_filepath = save_path.joinpath(str(obs_date) + '_stats.txt')        #.txt file to save results to
        
        minutedirs = [f for f in data_path.iterdir() if f.is_dir()]          #all minutes in night bias directory

        #create file and make header
        with open(save_filepath, 'w') as filehandle:
                    filehandle.write('filename    time    med    mean    mode    RMS      sensorTemp    baseTemp    FPGAtemp\n')  
                    
        readnoise_filepath = save_path.joinpath(str(obs_date) + '_readnoise.txt') # 08-15 Roman A. file for read noise data
        with open(readnoise_filepath, 'a') as filehandle:
                filehandle.write('minute    readnoise\n')

        print('number of subdirectories: ', len(minutedirs))
        
        '''----------loop through each minute and each image, calculate stats------------------'''
        Average_readnoises=[]
        for minute in minutedirs:
            print('------------------------------------')
            print('working on: ', minute.name)
            images = sorted(minute.glob('*.rcd'))   #list of images
            biasFileList=images
            pairs=0 #number of pairs that will be iterrated
            ReadNoise=[] 
            
             
            Average_stds=[]

            '''----------------------------READ OUT NOISE SECTION------------------------------------'''    
            #17-07-2022 Roman A.
            
            if len(biasFileList)% 2 == 0: #check for even or odd number of images in the directory
            
                for i in range(0,len(biasFileList),2): #iterrate through all pairs of subsequent bias images without dublicates
        
            
           
                #getting images for .rcd files - RAB 06212022:
                    FirstBias = lightcurve_maker.importFramesRCD(minute, biasFileList, i, 1, np.zeros((2048,2048)), gain)[0]
                    SecondBias = lightcurve_maker.importFramesRCD(minute, biasFileList, i+1, 1, np.zeros((2048,2048)), gain)[0]
        
                #get list of readnoises 
                    ReadNoise.append(get_ReadNoise(FirstBias,SecondBias,gain))
                    pairs+=1
                print(pairs," pairs iterrated")
                AverageRN=np.average(ReadNoise) 
                StdRn=np.std(ReadNoise)         #standard deviation of read noise
                
                Average_readnoises.append(AverageRN)
                Average_stds.append(StdRn)
                
            else:
                
                for i in range(0,len(biasFileList)-1,2):
        
            
                    FirstBias = lightcurve_maker.importFramesRCD(minute, biasFileList, i, 1, np.zeros((2048,2048)), gain)[0]
                    SecondBias = lightcurve_maker.importFramesRCD(minute, biasFileList, i+1, 1, np.zeros((2048,2048)), gain)[0]
        
            
                    ReadNoise.append(get_ReadNoise(FirstBias,SecondBias,gain))
                    pairs+=1
                print(pairs," pairs iterrated")
                AverageRN=np.average(ReadNoise)
                StdRn=np.std(ReadNoise) 
                
                Average_readnoises.append(AverageRN)
                Average_stds.append(StdRn)
            
            with open(readnoise_filepath, 'a') as filehandle:
                    filehandle.write('%s %f\n' %(minute, AverageRN))
            print("READ NOISE: ",'{:.4}'.format(AverageRN)+'+/-'+'{:.2}'.format(StdRn))
            
            
            '''----------------------------READ OUT NOISE SECTION END------------------------------------'''
            
            #loop through each image in minute directory
            for image in images:
                
                #import image
                data, time, temps = importFramesRCD(minute, [image], 0, 1, np.zeros((2048,2048)), gain)
                
                #flatten 2D array into 1D array for easier computation
                data_flat = data.flatten()
                
                #image statistics
                med = np.median(data_flat)                              #median pixel value of image
                mean = np.mean(data_flat)                               #mean pixel value of image
                mode = float(scipy.stats.mode(data_flat, axis = None).mode)   #mode pixel value of image
                
                RMS = np.sqrt(np.mean(np.square(data_flat)))            #RMS of entire image
                            
                #append image stats to file
                with open(save_filepath, 'a') as filehandle:
                    filehandle.write('%s %s %f %f %f %f %f %f %f\n' %(image, time[0], med, mean, mode, RMS, temps[0], temps[1], temps[2]))
            
        with open(readnoise_filepath, 'a') as filehandle: #write read noise data in txt file
                    filehandle.write('Total average Read Noise: %f\n' %(np.average(Average_readnoises)))
        print(" Total READ NOISE: ",'{:.4}'.format(np.average(Average_readnoises))+'+/-'+'{:.2}'.format(np.average(Average_stds)))
        
        '''-----------------------------Building graphs-------------------------------------'''
        
            #%%
        #one night version
        
        
        scope = telescope
        #night = '2022-08-06'
        
        print(save_filepath)
        data_file=save_path
        #data = pd.read_csv('./meanTests/' + night + '_mean_' + scope + '.txt', delim_whitespace = True, nrows = 16764)
        data = pd.read_csv(data_file.joinpath(str(obs_date)+'_stats.txt'), delim_whitespace = True)
        #data = pd.read_csv(('D:/ColibriArchive/20220731_diagnostics/Bias_Stats/2022-07-31_stats.txt'), delim_whitespace = True)
        data[['day','hour']] = data['time'].str.split('T', expand = True)

        #find time breaks in data (when a new folder was created)
        
        #get list of different bias folders, the indices of where these start in the data frame
        folders = data['filename'].str.split('\\', expand = True)[1]
        minutes = data['hour'].str.split(':', expand = True)[1]
        #folders, index = np.unique(folders, return_index = True)
        folders, index = np.unique(minutes, return_index = True)
        labels = data['hour'][index]
        labels = labels.str.split('.', expand = True)[0]
        
            #%%
        #plot with single nights data with temperature
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize = (16,8), gridspec_kw=dict(hspace = 0.05))
        
        #minimum and maximum value in one night
        lower_m = -3 #data['mean'].min()
        upper_m = 3 #data['mean'].max()
        
        ax1.scatter(data['hour'], np.mean(data['mean']) - data['mean'], label = 'mean', s = 2)
        
        ax1.set_title(scope + ' Bias Levels - ' + data.loc[0]['day'])
        ax1.set_ylabel('(Overall mean) - (mean bias pixel value)')
        ax1.vlines(index, lower_m, upper_m, color = 'black', linewidth = 1)
        ax1.set_xticks([])
        ax1.set_xticklabels([])
        ax1.set_ylim(np.min(np.mean(data['mean']) - data['mean'])-0.5,np.amax(np.mean(data['mean']) - data['mean'])+0.5) #modified to see whole Graph
        
        #ax1.legend()
        
        lower_t = -3 #48
        upper_t = 3 #52
        #ax2.scatter(data['hour'], data['baseTemp'], label = 'Base temp', s = 2)
        ax2.scatter(data['hour'], np.mean(data['FPGAtemp']) - data['FPGAtemp'], label = 'FGPA temp', s = 2)
        
        ax2.vlines(index, np.min(np.mean(data['FPGAtemp']) - data['FPGAtemp']), np.amax(np.mean(data['FPGAtemp']) - data['FPGAtemp']), color = 'black', linewidth = 1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('(mean temperature) - temperature (C)')
        ax2.set_xticks(index)
        ax2.set_xticklabels(labels,rotation=90)
        ax2.set_ylim(np.min(np.mean(data['FPGAtemp']) - data['FPGAtemp'])-0.5, np.amax(np.mean(data['FPGAtemp']) - data['FPGAtemp'])+0.5) #modified to see whole Graph
        
        #ax2.legend()
        
        plt.savefig(data_file.joinpath( str(obs_date) + 'meanofmean_bias_stats_' + scope + '.png'),bbox_inches = "tight",dpi=300)
    #    plt.show()
        plt.close()
            
        
            #%%
        lower = 70
        upper = 78
        #plot with single nights data of mean and median
        plt.scatter(data['hour'], data['med'], label = 'median', s = 2)
        plt.scatter(data['hour'], data['mean'], label = 'mean', s = 2)
        #plt.scatter(data['hour'], data['mode'], label = 'mode', s = 2)
        
        
        
        plt.title(scope + ' biases - ' + data.loc[0]['day'])
        plt.ylabel('image pixel value')
        plt.vlines(index, np.min(data['med'])-0.5, np.amax(data['med'])+0.5, color = 'black', linewidth = 1) #modified to see whole Graph
        plt.xlabel('time')
        plt.xticks(index, labels,rotation=20,fontsize=7)
        #plt.ylim(lower-0.2, upper+0.2)
        
        plt.legend()
        
        
        plt.savefig(data_file.joinpath(scope + '.png'),dpi=300)
    #    plt.show()
        plt.close()


    except TypeError:
        print("TypeError: Check your data structure. There may be a repeating header in the file.")