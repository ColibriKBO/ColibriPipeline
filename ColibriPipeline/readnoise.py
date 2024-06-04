# -*- coding: utf-8 -*-
"""
Author(s):
    Roman A.
Created:
    Tue Jun 21 10:18:43 2022
Updated:
    Thu May 23 2024 by Toni C. Almeida
Usage:
    Calculate read noise from a given image
Updates: 
    Small changes on comments to improve documentation. Toni C. Almeida
"""


import numpy as np
from astropy.io import fits
from pathlib import Path
import math
import lightcurve_maker     #RAB 06212022

'''main function'''
def get_ReadNoise(FirstDark,SecondDark,gain): 
    
    if gain=='high': #gain string to int
        gain=0.82
        
    else:
        gain=18.98
    
    
    diffImage=FirstDark-SecondDark #subtract one image from another. This results in a differential image of the darks.
    dev = np.std(diffImage)         #standard deviation of the differential image on a pixel per pixel basis
    ReadNoise=dev*gain/math.sqrt(2) #read noise by https://www.photometrics.com/wp-content/uploads/2019/10/read-noise-calculator.pdf
    return ReadNoise


if __name__ == '__main__':

    telescope='Green'
    gain='high'
    #ImageIndex=1

    RCDfiles = True             #option to process .rcd files - RAB 06212022

    #base_path = Path('/', 'C:','\\Users', 'Admin', 'Desktop', 'Colibri', 'GREEN-RED',telescope) #path to main directory
    base_path = Path('/', 'home', 'rbrown', 'Documents', 'Colibri', telescope)  #RAB 06212022


    #dark_path=base_path.joinpath('Dark')                                    #path to dark data
    dark_path = base_path.joinpath('ColibriData', '20210804', 'Dark', '20210804_00.42.30.541')  #RAB 06212022
    #darkFileList = sorted(dark_path.glob('*.fit'))                          #get dark files list
    darkFileList = sorted(dark_path.glob('*.rcd'))         #RAB 06212022

    #FirstDark=fits.getdata(darkFileList[ImageIndex])*(-1)*(-1) #convert uint to int by multiplying by (-1)
    #SecondDark=fits.getdata(darkFileList[ImageIndex+1])*(-1)*(-1)


    ReadNoise=[] #create an array of all read noises
    pairs=0     #number of pairs that will be iterrated


    for i in range(0,len(darkFileList),2): #iterrate through all pairs of subsequent dark images without dublicates

        
        #added if/else statement to give option for .rcd files - RAB 06212022
        if RCDfiles == False:
            FirstDark=fits.getdata(darkFileList[i])*(-1)*(-1) #convert uint16 to int32 by multiplying by (-1) and back
            SecondDark=fits.getdata(darkFileList[i+1])*(-1)*(-1)
        
        else:
            #for .rcd files - RAB 06212022:
            FirstDark = lightcurve_maker.importFramesRCD(dark_path, darkFileList, i, 1, np.zeros((2048,2048)), gain)[0]
            SecondDark = lightcurve_maker.importFramesRCD(dark_path, darkFileList, i+1, 1, np.zeros((2048,2048)), gain)[0]

        
        ReadNoise.append(get_ReadNoise(FirstDark,SecondDark,gain))
        pairs+=1
        
        
    AverageRN=np.average(ReadNoise) #average read noise
    StdRn=np.std(ReadNoise)         #standard deviation of read noise

    print(pairs," pairs iterrated")
    print("READ NOISE (",gain,"gain ) :",'{:.4}'.format(AverageRN)+'+/-'+'{:.2}'.format(StdRn))