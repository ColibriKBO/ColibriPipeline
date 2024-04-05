# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 10:18:43 2022

@author: Roman A.
"""


import numpy as np
from astropy.io import fits
from pathlib import Path
import math
import lightcurve_maker     #RAB 06212022

'''main function'''
def get_ReadNoise(FirstBias,SecondBias,gain): 
    
    if gain=='high': #gain string to int
        gain=0.82
        
    else:
        gain=18.98
    
    
    diffImage=FirstBias-SecondBias #subtract one image from another. This results in a differential image of the biases.
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


    #bias_path=base_path.joinpath('Bias')                                    #path to bias data
    bias_path = base_path.joinpath('ColibriData', '20210804', 'Bias', '20210804_00.42.30.541')  #RAB 06212022
    #biasFileList = sorted(bias_path.glob('*.fit'))                          #get bias files list
    biasFileList = sorted(bias_path.glob('*.rcd'))         #RAB 06212022

    #FirstBias=fits.getdata(biasFileList[ImageIndex])*(-1)*(-1) #convert uint to int by multiplying by (-1)
    #SecondBias=fits.getdata(biasFileList[ImageIndex+1])*(-1)*(-1)


    ReadNoise=[] #create an array of all read noises
    pairs=0     #number of pairs that will be iterrated


    for i in range(0,len(biasFileList),2): #iterrate through all pairs of subsequent bias images without dublicates

        
        #added if/else statement to give option for .rcd files - RAB 06212022
        if RCDfiles == False:
            FirstBias=fits.getdata(biasFileList[i])*(-1)*(-1) #convert uint16 to int32 by multiplying by (-1) and back
            SecondBias=fits.getdata(biasFileList[i+1])*(-1)*(-1)
        
        else:
            #for .rcd files - RAB 06212022:
            FirstBias = lightcurve_maker.importFramesRCD(bias_path, biasFileList, i, 1, np.zeros((2048,2048)), gain)[0]
            SecondBias = lightcurve_maker.importFramesRCD(bias_path, biasFileList, i+1, 1, np.zeros((2048,2048)), gain)[0]

        
        ReadNoise.append(get_ReadNoise(FirstBias,SecondBias,gain))
        pairs+=1
        
        
    AverageRN=np.average(ReadNoise) #average read noise
    StdRn=np.std(ReadNoise)         #standard deviation of read noise

    print(pairs," pairs iterrated")
    print("READ NOISE (",gain,"gain ) :",'{:.4}'.format(AverageRN)+'+/-'+'{:.2}'.format(StdRn))