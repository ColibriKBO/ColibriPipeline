# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:12:59 2021
Update: Jan. 24, 2022, 11:45

@author: Rachel A. Brown

for looking at Colibri pipeline output files
"""


import numpy as np
import matplotlib.pyplot as plt
from glob import glob


def read_plot(filename):
    '''reads .npy file and makes plot of star positions
    input: path to .npy file (path object)
    returns: shows (or saves) plot of star positions found in file'''
    

    star_pos1 = np.load(filename)

    print(star_pos1[:,0])
    
    plt.scatter(star_pos1[:,0], star_pos1[:,1])

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.show()
    plt.close()

    #plt.scatter(range(0, len(star_pos1[:,0])), star_pos1[:,2])
    #plt.xlabel('Star number')
    #plt.ylabel('Half light radius (px)')

    plt.show()
    plt.close()


def to_ds9(filename, savefile):
    '''make .txt file of star coords suitable for plotting in ds9
    input: filepath containing coords (path object), filepath to save new format to (path object)
    returns: saved file'''
    
    star_pos = np.load(filename)
    
    #ds9 coords have bottom left at (1,1), Colibri coords have bottom left at (0,0)
    star_pos_ds9 = star_pos + 1
    
    with open(savefile, 'w') as filehandle:
    
        for line in star_pos_ds9:
            #plot as circles with radius 5 px
            filehandle.write('%s %f %f %f\n' %('circle', line[0], line[1], 5.))

