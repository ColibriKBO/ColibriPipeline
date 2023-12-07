# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 11:12:59 2021
Update: Jan. 24, 2022, 11:45

@author: Rachel A. Brown

for looking at Colibri pipeline output files
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from glob import glob


def read_plot(filename):
    '''reads .npy file and makes plot of star positions
    Should update the polar projection plot according to :
        https://stackoverflow.com/questions/29525356/produce-a-ra-vs-dec-equatorial-coordinates-plot-with-python

    input: path to .npy file (path object)
    returns: shows (or saves) plot of star positions found in file'''
    

    # Extract star data
    star_pos = np.load(filename)

    # Plot detected stars from image
    plt.scatter(star_pos[:,0], star_pos[:,1])

    plt.title('Star Pixel Positions')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(0, 2048)
    plt.ylim(0, 2048)
    plt.grid()
    plt.show()

    # Plot ra and dec of stars on circular plot
    ra = star_pos[:,4]*np.pi/180.
    dec = star_pos[:,5]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='polar')
    ax.scatter(ra, dec)
    ax.set_rmax(90)
    ax.set_rticks([0, 30, 60, 90])
    ax.set_rlabel_position(0)
    ax.grid(True)
    ax.set_title('Star RA and Dec')
    plt.show()



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

