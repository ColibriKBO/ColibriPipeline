"""
Filename:   wcsmatching.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    June 19, 2022
Updated:    June 20, 2022
    
Usage: python wcsmatching.py <obs_date>
       *This script is intended to run only on BLUEBIRD
"""

# Module Imports
import os,sys
import argparse
import pathlib
import multiprocessing
import gc
import sep
import re
import numpy as np
import time as timer
from datetime import datetime,timedelta
from astropy.convolution import RickerWavelet1DKernel
from astropy.time import Time
from copy import deepcopy
from multiprocessing import Pool

# Custom Script Imports
#import getRAdec
import colibri_image_reader as cir
import colibri_photometry as cp
from coordsfinder import getTransform, getRAdec

# Disable Warnings
import warnings
import logging

#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = pathlib.Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'


#--------------------------------functions------------------------------------#

def npyFileWCSUpdate(npy_file_list):

    # Iterate through npy files and generate a WCS transformation to write
    # back into the WCS file
    for npy_file in npy_file_list:

        print(f"Processing {npy_file.name}")

        # Read in data from npy file
        # Format x | y | half-light radius
        star_table = np.load(npy_file)

        # If npy file already contains ra/dec information
        if star_table.shape[1] == 5: 
            continue

        # Get minute string from filename using regex
        timestamp = re.search("^(\d{6}\._\d{2}\.\d{2}\.\d{2}\.\d{3})", npy_file.name).group(1)
        print(timestamp)

        # Generate WCS transformation and the RA/Dec
        transform  = getTransform(timestamp, [npy_file], {})
        star_radec = getRAdec(transform, star_table)

        # Save the array of star positions as an .npy file again
        # Format: x  |  y  | half-light radius | ra | dec
        npy_file.unlink()
        np.save(npy_file, star_radec)



def sharedStars(telescope1, telescope2, tolerance=1E-2):
    """
    Find matching stars between telescope pairs and between the three (if supplied).

    This alorithm should be upgraded to the "sophisticated solution" of the following:
    https://www.cs.ubc.ca/~liorma/cpsc320/files/closest-points.pdf

    
    telescope1 (arr):
    telescope2 (arr):
    
    """

    # Get number of stars in each array
    num_stars1 = telescope1.shape[0]
    num_stars2 = telescope2.shape[0]

    # Broadcast star coordinates to do numpy operations later
    star1_broadcast = np.tile(telescope1, (num_stars2,1,1))
    star2_broadcast = np.tile(telescope2, (num_stars1,1,1))

    # Subtract the star coordinates from one another, then calculate hypotenuse
    coord_diff = star1_broadcast - star2_broadcast.transpose((1,0,2))
    hypot = np.hypot(coord_diff[:,:,0], coord_diff[:,:,1])

    # Find any stars within tolerance
    # Return should be similar values as telescope2[close_star_inds[1]]
    close_star_inds = np.where(hypot < tolerance)
    return telescope1[close_star_inds[0]]