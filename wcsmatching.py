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
import getRAdec
import colibri_image_reader as cir
import colibri_photometry as cp
from coordsfinder import getTransform

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
        # Get minute string from filename
        timestamp = re.search("^(\d{6}\._\d{2}\.\d{2}\.\d{2}\.\d{3})", npy_file.name)

        # Read in data from npy file
        

        # Generate WCS transformation and the RA/Dec
        transform  = getTransform(timestamp, npy_file, {})
        star_radec = getRAdec.getRAdec(transform, star_find_results)

        ## Save the array of star positions as an .npy file
        ## Format: x  |  y  | half light radius | ra | dec
        star_pos_file = BASE_PATH.joinpath('ColibriArchive', str(obs_date), minuteDir.name + '_' + str(detect_thresh) + 'sig_pos.npy')
        if star_pos_file.exists():
            star_pos_file.unlink()
        np.save(star_pos_file, star_radec)
