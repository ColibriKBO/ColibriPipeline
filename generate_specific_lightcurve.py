"""
Filename:   generate_specific_lightcurve.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Thu May 25, 2022
Updated:    Thu May 25, 2022
    
Usage: python generate_specific_lightcurve.py <obs_date> <time_of_interest> [RA,DEC]
"""

# Module Imports
import os,sys
import argparse
import pathlib
import multiprocessing
import datetime
import gc
import sep
import numpy as np
import time as timer
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
#warnings.filterwarnings("ignore",category=DeprecationWarning)
#warnings.filterwarnings("ignore",category=VisibleDeprecationWarning)

#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = pathlib.Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'

# Timestamp format
OBSDATE_FORMAT = '%Y%m%d'
MINDIR_FORMAT  = '%Y%m%d_%H.%M.%S.%u'
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%u'

# Processing parameters
POOL_SIZE  = multiprocessing.cpu_count() - 2  # cores to use for multiprocessing
CHUNK_SIZE = 10  # images to process at once
SEC_TO_SAVE = 1.  # seconds on either side of event to save

# Photometry parameters
EXCLUDE_IMAGES = 1
STARFINDING_STACK = 9
APERTURE_RADIUS = 3.0
EXPOSURE_TIME = 0.025
STAR_DETEC_THRESH = 4.0

# Drift limits
DRIFT_TOLERANCE = 1.0  # maximum drift (in px/s) without throwing an error
DRIFT_THRESHOLD = 0.025  # maximum drif (in px/s) without applying a drift correction


#--------------------------------functions------------------------------------#

def generateLightcurve(minute_dir, central_frame, master_bias_list, obsdate):

    ## Setup image pathing ## 

    # Get the dark image data
    dark = cir.chooseBias(minute_dir, master_bias_list, obsdate)

    # Get image dimensions & number of images
    x_length, y_length, num_images = cir.getSizeRCD(image_paths)

    # Determine frames to save
    min_frame = central_frame - (SEC_TO_SAVE // EXPOSURE_TIME)
    max_frame = central_frame + 1 + (SEC_TO_SAVE // EXPOSURE_TIME)
    if min_frame < EXCLUDE_IMAGES:
        min_frame = EXCLUDE_IMAGES
    if max_frame >= num_images:
        max_frame = None

    # Get image paths to process
    image_paths = sorted(minute_dir.glob('*.rcd'))
    num_images = len(image_paths)
    lightcurve_paths = image_paths[min_frame:max_frame]


    ## Star identification ##

    # Get the star list from the archive (if not found, fail gracefully)
    # TODO: Add a way to generate a star list if it doesn't exist
    star_list_path = (ARCHIVE_PATH / str(obs_date)).glob(minute_dir.name + '_*sig_pos.npy')
    if len(star_list_path) == 0:
        print(f"ERROR: No star list found for {minute_dir.name}")
        return None
    else:
        star_list_path = star_list_path[0]
        star_list = np.load(star_list_path)

        # Get the star positions & radii
        num_stars = len(star_list)
        initial_positions = star_list[:, [0, 1]]
        initial_radii = star_list[:, 2]
        mean_guass_sigma = 2*np.mean(initial_radii) / 2.355


    ## Setup drift correction ##

    # Read in first and final frames
    first_frame, first_time = cir.importFramesRCD(image_paths, EXCLUDE_IMAGES, 1, dark)
    final_frame, final_time = cir.importFramesRCD(image_paths, num_images-1, 1, dark)

    # Refine star positions for first and final frames
    first_positions, _ = cp.refineCentroid(first_frame, first_time[0], mean_guass_sigma)
    final_positions, _ = cp.refineCentroid(final_frame, final_time[0], mean_guass_sigma)

    # Calculate median drift rate [px/s] in x and y over the minute
    x_drift, y_drift = cp.averageDrift(first_positions, final_positions, first_time, final_time)

    # Check if drift is within tolerance and if a correction is needed
    if (abs(x_drift) > DRIFT_TOLERANCE) or (abs(y_drift) > DRIFT_TOLERANCE):
        print(f"WARNING: Drift rate is too high for {minute_dir.name}")
        return None
    elif (abs(x_drift) > DRIFT_THRESHOLD) or (abs(y_drift) > DRIFT_THRESHOLD):
        drift = True
    else:
        drift = False

    
    ## Generate WCS transformations ##

    


def findMinute(obsdate, timestamp):

    # Get all minute directories for the given date
    obs_path = DATA_PATH / obsdate
    if not obs_path.exists():
        print(f"ERROR: {obs_path} does not exist")
        return None
    else:
        minute_dirs = [item for item in obs_path.iterdir() if item.is_dir()]
        dir_timestamps = [datetime.datetime.strptime(item.name+'000', MINDIR_FORMAT)
                          for item in minute_dirs]

    # Find the minute directory that contains the timestamp
    timestamp = datetime.datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    for dir_timestamp in dir_timestamps:
        if dir_timestamp <= timestamp < dir_timestamp + datetime.timedelta(minutes=1):
            target_dir = minute_dirs[dir_timestamps.index(dir_timestamp)]

            # Calculate the number of frames to skip to line up with the directory
            timediff = timestamp - dir_timestamp
            skip_frame = timediff.total_seconds() // EXPOSURE_TIME
            break

    return target_dir, skip_frame


def reversePixelMapping(minute_dir, star_radec, obsdate):

    # Get the list of medstacked images
    # TODO: Add a way to generate a medstack if it doesn't exist
    medstack_list_path = (ARCHIVE_PATH / str(obs_date)).glob('*_medstacked.fits')

    # Get the WCS transformation
    transform = getTransform(minute_dir.name, medstack_list_path)

    # Find the coordinates of the stars in the medstack 
    star_wcs = getRAdec.getXYSingle(transform, star_radec)
    star_X   = star_wcs[0]
    star_Y   = star_wcs[1]

    return star_X, star_Y