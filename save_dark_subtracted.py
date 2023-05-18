"""
Filename:   save_dark_subtracted.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Tue Sep 27 10:25:21 2022
Updated:    Tue Sep 27 10:25:21 2022
    
Usage: python save_dark_subtracted_images.py <path_to_dark_subtracted_images> <path_to_save_directory>
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
from datetime import datetime
from astropy.io import fits
from astropy.time import Time
from copy import deepcopy
from multiprocessing import Pool

# Custom Script Imports
import colibri_image_reader as cir
import colibri_photometry as cp


#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = pathlib.Path('D:')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'

# Timestamp format
TIMESTAMP_FORMAT = '%Y%m%d_%H.%M.%S'

# Get name of telescope
try:
    TELESCOPE = os.environ['COMPUTERNAME']
except KeyError:
    TELESCOPE = "TEST"


#--------------------------------functions------------------------------------#

def check_date_regex(date):
    """Check if date is in YYYYMMDD format"""
    if re.match(r'^\d{8}$', date):
        return date
    else:
        raise argparse.ArgumentTypeError('Date must be in YYYYMMDD format')
    
def date_to_archive_format(date):
    """Convert date from YYYYMMDD to YYYY-MM-DD"""
    return date[:4] + '-' + date[4:6] + '-' + date[6:]

def format_timestamp_to_datetime(timestamp):
    """Convert timestamp from YYYYMMDD_HH.MM.SS to datetime object"""
    # NOTE: This ignores the microseconds and assumes UTC
    return datetime.strptime(timestamp, TIMESTAMP_FORMAT)

def get_bias_timestamps(bias_path):
    """Get timestamps of all bias images in directory for cir.chooseBias"""
    # Get list of all files in directory
    bias_images = bias_path.glob('*.fits')

    # Get timestamps
    timestamps = [(format_timestamp_to_datetime(image.name[:16]), image) 
                  for image in bias_images]

    return timestamps


#------------------------------------main-------------------------------------#

if __name__ == '__main__':


###########################
## Argument Parser Setup
###########################

    # Generate argument parser
    arg_parser = argparse.ArgumentParser(description="Save minute with most stars as dark-subtracted image",
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    # Add arguments
    arg_parser.add_argument('date', type=check_date_regex, 
                            help="Date of observation (YYYYMMDD)")
    arg_parser.add_argument('-t', '--time', 
                            help="Timestamp of minute directory to save. Default is minute with most stars.")

    # Process argparse list as useful variables
    cml_args = arg_parser.parse_args()


###########################
## Path management
###########################

    # Relevant paths
    DATE_PATH = DATA_PATH / cml_args.date
    MBIAS_PATH = ARCHIVE_PATH / date_to_archive_format(cml_args.date) / 'masterBiases'
    STARLIST_PATH = ARCHIVE_PATH / date_to_archive_format(cml_args.date) / f'{TELESCOPE}_done.txt'

    # Check if relevant directories exist
    if not DATE_PATH.exists():
        raise FileNotFoundError(f'Date directory {DATE_PATH} does not exist. No images to process.')
    if not MBIAS_PATH.exists():
        #TODO: Add option to create master biases
        raise FileNotFoundError(f'Master bias directory {MBIAS_PATH} does not exist. Cannot execute script.')
    if (not STARLIST_PATH.exists()) and (cml_args.time is None):
        raise FileNotFoundError(f'Archive directory {STARLIST_PATH} does not exist' +\
                                 'and no time was specified. No images to process.')
    
    # Identify minute to save
    if cml_args.time is None:
        # Read starlist
        starlist = np.loadtxt(STARLIST_PATH, dtype=str)

        # Get minute with most stars
        minute = starlist[np.argmax(starlist[:,1].astype(int)),0]

    else:
        minute = cml_args.time

    # Get path to minute directory
    MINUTE_PATH = DATE_PATH / minute
    if not MINUTE_PATH.exists():
        raise FileNotFoundError(f'Minute directory {MINUTE_PATH} does not exist. No images to process.')
    
    # Create save directory
    SAVE_PATH = IMGE_PATH / cml_args.date / minute
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True)
        print(f'Directory {SAVE_PATH} does not exist. Creating directory.')
    else:
        print(f'Directory {SAVE_PATH} already exists. Cleaning directory.')
        for file in SAVE_PATH.glob('*'):
            file.unlink()


###########################
## Image processing
###########################

    # Select bias image
    bias_timestamps = get_bias_timestamps(MBIAS_PATH)
    best_bias = cir.chooseBias(MINUTE_PATH, np.array(bias_timestamps), minute)
    
    # Read in science images one at a time and save bias-subtracted images as fits files
    image_paths = sorted(MINUTE_PATH.glob('*.rcd')) 
    for i in range(len(image_paths)):
        print(f'Processing image {i+1} of {len(image_paths)}')

        # Read in image and subtract bias
        subtracted_img,_ = cir.importFramesRCD(image_paths, i, 1, best_bias)

        # Save image as fits file
        fits_image_filename = SAVE_PATH / (image_paths[i].name).replace('.rcd', '.fits')
        hdu = fits.PrimaryHDU(subtracted_img)
        hdu.writeto(fits_image_filename, overwrite=True)


    print('Done.')