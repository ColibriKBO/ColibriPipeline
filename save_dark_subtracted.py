"""
Filename:   save_dark_subtracted.py
Author(s):  Peter Quigley
Contact:    pquigley@uwo.ca
Created:    Tue Sep 27 10:25:21 2022
Updated:    Tue Sep 27 10:25:21 2022
    
Usage: python save_dark_subtracted_images.py <path_to_dark_subtracted_images> <path_to_save_directory>
"""

# Module Imports
import os
import argparse
import pathlib
import re
import numpy as np
import pandas as pd
from datetime import datetime
from astropy.io import fits
from multiprocessing import cpu_count,Pool

# Custom Script Imports
import colibri_image_reader as cir
import colibri_photometry as cp


#-------------------------------global vars-----------------------------------#

# Path variables
BASE_PATH = pathlib.Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'

# Timestamp format
OBSDATE_FORMAT = '%Y%m%d'
TIMESTAMP_FORMAT = '%Y%m%d_%H.%M.%S'

# REGEX format
DET_REGEX = r'det_(\d{4}-\d{2}-\d{2})_(\d{6}_\d{6})'

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

def process_science_images(image_path_list, start_frame, bias_data, save_dir_path):
    """Read in science images one at a time and save bias-subtracted images as fits files"""
    #print(f'Processing image {start_frame+1} of {len(image_path_list)}')

    # Read in image and subtract bias
    subtracted_img,_ = cir.importFramesRCD(image_path_list, start_frame, 1, bias_data)

    # Save image as fits file
    fits_image_filename = save_dir_path / (image_path_list[start_frame].name).replace('.rcd', '.fits')
    hdu = fits.PrimaryHDU(subtracted_img)
    hdu.writeto(fits_image_filename, overwrite=True)


def parse_det_file(det_file):
    """Parse detection file name into date, timestamp, and list of 
    detection files"""

    # Check that the detection file exists
    if not det_file.exists():
        raise FileNotFoundError(f'Detection file {det_file} does not exist.')

    # Read the detection file
    det_df = pd.read_csv(det_file, sep='\s+', header=None, comment='#',
                         names=['filename', 'time', 'flux', 'conv_flux'])
    
    # Get parent directory of one of the detection files
    minute = det_df['filename'][0].split('\\')[-2]

    # Return detection files as list
    return minute,det_df['filename'].tolist()


#------------------------------------main-------------------------------------#

def _save_dark_subtracted_minute(date, minute):
    """Save dark-subtracted images from a minute directory as fits files
    *NOTE: This function must be called after main"""


    # Get path to minute directory
    MINUTE_PATH = DATE_PATH / minute
    if not MINUTE_PATH.exists():
        raise FileNotFoundError(f'Minute directory {MINUTE_PATH} does not exist. No images to process.')
    
    # Create save directory
    SAVE_PATH = IMGE_PATH / date / minute
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True)
        print(f'Directory {SAVE_PATH} does not exist. Creating directory.')
    else:
        print(f'Directory {SAVE_PATH} already exists. Cleaning directory.')
        for file in SAVE_PATH.glob('*'):
            file.unlink()


    ## Image processing

    # Select bias image
    bias_timestamps = get_bias_timestamps(MBIAS_PATH)
    best_bias = cir.chooseBias(MINUTE_PATH, np.array(bias_timestamps), obs_date)
    
    # Get list of all files in directory
    image_paths = sorted(MINUTE_PATH.glob('*.rcd')) 

    # Set up multiprocessing
    pool_size = cpu_count() - 2
    pool = Pool(pool_size)
    args = ((image_paths, i, best_bias, SAVE_PATH) for i in range(len(image_paths)))

    # Process images in parallel
    pool.starmap(process_science_images, args)

    # Close multiprocessing
    pool.close()
    pool.join()
    print('Done.')


def _save_dark_subtracted_detec(date, minute, detec_str, file_list):
    """Save dark-subtracted images from a minute directory as fits files
    *NOTE: This function must be called after main"""


    # Get path to minute directory
    MINUTE_PATH = DATE_PATH / minute
    if not MINUTE_PATH.exists():
        raise FileNotFoundError(f'Minute directory {MINUTE_PATH} does not exist. No images to process.')
    
    # Create save directory
    SAVE_PATH = IMGE_PATH / date / detec_str
    if not SAVE_PATH.exists():
        SAVE_PATH.mkdir(parents=True)
        print(f'Directory {SAVE_PATH} does not exist. Creating directory.')
    else:
        print(f'Directory {SAVE_PATH} already exists. Cleaning directory.')
        for file in SAVE_PATH.glob('*'):
            file.unlink()


    ## Image processing

    # Select bias image
    bias_timestamps = get_bias_timestamps(MBIAS_PATH)
    best_bias = cir.chooseBias(MINUTE_PATH, np.array(bias_timestamps), obs_date)
    
    # Get list of all files in directory
    image_paths = [pathlib.Path(file) for file in file_list]

    # Set up multiprocessing
    pool_size = cpu_count() - 2
    pool = Pool(pool_size)
    args = ((image_paths, i, best_bias, SAVE_PATH) for i in range(len(image_paths)))

    # Process images in parallel
    pool.starmap(process_science_images, args)

    # Close multiprocessing
    pool.close()
    pool.join()
    print('Done.')



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
    arg_parser.add_argument('-d','--det',
                            help="Detection file to save. Incompatible with -t option")

    # Process argparse list as useful variables
    cml_args = arg_parser.parse_args()

    # Make sure only one or the other option is used
    if (cml_args.time is not None) and (cml_args.det is not None):
        raise argparse.ArgumentTypeError('Cannot use both -t and -d options')


###########################
## Path management
###########################

    # Format date as datetime object
    obs_date = datetime.strptime(cml_args.date, OBSDATE_FORMAT)

    # Relevant paths
    DATE_PATH = DATA_PATH / cml_args.date
    MBIAS_PATH = ARCHIVE_PATH / date_to_archive_format(cml_args.date) / 'masterBiases'
    STARLIST_PATH = ARCHIVE_PATH / date_to_archive_format(cml_args.date) / 'primary_summary.txt'

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
    if cml_args.det is not None:
        # Parse detection file name
        det_file = pathlib.Path(cml_args.det)

        print(f"MODE: Save frames saved in {det_file.name}")

        # Get minute directory and list of image files
        minute,image_files = parse_det_file(det_file)
        _save_dark_subtracted_detec(cml_args.date, minute, det_file.stem, image_files)

    elif cml_args.time is not None:
        print(f"MODE: Saving specified minute {cml_args.time}")

        # Save specified minute directory
        minute = cml_args.time
        _save_dark_subtracted_minute(cml_args.date, minute)

    else:
        print("MODE: Saving minute with most stars detected")

        # Read starlist
        starlist = np.loadtxt(STARLIST_PATH, dtype=str, delimiter=',')

        # Get minute with most stars
        minute = starlist[np.argmax(starlist[:,1].astype(int)),0]
        _save_dark_subtracted_minute(cml_args.date, minute)