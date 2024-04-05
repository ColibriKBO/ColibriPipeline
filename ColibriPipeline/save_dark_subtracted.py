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
import imageio
import numpy as np
import pandas as pd
from datetime import datetime
from astropy.io import fits
from multiprocessing import cpu_count,Pool

# Custom Script Imports
import colibri_image_reader as cir


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

# GIF parameters
LOOPS_PER_SEC = 0.5


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

def get_dark_timestamps(dark_path):
    """Get timestamps of all dark images in directory for cir.chooseDark"""
    # Get list of all files in directory
    dark_images = dark_path.glob('*.fits')

    # Get timestamps
    timestamps = [(format_timestamp_to_datetime(image.name[:16]), image) 
                  for image in dark_images]

    return timestamps

def process_science_images(image_path_list, start_frame, dark_data, save_dir_path):
    """Read in science images one at a time and save dark-subtracted images as fits files"""
    #print(f'Processing image {start_frame+1} of {len(image_path_list)}')

    # Read in image and subtract dark
    subtracted_img,_ = cir.importFramesRCD(image_path_list, start_frame, 1, dark_data)

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


def compileImagesAsGIF(image_path_list, save_path, skip_frames=1):
    """Compile a list of images into a GIF"""
    
    # Check that the number of images is less than 100
    if len(image_path_list) > 100:
        raise ValueError('Cannot compile more than 100 images into GIF')
    
    # Read in every n images
    images_to_read = image_path_list[::1+skip_frames]
    images = [fits.getdata(image_path) for image_path in images_to_read]
    print(f'Compiling {len(images)} images into GIF...')

    # Save images as a looping GIF
    gif_duration = int(1000. / LOOPS_PER_SEC / len(images))
    imageio.mimsave(save_path, images, duration=gif_duration, loop=0)


#------------------------------------main-------------------------------------#

def _save_dark_subtracted_minute(date, minute, DATE_PATH, MBIAS_PATH):
    """Save dark-subtracted images from a minute directory as fits files
    *NOTE: This function must be called after main"""

    # Format date as datetime object
    obs_date = datetime.strptime(date, OBSDATE_FORMAT)

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

    # Select dark image
    dark_timestamps = get_dark_timestamps(MBIAS_PATH)
    best_dark = cir.chooseDark(MINUTE_PATH, np.array(dark_timestamps), obs_date)
    
    # Get list of all files in directory
    image_paths = sorted(MINUTE_PATH.glob('*.rcd')) 

    # Set up multiprocessing
    pool_size = cpu_count() - 2
    pool = Pool(pool_size)
    args = ((image_paths, i, best_dark, SAVE_PATH) for i in range(len(image_paths)))

    # Process images in parallel
    pool.starmap(process_science_images, args)

    # Close multiprocessing
    pool.close()
    pool.join()
    print('Done.')

    return SAVE_PATH


def _save_dark_subtracted_detec(date, minute, detec_str, file_list, DATE_PATH, MBIAS_PATH):
    """Save dark-subtracted images from a minute directory as fits files
    *NOTE: This function must be called after main"""

    # Format date as datetime object
    obs_date = datetime.strptime(date, OBSDATE_FORMAT)

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

    # Select dark image
    dark_timestamps = get_dark_timestamps(MBIAS_PATH)
    best_dark = cir.chooseDark(MINUTE_PATH, np.array(dark_timestamps), obs_date)
    
    # Get list of all files in directory
    image_paths = [pathlib.Path(file) for file in file_list]

    # Set up multiprocessing
    pool_size = cpu_count() - 2
    pool = Pool(pool_size)
    args = ((image_paths, i, best_dark, SAVE_PATH) for i in range(len(image_paths)))

    # Process images in parallel
    pool.starmap(process_science_images, args)

    # Close multiprocessing
    pool.close()
    pool.join()
    print('Done.')

    return SAVE_PATH


def main(date, time=None, det=None, gif=False, skip=False):

    # Make sure only one or the other option is used
    if (time is not None) and (det is not None):
        raise argparse.ArgumentTypeError('Cannot use both -t and -d options')            

    # Relevant paths
    DATE_PATH = DATA_PATH / date
    MBIAS_PATH = ARCHIVE_PATH / date_to_archive_format(date) / 'masterDarks'
    STARLIST_PATH = ARCHIVE_PATH / date_to_archive_format(date) / 'primary_summary.txt'

    # Check if relevant directories exist
    if not DATE_PATH.exists():
        raise FileNotFoundError(f'Date directory {DATE_PATH} does not exist. No images to process.')
    if not MBIAS_PATH.exists():
        #TODO: Add option to create master darks
        raise FileNotFoundError(f'Master dark directory {MBIAS_PATH} does not exist. Cannot execute script.')
    if (not STARLIST_PATH.exists()) and (time is None):
        raise FileNotFoundError(f'Archive directory {STARLIST_PATH} does not exist' +\
                                 'and no time was specified. No images to process.')
    
    # Identify minute to save
    if skip is True:
        print("MODE: Skipping saving images as fits files")
        
        # Define save directory for other modes
        if det is not None:
            # Parse detection file name
            det_file = pathlib.Path(det)

            # Get minute directory and list of image files
            minute,image_files = parse_det_file(det_file)
            save_dir = IMGE_PATH / date / det_file.stem
        elif time is not None:
            # Save specified minute directory
            minute = time
            save_dir = IMGE_PATH / date / minute
        else:
            # Read starlist
            starlist = np.loadtxt(STARLIST_PATH, dtype=str, delimiter=',')

            # Get minute with most stars
            minute = starlist[np.argmax(starlist[:,1].astype(int)),0]
            save_dir = IMGE_PATH / date / minute


    elif det is not None:
        # Parse detection file name
        det_file = pathlib.Path(det)

        print(f"MODE: Save frames saved in {det_file.name}")

        # Get minute directory and list of image files
        minute,image_files = parse_det_file(det_file)
        save_dir = _save_dark_subtracted_detec(date, minute, det_file.stem, image_files, DATE_PATH, MBIAS_PATH)

    elif time is not None:
        print(f"MODE: Saving specified minute {time}")

        # Save specified minute directory
        minute = time
        save_dir = _save_dark_subtracted_minute(date, minute, DATE_PATH, MBIAS_PATH)

    else:
        print("MODE: Saving minute with most stars detected")

        # Read starlist
        starlist = np.loadtxt(STARLIST_PATH, dtype=str, delimiter=',')

        # Get minute with most stars
        minute = starlist[np.argmax(starlist[:,1].astype(int)),0]
        save_dir = _save_dark_subtracted_minute(date, minute, DATE_PATH, MBIAS_PATH)


    # Save images as GIF in specified directory
    if gif is True:

        print(f"MODE: Saving images as GIF in {save_dir}")
        
        # If detection file is specified, only save 1/10 frames
        if cml_args.det is not None:
            GIF_SKIP_FRAMES = 9

        # Otherwise, save 1/1000 frames when saving an entire minute
        else:
            GIF_SKIP_FRAMES = 999

        # Get list of all files in directory
        image_paths = sorted(save_dir.glob('*.fits'))

        # Compile images into GIF
        gif_save_path = save_dir / f'{save_dir.name}.gif'
        compileImagesAsGIF(image_paths, gif_save_path, skip_frames=GIF_SKIP_FRAMES)


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
    arg_parser.add_argument('-s', '--skip', action='store_true',
                            help="Skip saving images as fits files. Default is False.")
    arg_parser.add_argument('-t', '--time', 
                            help="Timestamp of minute directory to save. Default is minute with most stars.")
    arg_parser.add_argument('-d','--det',
                            help="Detection file to save. Incompatible with -t option")
    arg_parser.add_argument('-g', '--gif', action='store_true',
                            help="Save images as a GIF. Default is False.")

    # Process argparse list as useful variables
    cml_args = arg_parser.parse_args()


###########################
## Setup and Main
###########################

    # Run main
    main(cml_args.date, cml_args.time, cml_args.det, cml_args.gif, cml_args.skip)