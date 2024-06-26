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
import gc
import sep
import re
import shutil
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
MINDIR_FORMAT  = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
BARE_FORMAT = '%Y-%m-%d_%H%M%S_%f'

# Processing parameters
POOL_SIZE  = multiprocessing.cpu_count() - 2  # cores to use for multiprocessing
CHUNK_SIZE = 10  # images to process at once
SEC_TO_SAVE = 1.  # seconds on either side of event to save

# Photometry parameters
EXCLUDE_IMAGES = 1
STARFINDING_STACK = 9
APERTURE_RADIUS = 3.0
DEFAULT_STAR_RADIUS = 2.0
EXPOSURE_TIME = 0.025
STAR_DETEC_THRESH = 4.0

# Drift limits
DRIFT_TOLERANCE = 1.0  # maximum drift (in px/s) without throwing an error
DRIFT_THRESHOLD = 0.025  # maximum drif (in px/s) without applying a drift correction

# Verbose print
VERBOSE = False
verboseprint = print if VERBOSE else lambda *a, **k: None


#--------------------------------functions------------------------------------#

###########################
## Lightcurves
###########################

def generateLightcurve(minute_dir, timestamp, master_dark_list, 
                       obsdate, xy, drift_correction=False):

    ## Setup image pathing ## 

    # Get the dark image data
    archive_dir = ARCHIVE_PATH / hyphonateDate(obsdate)
    dark = cir.chooseDark(minute_dir, master_dark_list, datetime.strptime(obsdate,OBSDATE_FORMAT))

    # Get image dimensions & number of images
    image_paths = sorted(minute_dir.glob('*.rcd'))
    x_length, y_length, num_images = cir.getSizeRCD(image_paths)

    # Calculate the number of frames to skip to line up with the directory
    timestamp = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    dir_timestamp = datetime.strptime(minute_dir.name+'000', MINDIR_FORMAT)
    timediff = (timestamp - dir_timestamp).total_seconds()
    central_frame = timediff // EXPOSURE_TIME
    
    # Read the central frame to get the timestamp
    _, central_time = cir.importFramesRCD(image_paths, central_frame, 1, dark)
    verboseprint(f"CENTRAL_TIME: {central_time}")

    # Calculate the number of frames to skip to line up with the timestamp
    # Since FLI returns nanoseconds, we strip the last 3 digits from central_timestamp
    timediff = (timestamp - datetime.strptime(central_time[0][:-3], TIMESTAMP_FORMAT)).total_seconds()
    central_frame += timediff // EXPOSURE_TIME

    # Determine frames to save
    min_frame = int(central_frame - 1 - (SEC_TO_SAVE // EXPOSURE_TIME))
    max_frame = int(central_frame + 1 + (SEC_TO_SAVE // EXPOSURE_TIME))
    if min_frame <= EXCLUDE_IMAGES:
        min_frame = EXCLUDE_IMAGES + 1
    if max_frame >= num_images:
        max_frame = None
    print(f"Saving frames from {min_frame} to {max_frame if max_frame is not None else num_images} " +\
          f"with central frame {central_frame}.")

    # Get image paths to process
    lightcurve_paths = image_paths[min_frame:max_frame]
    num_frames = len(lightcurve_paths)


    ## Star identification ##

    # Get the star list from the archive (if not found, fail gracefully)
    # TODO: Add a way to generate a star list if it doesn't exist
    star_list_path = list(archive_dir.glob(minute_dir.name + '_*sig_pos.npy'))
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

    verboseprint(f"First time: {first_time}")

    # Refine star positions for first and final frames
    # TODO: Fix the arguments here!!!
    first_positions, _ = cp.refineCentroid(first_frame, first_time[0], initial_positions, mean_guass_sigma)
    final_positions, _ = cp.refineCentroid(final_frame, final_time[0], initial_positions, mean_guass_sigma)

    # Calculate median drift rate [px/s] in x and y over the minute
    x_drift, y_drift = cp.averageDrift(first_positions, final_positions, 
                                       Time(first_time, precision=9).unix, 
                                       Time(final_time, precision=9).unix)

    # Check if drift is within tolerance and if a correction is needed
    if (abs(x_drift) > DRIFT_TOLERANCE) or (abs(y_drift) > DRIFT_TOLERANCE):
        print(f"ERROR: Drift rate is too high for {minute_dir.name}!")
        print(f"         Drifted by ({x_drift},{y_drift})")
        return None
    elif (abs(x_drift) > DRIFT_THRESHOLD) or (abs(y_drift) > DRIFT_THRESHOLD):
        print(f"WARNING: Drift detected in {minute_dir.name}!")
        print(f"         Drifted by ({x_drift},{y_drift})")
        drift = True
    else:
        print("No drift detected :)")
        drift = False

    
    ## Generate WCS transformations ##

    star_X,star_Y = xy
    print(f"(StarX, StarY) = ({star_X},{star_Y})")

    # Determine if star is visible in field of view and matches an existing star
    if (star_X < 0) or (star_X > x_length) or (star_Y < 0) or (star_Y > y_length):
        print(f"WARNING: Star is not visible in {minute_dir.name}")
        return None
    elif np.all(abs(first_positions[:,0] - star_X) > APERTURE_RADIUS) or \
         np.all(abs(first_positions[:,1] - star_Y) > APERTURE_RADIUS):
        print(f"WARNING: Star was not found by SEP in {minute_dir.name}")
    else:
        #target_star_index_x = np.where(abs(first_positions[:,0] - star_X) < APERTURE_RADIUS)
        #target_star_index_y = np.where(abs(first_positions[:,1] - star_Y) < APERTURE_RADIUS)
        print(f"Star has been found by SEP in {minute_dir.name}")

    
    ## Generate lightcurve ##

    # Initialize lightcurve container
    # [0] x coorinate; [1] y coordinate; [2] flux; [3] frame time
    star_data = np.empty((num_frames+1, 1, 4), dtype=np.float64)
    print(f"num_images = {num_frames}")

    # Seed lightcurve with first frame data
    star_initial_flux = sep.sum_circle(first_frame, [star_X], [star_Y], [APERTURE_RADIUS])
    star_data[0] = [[star_X, star_Y, star_initial_flux[0][0], Time(first_time[0], precision=9).unix]]

    # Loop through each frame and calculate flux
    header_times = [first_time[0]]
    bkg_medians  = []
    if drift: # If drift correction is needed

        print("\nGenerating lightcurve with drift correction...")

        for i in range(num_frames):
            # Raw frame data
            image_data,image_time = cir.importFramesRCD(lightcurve_paths,i,1,dark)

            # Analyze data
            #print("STARDATA: ",star_data[i])
            star_data[i+1] = cp.timeEvolve(image_data,
                                           deepcopy(star_data[i]),
                                           image_time[0],
                                           APERTURE_RADIUS,
                                           1,
                                           (x_length,y_length),
                                           (x_drift,y_drift))
            header_times = header_times + image_time
            bkg_medians.append(np.median(image_data))

    else: # if drift correction is not needed

        print("\nGenerating lightcurve without drift correction...")

        for i in range(num_frames):
            # Raw frame data
            image_data,image_time = cir.importFramesRCD(lightcurve_paths,i,1,dark)

            # Analyze data
            #print("STARDATA: ",star_data[i])
            star_data[i+1] = cp.getStationaryFlux(image_data.reshape(1,*image_data.shape),
                                                  deepcopy(star_data[i]),
                                                  image_time,
                                                  APERTURE_RADIUS,
                                                  1,
                                                  (x_length,y_length))
            header_times = header_times + image_time
            bkg_medians.append(np.median(image_data))


    # Return relevant data, optionally background median
    # Removes the zeroeth frame data used to seed the star_data
    #return lightcurve_paths, star_data, header_times, np.median(bkg_medians)
    return lightcurve_paths, star_data[1:], header_times[1:]


def saveLightcurve(lightcurve_paths, star_data, header_times,
                   obsdate, radec, xy, secondary_save_path=None):

    print(f"\n# Write lightcurve to file #")

    ## Analyze lightcurve ##

    # Get statistical parameters of lightcurve
    lightcurve_flux = star_data[:,0,2]
    lightcurve_mean = np.mean(lightcurve_flux)
    lightcurve_std  = np.std(lightcurve_flux)


    ## Save lightcurve data ##

    # Telescope name
    try:
        telescope = os.environ['COMPUTERNAME']
    except KeyError:
        telescope = "TEST"

    # Naming/indexing parameters
    center_index = len(lightcurve_paths)//2
    obsdate_hyphon = hyphonateDate(obsdate)

    center_timestamp = datetime.strptime(header_times[center_index][:-3], TIMESTAMP_FORMAT)
    timestamp_bare   = center_timestamp.strftime(BARE_FORMAT)
    
    field_name = (lightcurve_paths[center_index].name).split("_")[0]

    # Save names
    save_path = ARCHIVE_PATH / obsdate_hyphon
    save_file = save_path / "art_{}_{}.txt".format(timestamp_bare, telescope)
    print(f"Saving lightcurve to {save_file}")

    # Write header and data to detection file
    with open(save_file, 'w') as filehandle:
        # Write header
        print("Writing lightcurve header...")
        filehandle.write('#\n#\n#\n#\n')
        filehandle.write('#    Event File: %s\n' %(lightcurve_paths[center_index]))
        filehandle.write('#    Star Coords: %f %f\n' %(xy[0], xy[1]))
        filehandle.write('#    RA Dec Coords: %f %f\n' %(radec[0], radec[1]))
        filehandle.write('#    DATE-OBS: %s\n' %(header_times[center_index]))
        filehandle.write('#    Telescope: %s\n' %(telescope))
        filehandle.write('#    Field: %s\n' %(field_name))
        filehandle.write('#    significance: ARTIFICAL\n')
        filehandle.write('#    Raw lightcurve std: %.4f\n' %(lightcurve_std))
        filehandle.write('#    Raw lightcurve mean: %.4f\n' %(lightcurve_mean))
        filehandle.write('#    Convolution background std: NONE\n')
        filehandle.write('#    Convolution background mean: NONE\n')
        filehandle.write('#    Convolution minimal value: NONE\n')
        filehandle.write('#\n#\n')
        filehandle.write('#filename     time      flux     conv_flux\n')

        # Write data
        print("Writing lightcurve data...")
        for i in range(len(lightcurve_paths)):
            seconds = header_times[i].split(":")[-1]
            filehandle.write(f"{lightcurve_paths[i]} {seconds} {star_data[i,0,2]} N/A\n")
    
    # Save secondary file if requested
    if secondary_save_path is not None:
        print(f"Saving secondary lightcurve to {secondary_save_path}")
        shutil.copy(save_file, secondary_save_path)

    # Completed
    print(f"Finished writing.")


###########################
## Directory/Timestamp Manipulation
###########################

def findMinute(obsdate, timestamp):

    # Get all minute directories for the given date
    obs_path = DATA_PATH / obsdate
    if not obs_path.exists():
        print(f"ERROR: {obs_path} does not exist")
        return None
    
    # Define the minute directories and their timestamps
    minute_dirs    = [item for item in obs_path.iterdir() if item.is_dir()]
    dir_timestamps = [datetime.strptime(item.name+'000', MINDIR_FORMAT)
                        for item in minute_dirs if item.name != 'Dark']

    # Find the minute directory that contains the timestamp
    timestamp = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    for dir_timestamp in dir_timestamps:
        if dir_timestamp <= timestamp < dir_timestamp + timedelta(minutes=1):
            target_dir = minute_dirs[dir_timestamps.index(dir_timestamp)]
            verboseprint(f"TARGET: {target_dir.name}")
            return target_dir
        

def improveCentralFrame(target_timestamp, guessed_timestamp):

    # Convert the timestamps to datetime objects
    target_dt  = datetime.strptime(target_timestamp, TIMESTAMP_FORMAT)
    guessed_dt = datetime.strptime(guessed_timestamp, TIMESTAMP_FORMAT)

    # Calculate the difference in seconds
    timediff = (target_dt - guessed_dt).total_seconds()
    num_frames = int(timediff // EXPOSURE_TIME)

    # Return the number of frames to skip
    return num_frames
        

def findMatchedDir(obsdate, timestamp):

    # Set Green's basedir
    if os.environ['COMPUTERNAME'] == 'GREENBIRD':
        green_path = BASE_PATH
    else:
        green_path = pathlib.Path("G:/")

    # Path to matched directory
    matched_dir = green_path / "ColibriArchive" / hyphonateDate(obsdate) / "matched"

    # Convert the timestamp to the correct format
    timestamp = datetime.strptime(timestamp, TIMESTAMP_FORMAT)
    timestamp = timestamp.strftime(BARE_FORMAT)

    # Find the matched directory
    match_dir = list(matched_dir.glob(f"{timestamp}*"))
    if len(match_dir) == 0:
        print(f"ERROR: No matched directory found for {timestamp}")
        return None
    elif len(match_dir) > 1:
        print(f"WARNING: Multiple matched directories found for {timestamp}:")
        print(f"       {match_dir}")
        print(f"\nUsing {match_dir[0]}...")
        
        return match_dir[0]
    else:
        return match_dir[0]


###########################
## Misc Functions
###########################

def reversePixelMapping(minute_dir, obsdate, RA, DEC):

    # Get the list of medstacked images
    # TODO: Add a way to generate a medstack if it doesn't exist
    obsdate = hyphonateDate(obsdate)
    medstack_list_path = (ARCHIVE_PATH / str(obsdate)).glob('*_medstacked.fits')

    # Get the WCS transformation
    transform = getTransform(minute_dir.name, medstack_list_path, {})

    # Find the coordinates of the stars in the medstack 
    star_wcs = getRAdec.getXYSingle(transform, (RA, DEC))
    star_X   = star_wcs[0]
    star_Y   = star_wcs[1]

    return star_X, star_Y


def hyphonateDate(obsdate):

    # Convert the date to a datetime object
    obsdate = datetime.strptime(obsdate, OBSDATE_FORMAT)

    # Convert the date to a hyphonated string
    obsdate = obsdate.strftime('%Y-%m-%d')

    return obsdate


#------------------------------------main-------------------------------------#


def main(obsdate, timestamp, radec, matched_dir=None):

    # Trim timestamp as necessary
    if len(timestamp) > 26:
        timestamp = timestamp[:26]
        print(f"ALERT: Trimmed timestamp to {timestamp}")


###########################
## Generate Lightcurve
###########################

    # Find which minute we expect to build the lightcurve in
    minute_dir = findMinute(obsdate,timestamp)

    # Check that this passed
    if minute_dir is None:
        raise NotADirectoryError("ERROR: No valid minute directory found.")

    # Generate master dark set
    obs_archive = ARCHIVE_PATH / hyphonateDate(obsdate)
    mdark_path  = obs_archive / 'masterDarks'
    mdark_array = [(datetime.strptime(dark.name[:-5]+'000',MINDIR_FORMAT), dark) for
                   dark in mdark_path.iterdir()]
    mdark_array = np.array(mdark_array)

    # Get XY pixel coordinates for star of interest from WCS transform
    star_XY = reversePixelMapping(minute_dir, obsdate, radec[0], radec[1])

    # Generate lightcurve
    lightcurve = generateLightcurve(minute_dir, timestamp, mdark_array,
                                    obsdate,star_XY)
    if lightcurve is None:
        print("ERROR: Failed to generate lightcurve.")
        sys.exit()
    else:
        lightcurve_paths,star_data,header_times = lightcurve

    # Save lightcurve
    saveLightcurve(lightcurve_paths, star_data, header_times,
                   obsdate, radec, star_XY, secondary_save_path=matched_dir)



#----------------------------------standalone---------------------------------#


if __name__ == '__main__':


###########################
## Argument Parser Setup
###########################


    # Generate argument parser
    arg_parser = argparse.ArgumentParser(description="Force generate a lightcurve from raw images",
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    # Available argument functionality
    arg_parser.add_argument('date', help='Observation date (YYYY/MM/DD) of data to be processed.')
    arg_parser.add_argument('timestamp' , help='Timestamp of 2-telescope event\'s central peak.')
    arg_parser.add_argument('radec', help='RA and Dec of the 2-telescope event (in deg).',
                            nargs=2,type=float)
    arg_parser.add_argument('-m','--match_dir', help='Save to match directory with matching timestamp.',
                            action='store_true')
    arg_parser.add_argument('-v','--verbose', help='Print verbose output.',
                            action='store_true')

    # Process argparse list as useful variables
    cml_args = arg_parser.parse_args()

    # Assign cml args input as variables for readability
    obsdate = (cml_args.date).replace("/","")
    timestamp = cml_args.timestamp
    radec = cml_args.radec

    # Verbosity
    VERBOSE = cml_args.verbose
    verboseprint = print if VERBOSE else lambda *a, **k: None

    # If matched_dir is set, set the secondary save path to the matched directory
    # Find that directory
    if cml_args.match_dir:
        match_dir = findMatchedDir(obsdate, timestamp)
    else:
        match_dir = None

    # Run main
    main(obsdate, timestamp, radec, matched_dir=match_dir)
    verboseprint("Verbose")