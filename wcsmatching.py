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
import multiprocessing
import gc
import sep
import re
import itertools
import numpy as np
import time as timer
from pathlib import Path
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
BASE_PATH = Path('D:/')
DATA_PATH = BASE_PATH / 'ColibriData'
IMGE_PATH = BASE_PATH / 'ColibriImages'
ARCHIVE_PATH = BASE_PATH / 'ColibriArchive'

# Timestamp format
OBSDATE_FORMAT = '%Y%m%d'
MINDIR_FORMAT  = '%Y%m%d_%H.%M.%S.%f'
TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%S.%f'
BARE_FORMAT = '%Y-%m-%d_%H%M%S_%f'


#----------------------------------class--------------------------------------#

class Telescope:

    def __init__(self, name, base_path, obs_date):

        # Telescope identifiers
        print(f"Initializing {name}...")
        self.name = name

        # Error logging
        self.errors = []

        # Path variables
        self.base_path = Path(base_path)
        self.obs_archive = self.base_path / "ColibriArchive" / hyphonateDate(obs_date)
        self.npy_file_list = sorted(self.obs_archive.glob('*pos.npy'))
        self.medstack_file_list = sorted(self.obs_archive.glob('*medstacked.fits'))

        # Check that star tables exist
        if (not self.obs_archive.exists()) or (len(self.npy_file_list) == 0):
            self.addError(f"ERROR: No star tables exist for {name} on {obs_date}!")
            self.star_tables = False
        else:
            self.star_tables = True


    def addError(self, error_msg):

        self.errors.append(error_msg)
        print(error_msg)


    def genDatetimeList(self):

        # Initialize datetime list
        self.dt_list = []
        self.dt_dict = {}

        # Confirm that star tables exist
        if self.star_tables is False:
            #return self.dt_list
            return []
        
        # Iterate through all npy_files and get the datetime str from the name
        for npy_file in self.npy_file_list:
            #self.dt_list.append(regexNPYName(npy_file.name)[1])
            self.dt_dict[regexNPYName(npy_file.name)[1]] = npy_file.name
        else:
            #return sorted(self.dt_list)
            return sorted(list(self.dt_dict.keys()))



#--------------------------------functions------------------------------------#

def npyFileWCSUpdate(npy_file_list, medstack_file_list):

    # Iterate through npy files and generate a WCS transformation to write
    # back into the WCS file
    for npy_file in npy_file_list:

        print(f"Processing {npy_file.name}...")

        # Read in data from npy file
        # Format x | y | half-light radius
        star_table = np.load(npy_file)

        # If npy file already contains ra/dec information
        if star_table.shape[1] == 4:
            print("  Already proccessed.")
            continue

        # Get minute string from filename using regex
        timestamp,_ = regexNPYName(npy_file.name)
        print(f"  -> Associated timestamp: {timestamp}")

        # Generate WCS transformation and the RA/Dec
        transform  = getTransform(timestamp, medstack_file_list, {})
        star_radec = getRAdec(transform, npy_file)
        print("  Successfully generated RA/DEC.")

        # Save the array of star positions as an .npy file again
        # Format: x  |  y  | ra | dec :: No half-light radius data saved!
        npy_file.unlink()
        np.save(npy_file, star_radec)


def pairMinutes(minute_list1, minute_list2, spacing=33):
    """
    
    """

    # Cartesian product of the two sorted minute lists
    minute_pairs = list(itertools.product(minute_list1,minute_list2))
    minute_pairs = np.array(minute_pairs, dtype=object)

    # Find time difference between two times
    time_diff = np.array([abs((time_tuple[0] - time_tuple[1]).total_seconds()) \
                          for time_tuple in minute_pairs],
                          dtype=object)

    # Identify minute pairs within our timing tolerance
    valid_pair_inds = np.where(time_diff < spacing)[0]

    # Of the timestamps identified, find unique timestamps.
    # In the case of duplicates, default to earliest time in minute_list1 then minute_list2.
    _,unique1_inds = np.unique(minute_pairs[[valid_pair_inds],0],
                               return_index=True)
    _,unique2_inds = np.unique(minute_pairs[[valid_pair_inds[unique1_inds]],1],
                               return_index=True)
    paired_inds = valid_pair_inds[unique1_inds][unique2_inds]

    # Return only valid, unique minute pairs that satisfy our timing tolerance
    print(f"Minutes successfully paired. {len(paired_inds)} minute pairs found.")
    return minute_pairs[paired_inds]



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
    # Return two arrays of indices
    close_star_inds = np.where(hypot < tolerance)
    print(f"Stars successfully matched. {len(close_star_inds[0])} matched stars found.")
    return close_star_inds[0],close_star_inds[1]


def regexNPYName(npy_file_name):

    # Strip significance and file ext from filename
    time_str = npy_file_name[:21] 

    # Convert to datetime object
    time_obj = datetime.strptime(time_str + "000", MINDIR_FORMAT)

    return time_str, time_obj


def hyphonateDate(obsdate):

    # Convert the date to a datetime object
    obsdate = datetime.strptime(obsdate, OBSDATE_FORMAT)

    # Convert the date to a hyphonated string
    obsdate = obsdate.strftime('%Y-%m-%d')

    return obsdate


#------------------------------------main-------------------------------------#

if __name__ == '__main__':


###########################
## Argument Parser Setup
###########################

    # Generate argument parser
    description = "Generate WCS transform of observed stars and match to other telescopes"
    arg_parser = argparse.ArgumentParser(description=description,
                                         formatter_class=argparse.RawTextHelpFormatter)
    
    # Available argument functionality
    arg_parser.add_argument('date', help='Observation date (YYYYMMDD) of data to be processed.')
    arg_parser.add_argument('-a', '--all', help='Process for all telescopes. Blue only.', 
                            action='store_true')
    arg_parser.add_argument('-m', '--match', help='Match stars between telescopes. Blue only.', 
                            action='store_true')

    # Process argparse list as useful variables
    cml_args  = arg_parser.parse_args()
    obsdate   = cml_args.date
    proc_all  = cml_args.all
    match_all = cml_args.match

    # Environment variable to ensure BLUEBIRD is used when required
    current_name = os.environ['COMPUTERNAME']


###########################
## WCS Transforms
###########################

    print("\n## WCS Transformations ##")

    # Single-telescope mode
    if proc_all is False:
        # Process only for the current machine
        print("Processing current machine...\n")
        current = Telescope(current_name, BASE_PATH, obsdate)
        if (current.star_tables) and (current.medstack_file_list != []):
            npyFileWCSUpdate(current.npy_file_list,
                             current.medstack_file_list)
        else:
            print("ERROR: Could not process for current telescope!")

    # Multi-telescope mode (assumed to run only on Blue)
    elif current_name != "BLUEBIRD":
        print(f"ERROR: Multi-telescope mode is only designed to run on Blue!")
        sys.exit()

    else:
        # Initialize telescope classes
        Red   = Telescope("RED", "R:", obsdate)
        Green = Telescope("GREEN", "G:", obsdate)
        Blue  = Telescope("BLUE", "D:", obsdate)

        # Check that at least one star table set exists
        if not (Red.star_tables | Green.star_tables | Blue.star_tables):
            print(f"ERROR: No star tables exist for {obsdate}!")
            sys.exit()
        
        # Process for each of the machines
        for machine in (Red,Green,Blue):
            if (machine.star_tables) and (machine.medstack_file_list != []):
                print(f"Processing star tables for {machine.name}...")
                npyFileWCSUpdate(machine.npy_file_list,
                                 machine.medstack_file_list)
            else:
                print(f"Skipping {machine.name}...")


###########################
## Matching Stars
###########################

    # If not matching or not on Blue, we are done at this point
    if match_all is False:
        print("\nDone!")
        sys.exit()
    elif current_name != "BLUEBIRD":
        print("ERROR: Matching is only designed to run on Blue!")
        sys.exit()

    # Initialize telescope classes if that wasn't done in the last part
    if (proc_all is False) and (match_all):
        print("Trusting that all telescopes have run astrometry...")

        # Initialize telescope classes
        Red   = Telescope("RED", "R:", obsdate)
        Green = Telescope("GREEN", "G:", obsdate)
        Blue  = Telescope("BLUE", "D:", obsdate)


    ## Minute matching ## 

    print("\n## Minute Matching ##")

    # Generate datetime objects for all telescopes
    for machine in (Red,Green,Blue):
        machine.genDatetimeList()

    """ Currently revising to use dictionary
    # Try to find valid timestamp pairs between telescopes
    RB_pairs = pairMinutes(Red.dt_list, Blue.dt_list)
    BG_pairs = pairMinutes(Blue.dt_list, Green.dt_list)
    RG_pairs = pairMinutes(Red.dt_list, Green.dt_list)
    """

    # Try to find valid timestamp pairs between telescopes
    #print(sorted(list(Red.dt_dict.keys())))
    RG_pairs = pairMinutes(sorted(list(Red.dt_dict.keys())),
                           sorted(list(Green.dt_dict.keys())))
    GB_pairs = pairMinutes(sorted(list(Green.dt_dict.keys())),
                           sorted(list(Blue.dt_dict.keys())))
    RB_pairs = pairMinutes(sorted(list(Red.dt_dict.keys())),
                           sorted(list(Blue.dt_dict.keys())))
    
    print(f"RG = {len(RG_pairs)}\nGB = {len(GB_pairs)}\nRB = {len(RB_pairs)}")

    # If we found pairs between RB and BG, then we search for pairs between all 3
    if (RG_pairs.size == 0) or (GB_pairs.size == 0) or (RB_pairs.size == 0):
        print("Failed to match minutes for all 3 machines!")
        sys.exit()
    else:
        shared_matches, GB_inds, RB_inds = np.intersect1d(GB_pairs[:,1],
                                                          RB_pairs[:,1],
                                                          assume_unique=True,
                                                          return_indices=True)
        
        # Join shared matches into triples
        time_triplets = np.vstack((RB_pairs[RB_inds,0],
                                   GB_pairs[GB_inds,0],
                                   GB_pairs[GB_inds,1]))
        time_triplets = time_triplets.transpose()

        print(f"RGB = {len(time_triplets)}")


    ## Star matching ##

    print("\n## Star Matching ##")

    # Match stars from each telescope's star lists
    star_minutes = 0
    for minute in time_triplets:

        # Get npy file from each telescope
        Red_file   = Red.obs_archive / Red.dt_dict[minute[0]]
        Green_file = Green.obs_archive / Green.dt_dict[minute[1]]
        Blue_file  = Blue.obs_archive / Blue.dt_dict[minute[2]]

        # Get ra/dec from each npy file
        # TODO: WCS mapping if not already done
        try:
            Red_stars   = np.load(Red_file)[:,[2,3]]
        except IndexError:
            print(f"ERROR: {Red_file} has not been updated!")
            continue
        try:
            Green_stars = np.load(Green_file)[:,[2,3]]
        except IndexError:
            print(f"ERROR: {Green_file} has not been updated!")
            continue
        try:
            Blue_stars  = np.load(Blue_file)[:,[2,3]]
        except IndexError:
            print(f"ERROR: {Blue_file} has not been updated!")
            continue


        # Match stars between 2 and then 3
        BR_matched = sharedStars(Blue_stars, Red_stars)
        BG_matched = sharedStars(Blue_stars, Green_stars)
        shared_stars = np.intersect1d(BR_matched[0], BG_matched[0])

        star_minutes += len(shared_stars)
    
    print(f"\nDone star matching! {star_minutes/60.} star-hours detected.")